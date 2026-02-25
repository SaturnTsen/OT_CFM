from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any

import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import ot

Tensor = torch.Tensor

PathName = Literal[
    "flow_matching",              # Lipman et al. 2023
    "rectified_flow",             # Liu 2022
    "vp_stochastic_interpolant",  # Albergo & Vanden-Eijnden 2023 (σ_t=0)
    "independent_cfm",
    "ot_cfm",                     # Tong et al.
    "schrodinger_bridge_cfm",     # q(z)=π_{2σ^2}, σ_t=σ sqrt(t(1-t))
]


@torch.no_grad()
def ot_coupling_minibatch(x0: Tensor, x1: Tensor, num_samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    x0f, x1f = x0.reshape(x0.size(0), -1), x1.reshape(x1.size(0), -1)
    sol = ot.solve_sample(x0f, x1f)
    P = sol.plan
    if not torch.is_tensor(P):
        P = torch.as_tensor(P, device=x0.device, dtype=torch.float32)

    n, m = P.shape
    if num_samples is None:
        num_samples = n

    P_flat = (P / (P.sum() + 1e-12)).reshape(-1)
    idx = torch.multinomial(P_flat, num_samples, replacement=True)
    i = idx // m
    j = idx % m
    return x0[i], x1[j]

@torch.no_grad()
def sinkhorn_coupling_minibatch(
    x0: Tensor,
    x1: Tensor,
    eps: float,
    num_samples: Optional[int] = None,
    unbalanced: bool = True
) -> Tuple[Tensor, Tensor]:
    x0f, x1f = x0.reshape(x0.size(0), -1), x1.reshape(x1.size(0), -1)

    n, m = x0f.size(0), x1f.size(0)
    a = torch.full((n,), 1.0 / n, device=x0.device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=x0.device, dtype=torch.float32)
    C = ot.dist(x0, x1, metric='sqeuclidean', p=2, w=None, use_tensor=False)

    if unbalanced:
        P = ot.sinkhorn_unbalanced(a, b, C, eps, reg_m=1.0)
    else:
        P = ot.sinkhorn(a, b, C, eps)

    if num_samples is None:
        num_samples = n

    P_flat = (P / (P.sum() + 1e-12)).reshape(-1)
    idx = torch.multinomial(P_flat, num_samples, replacement=True)
    i = idx // m
    j = idx % m
    return x0[i], x1[j]

@dataclass
class PathConfig:
    name: PathName
    sigma: float = 0.005
    num_coupling_samples: Optional[int] = None


class ProbabilityPath:
    """
    Path:
      x_t = μ_t(x0,x1,t) + σ_t(t) * ε
      u_t = dμ_t/dt + dσ_t/dt * ε
    
    Objective:
      v_theta(x_t, t) ≈ u_t
    """

    def __init__(self, cfg: PathConfig):
        self.cfg = cfg

    def needs_pair(self) -> bool:
        return self.cfg.name in (
            "rectified_flow",
            "vp_stochastic_interpolant",
            "independent_cfm",
            "ot_cfm",
            "schrodinger_bridge_cfm",
        )

    def couple(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        name = self.cfg.name
        if name == "ot_cfm":
            return ot_coupling_minibatch(x0, x1, num_samples=self.cfg.num_coupling_samples)
        if name == "schrodinger_bridge_cfm":
            eps_reg = 2.0 * (float(self.cfg.sigma) ** 2)
            return sinkhorn_coupling_minibatch(x0, x1, eps=eps_reg, num_samples=self.cfg.num_coupling_samples)
        return x0, x1

    @torch.no_grad()
    def sample_xt_and_ut(self, x0: Optional[Tensor], x1: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
          x0, x1: (B, ...)
          t: (B, 1) in [0,1]
        Returns:
          x_t, u_t : both (B, ...)
        """
        name = self.cfg.name
        B = x1.shape[0]
        t = t.view(B, 1)

        if name == "rectified_flow":
            mu = (1.0 - t) * x0 + t * x1
            x_t = mu
            u_t = (x1 - x0)
            return x_t, u_t

        if name in ("independent_cfm", "ot_cfm"):
            mu = (1.0 - t) * x0 + t * x1
            eps = torch.randn_like(mu)
            x_t = mu + float(self.cfg.sigma) * eps
            u_t = (x1 - x0)  # dsigma/dt=0
            return x_t, u_t

        if name == "schrodinger_bridge_cfm":
            mu = (1.0 - t) * x0 + t * x1
            eps = torch.randn_like(mu)

            tt = torch.clamp(t * (1.0 - t), min=1e-12)
            sigma_t = float(self.cfg.sigma) * torch.sqrt(tt)
            # d/dt sqrt(t(1-t)) = (1-2t)/(2*sqrt(t(1-t)))
            d_sigma_t = float(self.cfg.sigma) * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(tt))

            x_t = mu + sigma_t * eps
            u_t = (x1 - x0) + d_sigma_t * eps
            return x_t, u_t

        if name == "vp_stochastic_interpolant":
            a = 0.5 * math.pi * t
            mu = torch.cos(a) * x0 + torch.sin(a) * x1
            x_t = mu
            u_t = (0.5 * math.pi) * (-torch.sin(a) * x0 + torch.cos(a) * x1)
            return x_t, u_t

        if name == "flow_matching":
            # q(x1), μ_t = t x1, σ_t = t*fm_sigma - t + 1
            mu = t * x1
            eps = torch.randn_like(mu)

            sigma_t = t * float(self.cfg.fm_sigma) - t + 1.0
            d_sigma_t = float(self.cfg.fm_sigma) - 1.0  # constant
            x_t = mu + sigma_t * eps
            u_t = x1 + d_sigma_t * eps
            return x_t, u_t

        raise ValueError(f"Unknown path: {name}")


class Trainer:
    """
    Trainer for flow-matching methods.
      - for flow_matching: batch = x1
      - for coupled methods: batch = (x0, x1)
    """

    def __init__(
        self,
        flow_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        path_cfg: PathConfig,
        n_epochs: int = 100,
        grad_clip_norm: Optional[float] = None,
        save_path: Optional[int] = None,
        save_period: int = 1
    ):
        self.flow_model = flow_model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.path = ProbabilityPath(path_cfg)
        self.n_epochs = n_epochs
        self.grad_clip_norm = grad_clip_norm
        self.save_period = save_period
        self.save_path = save_path

    def _unpack(self, batch: Any, device: torch.device) -> Tuple[Optional[Tensor], Tensor]:
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                return batch[0].to(device), batch[1].to(device)
            if len(batch) == 1:
                return None, batch[0].to(device)
            raise ValueError("Unsupported batch format.")
        return None, batch.to(device)

    def train(self) -> torch.nn.Module:
        device = next(self.flow_model.parameters()).device
        pbar = tqdm(range(self.n_epochs))

        for epoch in pbar:
            total_loss, total_n = 0.0, 0

            if hasattr(self.dataloader.dataset, "reshuffle"):
                self.dataloader.dataset.reshuffle()

            for batch in self.dataloader:

                # Get (x0, x1) from batch; x0 may be None for flow_matching
                x0, x1 = self._unpack(batch, device=device)

                # If the path requires coupling, sample (x0,x1) pairs accordingly
                if self.path.needs_pair():
                    if x0 is None: raise ValueError(f"Path {self.path.cfg.name} requires batch=(x0,x1).")
                    x0, x1 = self.path.couple(x0, x1)

                # Sample t and compute (x_t, u_t)
                B = x1.size(0)
                t = torch.rand(B, 1, device=device)
                x_t, u_t = self.path.sample_xt_and_ut(x0, x1, t)

                # Compute model output and loss
                v = self.flow_model(x_t, t)
                loss = F.mse_loss(v, u_t)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping if specified
                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.grad_clip_norm)
                
                self.optimizer.step()

                total_loss += float(loss.item()) * B
                total_n += B

            avg = total_loss / max(total_n, 1)
            pbar.set_description(f"Epoch [{epoch+1}/{self.n_epochs}] Path={self.path.cfg.name} Loss={avg:.6f}")

            if epoch % self.save_period == 0:
                torch.save(self.flow_model.state_dict(), self.save_path)

        return self.flow_model