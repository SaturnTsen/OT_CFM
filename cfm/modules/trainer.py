# universal_trainer.py
# Generalized Conditional Flow Matching trainer (all are flow models).
# Supports Table-1 paths without any "diffusion" naming.

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Any
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import ot
Tensor = torch.Tensor

@torch.no_grad()
def independent_coupling(x0: Tensor, x1: Tensor, **_) -> Tuple[Tensor, Tensor]:
    return x0, x1

@torch.no_grad()
def ot_coupling_minibatch(x0: Tensor, x1: Tensor, num_samples: Optional[int] = None) -> Tuple[Tensor, Tensor]:
    sol = ot.solve_sample(x0, x1)
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
    eps: float = 0.1,
    num_samples: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    n = x0.size(0)
    m = x1.size(0)
    a = torch.full((n,), 1.0 / n, device=x0.device, dtype=torch.float32)
    b = torch.full((m,), 1.0 / m, device=x0.device, dtype=torch.float32)

    C = torch.cdist(x0, x1, p=2).pow(2)
    P = ot.sinkhorn(a, b, C, eps)
    if not torch.is_tensor(P):
        P = torch.as_tensor(P, device=x0.device, dtype=torch.float32)

    if num_samples is None:
        num_samples = n

    P_flat = (P / (P.sum() + 1e-12)).reshape(-1)
    idx = torch.multinomial(P_flat, num_samples, replacement=True)
    i = idx // m
    j = idx % m
    return x0[i], x1[j]


# -----------------------------
# Variance perserving alpha schedule
# -----------------------------

@dataclass
class VPAlphaSchedule:
    """
    Provides alpha(s) for the VP path row in your table:
      mu_t = alpha_{1-t} x1
      sigma_t = sqrt(1 - alpha_{1-t}^2)

    Use a common smooth alpha(s)=exp(-0.5 ∫ beta) with linear beta(s).
    """
    beta_min: float = 0.1
    beta_max: float = 20.0

    def beta(self, s: Tensor) -> Tensor:
        return self.beta_min + (self.beta_max - self.beta_min) * s

    def integral_beta(self, s: Tensor) -> Tensor:
        b0, b1 = self.beta_min, self.beta_max
        return b0 * s + 0.5 * (b1 - b0) * s * s

    def alpha(self, s: Tensor) -> Tensor:
        return torch.exp(-0.5 * self.integral_beta(s))

    def d_alpha_ds(self, s: Tensor) -> Tensor:
        a = self.alpha(s)
        return a * (-0.5 * self.beta(s))


@dataclass
class VESigmaSchedule:
    """
    Provides sigma(s) for the VE path row in your table:
      mu_t = x1
      sigma_t = sigma_{1-t}

    Use sigma(s)=sigma_min*(sigma_max/sigma_min)^s.
    """
    sigma_min: float = 0.01
    sigma_max: float = 50.0

    def sigma(self, s: Tensor) -> Tensor:
        r = self.sigma_max / self.sigma_min
        return self.sigma_min * (r ** s)

    def d_sigma_ds(self, s: Tensor) -> Tensor:
        r = self.sigma_max / self.sigma_min
        return self.sigma(s) * math.log(r)


# -----------------------------
# Generalized probability path for FLOW matching
# -----------------------------

PathName = Literal[
    "ve", "vp", "flow_matching",
    "rectified_flow",
    "vp_stochastic_interpolant",
    "independent_cfm",
    "ot_cfm",
    "schrodinger_bridge_cfm",
]


@dataclass
class PathConfig:
    name: PathName

    # constant sigma used by Independent CFM / OT-CFM / (as base) SB-CFM
    sigma: float = 0.005

    # Flow Matching (Lipman): sigma_t = t*sigma_fm - t + 1
    sigma_fm: float = 1.0

    # VP/VE schedules (flow-style naming)
    vp_alpha: VPAlphaSchedule = VPAlphaSchedule()
    ve_sigma: VESigmaSchedule = VESigmaSchedule()

    # SB coupling: eps for sinkhorn approx ~ (sigma^2 * t(1-t)) * scale
    sb_eps_scale: float = 1.0

    # coupling sampling
    ot_num_samples: Optional[int] = None


class ProbabilityPath:
    """
    Define:
      x_t = mu_t(x0,x1,t) + sigma_t(t) * eps
      u_t = d/dt mu_t + d/dt sigma_t * eps
    Train v_theta(x_t, t) ~ u_t (generalized conditional flow matching).
    """

    def __init__(self, cfg: PathConfig):
        self.cfg = cfg

    def couple(self, x0: Tensor, x1: Tensor, tB1: Tensor) -> Tuple[Tensor, Tensor]:
        name = self.cfg.name

        if name in ("independent_cfm", "rectified_flow", "vp_stochastic_interpolant"):
            return independent_coupling(x0, x1)

        if name == "ot_cfm":
            return ot_coupling_minibatch(x0, x1, num_samples=self.cfg.ot_num_samples)

        if name == "schrodinger_bridge_cfm":
            # minibatch approximation: use mean t to set entropic regularization
            t_mean = float(tB1.mean().item())
            ent = self.cfg.sb_eps_scale * (self.cfg.sigma ** 2) * max(t_mean * (1.0 - t_mean), 1e-4)
            return sinkhorn_coupling_minibatch(x0, x1, eps=ent, num_samples=self.cfg.ot_num_samples)

        # ve/vp/flow_matching typically just use x1, keep x0/x1 as provided
        return x0, x1

    def mu_sigma_and_time_derivatives(
        self, x0: Tensor, x1: Tensor, tB1: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
          mu, sigma, dmu_dt, dsigma_dt
        sigma/dsigma_dt have shape (B,1) broadcastable.
        """
        name = self.cfg.name

        if name == "rectified_flow":
            mu = (1.0 - tB1) * x0 + tB1 * x1
            sigma = torch.zeros_like(tB1)
            dmu = (x1 - x0)
            dsigma = torch.zeros_like(tB1)
            return mu, sigma, dmu, dsigma

        if name in ("independent_cfm", "ot_cfm"):
            mu = (1.0 - tB1) * x0 + tB1 * x1
            sigma = torch.full_like(tB1, float(self.cfg.sigma))
            dmu = (x1 - x0)
            dsigma = torch.zeros_like(tB1)
            return mu, sigma, dmu, dsigma

        if name == "schrodinger_bridge_cfm":
            mu = (1.0 - tB1) * x0 + tB1 * x1
            sigma = float(self.cfg.sigma) * torch.sqrt(torch.clamp(tB1 * (1.0 - tB1), min=1e-12))
            denom = torch.sqrt(torch.clamp(tB1 * (1.0 - tB1), min=1e-12))
            dsigma = float(self.cfg.sigma) * (1.0 - 2.0 * tB1) / (2.0 * denom)
            dmu = (x1 - x0)
            return mu, sigma, dmu, dsigma

        if name == "vp_stochastic_interpolant":
            a = 0.5 * math.pi * tB1
            mu = torch.cos(a) * x0 + torch.sin(a) * x1
            sigma = torch.zeros_like(tB1)
            dmu = (0.5 * math.pi) * (-torch.sin(a) * x0 + torch.cos(a) * x1)
            dsigma = torch.zeros_like(tB1)
            return mu, sigma, dmu, dsigma

        if name == "flow_matching":
            mu = tB1 * x1
            sigma = tB1 * float(self.cfg.sigma_fm) - tB1 + 1.0
            dmu = x1
            dsigma = torch.full_like(tB1, float(self.cfg.sigma_fm) - 1.0)
            return mu, sigma, dmu, dsigma

        if name == "vp":
            s = 1.0 - tB1
            a = self.cfg.vp_alpha.alpha(s)
            da_ds = self.cfg.vp_alpha.d_alpha_ds(s)
            da_dt = -da_ds

            mu = a * x1
            sigma = torch.sqrt(torch.clamp(1.0 - a * a, min=1e-12))

            dmu = da_dt * x1
            dsigma = (-a * da_dt) / torch.clamp(sigma, min=1e-12)
            return mu, sigma, dmu, dsigma

        if name == "ve":
            s = 1.0 - tB1
            sig = self.cfg.ve_sigma.sigma(s)
            dsig_ds = self.cfg.ve_sigma.d_sigma_ds(s)
            dsig_dt = -dsig_ds

            mu = x1
            sigma = sig
            dmu = torch.zeros_like(x1)
            dsigma = dsig_dt
            return mu, sigma, dmu, dsigma

        raise ValueError(f"Unknown path: {name}")


# -----------------------------
# Universal trainer
# -----------------------------

@dataclass
class TrainerConfig:
    n_epochs: int = 100
    from_random_gaussian: bool = False
    t_shape: Literal["B1", "B"] = "B1"
    grad_clip_norm: Optional[float] = None


class UniversalCFMTrainer:
    """
    Dataset batch formats:
      - (x0, x1): source/target
      - (x1,) or x1: single distribution
        - set from_random_gaussian=True to synthesize x0 ~ N(0,I)
        - or keep False for VE/VP/FM paths that only need x1 (x0 will be dummy zeros)
    """

    def __init__(
        self,
        flow_model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        path_cfg: PathConfig,
        trainer_cfg: TrainerConfig = TrainerConfig(),
    ):
        self.flow_model = flow_model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.path = ProbabilityPath(path_cfg)
        self.trainer_cfg = trainer_cfg

    def _unpack_batch(self, batch: Any, device: torch.device) -> Tuple[Tensor, Tensor]:
        if isinstance(batch, (tuple, list)):
            if len(batch) >= 2:
                return batch[0].to(device), batch[1].to(device)
            if len(batch) == 1:
                x1 = batch[0].to(device)
            else:
                raise ValueError("Empty batch.")
        else:
            x1 = batch.to(device)

        if self.trainer_cfg.from_random_gaussian:
            x0 = torch.randn_like(x1)
            return x0, x1

        # for VE/VP/FM we only use x1; x0 is unused but required by interface
        return torch.zeros_like(x1), x1

    def train(self) -> torch.nn.Module:
        device = next(self.flow_model.parameters()).device
        pbar = tqdm(range(self.trainer_cfg.n_epochs))

        for epoch in pbar:
            total_loss = 0.0
            total_n = 0

            if hasattr(self.dataloader.dataset, "reshuffle"):
                self.dataloader.dataset.reshuffle()

            for batch in self.dataloader:
                x0, x1 = self._unpack_batch(batch, device)
                B = x1.size(0)

                if self.trainer_cfg.t_shape == "B1":
                    t = torch.rand(B, 1, device=device)
                else:
                    t = torch.rand(B, device=device)

                tB1 = t if t.ndim == 2 else t.view(B, 1)

                # coupling (OT / SB / independent)
                x0, x1 = self.path.couple(x0, x1, tB1)

                # construct training sample and target field
                mu, sigma, dmu_dt, dsigma_dt = self.path.mu_sigma_and_time_derivatives(x0, x1, tB1)
                eps = torch.randn_like(mu)
                x = mu + sigma * eps
                u = dmu_dt + dsigma_dt * eps  # generalized target velocity

                v = self.flow_model(x, t)
                loss = F.mse_loss(v, u)

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.trainer_cfg.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), self.trainer_cfg.grad_clip_norm)
                self.optimizer.step()

                total_loss += float(loss.item()) * B
                total_n += B

            avg = total_loss / max(total_n, 1)
            pbar.set_description(f"Epoch [{epoch+1}/{self.trainer_cfg.n_epochs}] Path={self.path.cfg.name} Loss={avg:.6f}")

        return self.flow_model


# -----------------------------
# Example usage
# -----------------------------
"""
# 1) Rectified Flow (deterministic line, needs (x0,x1) or Gaussian source)
path_cfg = PathConfig(name="rectified_flow")
trainer_cfg = TrainerConfig(n_epochs=200, from_random_gaussian=False)
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg, trainer_cfg)
trainer.train()

# 2) Independent CFM (stochastic line, constant sigma, needs (x0,x1))
path_cfg = PathConfig(name="independent_cfm", sigma=0.01)
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg)
trainer.train()

# 3) OT-CFM (needs (x0,x1) as two independent batches; coupling is minibatch OT)
path_cfg = PathConfig(name="ot_cfm", sigma=0.01, ot_num_samples=None)
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg)
trainer.train()

# 4) Schrödinger Bridge CFM (minibatch sinkhorn; coupling regularization ~ sigma^2 t(1-t))
path_cfg = PathConfig(name="schrodinger_bridge_cfm", sigma=0.01, sb_eps_scale=1.0)
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg)
trainer.train()

# 5) Flow Matching (Lipman) (uses x1 only; dataloader can yield x1; set from_random_gaussian=False)
path_cfg = PathConfig(name="flow_matching", sigma_fm=1.0)
trainer_cfg = TrainerConfig(n_epochs=200, from_random_gaussian=False)
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg, trainer_cfg)
trainer.train()

# 6) VE / VP paths (also x1-only training)
path_cfg = PathConfig(name="ve")
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg)
trainer.train()

path_cfg = PathConfig(name="vp")
trainer = UniversalCFMTrainer(model, dataloader, optimizer, path_cfg)
trainer.train()
"""