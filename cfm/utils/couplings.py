import torch
import ot
from typing import Tuple, Optional

def independent_coupling(x_s: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x_s, x_t

def icfm_coupling(x_s: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return x_s, x_t

def ot_coupling_minibatch(x_s, x_t, num_samples=None):
    """
    Sample (x_s, x_t) pairs according to OT coupling plan.

    Args:
        x_s: (n, d) torch tensor
        x_t: (m, d) torch tensor
        num_samples: number of sampled pairs (default = n)

    Returns:
        x_s_sampled, x_t_sampled
    """
    sol = ot.solve_sample(x_s, x_t)
    P = sol.plan

    n, m = P.shape
    if num_samples is None:
        num_samples = n

    # flatten joint distribution
    P_flat = P.reshape(-1)

    # sample indices from joint
    idx = torch.multinomial(P_flat, num_samples, replacement=True)

    # recover (i, j)
    i = idx // m
    j = idx % m

    return x_s[i], x_t[j]

def sinkhorn_coupling_minibatch(x_s: torch.Tensor, x_t: torch.Tensor, eps: float = 0.1, num_samples: Optional[int] = None):
    n = x_s.size(0)
    m = x_t.size(0)
    a = torch.full((n,), 1.0 / n, device=x_s.device)
    b = torch.full((m,), 1.0 / m, device=x_s.device)
    C = torch.cdist(x_s, x_t, p=2) ** 2
    P = ot.sinkhorn(a, b, C, eps)
    if num_samples is None:
        num_samples = n
    P_flat = P.reshape(-1)
    idx = torch.multinomial(P_flat, num_samples, replacement=True)
    i = idx // m
    j = idx % m
    return x_s[i], x_t[j]