import ot
import torch
from typing import Tuple

def compute_w2_squared(x_s: torch.Tensor, x_t: torch.Tensor) -> float:
    n = x_s.size(0)
    m = x_t.size(0)
    a = torch.full((n,), 1.0 / n, device=x_s.device)
    b = torch.full((m,), 1.0 / m, device=x_t.device)
    M = ot.dist(x_s, x_t, metric='euclidean') ** 2
    W2_squared = ot.emd2(a, b, M)
    return W2_squared.item()

def compute_path_energy(model: torch.nn.Module, x0: torch.Tensor, n_steps: int = 200) -> Tuple[float, torch.Tensor]
    device = next(model.parameters()).device
    x = x0.to(device)
    dt = 1.0 / n_steps
    energy = 0.0
    t_vals = torch.linspace(0, 1, steps=n_steps, device=device)
    with torch.no_grad():
        for t in t_vals:
            t_batch = t * torch.ones(x.size(0), 1, device=device)
            v = model(t_batch, x)
            energy += (v.pow(2).sum(dim=1).mean()) * dt
            x = x + v * dt
    return energy, x

def compute_w2_npe(model: torch.nn.Module, x0: torch.Tensor, x1: torch.Tensor, n_steps: int = 200) -> Tuple[float, float]:
    energy, x_out = compute_path_energy(model, x0, n_steps)
    energy = float(energy)
    w2_sq = compute_w2_squared(x0, x1)
    npe = (energy - w2_sq) / w2_sq
    w2_sq_xt_xout = compute_w2_squared(x_out, x1)
    return w2_sq_xt_xout, npe