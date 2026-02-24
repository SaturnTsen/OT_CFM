"""High-level experiment utilities for CFM models.

This module exposes helpers that were previously contained in standalone
scripts (run_baseline.py and evaluate_npe.py).  It provides:

* toy data samplers
* coupling functions (fm, icfm, ot, sb)
* training wrapper `train_baseline` that returns a trained model
* evaluation routines for path energy / W2 / NPE

The scripts still exist for command-line use, but they now import these
functions so that notebooks can also reuse them.
"""

import math
from typing import Callable, Optional, Tuple

import torch
import ot
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_circles, make_classification, make_moons
from .couplings import *

from .trainer import Trainer, sample_from_ot_coupling
from ..modules.simple_flow import SimpleFlowModel

def sample_toy(dataset_type, n_samples=1000, noise=0.1):
    if dataset_type == 'moons':
        X, _ = make_moons(n_samples=n_samples, noise=noise)
    elif dataset_type == 'circles':
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
        # normalize
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    elif dataset_type == 'classification':
        X, _ = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1)
    elif dataset_type == '5gaussians':
        centers = [[math.cos(2 * math.pi * i / 5), math.sin(2 * math.pi * i / 5)] for i in range(5)]
        X, _ = make_classification(n_samples=n_samples, n_features=2, n_informative=2,
                                   n_redundant=0, n_clusters_per_class=1, n_classes=5,
                                   class_sep=2.0, centers=centers)
    else:
        raise ValueError("Unknown dataset type")
    return X
    
COUPLING_DICT = {
    'fm': independent_coupling,
    'icfm': icfm_coupling,
    'ot': ot_coupling_minibatch,
    'sb': sinkhorn_coupling_minibatch,
}

def train_baseline(
    method: str,
    src: str,
    tgt: Optional[str],
    device: str = 'cpu',
    record_history: bool = False,
    **kwargs,
) -> nn.Module | Tuple[nn.Module, dict]:
    """Train a flow-matching model with a given coupling method.

    Parameters mirror those of the command-line script; see that file for
    documentation.  ``src`` is the name of the source distribution; ``tgt``
    is ignored for the default setting where samples are drawn from the
    target distribution directly during training.

    Args:
        record_history: if True, the returned value will be a tuple
            ``(model, history)`` where ``history`` contains training loss and
            target variance logs.
    Returns:
        Trained :class:`SimpleFlowModel` or (model, history) if ``record_history``.
    """
    # sample dataset from src (tgt unused here)
    X = sample_toy(src, n_samples=kwargs.pop('n_samples', 10000))
    ds = TensorDataset(X)
    loader = DataLoader(ds, batch_size=kwargs.get('batch_size', 64), shuffle=True)

    model = SimpleFlowModel(input_dim=2, time_dim=kwargs.get('time_dim', 8), hidden_dim=kwargs.get('hidden', 32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=kwargs.get('lr', 1e-3))
    coupling = COUPLING_DICT[method]
    trainer = Trainer(model, loader, opt, n_epochs=kwargs.get('epochs', 200), sigma=kwargs.get('sigma', 0.005), sample_from_coupling=coupling)
    if record_history:
        model, history = trainer.train(from_random_gaussian=True, record_history=True)
        return model, history
    else:
        trainer.train(from_random_gaussian=True)
        return model


# ---------- evaluation utilities ---------------------------------------------

def compute_w2_squared(x_s: torch.Tensor, x_t: torch.Tensor) -> float:
    n = x_s.size(0)
    a = torch.full((n,), 1.0 / n, device=x_s.device)
    b = torch.full((n,), 1.0 / n, device=x_s.device)
    C = torch.cdist(x_s, x_t, p=2) ** 2
    return ot.emd2(a, b, C)


def compute_path_energy(model: nn.Module, x0: torch.Tensor, n_steps: int = 200) -> float:
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
    return energy


# ---------- inference helpers -----------------------------------------------

def _counting_wrapper(model: nn.Module):
    class Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.nfe = 0

        def forward(self, t, x):
            self.nfe += 1
            return self.base(t, x)

    return Wrapper(model)


def evaluate_nfe(model: nn.Module, x_s: torch.Tensor, tol: float = 1e-3) -> int:
    """Integrate with dopri5 and return average number of function evals."""
    try:
        from torchdiffeq import odeint
    except ImportError:
        raise RuntimeError("torchdiffeq not installed; please `pip install torchdiffeq` for NFE evaluation")

    device = next(model.parameters()).device
    wrapper = _counting_wrapper(model.to(device))
    t = torch.tensor([0.0, 1.0], device=device)
    # odeint expects func(t, x)
    with torch.no_grad():
        _ = odeint(wrapper, x_s.to(device), t, rtol=tol, atol=tol, method='dopri5')
    return wrapper.nfe // x_s.size(0)

def euler_integration(model: nn.Module, x_s: torch.Tensor, n_steps: int = 100) -> torch.Tensor:
    """Fixed-step Euler integration from t=0 to 1."""
    device = next(model.parameters()).device
    x = x_s.to(device).clone()
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t_val = torch.full((x.size(0), 1), i * dt, device=device)
        v = model(t_val, x)
        x = x + v * dt
    return x


def compute_npe(model: nn.Module, x_s: torch.Tensor, x_t: torch.Tensor, n_steps: int = 200) -> float:
    w2 = compute_w2_squared(x_s, x_t)
    pe = compute_path_energy(model, x_s, n_steps=n_steps)
    return abs(pe - w2) / (w2 + 1e-12)
