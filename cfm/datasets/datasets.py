import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from sklearn.datasets import make_blobs, make_circles, make_moons
import math
from typing import Literal

def sample_toy(
    dataset_type: Literal['moons', 'circles', '6gaussians'],
    n_samples: int = 1000,
    noise: float = 0.1
):
    if dataset_type == 'moons':
        print(f"Sampling moons with noise={noise}")
        X, _ = make_moons(n_samples=n_samples, noise=noise)
        X[:, 0] -= 0.5
        X[:, 1] -= 0.25
        X[:, 0] *= 0.5
        X[:, 1] *= 0.8
    elif dataset_type == 'circles':
        print(f"Sampling circles with noise={noise}")
        X, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
    elif dataset_type == '6gaussians':
        centers = [[math.cos(2 * math.pi * i / 6), math.sin(2 * math.pi * i / 6)]
                   for i in range(6)]
        X, _ = make_blobs(n_samples=n_samples, centers=centers,
                          cluster_std=noise, n_features=2)
    else:
        raise ValueError("Unknown dataset type")
    return torch.from_numpy(X).float()


class EpochShufflePairDataset(Dataset):
    """
    A dataset that yields pairs of samples from two datasets,
    with a new random pairing each epoch.
    """

    def __init__(self, X0, X1):
        self.X0 = X0
        self.X1 = X1
        self.N = min(len(X0), len(X1))
        self.perm0 = torch.randperm(len(X0))[:self.N]
        self.perm1 = torch.randperm(len(X1))[:self.N]

    def reshuffle(self):
        self.perm0 = torch.randperm(len(self.X0))[:self.N]
        self.perm1 = torch.randperm(len(self.X1))[:self.N]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X0[self.perm0[idx]], self.X1[self.perm1[idx]]


class RandomPairIterableDataset(IterableDataset):
    """
    An infinite iterable dataset that yields random pairs of samples from two datasets.

    Each pair consists of one sample from X0 and one sample from X1, 
    where X0 and X1 are fixed datasets provided at initialization.
    """

    def __init__(self, X0, X1):
        self.X0 = X0
        self.X1 = X1

    def __iter__(self):
        worker_info = get_worker_info()

        if worker_info is None:
            rng = torch.Generator()
        else:
            rng = torch.Generator()
            rng.manual_seed(torch.initial_seed() + worker_info.id)

        while True:
            i0 = torch.randint(len(self.X0), (1,), generator=rng).item()
            i1 = torch.randint(len(self.X1), (1,), generator=rng).item()
            yield self.X0[i0], self.X1[i1]
