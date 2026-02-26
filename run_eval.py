import torch
import os
import random
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from cfm.datasets import EpochShufflePairDataset, sample_toy
from cfm import SimpleFlowModel, Trainer, PathConfig, PathName
from functools import partial
from cfm.utils.metrics import compute_w2_npe
from typing import Any, Optional, Tuple, Literal, cast

def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

def append_to_csv(row_dict: dict, csv_dir: str):
    csv_path = Path(csv_dir)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_dict])
    df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )

DatasetName = Literal['moons', 'circles', '6gaussians']

@torch.inference_mode()
def eval_callback(
    run_name: str,
    x_s_name: DatasetName,
    x_t_name: DatasetName,
    csv_path: str,
    model,
    epoch: int,
):
    model.eval()
    device = next(model.parameters()).device
    n_eval = 2048
    x0 = sample_toy(x_s_name, n_samples=n_eval, noise=0.05).to(device)
    xt = sample_toy(x_t_name, n_samples=n_eval, noise=0.1).to(device)
    
    w2_sq_xt_xout, npe = compute_w2_npe(model, x0, xt)
        
    print(
        f"[{run_name}] Epoch {epoch:04d} | "
        f"W2^2(x_out, x_t): {w2_sq_xt_xout:.6f} | NPE: {npe:.6f}"
    )
    
    dataset_name, method_name, seed_str = run_name.split("-")

    append_to_csv(
        {
            "run_name": run_name,
            "epoch": epoch,
            "x_source": x_s_name,
            "x_target": x_t_name,
            "method": method_name,
            "seed": int(seed_str),
            "w22_xout_xt": w2_sq_xt_xout,
            "npe": npe,
        },
        csv_path,
    )

def train_one_run(
    d_from: DatasetName,
    d_to: DatasetName,
    method: PathName,
    seed: int,
    config: dict,
    run_name: str,
):
    set_seed(seed)
    os.makedirs("./run/checkpoints", exist_ok=True)
    os.makedirs("./run/datasets", exist_ok=True)

    x_source = sample_toy(d_from, n_samples=100000, noise=0.05)
    x_target = sample_toy(d_to, n_samples=100000, noise=0.1)

    # visualize the datasets
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.title(f"Source: {d_from}")
    plt.scatter(x_source[:, 0], x_source[:, 1], s=1, alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.title(f"Target: {d_to}")
    plt.scatter(x_target[:, 0], x_target[:, 1], s=1, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"./run/datasets/{run_name}.png")
    plt.close()

    dataset_pair = EpochShufflePairDataset(x_source, x_target)
    dataloader = torch.utils.data.DataLoader(dataset_pair,batch_size=config["batch_size"], shuffle=False)

    flow_model = SimpleFlowModel(input_dim=2, time_dim=8, hidden_dim=128)
    path_cfg = PathConfig(name=method, sigma=0.05)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=config["lr"])

    ckpt_path = f"./run/checkpoints/{run_name.replace('/', '_')}.pth"
    csv_path = f"./run/results/metrics_{d_from}_{d_to}.csv"

    trainer = Trainer(flow_model, dataloader, optimizer=optimizer, n_epochs=config["epochs"], path_cfg=path_cfg,
        save_period=5, save_path=ckpt_path, eval_every=config["eval_every"], 
        callbacks=[partial(eval_callback, run_name, d_from, d_to, csv_path)])

    trainer.train()
    torch.save(flow_model.state_dict(), ckpt_path)

if __name__ == "__main__":
    DATASETS = [["moons", "6gaussians"],
                ["moons", "circles"]]
    METHODS = [
        # "independent_cfm",
        # "ot_cfm", 
        # "schrodinger_bridge_cfm",
        "flow_matching",
    ]
    SEEDS = [
        #0,
        1, 2, 3, 4]

    COMMON_CONFIG = dict(
        epochs=120,
        batch_size=128,
        lr=1e-3,
        eval_every=5,
    )

    for d_from, d_to in DATASETS:
        for method in METHODS:
            for seed in SEEDS:
                run_name = f"{d_from}.{d_to}-{method}-{seed}"
                train_one_run(
                    d_from=cast(DatasetName, d_from),
                    d_to=cast(DatasetName, d_to),
                    method=cast(PathName, method),
                    seed=seed,
                    config=COMMON_CONFIG,
                    run_name=run_name,
                )