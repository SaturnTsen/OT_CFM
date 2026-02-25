import torch
from cfm.datasets import EpochShufflePairDataset, sample_toy
from cfm import SimpleFlowModel, Trainer, PathConfig
from functools import partial
from cfm.utils.metrics import compute_w2_npe
import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd

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

def append_to_csv(row_dict: dict, csv_path: str):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row_dict])
    df.to_csv(
        csv_path,
        mode="a",
        header=not csv_path.exists(),
        index=False,
    )

@torch.inference_mode()
def eval_callback(
    run_name: str,
    x_s_name: str,
    x_t_name: str,
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
    dataset: str,
    method: str,
    seed: int,
    config: dict,
    run_name: str,
):
    set_seed(seed)
    os.makedirs("./run/checkpoints", exist_ok=True)
    if dataset == "moons.6gaussians":
        x_source = sample_toy("moons", n_samples=100000, noise=0.05)
        x_target = sample_toy("6gaussians", n_samples=100000, noise=0.1)
        x_s_name, x_t_name = "moons", "6gaussians"
    elif dataset == "gaussians":
        x_source = sample_toy("gaussians", n_samples=100000, noise=0.05)
        x_source[:, 0] -= 3
        x_target = sample_toy("gaussians", n_samples=100000, noise=0.1)
        x_s_name, x_t_name = "gaussians", "gaussians"
    elif dataset == "moons.circles":
        x_source = sample_toy("moons", n_samples=100000, noise=0.05)
        x_target = sample_toy("circles", n_samples=100000, noise=0.1)
        x_s_name, x_t_name = "moons", "circles"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    dataset_pair = EpochShufflePairDataset(x_source, x_target)
    dataloader = torch.utils.data.DataLoader(dataset_pair,batch_size=config["batch_size"], shuffle=False)

    flow_model = SimpleFlowModel(input_dim=2, time_dim=8, hidden_dim=128)
    path_cfg = PathConfig(name=method, sigma=0.05)
    optimizer = torch.optim.Adam(flow_model.parameters(), lr=config["lr"])

    ckpt_path = f"./run/checkpoints/{run_name.replace('/', '_')}.pth"
    csv_path = "./run/results/metrics.csv"

    trainer = Trainer(flow_model, dataloader, optimizer=optimizer, n_epochs=config["epochs"], path_cfg=path_cfg,
        save_period=5, save_path=ckpt_path, eval_every=config["eval_every"], 
        callbacks=[partial(eval_callback, run_name, x_s_name, x_t_name, csv_path)])

    trainer.train()
    torch.save(flow_model.state_dict(), ckpt_path)

if __name__ == "__main__":
    DATASET = ["moons.6gaussians"]
    METHODS = [
        "independent_cfm",
        "ot_cfm",
        "schrodinger_bridge_cfm",
        "flow_matching",
    ]
    SEEDS = [0, 1, 2, 3, 4]

    COMMON_CONFIG = dict(
        epochs=50,
        batch_size=128,
        lr=1e-3,
        eval_every=5,
    )

    for dataset in DATASET:
        for method in METHODS:
            for seed in SEEDS:
                run_name = f"{dataset}-{method}-{seed}"
                train_one_run(
                    dataset=dataset,
                    method=method,
                    seed=seed,
                    config=COMMON_CONFIG,
                    run_name=run_name,
                )