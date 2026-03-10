"""
Campus Flow Prediction Experiment
====================================
SITP Innovation Module – compares CfC vs LSTM on the Tongji campus
pedestrian-flow dataset with:

  • Irregular sampling (15 % missing steps)
  • 5-step rollout (recursive multi-step prediction)

Results saved to:
  results/campus_flow/loss_curves.png
  results/campus_flow/rollout_comparison.png
  results/campus_flow/metrics.csv
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.registry import build_model
from src.data.campus_flow import CampusFlowDataset, generate_campus_flow

# ---------------------------------------------------------------------------
SEED        = 42
UNITS       = 32          # hidden dimension (same for CfC and LSTM)
INPUT_SIZE  = 2           # [observed_flow, mask]
OUTPUT_SIZE = 1           # normalised flow
SEQ_LEN     = 48          # ~4 hours at 5-min intervals
ROLLOUT_K   = 5           # recursive prediction steps
LR          = 1e-3
EPOCHS      = 100
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/campus_flow"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(name: str, train_loader, test_loader, n_train_samples: int):
    model = build_model(name, input_size=INPUT_SIZE,
                        units=UNITS, output_size=OUTPUT_SIZE).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    losses, epoch_times = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * xb.size(0)
        losses.append(running / n_train_samples)
        epoch_times.append(time.time() - t0)
        if epoch % 25 == 0:
            print(f"  [{name.upper()}] epoch {epoch:3d}/{EPOCHS} "
                  f"loss={losses[-1]:.6f}")

    # ---- Evaluate on test set ----
    model.eval()
    test_mse = 0.0
    n_test   = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _ = model(xb)
            test_mse += crit(pred, yb).item() * xb.size(0)
            n_test   += xb.size(0)
    test_mse /= n_test

    n_params = sum(p.numel() for p in model.parameters())
    return model, {
        "MSE":              test_mse,
        "Params":           n_params,
        "Avg_epoch_time_s": np.mean(epoch_times),
        "losses":           losses,
    }


# ---------------------------------------------------------------------------
# 5-step rollout
# ---------------------------------------------------------------------------

def rollout(model, x_seed, steps: int = ROLLOUT_K):
    """
    Recursive (auto-regressive) prediction for `steps` future time-steps.

    x_seed : (1, seq_len, INPUT_SIZE) – seed window
    Returns : list of `steps` scalar predictions
    """
    model.eval()
    preds = []
    x = x_seed.clone().to(DEVICE)
    hx = None
    with torch.no_grad():
        for _ in range(steps):
            out, hx = model(x, hx)
            next_val = out[:, -1:, :]          # last time-step output (1,1,1)
            preds.append(next_val.cpu().item())
            # Shift window: append predicted value as "observed", mask=1
            next_input = torch.cat(
                [next_val, torch.ones(1, 1, 1, device=DEVICE)], dim=-1
            )
            x = torch.cat([x[:, 1:, :], next_input], dim=1)
    return preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    dataset = CampusFlowDataset(
        n_days=60, dt_minutes=5, seq_len=SEQ_LEN,
        missing_rate=0.15, seed=SEED
    )
    train_loader = dataset.get_dataloader(batch_size=BATCH_SIZE,
                                          split="train")
    test_loader  = dataset.get_dataloader(batch_size=BATCH_SIZE,
                                          split="test", shuffle=False)
    n_train = sum(1 for _ in train_loader) * BATCH_SIZE  # approx

    results = {}
    models  = {}
    for model_name in ["cfc", "lstm"]:
        print(f"\n{'='*52}")
        print(f"  Campus Flow – {model_name.upper()} ({UNITS} units)")
        print(f"{'='*52}")
        mdl, r = train(model_name, train_loader, test_loader, n_train)
        results[model_name] = r
        models[model_name]  = mdl
        print(f"  → Test MSE={r['MSE']:.6f}  Params={r['Params']}  "
              f"Avg epoch={r['Avg_epoch_time_s']*1e3:.1f} ms")

    # ---- Loss curves ----
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = {"cfc": "royalblue", "lstm": "tomato"}
    for name, r in results.items():
        ax.semilogy(r["losses"], label=f"{name.upper()} ({UNITS} units)",
                    color=colors[name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Campus Flow Prediction – Training Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # ---- 5-step rollout comparison ----
    # Take first test sample as seed
    first_test_x, first_test_y = next(iter(test_loader))
    x_seed  = first_test_x[:1]   # (1, seq_len, 2)
    y_truth = first_test_y[0, :ROLLOUT_K, 0].numpy()  # ground truth next steps

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(ROLLOUT_K), y_truth, "k.-", lw=2,
            label="Ground Truth", markersize=8)

    for name, mdl in models.items():
        rolled = rollout(mdl, x_seed, steps=ROLLOUT_K)
        ax.plot(range(ROLLOUT_K), rolled, ".--",
                color=colors[name], lw=1.5, markersize=8,
                label=f"{name.upper()} rollout")

    ax.set_xlabel("Rollout step")
    ax.set_ylabel("Normalised Flow")
    ax.set_title(f"{ROLLOUT_K}-Step Rollout – Campus Flow Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "rollout_comparison.png"), dpi=150)
    plt.close()

    # ---- CSV ----
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":            name.upper(),
            "Units":            UNITS,
            "Missing_rate":     0.15,
            "Test_MSE":         r["MSE"],
            "Parameters":       r["Params"],
            "Avg_epoch_time_s": r["Avg_epoch_time_s"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
