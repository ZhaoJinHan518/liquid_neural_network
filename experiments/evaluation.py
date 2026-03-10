"""
Evaluation & Metrics
======================
Performs multi-dimensional quantitative evaluation of all models:

1. Robustness analysis – adds noise (σ ∈ {0.0, 0.01, 0.05}) to the campus
   flow test set and measures degradation in RMSE.

2. Performance summary table – records for each model:
     • Total Parameters
     • Inference Latency (single-step, CPU)
     • RMSE on clean test set
     • R² Score

Results saved to:
  results/evaluation/robustness_curve.png
  results/evaluation/performance_table.md
  results/evaluation/performance_table.csv
"""

import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.registry import build_model
from src.data.campus_flow import CampusFlowDataset

# ---------------------------------------------------------------------------
SEED        = 42
UNITS       = 32
INPUT_SIZE  = 2
OUTPUT_SIZE = 1
SEQ_LEN     = 48
LR          = 1e-3
EPOCHS      = 80
BATCH_SIZE  = 32
NOISE_LEVELS = [0.0, 0.01, 0.05]
DEVICE      = torch.device("cpu")      # keep on CPU for latency measurement
RESULTS_DIR = "results/evaluation"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ["cfc", "lstm", "ltc", "gru"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_dataset_tensors(missing_rate: float = 0.15, seed: int = SEED):
    ds = CampusFlowDataset(
        n_days=60, dt_minutes=5, seq_len=SEQ_LEN,
        missing_rate=missing_rate, seed=seed
    )
    # Materialise all samples into tensors
    xs, ys = zip(*[ds[i] for i in range(len(ds))])
    X = torch.stack(xs)   # (N, SEQ_LEN, 2)
    Y = torch.stack(ys)   # (N, SEQ_LEN, 1)
    n_train = int(len(X) * 0.8)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def quick_train(name: str, X_train, Y_train):
    """Train for EPOCHS epochs and return the model."""
    model = build_model(name, input_size=INPUT_SIZE,
                        units=UNITS, output_size=OUTPUT_SIZE).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()
    ds     = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                         shuffle=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        if epoch % 20 == 0:
            print(f"    [{name.upper()}] epoch {epoch}/{EPOCHS}")
    return model


def evaluate_noisy(model, X_test, Y_test, sigma: float):
    """Add Gaussian noise σ to X_test[:, :, 0] (flow channel) and compute RMSE."""
    model.eval()
    X_noisy = X_test.clone()
    if sigma > 0:
        X_noisy[:, :, 0] += torch.randn_like(X_noisy[:, :, 0]) * sigma
        X_noisy[:, :, 0].clamp_(0.0, 1.0)

    with torch.no_grad():
        pred, _ = model(X_noisy.to(DEVICE))

    pred_np = pred.cpu().numpy().reshape(-1)
    true_np = Y_test.numpy().reshape(-1)
    rmse = math.sqrt(np.mean((pred_np - true_np) ** 2))
    return rmse, pred_np, true_np


def measure_latency(model, n_repeats: int = 200):
    """Single-step inference latency (ms) on CPU."""
    model.eval()
    x = torch.zeros(1, SEQ_LEN, INPUT_SIZE)
    # warm-up
    for _ in range(10):
        with torch.no_grad():
            model(x)
    t0 = time.perf_counter()
    for _ in range(n_repeats):
        with torch.no_grad():
            model(x)
    elapsed = (time.perf_counter() - t0) / n_repeats * 1000
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading dataset …")
    X_train, Y_train, X_test, Y_test = get_dataset_tensors()
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    trained_models = {}
    perf_rows      = []

    for name in MODEL_NAMES:
        print(f"\n{'='*52}")
        print(f"  Training {name.upper()} for evaluation …")
        print(f"{'='*52}")
        model = quick_train(name, X_train, Y_train)
        trained_models[name] = model

        # Baseline metrics (no noise)
        rmse_clean, pred_np, true_np = evaluate_noisy(model, X_test, Y_test, 0.0)
        r2 = r2_score(true_np, pred_np)
        latency_ms = measure_latency(model)
        n_params = sum(p.numel() for p in model.parameters())

        perf_rows.append({
            "Model":            name.upper(),
            "Parameters":       n_params,
            "Latency_ms":       round(latency_ms, 3),
            "RMSE_clean":       round(rmse_clean, 6),
            "R2_Score":         round(r2, 4),
        })
        print(f"  RMSE={rmse_clean:.5f}  R²={r2:.4f}  "
              f"Latency={latency_ms:.2f}ms  Params={n_params}")

    # ---- Robustness curve ----
    rob_results = {name: [] for name in MODEL_NAMES}
    for sigma in NOISE_LEVELS:
        for name, model in trained_models.items():
            rmse, _, _ = evaluate_noisy(model, X_test, Y_test, sigma)
            rob_results[name].append(rmse)

    fig, ax = plt.subplots(figsize=(8, 5))
    line_styles = {"cfc": ("royalblue", "o-"), "lstm": ("tomato", "s--"),
                   "ltc": ("seagreen", "^:"),  "gru":  ("darkorange", "D-.")}
    for name, rmses in rob_results.items():
        color, style = line_styles.get(name, ("gray", "x-"))
        ax.plot(NOISE_LEVELS, rmses, style, color=color,
                label=name.upper(), lw=2, markersize=7)

    ax.set_xlabel("Noise level σ")
    ax.set_ylabel("RMSE on test set")
    ax.set_title("Robustness to Input Noise – Campus Flow Prediction")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "robustness_curve.png"), dpi=150)
    plt.close()
    print(f"\nSaved robustness_curve.png")

    # ---- Performance table (CSV + Markdown) ----
    df = pd.DataFrame(perf_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "performance_table.csv"), index=False)

    # Markdown
    md_lines = ["# Model Performance Summary\n",
                "| Model | Parameters | Latency (ms) | RMSE | R² Score |",
                "|-------|-----------|-------------|------|---------|"]
    for _, row in df.iterrows():
        md_lines.append(
            f"| {row['Model']} | {row['Parameters']:,} | "
            f"{row['Latency_ms']:.3f} | {row['RMSE_clean']:.5f} | "
            f"{row['R2_Score']:.4f} |"
        )
    md_lines.append("")

    # Robustness section
    md_lines.append("## Robustness Analysis (RMSE at different noise levels)\n")
    md_lines.append("| Model | σ=0.00 | σ=0.01 | σ=0.05 |")
    md_lines.append("|-------|--------|--------|--------|")
    for name in MODEL_NAMES:
        vals = rob_results[name]
        md_lines.append(
            f"| {name.upper()} | {vals[0]:.5f} | {vals[1]:.5f} | {vals[2]:.5f} |"
        )

    md_path = os.path.join(RESULTS_DIR, "performance_table.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    print(f"Results saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
