"""
Damped Sine Wave Experiment
============================
Reproduces the 1-step prediction experiment from Fig 6 of arXiv:2510.07578v1.

Two models (LNN / GRU, 32 units) are trained to predict the next value of a
damped sine wave:  y(t) = A·exp(-γt)·sin(ωt + φ)

Results are saved to:
    results/damped_sine/predictions.png
    results/damped_sine/loss_curves.png
    results/damped_sine/metrics.csv
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

# Ensure project root is in sys.path when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.registry import build_model

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters  (matching paper setup)
# ---------------------------------------------------------------------------
UNITS        = 32
SEQ_LEN      = 100
N_TRAIN      = 1000
N_TEST       = 200
LR           = 1e-3
EPOCHS       = 200
BATCH_SIZE   = 32
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "results/damped_sine"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_damped_sine(n_samples: int, seq_len: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    A   = rng.uniform(0.5, 1.5, n_samples)
    gamma = rng.uniform(0.005, 0.02, n_samples)
    omega = rng.uniform(0.5, 2.0, n_samples)
    phi   = rng.uniform(0, 2 * np.pi, n_samples)
    t = np.linspace(0, 10, seq_len + 1)
    data = []
    for i in range(n_samples):
        s = A[i] * np.exp(-gamma[i] * t) * np.sin(omega[i] * t + phi[i])
        data.append(s)
    data = np.array(data, dtype=np.float32)  # (N, seq_len+1)
    x = data[:, :-1, np.newaxis]             # (N, seq_len, 1)
    y = data[:, 1:, np.newaxis]              # (N, seq_len, 1) – 1-step target
    return (
        torch.from_numpy(x),
        torch.from_numpy(y),
        t[:-1],
    )


X_train, Y_train, t_axis = make_damped_sine(N_TRAIN, SEQ_LEN, seed=0)
X_test,  Y_test,  _      = make_damped_sine(N_TEST,  SEQ_LEN, seed=1)

train_ds = torch.utils.data.TensorDataset(X_train, Y_train)
train_loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True
)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_model(name: str):
    model = build_model(name, input_size=1, units=UNITS, output_size=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    loss_history = []
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(train_ds)
        epoch_times.append(time.time() - t0)
        loss_history.append(epoch_loss)
        if epoch % 50 == 0:
            print(f"  [{name.upper()}] Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={epoch_loss:.6f}")

    return model, loss_history, epoch_times


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, name: str):
    model.eval()
    criterion = nn.MSELoss()
    with torch.no_grad():
        xb = X_test.to(DEVICE)
        yb = Y_test.to(DEVICE)
        pred, _ = model(xb)
        mse = criterion(pred, yb).item()
    return mse, pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = {}
    preds   = {}

    for model_name in ["ltc", "gru"]:
        print(f"\n{'='*50}")
        print(f"  Training {model_name.upper()} ({UNITS} units) – Damped Sine")
        print(f"{'='*50}")
        model, losses, times = train_one_model(model_name)
        mse, pred = evaluate(model, model_name)

        # Parameter count
        n_params = sum(p.numel() for p in model.parameters())

        results[model_name] = {
            "MSE": mse,
            "Params": n_params,
            "Avg_epoch_time_s": np.mean(times),
            "losses": losses,
        }
        preds[model_name] = pred
        print(f"  → MSE={mse:.6f}  Params={n_params}  "
              f"Avg epoch={np.mean(times)*1000:.1f}ms")

    # ---- Plot: loss curves ----
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        ax.semilogy(r["losses"], label=f"{name.upper()} ({UNITS} units)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training Loss – Damped Sine Wave (1-step prediction)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # ---- Plot: prediction vs ground truth (first test sample) ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    sample_idx = 0
    y_true = Y_test[sample_idx, :, 0].numpy()
    colors = {"ltc": "royalblue", "gru": "tomato"}

    for i, (name, pred) in enumerate(preds.items()):
        axes[i].plot(t_axis, y_true, "k-", lw=1.5, label="Ground Truth")
        axes[i].plot(t_axis, pred[sample_idx, :, 0], "--",
                     color=colors[name], lw=1.5,
                     label=f"{name.upper()} prediction")
        axes[i].set_title(f"{name.upper()} – 1-step Prediction")
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Amplitude")
        axes[i].legend(fontsize=9)

    plt.suptitle("Damped Sine Wave Prediction (Fig 6 reproduction)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictions.png"), dpi=150)
    plt.close()

    # ---- Save metrics CSV ----
    rows = []
    for name, r in results.items():
        rows.append({
            "Model": name.upper(),
            "Units": UNITS,
            "MSE": r["MSE"],
            "Parameters": r["Params"],
            "Avg_Epoch_Time_s": r["Avg_epoch_time_s"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
