"""
Walker2d Trajectory Experiment
================================
Reproduces the Section 4.1 experiment from arXiv:2510.07578v1.

Because the minari Walker2d-medium dataset requires a Gymnasium / MuJoCo
installation that may not be available in all environments, this script
provides two modes:

  1. SYNTHETIC mode (default): generates synthetic 17-dim dynamics data
     that mirrors the statistical properties of the Walker2d state space
     (positions + velocities for a bipedal walker), so the experiment can
     always run without extra system dependencies.

  2. MINARI mode: activated by passing --use-minari as CLI argument.
     Loads the 'D4RL/walker2d/medium-v2' dataset from minari.

Task
----
  Input  : 10 time-steps × 17 dims
  Output : 17 dims (next-step state prediction)

Models compared
---------------
  - LTC  (64 units)
  - LSTM (64 units)

Results saved to
----------------
  results/walker2d/loss_curves.png
  results/walker2d/metrics.csv
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.registry import build_model

# ---------------------------------------------------------------------------
SEED         = 42
UNITS        = 64
INPUT_DIM    = 17
OUTPUT_DIM   = 17
SEQ_LEN      = 10          # 10-step input
LR           = 1e-3
EPOCHS       = 100
BATCH_SIZE   = 64
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR  = "results/walker2d"

torch.manual_seed(SEED)
np.random.seed(SEED)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_synthetic_walker(n_samples: int = 5000, seq_len: int = SEQ_LEN,
                          seed: int = 0):
    """
    Synthetic Walker2d-like data:
      - Joint positions (8 dims): bounded sinusoidal motion
      - Joint velocities (9 dims): derivatives with Gaussian noise
    """
    rng = np.random.default_rng(seed)
    T = seq_len + 1
    t = np.linspace(0, 2 * np.pi, T)

    freqs = rng.uniform(0.5, 3.0, size=(n_samples, 8))
    phases = rng.uniform(0, 2 * np.pi, size=(n_samples, 8))
    amps   = rng.uniform(0.2, 1.0, size=(n_samples, 8))

    # positions: (N, T, 8)
    pos = amps[:, np.newaxis, :] * np.sin(
        freqs[:, np.newaxis, :] * t[np.newaxis, :, np.newaxis]
        + phases[:, np.newaxis, :]
    )
    # velocities: forward difference for first step, central differences otherwise
    # Result: (N, T, 8)
    vel8 = np.concatenate([
        pos[:, 1:2, :] - pos[:, 0:1, :],          # forward diff at t=0
        (pos[:, 2:, :] - pos[:, :-2, :]) / 2,     # central diff for t=1..T-2
        pos[:, -1:, :] - pos[:, -2:-1, :],         # backward diff at t=T-1
    ], axis=1)  # (N, T, 8)
    extra = rng.normal(0, 0.05, size=(n_samples, T, 1))
    vel = np.concatenate([vel8, extra], axis=-1)  # (N, T, 9)
    vel += rng.normal(0, 0.02, size=vel.shape)

    data = np.concatenate([pos, vel], axis=-1).astype(np.float32)   # (N, T, 17)

    X = torch.from_numpy(data[:, :seq_len, :])    # (N, seq_len, 17)
    Y = torch.from_numpy(data[:, 1:seq_len+1, :]) # (N, seq_len, 17)
    return X, Y


def load_minari_walker(seq_len: int = SEQ_LEN):
    """Load Walker2d-medium-v2 from minari (requires minari package)."""
    import minari
    dataset = minari.load_dataset("D4RL/walker2d/medium-v2", download=True)
    obs_list, next_obs_list = [], []
    for ep in dataset:
        obs = ep.observations[:-1]
        next_obs = ep.observations[1:]
        T = len(obs) - seq_len
        for start in range(0, T, seq_len):
            obs_list.append(obs[start:start + seq_len])
            next_obs_list.append(next_obs[start:start + seq_len])

    X = torch.tensor(np.stack(obs_list), dtype=torch.float32)
    Y = torch.tensor(np.stack(next_obs_list), dtype=torch.float32)
    return X, Y


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(name: str, X_train, Y_train, X_test, Y_test):
    model = build_model(name, input_size=INPUT_DIM,
                        units=UNITS, output_size=OUTPUT_DIM).to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    crit  = nn.MSELoss()

    ds = torch.utils.data.TensorDataset(X_train, Y_train)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                         shuffle=True)

    losses, epoch_times = [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0, running = time.time(), 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optim.zero_grad()
            pred, _ = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            running += loss.item() * xb.size(0)
        losses.append(running / len(ds))
        epoch_times.append(time.time() - t0)
        if epoch % 25 == 0:
            print(f"  [{name.upper()}] epoch {epoch:3d}/{EPOCHS} "
                  f"loss={losses[-1]:.6f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        pred, _ = model(X_test.to(DEVICE))
        test_mse = crit(pred, Y_test.to(DEVICE)).item()

    n_params = sum(p.numel() for p in model.parameters())
    return {
        "MSE": test_mse,
        "Params": n_params,
        "Avg_epoch_time_s": np.mean(epoch_times),
        "losses": losses,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-minari", action="store_true",
                        help="Load Walker2d-medium-v2 from minari.")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    if args.use_minari:
        print("Loading Walker2d-medium-v2 from minari …")
        X, Y = load_minari_walker()
    else:
        print("Generating synthetic Walker2d data …")
        X, Y = load_synthetic_walker(n_samples=6000)

    # Train / test split (80/20)
    n_train = int(len(X) * 0.8)
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test,  Y_test  = X[n_train:], Y[n_train:]
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    results = {}
    for model_name in ["ltc", "lstm"]:
        print(f"\n{'='*52}")
        print(f"  Walker2d – {model_name.upper()} ({UNITS} units)")
        print(f"{'='*52}")
        results[model_name] = train(model_name, X_train, Y_train,
                                    X_test, Y_test)
        r = results[model_name]
        print(f"  → Test MSE={r['MSE']:.6f}  Params={r['Params']}  "
              f"Avg epoch={r['Avg_epoch_time_s']*1e3:.1f}ms")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, r in results.items():
        ax.semilogy(r["losses"], label=f"{name.upper()} ({UNITS} units)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training MSE (log scale)")
    ax.set_title("Walker2d Next-Step Prediction (Section 4.1 reproduction)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # CSV
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":            name.upper(),
            "Units":            UNITS,
            "Input_dim":        INPUT_DIM,
            "Output_dim":       OUTPUT_DIM,
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
