"""
UCI HAR Sequence Classification Experiment
==========================================
Downloads the UCI Human Activity Recognition (HAR) dataset and trains
multiple RNN variants on the 6-class activity classification task.

Dataset source:
    https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip

Falls back to synthetic data if the download fails (offline CI).

Models  : LTC, CfC, LSTM, GRU, RNN
Task    : 128-step inertial-sensor sequence → 6-class activity label
Metrics : Accuracy (%), parameter count

Results saved to:
    results/uci_har/accuracy_bar.png
    results/uci_har/loss_curves.png
    results/uci_har/metrics.csv
"""

import os
import sys
import io
import time
import zipfile
import urllib.request
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
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
UNITS       = 32
N_CLASSES   = 6
LR          = 1e-3
EPOCHS      = 80
BATCH_SIZE  = 64
# Downsample temporal resolution (step > 1) to reduce LTC computation time.
# step=2 halves the sequence length from 128 → 64 steps while keeping the
# same sensor signals; all models use the same downsampled data.
TEMPORAL_STEP = 2
# Cap training samples so LTC remains tractable on CPU.
# Set to None to use the full dataset.
MAX_TRAIN_SAMPLES = 2000
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/uci_har"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ["ltc", "cfc", "lstm", "gru", "rnn"]
COLORS = {
    "ltc":  "royalblue",
    "cfc":  "darkorange",
    "lstm": "tomato",
    "gru":  "seagreen",
    "rnn":  "purple",
}

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00240/UCI%20HAR%20Dataset.zip"
)
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".cache", "uci_har")


def _load_split(base_dir: str, split: str):
    """
    Load one split ('train' or 'test') from the extracted UCI HAR directory.
    Returns X: (N, 128, 9), y: (N,) with labels in [0, 5].
    """
    split_dir  = os.path.join(base_dir, split)
    signal_dir = os.path.join(split_dir, "Inertial Signals")

    # 9 raw inertial channels
    channels = [
        "body_acc_x", "body_acc_y", "body_acc_z",
        "body_gyro_x", "body_gyro_y", "body_gyro_z",
        "total_acc_x", "total_acc_y", "total_acc_z",
    ]

    arrays = []
    for ch in channels:
        fname = os.path.join(signal_dir, f"{ch}_{split}.txt")
        arr = np.loadtxt(fname, dtype=np.float32)   # (N, 128)
        arrays.append(arr)

    X = np.stack(arrays, axis=-1)  # (N, 128, 9)

    label_path = os.path.join(split_dir, f"y_{split}.txt")
    y = np.loadtxt(label_path, dtype=np.int64) - 1  # 0-indexed

    return X, y


def _try_download_uci() -> str:
    """Download & unzip UCI HAR dataset; returns path to extracted root."""
    os.makedirs(_CACHE_DIR, exist_ok=True)
    extracted = os.path.join(_CACHE_DIR, "UCI HAR Dataset")
    if os.path.isdir(extracted):
        return extracted

    print("  Downloading UCI HAR dataset …")
    try:
        with urllib.request.urlopen(_UCI_URL, timeout=60) as resp:
            data = resp.read()
        zf = zipfile.ZipFile(io.BytesIO(data))
        zf.extractall(_CACHE_DIR)
        print(f"  Extracted to {extracted}")
        return extracted
    except Exception as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


def _synthetic_har(n_train: int = 7352, n_test: int = 2947,
                   seq_len: int = 128, n_feat: int = 9,
                   n_classes: int = 6) -> tuple:
    """
    Generate synthetic classification data that mimics UCI HAR shape.
    Each class has a distinct spectral signature (dominant frequency)
    so that all models can learn some structure.
    Used as fallback when the real dataset cannot be downloaded.
    """
    rng = np.random.default_rng(SEED)

    def _make_class_signal(label: int, n: int) -> np.ndarray:
        # Each class has a different dominant frequency
        t = np.linspace(0, 1, seq_len, dtype=np.float32)
        freq   = 1.0 + label * 1.5      # classes: 1, 2.5, 4, 5.5, 7, 8.5 Hz
        amp    = 0.5 + 0.1 * label
        noise  = rng.normal(0, 0.3, size=(n, seq_len, n_feat)).astype(np.float32)
        signal = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        # Modulate different features with the class signal
        X = noise.copy()
        for f in range(n_feat):
            X[:, :, f] += signal * (0.5 + 0.1 * ((f + label) % n_feat))
        return X

    all_X_tr, all_y_tr, all_X_te, all_y_te = [], [], [], []
    per_class_tr = n_train // n_classes
    per_class_te = n_test  // n_classes

    for cls in range(n_classes):
        X_tr_cls = _make_class_signal(cls, per_class_tr)
        X_te_cls = _make_class_signal(cls, per_class_te)
        all_X_tr.append(X_tr_cls)
        all_y_tr.append(np.full(per_class_tr, cls, dtype=np.int64))
        all_X_te.append(X_te_cls)
        all_y_te.append(np.full(per_class_te, cls, dtype=np.int64))

    X_tr = np.concatenate(all_X_tr, axis=0)
    y_tr = np.concatenate(all_y_tr, axis=0)
    X_te = np.concatenate(all_X_te, axis=0)
    y_te = np.concatenate(all_y_te, axis=0)

    # Shuffle training set
    idx = rng.permutation(len(X_tr))
    return X_tr[idx], y_tr[idx], X_te, y_te


def load_data():
    """
    Load UCI HAR dataset (download if needed) or fall back to synthetic data.
    Returns tensors: X_train, y_train, X_test, y_test.
    """
    try:
        root   = _try_download_uci()
        X_tr, y_tr = _load_split(root, "train")
        X_te, y_te = _load_split(root, "test")
        print(f"  UCI HAR – train: {X_tr.shape}  test: {X_te.shape}")
    except Exception as exc:
        print(f"  [WARNING] UCI HAR load failed ({exc}). "
              "Using synthetic data instead.")
        X_tr, y_tr, X_te, y_te = _synthetic_har()

    # Per-feature z-score normalisation (fit on train)
    mu  = X_tr.mean(axis=(0, 1), keepdims=True)
    std = X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
    X_tr = (X_tr - mu) / std
    X_te = (X_te - mu) / std

    # Temporal downsampling (reduces seq_len, speeds up LTC significantly)
    if TEMPORAL_STEP > 1:
        X_tr = X_tr[:, ::TEMPORAL_STEP, :]
        X_te = X_te[:, ::TEMPORAL_STEP, :]

    # Optionally cap training samples (makes LTC tractable on CPU)
    if MAX_TRAIN_SAMPLES is not None and len(X_tr) > MAX_TRAIN_SAMPLES:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(X_tr), size=MAX_TRAIN_SAMPLES, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]
        print(f"  Capped training set to {MAX_TRAIN_SAMPLES} samples")

    return (
        torch.from_numpy(X_tr),
        torch.from_numpy(y_tr),
        torch.from_numpy(X_te),
        torch.from_numpy(y_te),
    )


# ---------------------------------------------------------------------------
# Classification wrapper
# ---------------------------------------------------------------------------

class ClassifierWrapper(nn.Module):
    """
    Wraps any sequence model so it outputs N_CLASSES logits from the
    last hidden state.
    """

    def __init__(self, base_model: nn.Module, units: int, n_classes: int):
        super().__init__()
        self.base     = base_model
        self.head     = nn.Linear(units, n_classes)
        self._units   = units
        self._classes = n_classes

    def forward(self, x, hx=None):
        # base_model forward returns (output_seq, hx)
        # output_seq: (B, T, output_size)  — we need the last hidden rep
        out, hx = self.base(x, hx)
        last = out[:, -1, :]           # (B, output_size)
        return self.head(last), hx     # (B, n_classes)


def _build_classifier(name: str, input_size: int) -> nn.Module:
    """Build a classification model that outputs N_CLASSES logits."""
    # We use UNITS as both hidden size and output_size for the base model
    # so the last step of the sequence output has UNITS dimensions.
    base = build_model(name, input_size=input_size,
                       units=UNITS, output_size=UNITS)

    # For CfC the output is already projected to `output_size` (UNITS here),
    # so ClassifierWrapper's head maps UNITS → N_CLASSES correctly.
    # For LTC (FullyConnected wiring), the output includes both hidden and
    # motor neurons; we need just UNITS dimensions.  FullyConnected with
    # output_size=UNITS gives exactly UNITS outputs.
    return ClassifierWrapper(base, UNITS, N_CLASSES)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(name: str, X_train, y_train, X_test, y_test):
    input_size = X_train.shape[-1]
    model      = _build_classifier(name, input_size).to(DEVICE)

    # Collect all parameters across wrapper and sub-modules
    optimizer  = torch.optim.Adam(list(model.parameters()), lr=LR,
                                  weight_decay=1e-4)
    criterion  = nn.CrossEntropyLoss()

    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    loader   = torch.utils.data.DataLoader(train_ds,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

    losses, accs, times = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0       = time.time()
        ep_loss  = 0.0
        n_samples = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss   += loss.item() * xb.size(0)
            n_samples += xb.size(0)
        ep_loss /= n_samples
        losses.append(ep_loss)
        times.append(time.time() - t0)

        # Accuracy on test set (no gradient)
        model.eval()
        with torch.no_grad():
            logits, _ = model(X_test.to(DEVICE))
            preds  = logits.argmax(dim=1).cpu()
            acc    = (preds == y_test).float().mean().item() * 100
        accs.append(acc)

        if epoch % 20 == 0:
            print(f"  [{name.upper()}] Epoch {epoch:3d}/{EPOCHS}  "
                  f"loss={ep_loss:.4f}  test_acc={acc:.2f}%")

    # Count all trainable params
    n_params = sum(p.numel() for p in model.parameters())
    return model, losses, accs, times, n_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    X_train, y_train, X_test, y_test = load_data()
    print(f"  Train: {X_train.shape}  Test: {X_test.shape}")

    results = {}

    for name in MODEL_NAMES:
        print(f"\n{'='*54}")
        print(f"  UCI HAR – {name.upper()} ({UNITS} units)")
        print(f"{'='*54}")
        _, losses, accs, times, n_params = train_model(
            name, X_train, y_train, X_test, y_test
        )
        final_acc = accs[-1]
        results[name] = {
            "Accuracy_%":       final_acc,
            "Parameters":       n_params,
            "Avg_epoch_time_s": float(np.mean(times)),
            "losses":           losses,
            "accs":             accs,
        }
        print(f"  → Final accuracy={final_acc:.2f}%  "
              f"Params={n_params}  "
              f"Avg epoch={np.mean(times)*1e3:.1f} ms")

    # ---- Plot: accuracy bar chart ----
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = [n.upper() for n in MODEL_NAMES]
    values = [results[n]["Accuracy_%"] for n in MODEL_NAMES]
    bars   = ax.bar(names, values,
                    color=[COLORS[n] for n in MODEL_NAMES],
                    edgecolor="black", linewidth=0.7)
    ax.bar_label(bars, fmt="%.2f%%", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("UCI HAR – 6-class Activity Recognition Accuracy")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "accuracy_bar.png"), dpi=150)
    plt.close()

    # ---- Plot: loss curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for name in MODEL_NAMES:
        axes[0].semilogy(results[name]["losses"],
                         label=name.upper(), color=COLORS[name], lw=1.5)
        axes[1].plot(results[name]["accs"],
                     label=name.upper(), color=COLORS[name], lw=1.5)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss (log)")
    axes[0].set_title("Training Loss")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test Accuracy (%)")
    axes[1].set_title("Test Accuracy over Training")
    axes[1].legend(fontsize=8)
    plt.suptitle("UCI HAR Sequence Classification", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # ---- Save metrics CSV ----
    rows = []
    for name in MODEL_NAMES:
        r = results[name]
        rows.append({
            "Model":            name.upper(),
            "Units":            UNITS,
            "Accuracy_%":       round(r["Accuracy_%"], 2),
            "Parameters":       r["Parameters"],
            "Avg_epoch_time_s": round(r["Avg_epoch_time_s"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
