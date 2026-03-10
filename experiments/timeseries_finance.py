"""
Financial Time-Series Prediction Experiment
=============================================
Downloads closing-price data for the NASDAQ Composite (^IXIC) via yfinance
(falls back to synthetic data if the download fails, e.g., in offline CI).

Task  : given the past 30 trading-day closing prices, predict the next 5.
Models: CfC  vs  LSTM  vs  GRU
Metrics: MSE and MAE on the test set.

Results saved to:
    results/timeseries_finance/loss_curves.png
    results/timeseries_finance/predictions.png
    results/timeseries_finance/metrics.csv
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
INPUT_LEN   = 30    # look-back window (trading days)
PRED_LEN    = 5     # forecast horizon
INPUT_SIZE  = 1     # univariate: one closing-price feature
OUTPUT_SIZE = PRED_LEN
LR          = 1e-3
EPOCHS      = 150
BATCH_SIZE  = 32
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "results/timeseries_finance"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ["cfc", "lstm", "gru"]
COLORS      = {"cfc": "royalblue", "lstm": "tomato", "gru": "seagreen"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _download_prices(ticker: str = "^IXIC", years: int = 5) -> np.ndarray:
    """Download ~5 years of closing prices. Returns 1-D float32 array."""
    try:
        import yfinance as yf
        import datetime
        end   = datetime.date.today()
        start = end - datetime.timedelta(days=int(years * 365.25))
        df = yf.download(ticker, start=str(start), end=str(end),
                         progress=False, auto_adjust=True)
        prices = df["Close"].dropna().values.astype(np.float32)
        if len(prices) < INPUT_LEN + PRED_LEN + 50:
            raise ValueError("Too few data points downloaded")
        print(f"  Downloaded {len(prices)} trading days for {ticker}")
        return prices
    except Exception as exc:
        print(f"  [WARNING] yfinance download failed ({exc}). "
              "Using synthetic log-normal price series instead.")
        return _synthetic_prices(1260)   # ~5 years of trading days


def _synthetic_prices(n: int = 1260) -> np.ndarray:
    """Geometric Brownian Motion price series (fallback when offline)."""
    rng  = np.random.default_rng(SEED)
    dt   = 1 / 252
    mu   = 0.08
    sigma = 0.20
    log_returns = rng.normal((mu - 0.5 * sigma ** 2) * dt,
                              sigma * np.sqrt(dt), size=n)
    prices = 10000 * np.exp(np.cumsum(log_returns))
    return prices.astype(np.float32)


def _make_windows(prices: np.ndarray):
    """
    Slide a window of size (INPUT_LEN + PRED_LEN) over normalised
    log-return series to produce (X, Y) pairs.

    X : (N, INPUT_LEN, 1)  – normalised log-returns
    Y : (N, PRED_LEN)      – next PRED_LEN log-returns (target)
    """
    log_ret = np.diff(np.log(prices)).astype(np.float32)   # length N-1

    # Standardise
    mu  = log_ret.mean()
    std = log_ret.std() + 1e-8
    log_ret = (log_ret - mu) / std

    total = INPUT_LEN + PRED_LEN
    X_list, Y_list = [], []
    for i in range(len(log_ret) - total + 1):
        window = log_ret[i : i + total]
        X_list.append(window[:INPUT_LEN])
        Y_list.append(window[INPUT_LEN:])

    X = np.stack(X_list)[:, :, np.newaxis]  # (N, INPUT_LEN, 1)
    Y = np.stack(Y_list)                    # (N, PRED_LEN)
    return (
        torch.from_numpy(X),
        torch.from_numpy(Y),
        mu,
        std,
    )


def _prepare_data():
    """Download / generate data and split into train / val / test sets."""
    prices = _download_prices()
    X_all, Y_all, mu, std = _make_windows(prices)

    n_total = len(X_all)
    n_train = int(n_total * 0.7)
    n_val   = int(n_total * 0.15)
    n_test  = n_total - n_train - n_val

    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train]
    X_val   = X_all[n_train:n_train + n_val]
    Y_val   = Y_all[n_train:n_train + n_val]
    X_test  = X_all[n_train + n_val:]
    Y_test  = Y_all[n_train + n_val:]

    print(f"  Windows – train: {n_train}  val: {n_val}  test: {n_test}")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    return X_train, Y_train, X_val, Y_val, X_test, Y_test, train_loader, n_train


# ---------------------------------------------------------------------------
# Seq-to-vec wrapper
# ---------------------------------------------------------------------------

class Seq2VecWrapper(nn.Module):
    """
    Wraps a sequence model (output: (B, T, units)) so that only
    the LAST time-step hidden state is projected to PRED_LEN outputs.
    """

    def __init__(self, base: nn.Module, units: int, pred_len: int):
        super().__init__()
        self.base    = base
        self.head    = nn.Linear(units, pred_len)
        self._units  = units
        self._pred   = pred_len

    def forward(self, x, hx=None):
        out, hx = self.base.rnn(x, hx) if hasattr(self.base, "rnn") \
                  else self.base(x, hx)
        # out: (B, T, units)  →  take last step
        last = out[:, -1, :]       # (B, units)
        return self.head(last), hx  # (B, PRED_LEN)

    def parameters(self, **kw):
        return list(self.base.parameters()) + list(self.head.parameters())

    def train(self, mode=True):
        self.base.train(mode)
        self.head.train(mode)
        return self

    def eval(self):
        return self.train(False)


def _build_seq2vec(name: str) -> nn.Module:
    """Build a model that maps (B, INPUT_LEN, 1) → (B, PRED_LEN)."""
    base = build_model(name, input_size=INPUT_SIZE,
                       units=UNITS, output_size=UNITS)
    # Override the readout to use a PRED_LEN head instead
    if name == "cfc":
        # CfCModel has self.rnn (CfC) + self.fc; we replace self.fc
        base.fc = nn.Linear(UNITS, PRED_LEN)

        def _fwd(x, hx=None):
            out, hx = base.rnn(x, hx)
            return base.fc(out[:, -1, :]), hx

        base.forward = _fwd
    elif name in ("lstm", "gru"):
        # LSTMModel / GRUModel have self.lstm/self.gru + self.fc
        inner = base.lstm if name == "lstm" else base.gru
        base.fc = nn.Linear(UNITS, PRED_LEN)

        def _make_fwd(inner_rnn, fc):
            def _fwd(x, hx=None):
                out, hx = inner_rnn(x, hx)
                return fc(out[:, -1, :]), hx
            return _fwd

        base.forward = _make_fwd(inner, base.fc)
    return base


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(name: str, train_loader, X_val, Y_val, n_train: int):
    model = _build_seq2vec(name).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    losses, val_losses, times = [], [], []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0     = time.time()
        ep_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * xb.size(0)
        ep_loss /= n_train
        losses.append(ep_loss)
        times.append(time.time() - t0)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, _ = model(X_val.to(DEVICE))
            val_loss = criterion(val_pred, Y_val.to(DEVICE)).item()
        val_losses.append(val_loss)

        if epoch % 50 == 0:
            print(f"  [{name.upper()}] Epoch {epoch:3d}/{EPOCHS}  "
                  f"train={ep_loss:.6f}  val={val_loss:.6f}")

    return model, losses, val_losses, times


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, Y_test):
    model.eval()
    with torch.no_grad():
        pred, _ = model(X_test.to(DEVICE))
        pred    = pred.cpu().numpy()
    y_true  = Y_test.numpy()
    mse = float(np.mean((pred - y_true) ** 2))
    mae = float(np.mean(np.abs(pred - y_true)))
    return mse, mae, pred


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # Load / generate data
    (X_train, Y_train, X_val, Y_val,
     X_test, Y_test, train_loader, n_train) = _prepare_data()

    results = {}
    preds   = {}

    for name in MODEL_NAMES:
        print(f"\n{'='*54}")
        print(f"  Financial TS – {name.upper()} ({UNITS} units)")
        print(f"{'='*54}")
        model, losses, val_losses, times = train_model(
            name, train_loader, X_val, Y_val, n_train
        )
        mse, mae, pred = evaluate(model, X_test, Y_test)
        n_params = sum(p.numel() for p in model.parameters())
        results[name] = {
            "MSE": mse,
            "MAE": mae,
            "Params": n_params,
            "Avg_epoch_time_s": float(np.mean(times)),
            "losses": losses,
            "val_losses": val_losses,
        }
        preds[name] = pred
        print(f"  → MSE={mse:.6f}  MAE={mae:.6f}  "
              f"Params={n_params}  Avg epoch={np.mean(times)*1e3:.1f} ms")

    # ---- Plot: training loss curves ----
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, r in results.items():
        ax.semilogy(r["losses"],     label=f"{name.upper()} train",
                    color=COLORS[name], lw=1.5)
        ax.semilogy(r["val_losses"], label=f"{name.upper()} val",
                    color=COLORS[name], lw=1.0, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Financial Time-Series – Training / Validation Loss")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # ---- Plot: predictions on test window ----
    n_show = min(60, X_test.shape[0])
    fig, axes = plt.subplots(len(MODEL_NAMES), 1,
                             figsize=(12, 3 * len(MODEL_NAMES)),
                             sharex=True)
    for ax, name in zip(axes, MODEL_NAMES):
        true_vals = Y_test[:n_show].numpy().flatten()
        pred_vals = preds[name][:n_show].flatten()
        t = np.arange(len(true_vals))
        ax.plot(t, true_vals, "k-",  lw=1.2, label="Ground Truth")
        ax.plot(t, pred_vals, "--",  lw=1.2, color=COLORS[name],
                label=f"{name.upper()} prediction")
        ax.set_ylabel("Normalised log-return")
        ax.set_title(f"{name.upper()}  MSE={results[name]['MSE']:.5f}  "
                     f"MAE={results[name]['MAE']:.5f}")
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Test sample index × PRED_LEN")
    plt.suptitle("Financial Time-Series Prediction (NASDAQ, 30→5 days)",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "predictions.png"), dpi=150)
    plt.close()

    # ---- Save metrics CSV ----
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":            name.upper(),
            "Units":            UNITS,
            "Input_len":        INPUT_LEN,
            "Pred_len":         PRED_LEN,
            "MSE":              r["MSE"],
            "MAE":              r["MAE"],
            "Parameters":       r["Params"],
            "Avg_Epoch_Time_s": r["Avg_epoch_time_s"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
