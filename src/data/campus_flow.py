"""
Campus Flow Data Generator
===========================
Simulates pedestrian flow dynamics on the Tongji University campus.

Key physical features modelled
-------------------------------
1. Periodicity      : 24-hour sinusoidal baseline.
2. Stiff Dynamics   : Instantaneous spikes at class-break times
                      (mimicking stiff ODE behaviour that LNNs handle well).
3. Irregular Sampling: 15 % of time-steps are randomly masked (zeroed out,
                       with a companion binary mask feature) to test resilience
                       to missing sensor data.

Usage
-----
    from src.data.campus_flow import CampusFlowDataset, generate_campus_flow

    dataset = CampusFlowDataset(n_days=30, dt=5, seed=42)
    loader  = dataset.get_dataloader(seq_len=48, batch_size=32)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Physical simulation helpers
# ---------------------------------------------------------------------------

# Class-break times expressed as fraction of a 24-hour day (in minutes)
_CLASS_BREAK_MINUTES = [
    10 * 60,        # 10:00
    11 * 60 + 45,   # 11:45
    13 * 60 + 30,   # 13:30  afternoon break
    15 * 60 + 15,   # 15:15
    17 * 60,        # 17:00  after-school rush
    18 * 60 + 30,   # 18:30  dinner rush
]


def _base_signal(t_min: np.ndarray) -> np.ndarray:
    """
    Smooth 24-h periodic baseline (normalised to [0, 1]).
    t_min : time in minutes (can be multi-day).
    """
    phase = (t_min % 1440) / 1440.0 * 2 * np.pi
    signal = (
        0.35 * np.sin(phase - np.pi / 2)        # main 24-h cycle
        + 0.15 * np.sin(2 * phase - np.pi / 3)  # morning/evening peaks
        + 0.50                                    # DC offset → always ≥ 0
    )
    return signal


def _class_break_spikes(
    t_min: np.ndarray,
    spike_amplitude: float = 1.5,
    spike_width_min: float = 3.0,
) -> np.ndarray:
    """
    Sum of narrow Gaussian spikes centred on each class-break minute.
    This models the stiff ODE dynamics that occur at bell times.
    """
    spikes = np.zeros_like(t_min, dtype=float)
    day_t = t_min % 1440
    for cb in _CLASS_BREAK_MINUTES:
        diff = day_t - cb
        spikes += spike_amplitude * np.exp(-0.5 * (diff / spike_width_min) ** 2)
    return spikes


def generate_campus_flow(
    n_days: int = 30,
    dt_minutes: int = 5,
    missing_rate: float = 0.15,
    noise_std: float = 0.02,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic campus pedestrian-flow time series.

    Returns
    -------
    t         : shape (T,)  – time axis in minutes
    flow      : shape (T,)  – clean flow signal
    flow_obs  : shape (T,)  – observed signal with missing steps zeroed out
    mask      : shape (T,)  – 1 = observed, 0 = missing
    """
    rng = np.random.default_rng(seed)
    total_minutes = n_days * 1440
    t = np.arange(0, total_minutes, dt_minutes, dtype=float)

    flow = _base_signal(t) + _class_break_spikes(t)
    flow += rng.normal(0.0, noise_std, size=t.shape)
    flow = np.clip(flow, 0.0, None)

    # Normalise to [0, 1]
    flow = (flow - flow.min()) / (flow.max() - flow.min() + 1e-8)

    # Irregular sampling: randomly mask 15 % of steps
    mask = (rng.random(size=t.shape) > missing_rate).astype(float)
    flow_obs = flow * mask   # masked-out steps become 0

    return t, flow, flow_obs, mask


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class CampusFlowDataset(Dataset):
    """
    Sliding-window dataset over the campus flow time series.

    Each sample is:
        x  : (seq_len, 2)  – [observed_flow, mask]  (input features)
        y  : (seq_len, 1)  – clean flow target

    Parameters
    ----------
    n_days       : number of simulated days
    dt_minutes   : sampling interval in minutes
    seq_len      : sliding window length
    missing_rate : fraction of steps to mask
    noise_std    : additive Gaussian noise on the flow signal
    seed         : random seed
    """

    def __init__(
        self,
        n_days: int = 30,
        dt_minutes: int = 5,
        seq_len: int = 48,
        missing_rate: float = 0.15,
        noise_std: float = 0.02,
        seed: int = 42,
    ):
        self.seq_len = seq_len
        t, flow, flow_obs, mask = generate_campus_flow(
            n_days=n_days,
            dt_minutes=dt_minutes,
            missing_rate=missing_rate,
            noise_std=noise_std,
            seed=seed,
        )
        # Store as float32 tensors
        self.flow = torch.tensor(flow, dtype=torch.float32)
        self.flow_obs = torch.tensor(flow_obs, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.t = t

    def __len__(self) -> int:
        return len(self.flow) - self.seq_len

    def __getitem__(self, idx: int):
        sl = slice(idx, idx + self.seq_len)
        x = torch.stack([self.flow_obs[sl], self.mask[sl]], dim=-1)  # (T, 2)
        y = self.flow[sl].unsqueeze(-1)                               # (T, 1)
        return x, y

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        train_ratio: float = 0.8,
        split: str = "train",
    ) -> DataLoader:
        n_total = len(self)
        n_train = int(n_total * train_ratio)
        if split == "train":
            indices = list(range(n_train))
        else:
            indices = list(range(n_train, n_total))

        from torch.utils.data import Subset
        subset = Subset(self, indices)
        return DataLoader(subset, batch_size=batch_size,
                          shuffle=(shuffle and split == "train"))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import os

    t, flow, flow_obs, mask = generate_campus_flow(n_days=3, dt_minutes=5)
    minutes_per_day = 1440 // 5   # samples per day

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t / 60, flow, lw=1, label="Clean flow")
    axes[0].scatter(
        t[mask == 0] / 60, flow[mask == 0],
        s=4, color="red", label="Missing (masked)", zorder=5
    )
    axes[0].set_ylabel("Normalised Flow")
    axes[0].set_title("Tongji Campus Pedestrian Flow – Simulated (3 days)")
    axes[0].legend(fontsize=8)

    axes[1].plot(t / 60, flow_obs, lw=1, color="orange",
                 label="Observed (with missing)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Normalised Flow")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    os.makedirs("results/campus_flow", exist_ok=True)
    plt.savefig("results/campus_flow/campus_flow_preview.png", dpi=150)
    print("Saved results/campus_flow/campus_flow_preview.png")
