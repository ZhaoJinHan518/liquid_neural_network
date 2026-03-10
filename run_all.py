#!/usr/bin/env python3
"""
Run all SITP experiments sequentially.

Usage:
    python run_all.py              # uses default hyperparameters (full training)
    python run_all.py --fast       # quick smoke-test with reduced epochs
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Default epoch counts per experiment
_DEFAULT_EPOCHS = {
    "damped_sine":        200,
    "walker2d":           100,
    "campus_flow":        100,
    "evaluation":          80,
    "timeseries_finance": 150,
    "uci_har":             80,
    "cartpole_rl":        500,   # episodes, not epochs
}
_FAST_EPOCHS = 5


def _run_damped_sine(epochs: int):
    import experiments.damped_sine as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved


def _run_walker2d(epochs: int):
    import experiments.walker2d as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    saved_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved
        sys.argv = saved_argv


def _run_campus_flow(epochs: int):
    import experiments.campus_flow_exp as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved


def _run_evaluation(epochs: int):
    import experiments.evaluation as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved


def _run_timeseries_finance(epochs: int):
    import experiments.timeseries_finance as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved


def _run_uci_har(epochs: int):
    import experiments.uci_har as mod
    saved = mod.EPOCHS
    mod.EPOCHS = epochs
    try:
        mod.main()
    finally:
        mod.EPOCHS = saved


def _run_cartpole_rl(episodes: int):
    import experiments.cartpole_rl as mod
    saved = mod.N_EPISODES
    mod.N_EPISODES = episodes
    try:
        mod.main()
    finally:
        mod.N_EPISODES = saved


def main():
    parser = argparse.ArgumentParser(
        description="Run all LNN vs RNN comparison experiments"
    )
    parser.add_argument(
        "--fast", action="store_true",
        help=f"Use {_FAST_EPOCHS} epochs per experiment for a quick smoke-test"
    )
    args = parser.parse_args()

    epochs = {k: (_FAST_EPOCHS if args.fast else v)
              for k, v in _DEFAULT_EPOCHS.items()}

    print("\n" + "=" * 60)
    print("  Experiment 1: Damped Sine Wave (Fig 6 reproduction)")
    print("=" * 60)
    _run_damped_sine(epochs["damped_sine"])

    print("\n" + "=" * 60)
    print("  Experiment 2: Walker2d Trajectory (Section 4.1 reproduction)")
    print("=" * 60)
    _run_walker2d(epochs["walker2d"])

    print("\n" + "=" * 60)
    print("  Experiment 3: Campus Flow Prediction (SITP Innovation)")
    print("=" * 60)
    _run_campus_flow(epochs["campus_flow"])

    print("\n" + "=" * 60)
    print("  Experiment 4: Multi-dimensional Evaluation")
    print("=" * 60)
    _run_evaluation(epochs["evaluation"])

    print("\n" + "=" * 60)
    print("  Experiment 5: Financial Time-Series Prediction")
    print("=" * 60)
    _run_timeseries_finance(epochs["timeseries_finance"])

    print("\n" + "=" * 60)
    print("  Experiment 6: UCI HAR Sequence Classification")
    print("=" * 60)
    _run_uci_har(epochs["uci_har"])

    print("\n" + "=" * 60)
    print("  Experiment 7: CartPole-v1 RL (REINFORCE)")
    print("=" * 60)
    _run_cartpole_rl(epochs["cartpole_rl"])

    print("\n" + "=" * 60)
    print("  All experiments completed!")
    print("  Results in: results/")
    print("  Logs in:    docs/SITP_Log.md")
    print("  Report in:  docs/Final_Report_SITP.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
