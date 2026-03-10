"""
CartPole-v1 Reinforcement Learning Experiment
=============================================
Trains policy-gradient (REINFORCE) agents with RNN-based policy networks
on the CartPole-v1 environment (gymnasium).

Models: CfC policy  vs  LSTM policy
Metrics: Convergence speed (episodes to reach 200 reward) and final reward.

Results saved to:
    results/cartpole/reward_curves.png
    results/cartpole/metrics.csv
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
UNITS         = 32
N_EPISODES    = 500       # max training episodes
GAMMA         = 0.99      # discount factor
LR            = 2e-3      # policy learning rate
ENTROPY_COEF  = 0.01      # entropy bonus coefficient (reduces policy collapse)
LOG_INTERVAL  = 20        # print every N episodes
SMOOTH_WINDOW = 20        # rolling-average window for plotting
SOLVED_REWARD = 195.0     # CartPole-v1 "solved" threshold
DEVICE        = torch.device("cpu")   # RL loops run faster on CPU
RESULTS_DIR   = "results/cartpole"

os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_NAMES = ["cfc", "lstm"]
COLORS      = {"cfc": "royalblue", "lstm": "tomato"}

# CartPole-v1 state/action dimensions
OBS_SIZE    = 4
ACT_SIZE    = 2


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class RNNPolicy(nn.Module):
    """
    Stateful RNN policy network for REINFORCE.

    The RNN processes the observation sequence step-by-step; at each step
    the last hidden state is projected to action logits.

    Gradients flow through the entire episode trajectory (no hidden-state
    detachment within an episode) to enable proper credit assignment.
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        # Base RNN: input=OBS_SIZE, output=UNITS (hidden dim)
        self.rnn = build_model(name, input_size=OBS_SIZE,
                               units=UNITS, output_size=UNITS)
        self.actor = nn.Linear(UNITS, ACT_SIZE)

    def forward(self, obs, hx=None):
        """
        obs : (1, 1, OBS_SIZE) – single time-step, batch=1
        hx  : hidden state from previous step (or None)
        Returns (action_int, log_prob, entropy, new_hx).
        """
        # Pass through RNN (seq_len=1) – keep gradients through hx
        out, hx = self.rnn(obs, hx)    # out: (1, 1, UNITS)
        logits   = self.actor(out[:, -1, :])   # (1, ACT_SIZE)
        probs    = torch.softmax(logits, dim=-1)
        dist     = torch.distributions.Categorical(probs)
        action   = dist.sample()
        log_p    = dist.log_prob(action)
        entropy  = dist.entropy()
        return action.item(), log_p, entropy, hx


# ---------------------------------------------------------------------------
# REINFORCE training
# ---------------------------------------------------------------------------

def discount_returns(rewards, gamma: float, baseline: float = 0.0) -> torch.Tensor:
    """
    Compute discounted returns with an optional cross-episode baseline.
    Returns are NOT normalized per-episode so that the baseline can
    provide meaningful cross-episode variance reduction.
    """
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G - baseline)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Standardise within the episode for numerical stability
    returns = returns / (returns.abs().max() + 1e-8)
    return returns


def _make_env():
    """Create the CartPole-v1 environment (gymnasium or gym fallback)."""
    try:
        import gymnasium as gym
        env = gym.make("CartPole-v1")
    except ImportError:
        try:
            import gym as gym_legacy
            env = gym_legacy.make("CartPole-v1")
        except ImportError as e:
            raise ImportError(
                "Neither 'gymnasium' nor 'gym' is installed. "
                "Install with: pip install gymnasium"
            ) from e
    return env


def train_agent(name: str):
    env   = _make_env()
    env.action_space.seed(SEED)

    policy    = RNNPolicy(name).to(DEVICE)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    episode_rewards = []
    times           = []
    first_solved    = None   # episode index when avg reward ≥ SOLVED_REWARD
    baseline        = 0.0    # exponential moving average of episode returns

    for ep in range(1, N_EPISODES + 1):
        t0 = time.time()

        obs, _ = env.reset(seed=SEED + ep)
        log_probs, entropies, rewards = [], [], []
        hx = None  # reset hidden state at each episode boundary

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32,
                                 device=DEVICE).unsqueeze(0).unsqueeze(0)
            # obs_t: (1, 1, 4)
            action, log_p, entropy, hx = policy(obs_t, hx)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            log_probs.append(log_p)
            entropies.append(entropy)
            rewards.append(reward)

        ep_reward = sum(rewards)
        episode_rewards.append(ep_reward)
        times.append(time.time() - t0)

        # Update exponential moving average baseline (variance reduction)
        baseline = 0.95 * baseline + 0.05 * ep_reward

        # Check solved
        if (ep >= SMOOTH_WINDOW and first_solved is None and
                np.mean(episode_rewards[-SMOOTH_WINDOW:]) >= SOLVED_REWARD):
            first_solved = ep
            print(f"  [{name.upper()}] Solved at episode {ep}! "
                  f"(avg{SMOOTH_WINDOW}={np.mean(episode_rewards[-SMOOTH_WINDOW:]):.1f})")

        # REINFORCE update (with baseline + entropy bonus for exploration)
        returns      = discount_returns(rewards, GAMMA, baseline).to(DEVICE)
        log_probs_t  = torch.stack(log_probs)    # (T,)
        entropies_t  = torch.stack(entropies)    # (T,)
        policy_loss  = -(log_probs_t * returns).mean()
        entropy_loss = -ENTROPY_COEF * entropies_t.mean()
        loss         = policy_loss + entropy_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()

        if ep % LOG_INTERVAL == 0:
            avg = np.mean(episode_rewards[-LOG_INTERVAL:])
            print(f"  [{name.upper()}] Episode {ep:4d}/{N_EPISODES}  "
                  f"avg_reward={avg:.1f}")

    env.close()
    n_params = sum(p.numel() for p in policy.parameters())
    win = min(SMOOTH_WINDOW, len(episode_rewards))
    final_avg = float(np.mean(episode_rewards[-win:]))
    return episode_rewards, times, n_params, first_solved, final_avg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")
    results = {}

    for name in MODEL_NAMES:
        print(f"\n{'='*54}")
        print(f"  CartPole RL – {name.upper()} policy ({UNITS} units)")
        print(f"{'='*54}")
        ep_rewards, times, n_params, first_solved, final_avg = train_agent(name)
        results[name] = {
            "episode_rewards":  ep_rewards,
            "Avg_episode_time_s": float(np.mean(times)),
            "Parameters":       n_params,
            "First_solved_ep":  first_solved if first_solved else "N/A",
            "Final_avg_reward": final_avg,
        }
        print(f"  → Final avg={final_avg:.1f}  Params={n_params}  "
              f"Solved ep={first_solved}")

    # ---- Plot: reward curves ----
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, r in results.items():
        raw   = np.array(r["episode_rewards"])
        win   = min(SMOOTH_WINDOW, len(raw))
        if win > 0:
            smooth = np.convolve(raw, np.ones(win) / win, mode="valid")
            ep_idx = np.arange(win - 1, len(raw))
            ax.plot(ep_idx, smooth, lw=2, color=COLORS[name],
                    label=f"{name.upper()} (avg{win})")
        ax.fill_between(np.arange(len(raw)), raw,
                        alpha=0.15, color=COLORS[name])
    ax.axhline(SOLVED_REWARD, color="black", lw=1.0, linestyle="--",
               label=f"Solved ({SOLVED_REWARD:.0f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward")
    ax.set_title("CartPole-v1 REINFORCE – CfC vs LSTM Policy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "reward_curves.png"), dpi=150)
    plt.close()

    # ---- Save metrics CSV ----
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":              name.upper(),
            "Units":              UNITS,
            "Parameters":         r["Parameters"],
            "Final_avg_reward":   round(r["Final_avg_reward"], 2),
            "First_solved_ep":    r["First_solved_ep"],
            "Avg_episode_time_s": round(r["Avg_episode_time_s"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "metrics.csv"), index=False)
    print(f"\nResults saved to {RESULTS_DIR}/")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
