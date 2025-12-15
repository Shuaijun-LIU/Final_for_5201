"""
Plot training curves and performance bound bars from metrics JSONs.

Inputs:
- Training metrics: reproduce/outputs/metrics/train_{algo}_seed{seed}.json
- Eval metrics: reproduce/outputs/metrics/eval_{algo}_seed{seed}.json

Outputs:
- reproduce/outputs/figures/train_reward_penalty.png
- reproduce/outputs/figures/performance_bound.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reproduce" / "outputs" / "metrics"
FIG_DIR = ROOT / "reproduce" / "outputs" / "figures"


def load_training_metrics() -> List[dict]:
    metrics = []
    for path in METRICS_DIR.glob("train_*.json"):
        with open(path, "r") as f:
            metrics.append(json.load(f))
    return metrics


def load_eval_metrics() -> List[dict]:
    metrics = []
    for path in METRICS_DIR.glob("eval_*.json"):
        with open(path, "r") as f:
            metrics.append(json.load(f))
    return metrics


def moving_average(x: List[float], window: int = 200) -> np.ndarray:
    if len(x) == 0:
        return np.array([])
    x = np.array(x, dtype=float)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def plot_train_curves(metrics: List[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    for m in metrics:
        rewards = m.get("trace", {}).get("reward", [])
        penalties = m.get("trace", {}).get("penalty", [])
        algo = m.get("algo", "algo")
        seed = m.get("seed", 0)
        r_sm = moving_average(rewards, window=200)
        p_sm = moving_average(penalties, window=200)
        plt.plot(r_sm, label=f"{algo}-seed{seed}-reward")
        plt.plot(p_sm, label=f"{algo}-seed{seed}-penalty", linestyle="--")
    plt.title("Training reward and penalty (moving average)")
    plt.xlabel("Steps")
    plt.ylabel("Value")
    plt.legend()
    out = FIG_DIR / "train_reward_penalty.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_performance_bound(eval_metrics: List[dict]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    bounds = []
    for m in eval_metrics:
        algo = m.get("algo", "algo")
        seed = m.get("seed", 0)
        b = m.get("performance_bound", np.nan)
        labels.append(f"{algo}-s{seed}")
        bounds.append(b)
    if not bounds:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(labels, bounds)
    plt.axhline(0, color="k", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Performance bound (lower is better)")
    plt.title("DRL vs NLP performance bound")
    plt.tight_layout()
    out = FIG_DIR / "performance_bound.png"
    plt.savefig(out, dpi=200)
    plt.close()


def main():
    train_metrics = load_training_metrics()
    eval_metrics = load_eval_metrics()
    if train_metrics:
        plot_train_curves(train_metrics)
        print("Saved train_reward_penalty.png")
    else:
        print("No training metrics found.")
    if eval_metrics:
        plot_performance_bound(eval_metrics)
        print("Saved performance_bound.png")
    else:
        print("No eval metrics found.")


if __name__ == "__main__":
    main()

