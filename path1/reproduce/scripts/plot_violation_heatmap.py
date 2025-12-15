"""
Plot voltage violation heatmap from eval trajectory.

Uses eval metrics produced by scripts/eval_drl.py (traj with voltages).
Outputs: reproduce/outputs/figures/violation_heatmap_{algo}_seed{seed}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reproduce" / "outputs" / "metrics"
FIG_DIR = ROOT / "reproduce" / "outputs" / "figures"


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episode_idx", type=int, default=0, help="which episode in eval traj")
    args = parser.parse_args()

    eval_path = METRICS_DIR / f"eval_{args.algo}_seed{args.seed}.json"
    data = load_json(eval_path)
    traj_list = data.get("rollout", {}).get("traj", [])
    if not traj_list:
        print("No trajectory found in eval metrics.")
        return
    ep = traj_list[min(args.episode_idx, len(traj_list) - 1)]
    voltages = np.array(ep.get("voltages", []))  # shape (T, B)
    if voltages.size == 0:
        print("No voltages in trajectory.")
        return
    over = np.clip(voltages - 1.05, 0, None)
    under = np.clip(0.95 - voltages, 0, None)
    violation = over + under  # shape (T, B)

    plt.figure(figsize=(10, 6))
    plt.imshow(violation.T, aspect="auto", origin="lower", cmap="Reds")
    plt.colorbar(label="Voltage violation (p.u.)")
    plt.xlabel("Time steps")
    plt.ylabel("Bus index")
    plt.title(f"Voltage violations - {args.algo.upper()} seed {args.seed}")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"violation_heatmap_{args.algo}_seed{args.seed}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

