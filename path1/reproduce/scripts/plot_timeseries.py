"""
Plot typical-day dispatch (ESS power & SOC) comparing DRL vs NLP baseline.

Inputs:
- DRL training metrics (with trace): reproduce/outputs/metrics/train_{algo}_seed{seed}.json
- NLP baseline results: reproduce/outputs/metrics/nlp_baseline_{mode}_day{day}.json

Outputs:
- reproduce/outputs/figures/timeseries_{algo}_seed{seed}_day{day}.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METRICS_DIR = ROOT / "reproduce" / "outputs" / "metrics"
FIG_DIR = ROOT / "reproduce" / "outputs" / "figures"


def load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="ddpg/td3/sac/ppo")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--day", type=int, default=0)
    parser.add_argument("--nlp_mode", choices=["simplified", "linear"], default="simplified")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # DRL eval metrics with trajectory
    eval_path = METRICS_DIR / f"eval_{args.algo}_seed{args.seed}.json"
    eval_data = load_json(eval_path)
    traj_list = eval_data.get("rollout", {}).get("traj", [])
    traj = traj_list[0] if traj_list else {}
    actions = np.array(traj.get("actions", []))
    voltages = np.array(traj.get("voltages", []))
    soc_traj = np.array(traj.get("soc", []))

    # NLP baseline
    nlp_path = METRICS_DIR / f"nlp_baseline_{args.nlp_mode}_day{args.day}.json"
    nlp_data = load_json(nlp_path)
    Pbatt = np.array(nlp_data.get("Pbatt_kw") or nlp_data.get("P_kw", []))
    SOC = np.array(nlp_data.get("SOC", []))

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(Pbatt, label="NLP Pbatt (kW)")
    if actions.size > 0:
        axes[0].plot(np.sum(actions, axis=1), label="DRL Pbatt sum (kW)", alpha=0.7)
    axes[0].set_ylabel("Power (kW)")
    axes[0].legend()
    axes[0].set_title(f"{args.algo.upper()} seed {args.seed} vs NLP ({args.nlp_mode}) - day {args.day}")

    axes[1].plot(SOC, label="NLP SOC")
    if soc_traj.size > 0:
        axes[1].plot(soc_traj, label="DRL SOC", alpha=0.7)
    axes[1].set_ylabel("SOC")
    axes[1].legend()

    if voltages.size > 0:
        axes[2].plot(voltages, label="DRL voltages (p.u.)", alpha=0.7)
    axes[2].axhline(1.05, color="r", linestyle="--", linewidth=1)
    axes[2].axhline(0.95, color="r", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Voltage (p.u.)")
    axes[2].set_xlabel("Time steps (15min)")
    axes[2].legend()

    out = FIG_DIR / f"timeseries_{args.algo}_seed{args.seed}_day{args.day}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

