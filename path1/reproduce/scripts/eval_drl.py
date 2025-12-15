"""
Evaluate trained DRL models against NLP baseline to compute performance bound.

Outputs:
- metrics JSON at reproduce/outputs/metrics/eval_{algo}_seed{seed}.json
  containing episode reward, penalty, estimated cost, and performance bound
  vs. NLP (simplified or linear).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv

from reproduce.src.data_manager import DataSplitConfig, TimeSeriesDataManager
from reproduce.src.env_pandapower import EnvConfig, PandapowerDispatchEnv
from reproduce.src.env_wrapper import GymDispatchEnv
from reproduce.src.ess import ESSConfig
from reproduce.src.network import NetworkFiles, load_network_frames
from reproduce.src.metrics import performance_bound
from reproduce.src.nlp_baseline import run_simplified_nlp, run_linear_pf_nlp


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "RL-ADN" / "rl_adn" / "data_sources"
OUTPUT_METRICS = ROOT / "reproduce" / "outputs" / "metrics"
OUTPUT_MODELS = ROOT / "reproduce" / "outputs" / "models"

ALGOS = {
    "ddpg": DDPG,
    "td3": TD3,
    "sac": SAC,
    "ppo": PPO,
}


def make_env(split: str) -> gym.Env:
    ts_csv = DATA_DIR / "time_series_data" / "34_node_time_series.csv"
    dm = TimeSeriesDataManager(DataSplitConfig(csv_path=ts_csv, test_days=30))
    nodes_csv = DATA_DIR / "network_data" / "node_34" / "Nodes_34.csv"
    lines_csv = DATA_DIR / "network_data" / "node_34" / "Lines_34.csv"
    network_files = NetworkFiles(nodes_csv=nodes_csv, lines_csv=lines_csv)
    ess_cfg = ESSConfig(nodes=[12, 16, 27, 34])
    env_cfg = EnvConfig(data_split=split, max_steps=None)
    core_env = PandapowerDispatchEnv(dm, network_files, ess_cfg, env_cfg)
    return GymDispatchEnv(core_env), dm, nodes_csv, lines_csv


def rollout(model, env: DummyVecEnv, episodes: int = 1):
    rewards = []
    penalties = []
    price_energy_mwh = []
    traj = []
    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_penalty = 0.0
        ep_energy = 0.0
        ep_traj = {"voltages": [], "actions": [], "loads_kw": [], "res_kw": [], "soc": []}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            info0 = info[0]
            ep_penalty += float(info0.get("penalty", 0.0))
            price = float(info0.get("price", 0.0))
            p_kw = np.sum(info0.get("applied_action_kw", 0.0))
            ep_energy += price * (p_kw * 0.25 / 1000.0)  # approximate cost with price*energy
            # collect trajectories
            ep_traj["voltages"].append(info0.get("voltages"))
            ep_traj["actions"].append(info0.get("applied_action_kw"))
            ep_traj["loads_kw"].append(info0.get("loads_kw"))
            ep_traj["res_kw"].append(info0.get("res_kw"))
            ep_traj["soc"].append(info0.get("soc"))
            done = done[0]
        rewards.append(ep_reward)
        penalties.append(ep_penalty)
        price_energy_mwh.append(ep_energy)
        traj.append(ep_traj)
    return {
        "reward": float(np.mean(rewards)),
        "penalty": float(np.mean(penalties)),
        "cost_est": float(np.mean(price_energy_mwh)),
        "traj": traj,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS.keys()), default="ddpg")
    parser.add_argument("--seed", type=int, default=521)
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--nlp_mode", choices=["simplified", "linear"], default="simplified")
    args = parser.parse_args()

    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)

    # load env and model
    env, dm, nodes_csv, lines_csv = make_env(args.split)
    vec_env = DummyVecEnv([lambda: env])
    model_path = OUTPUT_MODELS / f"{args.algo}_seed{args.seed}.zip"
    model = ALGOS[args.algo].load(model_path, env=vec_env)

    # rollout
    rollout_res = rollout(model, vec_env, episodes=args.episodes)

    # NLP baseline on same horizon (use first day of test split)
    df = dm.get_test_df() if args.split == "test" else dm.get_train_df()
    steps_per_day = int(24 / 0.25)
    df_day = df.iloc[:steps_per_day]
    ess_cfg = ESSConfig(nodes=[12, 16, 27, 34])
    load_cols = dm.column_groups["load"]
    res_cols = dm.column_groups["renewable"]
    price_col = dm.column_groups["price"][0]
    nodes_df = None
    lines_df = None

    if args.nlp_mode == "simplified":
        sol = run_simplified_nlp(df_day, ess_cfg, 0.25, load_cols, res_cols, price_col)
        cost_opt = sol["objective"]
    else:
        from reproduce.src.network import load_network_frames

        nodes_df_frame, lines_df_frame = load_network_frames(NetworkFiles(nodes_csv=nodes_csv, lines_csv=lines_csv))
        sol = run_linear_pf_nlp(
            df_day,
            nodes_df_frame,
            lines_df_frame,
            ess_cfg,
            0.25,
            load_cols,
            res_cols,
            price_col,
        )
        cost_opt = sol["objective"]

    bound = performance_bound(rollout_res["cost_est"], cost_opt)

    out = {
        "algo": args.algo,
        "seed": args.seed,
        "split": args.split,
        "episodes": args.episodes,
        "nlp_mode": args.nlp_mode,
        "rollout": rollout_res,
        "nlp_cost": float(cost_opt),
        "performance_bound": float(bound),
    }

    out_path = OUTPUT_METRICS / f"eval_{args.algo}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Saved eval metrics to {out_path}")


if __name__ == "__main__":
    main()

