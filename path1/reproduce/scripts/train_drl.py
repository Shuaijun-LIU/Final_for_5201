"""
Train DRL agent (DDPG/TD3/SAC/PPO) on the PandaPower dispatch env using SB3.

Outputs:
- metrics JSON at reproduce/outputs/metrics/train_{algo}_seed{seed}.json
- saved model at reproduce/outputs/models/{algo}_seed{seed}.zip
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
from reproduce.src.training_logger import RewardPenaltyLogger


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


def make_env(split: str, day_idx: int | None = None) -> gym.Env:
    ts_csv = DATA_DIR / "time_series_data" / "34_node_time_series.csv"
    dm = TimeSeriesDataManager(DataSplitConfig(csv_path=ts_csv, test_days=30))
    nodes_csv = DATA_DIR / "network_data" / "node_34" / "Nodes_34.csv"
    lines_csv = DATA_DIR / "network_data" / "node_34" / "Lines_34.csv"
    network_files = NetworkFiles(nodes_csv=nodes_csv, lines_csv=lines_csv)
    ess_cfg = ESSConfig(nodes=[12, 16, 27, 34])
    env_cfg = EnvConfig(data_split=split, max_steps=None)
    core_env = PandapowerDispatchEnv(dm, network_files, ess_cfg, env_cfg)
    return GymDispatchEnv(core_env)


def train(algo_name: str, total_steps: int, seed: int, split: str):
    algo_cls = ALGOS[algo_name]
    env = DummyVecEnv([lambda: make_env(split)])
    callback = RewardPenaltyLogger()
    model = algo_cls("MlpPolicy", env, seed=seed, verbose=0)
    model.learn(total_timesteps=total_steps, callback=callback)
    return model, env, callback


def evaluate(model, env: DummyVecEnv, episodes: int = 5):
    rewards = []
    penalties = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        ep_penalty = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            ep_reward += reward[0]
            info0 = info[0]
            ep_penalty += float(info0.get("penalty", 0.0))
            done = done[0]
        rewards.append(ep_reward)
        penalties.append(ep_penalty)
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_penalty": float(np.mean(penalties)),
        "std_penalty": float(np.std(penalties)),
        "episodes": episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGOS.keys()), default="ddpg")
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--seeds", type=str, default="521")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--eval_episodes", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT_MODELS.mkdir(parents=True, exist_ok=True)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    for seed in seeds:
        model, env, cb = train(args.algo, args.total_steps, seed, args.split)
        eval_res = evaluate(model, env, episodes=args.eval_episodes)

        model_path = OUTPUT_MODELS / f"{args.algo}_seed{seed}.zip"
        model.save(model_path)

        metrics = {
            "algo": args.algo,
            "total_steps": args.total_steps,
            "seed": seed,
            "split": args.split,
            "eval": eval_res,
            "trace": {
                "reward": cb.rewards,
                "penalty": cb.penalties,
            },
        }
        metrics_path = OUTPUT_METRICS / f"train_{args.algo}_seed{seed}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Saved model to {model_path}")
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

