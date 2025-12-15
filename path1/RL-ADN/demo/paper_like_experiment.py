"""
Paper-style experiment runner (toward figures like in paper.txt):

1) Training: multi-algo (DDPG/TD3/SAC), multi-seed, long episodes
2) Logging: per-eval return/reward breakdown/violation counts/loss/wall-time
3) Outputs: each run saves metrics.json + final eval trajectory (eval_final.npz)
4) Plotting: companion script demo/paper_like_plots.py can render paper-style figs

Example (from project root):
    python demo/paper_like_experiment.py --algos ddpg,td3,sac --seeds 521,522,523 --num-episode 120 --eval-every 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import trange

# allow running without pip install
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import rl_adn
from rl_adn.DRL_algorithms.Agent import AgentDDPG, AgentSAC, AgentTD3
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer
from rl_adn.environments.env import PowerNetEnv


@dataclass
class ExpConfig:
    algos: Tuple[str, ...] = ("ddpg", "td3", "sac")
    seeds: Tuple[int, ...] = (521, 522, 523)
    num_episode: int = 200
    target_step: int = 1024
    warm_up: int = 8192
    batch_size: int = 256
    repeat_times: int = 1
    gamma: float = 0.99
    learning_rate: float = 6e-4
    net_dims: Tuple[int, int, int] = (256, 256, 256)
    buffer_size: int = int(3e5)
    gpu_id: int = -1
    eval_every: int = 2  # eval interval (smaller = denser curves, larger = faster)
    pf: str = "Laurent"  # Laurent | PandaPower (PandaPower is slower)
    out_root: str = "demo/outputs/paper_like"


AGENT_MAP = {
    "ddpg": AgentDDPG,
    "td3": AgentTD3,
    "sac": AgentSAC,
}


def build_abs_env_config(pf: str) -> Dict:
    pkg_dir = Path(rl_adn.__file__).resolve().parent
    data_dir = pkg_dir / "data_sources"
    bus = data_dir / "network_data" / "node_34" / "Nodes_34.csv"
    branch = data_dir / "network_data" / "node_34" / "Lines_34.csv"
    ts = data_dir / "time_series_data" / "34_node_time_series.csv"
    return {
        "voltage_limits": [0.95, 1.05],
        "algorithm": pf,
        "battery_list": [11, 15, 26, 29, 33],
        "year": 2020,
        "month": 1,
        "day": 1,
        "train": True,
        "state_pattern": "default",
        "network_info": {
            "vm_pu": 1.0,
            "s_base": 1000,
            "bus_info_file": str(bus),
            "branch_info_file": str(branch),
        },
        "time_series_data_path": str(ts),
    }


def eval_episode(env: PowerNetEnv, policy, device: torch.device):
    """Run one full eval episode; log action/SOC/voltage/violation for plotting."""
    env.train = False
    state = env.reset()

    T = int(env.episode_length)
    B = len(env.battery_list)

    actions = np.zeros((T, B), dtype=np.float32)
    actions_kw = np.zeros((T, B), dtype=np.float32)
    soc = np.zeros((T, B), dtype=np.float32)
    price = np.zeros((T,), dtype=np.float32)
    volt = np.zeros((T, B), dtype=np.float32)
    reward = np.zeros((T,), dtype=np.float32)
    reward_power = np.zeros((T,), dtype=np.float32)
    reward_penalty = np.zeros((T,), dtype=np.float32)
    violation_mag = np.zeros((T, B), dtype=np.float32)

    for t in range(T):
        den = env._denormalize_state(state.copy())
        _, soc_all, p, _, _ = env._split_state(den)
        soc[t] = soc_all
        price[t] = float(p.item())

        s_tensor = torch.as_tensor((state,), device=device, dtype=torch.float32)
        with torch.no_grad():
            a_tensor = policy(s_tensor)
        a = a_tensor.detach().cpu().numpy()[0].reshape(-1)
        actions[t] = a
        actions_kw[t] = a * 50.0  # max_charge=50kW

        next_state, r, done, _ = env.step(a)
        reward[t] = float(r)
        reward_power[t] = float(getattr(env, "reward_for_power", 0.0))
        reward_penalty[t] = float(getattr(env, "reward_for_penalty", 0.0))

        v_after = np.asarray(env.after_control)[env.battery_list]
        volt[t] = v_after

        low, high = env.voltage_low_boundary, env.voltage_high_boundary
        below = np.clip(low - v_after, 0, None)
        above = np.clip(v_after - high, 0, None)
        violation_mag[t] = below + above

        state = next_state
        if done:
            break

    ep_ret = float(reward.sum())
    violation_time = int((violation_mag.sum(axis=1) > 0).sum())
    return {
        "episode_return": ep_ret,
        "reward_for_power": float(reward_power.sum()),
        "reward_for_penalty": float(reward_penalty.sum()),
        "violation_time": violation_time,
        "actions": actions,
        "actions_kw": actions_kw,
        "soc": soc,
        "price": price,
        "volt": volt,
        "violation_mag": violation_mag,
    }


def train_one(cfg: ExpConfig, algo: str, seed: int) -> Path:
    assert algo in AGENT_MAP, f"Unsupported algo: {algo}"
    agent_class = AGENT_MAP[algo]

    np.random.seed(seed)
    torch.manual_seed(seed)

    out_dir = Path(cfg.out_root) / f"pf_{cfg.pf.lower()}" / algo / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # env
    env_train_cfg = build_abs_env_config(cfg.pf)
    env_eval_cfg = build_abs_env_config(cfg.pf)
    env_train_cfg["train"] = True
    env_eval_cfg["train"] = False
    env_train = PowerNetEnv(env_train_cfg)
    env_eval = PowerNetEnv(env_eval_cfg)

    env_args = {
        "env_name": "PowerNetEnv",
        "state_dim": env_train.state_space.shape[0],
        "action_dim": env_train.action_space.shape[0],
        "if_discrete": False,
        "num_envs": 1,
        "max_step": int(env_train.episode_length),
    }

    args = Config(agent_class=agent_class, env_class=None, env_args=env_args)
    args.run_name = f"{algo}_paper_like_seed{seed}"
    args.gamma = cfg.gamma
    args.target_step = cfg.target_step
    args.warm_up = cfg.warm_up
    args.buffer_size = int(cfg.buffer_size)
    args.repeat_times = int(cfg.repeat_times)
    args.batch_size = int(cfg.batch_size)
    args.net_dims = cfg.net_dims
    args.learning_rate = float(cfg.learning_rate)
    args.num_episode = int(cfg.num_episode)
    args.gpu_id = int(cfg.gpu_id)
    args.num_workers = 1
    args.if_remove = False  # do not delete cwd to avoid removing other outputs
    args.random_seed = int(seed)
    args.init_before_training()

    agent = args.agent_class(args.net_dims, args.state_dim, args.action_dim, gpu_id=args.gpu_id, args=args)

    buffer = ReplayBuffer(
        gpu_id=args.gpu_id,
        num_seqs=args.num_envs,
        max_size=args.buffer_size,
        state_dim=args.state_dim,
        action_dim=1 if args.if_discrete else args.action_dim,
        if_use_per=args.if_use_per,
        args=args,
    )

    # warmup
    start_wall = time.perf_counter()
    with torch.no_grad():
        while buffer.cur_size < args.warm_up:
            items = agent.explore_env(env_train, args.target_step, if_random=True)
            buffer.update(items)

    history = {
        "config": asdict(cfg),
        "algo": algo,
        "seed": seed,
        "episode": [],
        "episode_return": [],
        "reward_for_power": [],
        "reward_for_penalty": [],
        "violation_time": [],
        "critic_loss": [],
        "actor_loss": [],
        "buffer_size": [],
        "elapsed_sec": [],
    }

    last_eval = None
    for ep in trange(args.num_episode, desc=f"train {algo} seed{seed}", leave=False):
        torch.set_grad_enabled(True)
        update_out = agent.update_net(buffer)
        # SAC returns (critic, actor, alpha); others return (critic, actor)
        if isinstance(update_out, tuple) and len(update_out) == 3:
            critic_loss, actor_loss, alpha_term = update_out
        elif isinstance(update_out, tuple) and len(update_out) == 2:
            critic_loss, actor_loss = update_out
            alpha_term = None
        else:
            critic_loss = actor_loss = update_out if not isinstance(update_out, tuple) else update_out[0]
            alpha_term = None
        torch.set_grad_enabled(False)

        with torch.no_grad():
            items = agent.explore_env(env_train, args.target_step, if_random=False)
            buffer.update(items)

        if (ep % cfg.eval_every) == 0 or ep == args.num_episode - 1:
            last_eval = eval_episode(env_eval, agent.act, agent.device)
            history["episode"].append(int(ep))
            history["episode_return"].append(float(last_eval["episode_return"]))
            history["reward_for_power"].append(float(last_eval["reward_for_power"]))
            history["reward_for_penalty"].append(float(last_eval["reward_for_penalty"]))
            history["violation_time"].append(int(last_eval["violation_time"]))
            history["critic_loss"].append(float(critic_loss))
            history["actor_loss"].append(float(actor_loss))
            history["buffer_size"].append(int(buffer.cur_size))
            history["elapsed_sec"].append(float(time.perf_counter() - start_wall))
            if alpha_term is not None:
                history.setdefault("alpha_term", []).append(float(alpha_term))

    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    if last_eval is not None:
        np.savez_compressed(out_dir / "eval_final.npz", **{k: v for k, v in last_eval.items() if k.endswith(("kw", "soc", "price", "volt", "violation_mag"))})

    return out_dir


def parse_args() -> ExpConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--algos", type=str, default="ddpg,td3,sac")
    p.add_argument("--seeds", type=str, default="521,522,523")
    p.add_argument("--num-episode", type=int, default=ExpConfig.num_episode)
    p.add_argument("--target-step", type=int, default=ExpConfig.target_step)
    p.add_argument("--warm-up", type=int, default=ExpConfig.warm_up)
    p.add_argument("--batch-size", type=int, default=ExpConfig.batch_size)
    p.add_argument("--repeat-times", type=int, default=ExpConfig.repeat_times)
    p.add_argument("--gamma", type=float, default=ExpConfig.gamma)
    p.add_argument("--learning-rate", type=float, default=ExpConfig.learning_rate)
    p.add_argument("--buffer-size", type=int, default=ExpConfig.buffer_size)
    p.add_argument("--gpu-id", type=int, default=ExpConfig.gpu_id)
    p.add_argument("--eval-every", type=int, default=ExpConfig.eval_every)
    p.add_argument("--pf", type=str, default=ExpConfig.pf, choices=["Laurent", "PandaPower"])
    p.add_argument("--out-root", type=str, default=ExpConfig.out_root)
    ns = p.parse_args()

    algos = tuple(a.strip().lower() for a in ns.algos.split(",") if a.strip())
    seeds = tuple(int(s.strip()) for s in ns.seeds.split(",") if s.strip())
    return ExpConfig(
        algos=algos,
        seeds=seeds,
        num_episode=ns.num_episode,
        target_step=ns.target_step,
        warm_up=ns.warm_up,
        batch_size=ns.batch_size,
        repeat_times=ns.repeat_times,
        gamma=ns.gamma,
        learning_rate=ns.learning_rate,
        buffer_size=ns.buffer_size,
        gpu_id=ns.gpu_id,
        eval_every=ns.eval_every,
        pf=ns.pf,
        out_root=ns.out_root,
    )


def main() -> None:
    cfg = parse_args()
    print("Experiment config:", cfg)

    for algo in cfg.algos:
        if algo not in AGENT_MAP:
            raise ValueError(f"algo={algo} is not supported. Available: {sorted(AGENT_MAP.keys())}")

    all_out_dirs: List[Path] = []
    for algo in cfg.algos:
        for seed in cfg.seeds:
            out_dir = train_one(cfg, algo=algo, seed=seed)
            all_out_dirs.append(out_dir)
            print(f"Saved: {out_dir}")

    print("Done. Outputs:")
    for d in all_out_dirs:
        print(" -", d)


if __name__ == "__main__":
    main()


