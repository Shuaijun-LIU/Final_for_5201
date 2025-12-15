"""
A minimal-but-usable RL-ADN demo with richer plots:
- Uses Laurent/Tensor PF (no PandaPower required for this demo)
- Supports longer training and more detailed visualizations
- Outputs informative figures: training curves (loss/violation/reward
  breakdown), eval episode trajectories (action/SOC/voltage), and a
  voltage-violation heatmap

Usage (from project root is recommended):
    python demo/run_ddpg_quick_demo.py --num-episode 50
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running without `pip install`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for scripts/servers

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

import rl_adn
from rl_adn.DRL_algorithms.Agent import AgentDDPG
from rl_adn.DRL_algorithms.utility import Config, ReplayBuffer
from rl_adn.environments.env import PowerNetEnv


@dataclass
class DemoConfig:
    # Default to a longer run for smoother curves (you can override via CLI)
    num_episode: int = 50
    target_step: int = 1024
    warm_up: int = 4096
    batch_size: int = 256
    repeat_times: int = 1
    gamma: float = 0.99
    learning_rate: float = 6e-4
    net_dims: Tuple[int, int, int] = (256, 256, 256)
    random_seed: int = 521
    gpu_id: int = -1  # default CPU; set 0 to use GPU if available
    buffer_size: int = int(2e5)


def parse_args() -> DemoConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--num-episode", type=int, default=DemoConfig.num_episode)
    p.add_argument("--target-step", type=int, default=DemoConfig.target_step)
    p.add_argument("--warm-up", type=int, default=DemoConfig.warm_up)
    p.add_argument("--batch-size", type=int, default=DemoConfig.batch_size)
    p.add_argument("--repeat-times", type=int, default=DemoConfig.repeat_times)
    p.add_argument("--learning-rate", type=float, default=DemoConfig.learning_rate)
    p.add_argument("--gamma", type=float, default=DemoConfig.gamma)
    p.add_argument("--gpu-id", type=int, default=DemoConfig.gpu_id)
    p.add_argument("--seed", type=int, default=DemoConfig.random_seed)
    p.add_argument("--buffer-size", type=int, default=DemoConfig.buffer_size)
    ns = p.parse_args()
    return DemoConfig(
        num_episode=ns.num_episode,
        target_step=ns.target_step,
        warm_up=ns.warm_up,
        batch_size=ns.batch_size,
        repeat_times=ns.repeat_times,
        gamma=ns.gamma,
        learning_rate=ns.learning_rate,
        net_dims=DemoConfig.net_dims,
        random_seed=ns.seed,
        gpu_id=ns.gpu_id,
        buffer_size=ns.buffer_size,
    )


def build_abs_env_config() -> Dict:
    """Turn relative paths in env_config into absolute ones for robust execution."""
    pkg_dir = Path(rl_adn.__file__).resolve().parent
    data_dir = pkg_dir / "data_sources"

    bus = data_dir / "network_data" / "node_34" / "Nodes_34.csv"
    branch = data_dir / "network_data" / "node_34" / "Lines_34.csv"
    ts = data_dir / "time_series_data" / "34_node_time_series.csv"

    return {
        "voltage_limits": [0.95, 1.05],
        "algorithm": "Laurent",
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


def split_denorm_state(env: PowerNetEnv, denorm_state: np.ndarray):
    """Split denormalized state into netload, soc, price, time, vm_pu_bat."""
    # env._split_state is internal but handy for plotting
    return env._split_state(denorm_state)


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="valid")


def eval_episode(env: PowerNetEnv, policy, device: torch.device):
    """
    Run a full eval episode and record rich timeseries for plotting:
    - normalized_state / denorm_state
    - action ([-1,1]) and converted battery power (kW)
    - SOC / price / voltage / reward terms / violation magnitude
    """
    env.train = False
    state = env.reset()

    T = int(env.episode_length)
    B = len(env.battery_list)

    states_norm = np.zeros((T, env.state_space.shape[0]), dtype=np.float32)
    actions = np.zeros((T, B), dtype=np.float32)
    actions_kw = np.zeros((T, B), dtype=np.float32)
    soc = np.zeros((T, B), dtype=np.float32)
    price = np.zeros((T,), dtype=np.float32)
    volt = np.zeros((T, B), dtype=np.float32)
    reward = np.zeros((T,), dtype=np.float32)
    reward_power = np.zeros((T,), dtype=np.float32)
    reward_penalty = np.zeros((T,), dtype=np.float32)
    violation_mag = np.zeros((T, B), dtype=np.float32)  # amount beyond voltage bounds (p.u.)

    for t in range(T):
        states_norm[t] = state

        den = env._denormalize_state(state.copy())
        _, soc_all, p, _, v_bat = split_denorm_state(env, den)
        soc[t] = soc_all
        price[t] = float(p.item())

        s_tensor = torch.as_tensor((state,), device=device, dtype=torch.float32)
        with torch.no_grad():
            a_tensor = policy(s_tensor)
        a = a_tensor.detach().cpu().numpy()[0]
        a = np.asarray(a).reshape(-1)
        actions[t] = a
        actions_kw[t] = a * 50.0  # max_charge=50kW（battery.py）

        next_state, r, done, _ = env.step(a)
        reward[t] = float(r)
        reward_power[t] = float(getattr(env, "reward_for_power", 0.0))
        reward_penalty[t] = float(getattr(env, "reward_for_penalty", 0.0))

        v_after = np.asarray(env.after_control)[env.battery_list]
        volt[t] = v_after

        low = env.voltage_low_boundary
        high = env.voltage_high_boundary
        below = np.clip(low - v_after, 0, None)
        above = np.clip(v_after - high, 0, None)
        violation_mag[t] = below + above

        state = next_state
        if done:
            break

    return {
        "states_norm": states_norm,
        "actions": actions,
        "actions_kw": actions_kw,
        "soc": soc,
        "price": price,
        "volt": volt,
        "reward": reward,
        "reward_power": reward_power,
        "reward_penalty": reward_penalty,
        "violation_mag": violation_mag,
    }


def plot_training_summary(out_dir: Path, history: Dict, smooth_w: int = 5) -> None:
    # Try multiple styles for compatibility across matplotlib versions
    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(s)
            break
        except Exception:
            continue
    ep = np.asarray(history["episode"])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    ax = axes[0, 0]
    y = np.asarray(history["episode_return"])
    ax.plot(ep, y, alpha=0.35, label="episode_return (raw)")
    if len(y) >= smooth_w:
        ax.plot(ep[smooth_w - 1 :], _moving_avg(y, smooth_w), linewidth=2, label=f"moving_avg(w={smooth_w})")
    ax.set_title("Episode return")
    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(ep, history["violation_time"], marker=".", linewidth=1)
    ax.set_title("Voltage violation count (battery nodes, per episode)")
    ax.set_xlabel("episode")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    ax.plot(ep, history["reward_for_power"], label="reward_for_power", linewidth=1.5)
    ax.plot(ep, history["reward_for_penalty"], label="reward_for_penalty", linewidth=1.5)
    ax.plot(ep, history["episode_return"], label="episode_return", linewidth=1.5, alpha=0.8)
    ax.set_title("Reward decomposition (per episode)")
    ax.set_xlabel("episode")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(ep, history["critic_loss"], label="critic_loss", linewidth=1.5)
    ax.plot(ep, history["actor_loss"], label="actor_loss", linewidth=1.5)
    ax.set_title("Training losses (per episode)")
    ax.set_xlabel("episode")
    ax.legend()

    fig.suptitle("RL-ADN DDPG Training Summary (Laurent / Tensor PF)", fontsize=14)
    fig.savefig(out_dir / "training_summary.png", dpi=220)
    plt.close(fig)


def plot_eval_timeseries(out_dir: Path, battery_list: List[int], eval_data: Dict, vmin: float, vmax: float) -> None:
    for s in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(s)
            break
        except Exception:
            continue
    T = eval_data["reward"].shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, constrained_layout=True)

    axes[0].plot(t, eval_data["price"], color="#333333", linewidth=2)
    axes[0].set_title("Price (raw)")
    axes[0].set_ylabel("price")

    for i, node in enumerate(battery_list):
        axes[1].plot(t, eval_data["actions_kw"][:, i], linewidth=1.5, label=f"node{node}")
    axes[1].axhline(0, color="k", linewidth=1, alpha=0.6)
    axes[1].set_title("Battery action (kW)  (positive=charge, negative=discharge)")
    axes[1].set_ylabel("kW")
    axes[1].legend(ncol=5, fontsize=9, loc="upper right")

    for i, node in enumerate(battery_list):
        axes[2].plot(t, eval_data["soc"][:, i], linewidth=1.5, label=f"node{node}")
    axes[2].set_title("SOC")
    axes[2].set_ylabel("SOC")

    for i, node in enumerate(battery_list):
        axes[3].plot(t, eval_data["volt"][:, i], linewidth=1.5, label=f"V@node{node}")
    axes[3].axhline(vmin, color="r", linestyle="--", linewidth=1.5, label="Vmin")
    axes[3].axhline(vmax, color="r", linestyle="--", linewidth=1.5, label="Vmax")
    axes[3].set_title("Voltage at battery nodes (p.u.)")
    axes[3].set_ylabel("V (p.u.)")
    axes[3].set_xlabel("timestep (15min)")

    # Shade regions where any battery node has voltage violation
    any_violation = (eval_data["violation_mag"].sum(axis=1) > 0).astype(int)
    if any_violation.any():
        for ax in axes:
            ax.fill_between(t, ax.get_ylim()[0], ax.get_ylim()[1], where=any_violation > 0, color="red", alpha=0.06)

    fig.savefig(out_dir / "eval_timeseries.png", dpi=220)
    plt.close(fig)


def plot_violation_heatmap(out_dir: Path, battery_list: List[int], violation_mag: np.ndarray) -> None:
    for s in ("seaborn-v0_8-white", "seaborn-white", "ggplot"):
        try:
            plt.style.use(s)
            break
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    im = ax.imshow(violation_mag.T, aspect="auto", interpolation="nearest", cmap="Reds")
    ax.set_title("Voltage violation magnitude heatmap (p.u.)  (rows=battery nodes, cols=time)")
    ax.set_xlabel("timestep (15min)")
    ax.set_yticks(np.arange(len(battery_list)))
    ax.set_yticklabels([f"node{n}" for n in battery_list])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("violation magnitude (p.u.)")
    fig.savefig(out_dir / "violation_heatmap.png", dpi=220)
    plt.close(fig)


def main() -> None:
    cfg = parse_args()
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)

    # Create a unique output dir per run to avoid overwriting
    out_dir = Path(__file__).resolve().parent / "outputs" / f"run_ep{cfg.num_episode}_step{cfg.target_step}_seed{cfg.random_seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env_train_cfg = build_abs_env_config()
    env_eval_cfg = build_abs_env_config()
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

    args = Config(agent_class=AgentDDPG, env_class=None, env_args=env_args)
    args.run_name = "DDPG_quick_demo"
    args.gamma = cfg.gamma
    args.target_step = cfg.target_step
    args.warm_up = cfg.warm_up
    args.buffer_size = int(cfg.buffer_size)
    args.repeat_times = cfg.repeat_times
    args.batch_size = cfg.batch_size
    args.net_dims = cfg.net_dims
    args.learning_rate = cfg.learning_rate
    args.num_episode = cfg.num_episode
    args.gpu_id = cfg.gpu_id
    args.num_workers = 1
    args.if_remove = True
    args.random_seed = cfg.random_seed
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

    # Warmup (random) to fill buffer for stabler losses
    collected = 0
    with torch.no_grad():
        while collected < args.warm_up:
            buffer_items = agent.explore_env(env_train, args.target_step, if_random=True)
            buffer.update(buffer_items)
            collected = buffer.cur_size

    history = {
        "episode": [],
        "episode_return": [],
        "violation_time": [],
        "reward_for_power": [],
        "reward_for_penalty": [],
        "critic_loss": [],
        "actor_loss": [],
        "buffer_size": [],
    }

    last_eval = None
    for ep in trange(args.num_episode, desc="Training"):
        torch.set_grad_enabled(True)
        critic_loss, actor_loss = agent.update_net(buffer)
        torch.set_grad_enabled(False)

        # Training sampling (non-random)
        with torch.no_grad():
            buffer_items = agent.explore_env(env_train, args.target_step, if_random=False)
            buffer.update(buffer_items)

        # Eval every episode for rich plotting
        last_eval = eval_episode(env_eval, agent.act, agent.device)

        ep_ret = float(last_eval["reward"].sum())
        v_time = int((last_eval["violation_mag"].sum(axis=1) > 0).sum())
        r_power = float(last_eval["reward_power"].sum())
        r_pen = float(last_eval["reward_penalty"].sum())

        print(
            f"[ep {ep}] return={ep_ret:.4f}  violation_time={v_time}  "
            f"reward_power={r_power:.4f}  reward_penalty={r_pen:.4f}  "
            f"critic_loss={critic_loss:.4f}  actor_loss={actor_loss:.4f}"
        )

        history["episode"].append(int(ep))
        history["episode_return"].append(float(ep_ret))
        history["violation_time"].append(int(v_time))
        history["reward_for_power"].append(float(r_power))
        history["reward_for_penalty"].append(float(r_pen))
        history["critic_loss"].append(float(critic_loss))
        history["actor_loss"].append(float(actor_loss))
        history["buffer_size"].append(int(buffer.cur_size))

    # Training summary (higher information density)
    plot_training_summary(out_dir, history, smooth_w=7)

    # Last eval episode timeseries (actions/SOC/voltage with violation shading)
    if last_eval is not None:
        plot_eval_timeseries(
            out_dir,
            battery_list=env_eval.battery_list,
            eval_data=last_eval,
            vmin=env_eval.voltage_low_boundary,
            vmax=env_eval.voltage_high_boundary,
        )
        plot_violation_heatmap(out_dir, env_eval.battery_list, last_eval["violation_mag"])

    (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done. Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()


