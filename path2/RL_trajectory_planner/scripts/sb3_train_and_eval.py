"""
Train and evaluate TD3 using Stable-Baselines3, then export a single path to data/paths.json.
"""

import sys
from pathlib import Path

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from env.sb3_env import TerrainSB3Env  # noqa: E402
from env.config import ACTION_SCALE  # noqa: E402
from scripts.render_paths import save_paths, DEFAULT_OUT  # noqa: E402


def make_env():
    return TerrainSB3Env()


def evaluate(env: gym.Env, model: TD3, max_steps=800):
    obs, _ = env.reset()
    path = [obs_to_point(obs)]
    for _ in range(max_steps):
        action, _ = model.predict(np.array(obs), deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        path.append(obs_to_point(obs))
        if terminated or truncated:
            break
    return path


def obs_to_point(obs):
    # obs = (x, y, z, dx, dy, dz, dist, terrain_h)
    return {"x": float(obs[0]), "y": float(obs[1]), "z": float(obs[2])}


def main():
    env = DummyVecEnv([make_env])
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        batch_size=128,
        train_freq=1,
        gradient_steps=1,
        verbose=1,
    )

    # quick training for demo; adjust timesteps as needed
    model.learn(total_timesteps=5000)

    # evaluate on a fresh env
    eval_env = make_env()
    path = evaluate(eval_env, model, max_steps=eval_env.MAX_STEPS if hasattr(eval_env, "MAX_STEPS") else 800)

    # save single path to paths.json
    save_paths([path], DEFAULT_OUT)
    print(f"Saved path with {len(path)} points to {DEFAULT_OUT}")


if __name__ == "__main__":
    main()

