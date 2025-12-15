"""
Gymnasium wrapper for the PandaPower dispatch environment.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from .env_pandapower import PandapowerDispatchEnv


class GymDispatchEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, core_env: PandapowerDispatchEnv):
        super().__init__()
        self.core = core_env
        self.action_space = gym.spaces.Box(
            low=self.core.ess_cfg.p_min_kw,
            high=self.core.ess_cfg.p_max_kw,
            shape=(len(self.core.ess_cfg.nodes),),
            dtype=np.float32,
        )
        # Observation: loads + price + soc
        obs_size = len(self.core.data_mgr.column_groups["load"]) + 1 + len(self.core.ess_cfg.nodes)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        day_idx = None
        if options and "day_idx" in options:
            day_idx = options["day_idx"]
        obs = self.core.reset(day_idx=day_idx)
        return obs.astype(np.float32), {}

    def step(self, action):
        obs, reward, terminated, info = None, None, None, None
        obs, reward, done, info = self.core.step(action)
        terminated = done
        truncated = False
        return obs.astype(np.float32), float(reward), terminated, truncated, info


__all__ = ["GymDispatchEnv"]

