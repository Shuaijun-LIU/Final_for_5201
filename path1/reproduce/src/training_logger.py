"""
Callbacks and helpers to log training traces for SB3.
"""

from __future__ import annotations

from typing import List

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardPenaltyLogger(BaseCallback):
    """
    Logs per-step reward and penalty (from info['penalty']) into lists.
    Use after training to export traces.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.rewards: List[float] = []
        self.penalties: List[float] = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        infos = self.locals.get("infos")
        if rewards is not None:
            self.rewards.extend([float(r) for r in rewards])
        if infos is not None:
            for info in infos:
                self.penalties.append(float(info.get("penalty", 0.0)))
        return True


__all__ = ["RewardPenaltyLogger"]

