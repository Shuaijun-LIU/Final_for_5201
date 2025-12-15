"""
ESS model helpers for RL-ADN reproduction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class ESSConfig:
    nodes: List[int]  # node indices where ESSs are connected (1-based, as in CSV)
    p_max_kw: float = 50.0
    p_min_kw: float = -50.0
    soc_min: float = 0.2
    soc_max: float = 0.8
    eta_charge: float = 0.95
    eta_discharge: float = 0.95
    e_capacity_kwh: float = 100.0  # placeholder; adjust if paper provides exact values


class ESSState:
    def __init__(self, cfg: ESSConfig):
        self.cfg = cfg
        self.soc = np.full(len(cfg.nodes), (cfg.soc_min + cfg.soc_max) / 2.0, dtype=float)

    def reset(self, soc_init: float | None = None):
        val = soc_init if soc_init is not None else (self.cfg.soc_min + self.cfg.soc_max) / 2.0
        self.soc[:] = np.clip(val, self.cfg.soc_min, self.cfg.soc_max)

    def apply_action(self, action_kw: np.ndarray, dt_hours: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply actions (kW) with SOC updates and bounds.
        Returns:
            clipped_action_kw: applied power (kW), positive = discharge to grid
            new_soc: updated SOC array
        """
        act = np.asarray(action_kw, dtype=float)
        act = np.clip(act, self.cfg.p_min_kw, self.cfg.p_max_kw)

        soc = self.soc.copy()
        for i, p_kw in enumerate(act):
            if p_kw >= 0:  # discharge
                delta_e = p_kw * dt_hours / self.cfg.eta_discharge
                soc[i] -= delta_e / self.cfg.e_capacity_kwh
            else:  # charge
                delta_e = -p_kw * dt_hours * self.cfg.eta_charge
                soc[i] += delta_e / self.cfg.e_capacity_kwh

            if soc[i] > self.cfg.soc_max:
                # reduce charge to hit soc_max
                excess = soc[i] - self.cfg.soc_max
                energy_to_remove = excess * self.cfg.e_capacity_kwh
                p_correction = energy_to_remove / dt_hours / self.cfg.eta_charge
                act[i] -= p_correction
                soc[i] = self.cfg.soc_max
            if soc[i] < self.cfg.soc_min:
                # reduce discharge to hit soc_min
                deficit = self.cfg.soc_min - soc[i]
                energy_to_add = deficit * self.cfg.e_capacity_kwh
                p_correction = energy_to_add / dt_hours * self.cfg.eta_discharge
                act[i] += p_correction
                soc[i] = self.cfg.soc_min

        # final clip
        act = np.clip(act, self.cfg.p_min_kw, self.cfg.p_max_kw)
        self.soc = soc
        return act, self.soc.copy()


__all__ = ["ESSConfig", "ESSState"]

