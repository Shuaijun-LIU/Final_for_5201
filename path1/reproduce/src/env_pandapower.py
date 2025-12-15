"""
PandaPower-based environment for RL-ADN reproduction (single-phase approximation).

State (default):
    - Net load per node (active power, kW)
    - Price
    - ESS SOCs
Action:
    - ESS power setpoints (kW), positive = discharge to grid, negative = charge.
Reward:
    - price * sum(P_batt) * dt_hours - sigma * voltage_violation_penalty
Penalty:
    - sum of violations beyond [v_min, v_max] per bus (L1 norm).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pandapower as pp

from .data_manager import TimeSeriesDataManager
from .ess import ESSConfig, ESSState
from .network import NetworkFiles, build_pandapower_net, load_network_frames


@dataclass
class EnvConfig:
    data_split: str = "train"  # "train" or "test"
    ess_nodes: List[int] = None
    sigma: float = 400.0
    v_min: float = 0.95
    v_max: float = 1.05
    dt_hours: float = 0.25  # 15-minute resolution
    v_base_kv: float = 11.0
    max_steps: Optional[int] = None  # if None, run full split length


class PandapowerDispatchEnv:
    def __init__(
        self,
        data_mgr: TimeSeriesDataManager,
        network_files: NetworkFiles,
        ess_config: ESSConfig,
        env_config: Optional[EnvConfig] = None,
    ) -> None:
        self.data_mgr = data_mgr
        self.network_files = network_files
        self.env_cfg = env_config or EnvConfig(ess_nodes=ess_config.nodes)
        if self.env_cfg.ess_nodes is None:
            self.env_cfg.ess_nodes = ess_config.nodes

        # Build network
        nodes_df, lines_df = load_network_frames(network_files)
        self.net, self.bus_lookup = build_pandapower_net(nodes_df, lines_df, v_base_kv=self.env_cfg.v_base_kv)

        # Prepare ESS
        self.ess_cfg = ess_config
        self.ess_state = ESSState(ess_config)
        self.ess_bus_idx = [self.bus_lookup[n] for n in self.ess_cfg.nodes]
        self.data_mgr = data_mgr

        # Pre-create loads, RES, ESS injections
        self.load_elements = []
        self.res_elements = []
        self.batt_elements = []

        for node, bus in self.bus_lookup.items():
            # create zero loads; will be updated each step
            load_idx = pp.create_load(self.net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"load_{node}")
            self.load_elements.append((node, load_idx))
            # renewables as negative sgen (injecting)
            res_idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, name=f"res_{node}")
            self.res_elements.append((node, res_idx))
        for node, bus in zip(self.ess_cfg.nodes, self.ess_bus_idx):
            batt_idx = pp.create_sgen(self.net, bus=bus, p_mw=0.0, name=f"batt_{node}")
            self.batt_elements.append((node, batt_idx))

        # Data
        self.split_df = data_mgr.get_train_df() if self.env_cfg.data_split == "train" else data_mgr.get_test_df()
        self.max_steps = self.env_cfg.max_steps or len(self.split_df)
        self.t = 0
        self.done = False

    # -------------
    # Core API
    # -------------
    def reset(self, *, day_idx: Optional[int] = None, t0: int = 0, soc_init: Optional[float] = None) -> np.ndarray:
        self.ess_state.reset(soc_init)
        self.done = False
        if day_idx is not None:
            steps_per_day = int(24 / self.env_cfg.dt_hours)
            self.t = day_idx * steps_per_day
        else:
            self.t = t0
        return self._build_state(self.t)

    def step(self, action_kw: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if self.done:
            raise RuntimeError("Call reset() before stepping a finished episode.")

        # Apply ESS action & SOC update
        applied_action_kw, new_soc = self.ess_state.apply_action(action_kw, self.env_cfg.dt_hours)

        # Update loads/RES/battery injections for current timestep
        row = self.split_df.iloc[self.t]
        self._update_power_injections(row, applied_action_kw)

        # Run power flow
        success = self._run_pf()
        voltages = self.net.res_bus.vm_pu.values if success else np.ones(len(self.bus_lookup))

        # Reward
        reward, penalty = self._compute_reward(applied_action_kw, row, voltages, success)

        # Capture per-bus voltage and load info for visualization
        load_cols = self.data_mgr.column_groups["load"]
        res_cols = self.data_mgr.column_groups["renewable"]
        load_vals = row[load_cols].values.astype(float) if len(load_cols) else None
        res_vals = row[res_cols].values.astype(float) if len(res_cols) else None

        self.t += 1
        if self.t >= self.max_steps:
            self.done = True

        info = {
            "soc": new_soc.copy(),
            "applied_action_kw": applied_action_kw.copy(),
            "voltages": voltages,
            "pf_success": success,
            "penalty": penalty,
            "price": float(row["price"]),
            "loads_kw": load_vals,
            "res_kw": res_vals,
        }
        return self._build_state(self.t), reward, self.done, info

    # -------------
    # Internals
    # -------------
    def _build_state(self, t_idx: int) -> np.ndarray:
        t_idx = min(t_idx, len(self.split_df) - 1)
        row = self.split_df.iloc[t_idx]
        loads = row[self.data_mgr.column_groups["load"]].values.astype(float)
        price = float(row["price"])
        state = np.concatenate([loads, [price], self.ess_state.soc])
        return state

    def _update_power_injections(self, row: pd.Series, batt_kw: np.ndarray) -> None:
        load_cols = self.data_mgr.column_groups["load"]
        res_cols = self.data_mgr.column_groups["renewable"]
        load_vals = row[load_cols].values.astype(float)  # kW
        res_vals = row[res_cols].values.astype(float) if len(res_cols) else np.zeros_like(load_vals)

        # Update loads
        for (node, idx) in self.load_elements:
            bus_load_kw = load_vals[node - 1] if node - 1 < len(load_vals) else 0.0
            self.net.load.at[idx, "p_mw"] = bus_load_kw / 1000.0
            self.net.load.at[idx, "q_mvar"] = 0.0

        # Update renewables (as negative generation)
        for (node, idx) in self.res_elements:
            res_kw = res_vals[node - 1] if node - 1 < len(res_vals) else 0.0
            self.net.sgen.at[idx, "p_mw"] = -res_kw / 1000.0

        # Update batteries
        for i, (node, idx) in enumerate(self.batt_elements):
            p_kw = batt_kw[i] if i < len(batt_kw) else 0.0
            self.net.sgen.at[idx, "p_mw"] = -p_kw / 1000.0

    def _run_pf(self) -> bool:
        try:
            pp.runpp(self.net, algorithm="nr", calculate_voltage_angles=False, init="flat", numba=False)
            return True
        except Exception:
            return False

    def _compute_reward(self, batt_kw: np.ndarray, row: pd.Series, voltages: np.ndarray, pf_success: bool) -> Tuple[float, float]:
        price = float(row["price"])
        energy_mwh = np.sum(batt_kw) * self.env_cfg.dt_hours / 1000.0  # convert kW to MW*h
        revenue = price * energy_mwh

        if not pf_success:
            penalty = 1e3
        else:
            over = np.clip(voltages - self.env_cfg.v_max, 0, None)
            under = np.clip(self.env_cfg.v_min - voltages, 0, None)
            penalty = float(np.sum(over + under))

        reward = revenue - self.env_cfg.sigma * penalty
        return reward, penalty


__all__ = ["EnvConfig", "PandapowerDispatchEnv"]

