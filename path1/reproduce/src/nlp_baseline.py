"""
Step-1 NLP baseline (simplified, no grid constraints) for RL-ADN reproduction.

What it does:
- Builds a Pyomo model for a single-day (or given horizon) ESS dispatch with perfect
  knowledge of load/renewable/price time series.
- Constraints: ESS power bounds, SOC dynamics, SOC bounds.
- Objective: minimize purchase cost = sum_t price_t * (net_demand_t) * dt_hours,
  where net_demand_t = total_load_kw - total_res_kw + sum(P_batt_kw).

Notes:
- This is the first-pass baseline without network voltage/current constraints.
  A second pass will extend to include grid constraints consistent with the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .ess import ESSConfig


@dataclass
class NLPSolverConfig:
    solver: str = "glpk"
    solver_io: Optional[str] = None
    tee: bool = False


def build_simplified_nlp(
    df: pd.DataFrame,
    ess_cfg: ESSConfig,
    dt_hours: float,
    load_cols: List[str],
    res_cols: List[str],
    price_col: str = "price",
) -> pyo.ConcreteModel:
    """
    df: time-indexed dataframe with columns load_cols, res_cols, price_col.
    ess_cfg: ESSConfig with nodes aligned to entries in ess_cfg.nodes.
    """
    T = len(df)
    M = len(ess_cfg.nodes)

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.M = pyo.RangeSet(0, M - 1)

    # Parameters
    load_kw = df[load_cols].sum(axis=1).values.astype(float)
    res_kw = df[res_cols].sum(axis=1).values.astype(float) if len(res_cols) else np.zeros(T)
    price = df[price_col].values.astype(float)

    model.load_kw = pyo.Param(model.T, initialize=lambda m, t: load_kw[t], within=pyo.Reals)
    model.res_kw = pyo.Param(model.T, initialize=lambda m, t: res_kw[t], within=pyo.Reals)
    model.price = pyo.Param(model.T, initialize=lambda m, t: price[t], within=pyo.Reals)

    # Decision variables
    model.P = pyo.Var(model.T, model.M, within=pyo.Reals)  # kW; positive = discharge
    model.SOC = pyo.Var(model.T, model.M, bounds=(ess_cfg.soc_min, ess_cfg.soc_max))

    # Constraints
    def power_bounds_rule(m, t, i):
        return pyo.inequality(ess_cfg.p_min_kw, m.P[t, i], ess_cfg.p_max_kw)

    model.power_bounds = pyo.Constraint(model.T, model.M, rule=power_bounds_rule)

    def soc_init_rule(m, i):
        # initialize at mid-point if not specified externally; could be parameterized later
        return m.SOC[0, i] == (ess_cfg.soc_min + ess_cfg.soc_max) / 2.0

    model.soc_init = pyo.Constraint(model.M, rule=soc_init_rule)

    def soc_dyn_rule(m, t, i):
        if t == 0:
            return pyo.Constraint.Skip
        p_kw = m.P[t - 1, i]
        # charge/discharge efficiency
        charge = (-p_kw) * ess_cfg.eta_charge * dt_hours
        discharge = (p_kw / ess_cfg.eta_discharge) * dt_hours
        delta_soc = (charge - discharge) / ess_cfg.e_capacity_kwh
        return m.SOC[t, i] == m.SOC[t - 1, i] + delta_soc

    model.soc_dyn = pyo.Constraint(model.T, model.M, rule=soc_dyn_rule)

    # Objective
    def obj_rule(m):
        total = 0.0
        for t in m.T:
            net_kw = m.load_kw[t] - m.res_kw[t] + sum(m.P[t, i] for i in m.M)
            total += m.price[t] * (net_kw * dt_hours / 1000.0)  # price is per MWh
        return total

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return model


def solve_model(model: pyo.ConcreteModel, solver_cfg: NLPSolverConfig) -> pyo.SolverResults:
    solver = pyo.SolverFactory(solver_cfg.solver, solver_io=solver_cfg.solver_io)
    results = solver.solve(model, tee=solver_cfg.tee)
    return results


def extract_solution(model: pyo.ConcreteModel) -> Dict[str, np.ndarray]:
    T = len(model.T)
    M = len(model.M)
    P = np.zeros((T, M))
    SOC = np.zeros((T, M))
    for t in model.T:
        for i in model.M:
            P[t, i] = pyo.value(model.P[t, i])
            SOC[t, i] = pyo.value(model.SOC[t, i])
    obj = pyo.value(model.obj)
    return {"P_kw": P, "SOC": SOC, "objective": obj}


def run_simplified_nlp(
    df: pd.DataFrame,
    ess_cfg: ESSConfig,
    dt_hours: float,
    load_cols: List[str],
    res_cols: List[str],
    price_col: str = "price",
    solver_cfg: Optional[NLPSolverConfig] = None,
) -> Dict[str, np.ndarray]:
    """
    Convenience wrapper: build, solve, and return solution dict.
    """
    solver_cfg = solver_cfg or NLPSolverConfig()
    model = build_simplified_nlp(df, ess_cfg, dt_hours, load_cols, res_cols, price_col)
    _ = solve_model(model, solver_cfg)
    return extract_solution(model)


__all__ = [
    "NLPSolverConfig",
    "build_simplified_nlp",
    "solve_model",
    "extract_solution",
    "run_simplified_nlp",
]


# ------------------------------------------------------------
# Step-2 (approximate grid-aware) using linearized DistFlow
# ------------------------------------------------------------


def _build_incidence(lines: pd.DataFrame) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Build parent->children map and parent map assuming radial structure."""
    children: Dict[int, List[int]] = {}
    parent: Dict[int, int] = {}
    for _, row in lines.iterrows():
        i = int(row["FROM"])
        j = int(row["TO"])
        children.setdefault(i, []).append(j)
        parent[j] = i
    return children, parent


def build_linear_pf_nlp(
    df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    ess_cfg: ESSConfig,
    dt_hours: float,
    load_cols: List[str],
    res_cols: List[str],
    price_col: str = "price",
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> pyo.ConcreteModel:
    """
    Linearized radial DistFlow (active power only, Q≈0, ignore losses).
    Voltage drop: V_j = V_i - 2 * R_ij * P_ij / V_base^2 (with V_base≈1 p.u.).
    """
    T = len(df)
    M = len(ess_cfg.nodes)

    # Map node index (1-based) to position
    node_ids = nodes_df["NODES"].astype(int).tolist()
    node_to_idx = {n: k for k, n in enumerate(node_ids)}
    children, parent = _build_incidence(lines_df)

    # Time-varying net demand per node (kW)
    load_mat = df[load_cols].values  # shape (T, N_load)
    res_mat = df[res_cols].values if len(res_cols) else np.zeros_like(load_mat)
    net_load = load_mat - res_mat  # kW consumption positive
    price = df[price_col].values.astype(float)

    # Line parameters
    line_list = []
    for _, row in lines_df.iterrows():
        i = int(row["FROM"])
        j = int(row["TO"])
        R = float(row["R"])
        X = float(row.get("X", 0.0))
        line_list.append((i, j, R, X))

    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)
    model.B = pyo.RangeSet(0, len(node_ids) - 1)  # buses
    model.M = pyo.RangeSet(0, M - 1)  # ESS index
    model.L = pyo.RangeSet(0, len(line_list) - 1)  # lines

    # Parameters
    model.price = pyo.Param(model.T, initialize=lambda m, t: price[t], within=pyo.Reals)
    model.net_load = pyo.Param(
        model.T, model.B, initialize=lambda m, t, b: net_load[t, b], within=pyo.Reals
    )

    # Decision variables
    model.Pbatt = pyo.Var(model.T, model.M, within=pyo.Reals)  # kW
    model.SOC = pyo.Var(model.T, model.M, bounds=(ess_cfg.soc_min, ess_cfg.soc_max))
    model.Pline = pyo.Var(model.T, model.L, within=pyo.Reals)  # kW (flow from from->to)
    model.V = pyo.Var(model.T, model.B, bounds=(v_min ** 2, v_max ** 2))

    # ESS constraints
    def power_bounds_rule(m, t, i):
        return pyo.inequality(ess_cfg.p_min_kw, m.Pbatt[t, i], ess_cfg.p_max_kw)

    model.power_bounds = pyo.Constraint(model.T, model.M, rule=power_bounds_rule)

    def soc_init_rule(m, i):
        return m.SOC[0, i] == (ess_cfg.soc_min + ess_cfg.soc_max) / 2.0

    model.soc_init = pyo.Constraint(model.M, rule=soc_init_rule)

    def soc_dyn_rule(m, t, i):
        if t == 0:
            return pyo.Constraint.Skip
        p_kw = m.Pbatt[t - 1, i]
        charge = (-p_kw) * ess_cfg.eta_charge * dt_hours
        discharge = (p_kw / ess_cfg.eta_discharge) * dt_hours
        delta_soc = (charge - discharge) / ess_cfg.e_capacity_kwh
        return m.SOC[t, i] == m.SOC[t - 1, i] + delta_soc

    model.soc_dyn = pyo.Constraint(model.T, model.M, rule=soc_dyn_rule)

    # Network constraints (linearized, active-only)
    # Slack bus voltage fixed at 1.0 p.u. squared
    slack_bus_idx = node_to_idx.get(1, 0)

    def slack_voltage_rule(m, t):
        return m.V[t, slack_bus_idx] == 1.0

    model.slack_voltage = pyo.Constraint(model.T, rule=slack_voltage_rule)

    # Power balance per bus: sum inflow - sum outflow = net_load + batt at that bus
    def power_balance_rule(m, t, b):
        node = node_ids[b]
        inflow = 0.0
        outflow = 0.0
        for l_idx, (i, j, _, _) in enumerate(line_list):
            if j == node:
                inflow += m.Pline[t, l_idx]
            if i == node:
                outflow += m.Pline[t, l_idx]

        batt_injection = 0.0
        if node in ess_cfg.nodes:
            m_idx = ess_cfg.nodes.index(node)
            batt_injection = m.Pbatt[t, m_idx]

        return inflow - outflow == m.net_load[t, b] + batt_injection

    model.power_balance = pyo.Constraint(model.T, model.B, rule=power_balance_rule)

    # Voltage drop along lines: V_j = V_i - 2 * R * P_ij (lossless, Q=0)
    def voltage_drop_rule(m, t, l):
        i, j, R, _ = line_list[l]
        vi = m.V[t, node_to_idx[i]]
        vj = m.V[t, node_to_idx[j]]
        return vj == vi - 2 * R * m.Pline[t, l] / 1000.0  # convert kW to MW in R*P if R is in Ohm? approximate

    model.voltage_drop = pyo.Constraint(model.T, model.L, rule=voltage_drop_rule)

    # Objective: minimize purchase cost at slack (sum of all loads + batt injections)
    def obj_rule(m):
        total = 0.0
        for t in m.T:
            net_kw_total = sum(m.net_load[t, b] for b in m.B) + sum(m.Pbatt[t, i] for i in m.M)
            total += m.price[t] * (net_kw_total * dt_hours / 1000.0)
        return total

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return model


def run_linear_pf_nlp(
    df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    lines_df: pd.DataFrame,
    ess_cfg: ESSConfig,
    dt_hours: float,
    load_cols: List[str],
    res_cols: List[str],
    price_col: str = "price",
    solver_cfg: Optional[NLPSolverConfig] = None,
    v_min: float = 0.95,
    v_max: float = 1.05,
) -> Dict[str, np.ndarray]:
    """
    Build and solve the linearized grid-aware NLP; returns solution dict.
    """
    solver_cfg = solver_cfg or NLPSolverConfig()
    model = build_linear_pf_nlp(
        df,
        nodes_df,
        lines_df,
        ess_cfg,
        dt_hours,
        load_cols,
        res_cols,
        price_col,
        v_min,
        v_max,
    )
    _ = solve_model(model, solver_cfg)

    T = len(model.T)
    M = len(model.M)
    L = len(model.L)
    P = np.zeros((T, M))
    SOC = np.zeros((T, M))
    V = np.zeros((T, len(nodes_df)))
    Pline = np.zeros((T, L))

    for t in model.T:
        for i in model.M:
            P[t, i] = pyo.value(model.Pbatt[t, i])
            SOC[t, i] = pyo.value(model.SOC[t, i])
        for l in model.L:
            Pline[t, l] = pyo.value(model.Pline[t, l])
        for b in model.B:
            V[t, b] = pyo.value(model.V[t, b])

    obj = pyo.value(model.obj)
    return {"P_kw": P, "SOC": SOC, "Pline_kw": Pline, "V_pu2": V, "objective": obj}

