"""
Network loader for RL-ADN reproduction using PandaPower.

Responsibilities:
- Read node/line CSVs provided by the authors (e.g., Nodes_34.csv, Lines_34.csv).
- Build a PandaPower network with buses, an external grid at the slack, and placeholder
  load/sgen elements (updated each environment step).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import pandapower as pp


@dataclass
class NetworkFiles:
    nodes_csv: Path
    lines_csv: Path
    v_base_kv: float = 11.0  # per-phase base voltage (approx. from paper/code defaults)


def load_network_frames(files: NetworkFiles) -> Tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(files.nodes_csv)
    lines = pd.read_csv(files.lines_csv)
    return nodes, lines


def build_pandapower_net(
    nodes: pd.DataFrame,
    lines: pd.DataFrame,
    v_base_kv: float = 11.0,
) -> Tuple[pp.pandapowerNet, Dict[int, int]]:
    """
    Returns:
        net: pandapower network
        bus_lookup: map from node index (1-based in CSV) to pandapower bus index
    """
    net = pp.create_empty_network(sn_mva=1.0, f_hz=50.0)

    bus_lookup: Dict[int, int] = {}
    for _, row in nodes.iterrows():
        node = int(row["NODES"])
        is_slack = bool(row["Tb"] == 1)
        bus = pp.create_bus(net, vn_kv=v_base_kv, name=f"bus_{node}")
        bus_lookup[node] = bus
        if is_slack:
            pp.create_ext_grid(net, bus=bus, vm_pu=1.0, name=f"slack_{node}")

    # Create line elements; assume per-line parameters are total (length=1 km equivalent).
    for _, row in lines.iterrows():
        from_bus = bus_lookup[int(row["FROM"])]
        to_bus = bus_lookup[int(row["TO"])]
        r_ohm = float(row["R"])
        x_ohm = float(row["X"])
        b_total = float(row.get("B", 0.0))  # shunt susceptance, total
        status = int(row.get("STATUS", 1))
        tap = float(row.get("TAP", 1.0))

        if status == 0:
            continue

        # pandapower requires per-km parameters; we treat given R/X as per-line with len=1 km.
        pp.create_line_from_parameters(
            net,
            from_bus=from_bus,
            to_bus=to_bus,
            length_km=1.0,
            r_ohm_per_km=r_ohm,
            x_ohm_per_km=x_ohm,
            c_nf_per_km=b_total * 1e9 if b_total != 0 else 0.0,
            max_i_ka=1.0,  # placeholder; will not enforce thermal limits in this stage
            name=f"line_{from_bus}_{to_bus}",
            tap_side=None,
            df=1.0,
            parallel=1,
            in_service=True,
            max_loading_percent=100.0,
        )

        # Apply tap ratio by scaling line impedance if tap != 1.0
        if tap != 1.0:
            line_idx = net.line.index[-1]
            net.line.at[line_idx, "tap_pos"] = tap

    return net, bus_lookup


__all__ = ["NetworkFiles", "load_network_frames", "build_pandapower_net"]

