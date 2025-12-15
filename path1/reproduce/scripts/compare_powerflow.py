"""
Compare PandaPower vs Tensor Power Flow (if available) on the 34-node case.

Outputs:
- JSON at reproduce/outputs/metrics/pf_compare.json with timing and voltage diff stats.
Note: Tensor PF uses rl_adn.utility.grid.GridTensor; if import fails, only PandaPower is reported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp

from reproduce.src.network import NetworkFiles, build_pandapower_net, load_network_frames


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "RL-ADN" / "rl_adn" / "data_sources"
OUT_DIR = ROOT / "reproduce" / "outputs" / "metrics"


def add_base_loads(net: pp.pandapowerNet, nodes_df: pd.DataFrame) -> None:
    for _, row in nodes_df.iterrows():
        node = int(row["NODES"])
        pd_kw = float(row.get("PD", 0.0))
        qd_kvar = float(row.get("QD", 0.0))
        if pd_kw == 0 and qd_kvar == 0:
            continue
        bus = net.bus[net.bus.name == f"bus_{node}"].index[0]
        pp.create_load(net, bus=bus, p_mw=pd_kw / 1000.0, q_mvar=qd_kvar / 1000.0)


def run_pandapower(nodes_df: pd.DataFrame, lines_df: pd.DataFrame, repeat: int = 50):
    net, _ = build_pandapower_net(nodes_df, lines_df)
    add_base_loads(net, nodes_df)
    vm_all = []
    t0 = time.time()
    for _ in range(repeat):
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=False, init="flat", numba=False)
        vm_all.append(net.res_bus.vm_pu.to_numpy().copy())
    t1 = time.time()
    return {
        "time_s": t1 - t0,
        "per_run_ms": (t1 - t0) * 1000 / repeat,
        "voltages": np.array(vm_all),
    }


def run_tensor(nodes_df: pd.DataFrame, lines_df: pd.DataFrame, repeat: int = 50):
    try:
        from rl_adn.utility.grid import GridTensor
    except Exception as e:
        return {"error": str(e)}

    grid = GridTensor(nodes_frame=nodes_df, lines_frame=lines_df, from_file=False, numba=True, gpu_mode=False)
    vm_all = []
    t0 = time.time()
    for _ in range(repeat):
        # for constant-power-only case, use precompiled solver
        grid.power_flow_tensor_constant_power_only()
        vm_all.append(grid.v_bus.copy())  # v_bus updated inside solver
    t1 = time.time()
    return {
        "time_s": t1 - t0,
        "per_run_ms": (t1 - t0) * 1000 / repeat,
        "voltages": np.array(vm_all),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_csv = DATA_DIR / "network_data" / "node_34" / "Nodes_34.csv"
    lines_csv = DATA_DIR / "network_data" / "node_34" / "Lines_34.csv"
    nodes_df, lines_df = load_network_frames(NetworkFiles(nodes_csv=nodes_csv, lines_csv=lines_csv))

    repeat = 50
    pp_res = run_pandapower(nodes_df, lines_df, repeat=repeat)
    tensor_res = run_tensor(nodes_df, lines_df, repeat=repeat)

    # voltage diff if tensor available
    dv = None
    if "voltages" in tensor_res and "voltages" in pp_res:
        v_pp = pp_res["voltages"]
        v_te = tensor_res["voltages"]
        n = min(len(v_pp), len(v_te))
        diff = np.abs(v_pp[:n] - v_te[:n])
        dv = {
            "max": float(np.max(diff)),
            "mean": float(np.mean(diff)),
        }

    out = {
        "repeat": repeat,
        "pandapower_time_s": pp_res["time_s"],
        "pandapower_per_run_ms": pp_res["per_run_ms"],
        "tensor_time_s": tensor_res.get("time_s"),
        "tensor_per_run_ms": tensor_res.get("per_run_ms"),
        "tensor_error": tensor_res.get("error"),
        "voltage_diff": dv,
    }

    out_path = OUT_DIR / "pf_compare.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved comparison to {out_path}")


if __name__ == "__main__":
    main()

