"""
Run NLP baselines (simplified or linearized grid-aware) on 34-node data.

Outputs:
- metrics JSON at reproduce/outputs/metrics/nlp_baseline_{mode}_day{day_idx}.json
  containing objective, Pbatt, SOC, (optionally Pline/V for linear), and config.

Usage:
  python run_nlp_baseline.py --mode simplified --day_idx 0 --solver glpk
  python run_nlp_baseline.py --mode linear --day_idx 0 --solver glpk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reproduce.src.data_manager import DataSplitConfig, TimeSeriesDataManager
from reproduce.src.ess import ESSConfig
from reproduce.src.network import NetworkFiles, load_network_frames
from reproduce.src.nlp_baseline import (
    NLPSolverConfig,
    run_linear_pf_nlp,
    run_simplified_nlp,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "RL-ADN" / "rl_adn" / "data_sources"
OUTPUT_DIR = ROOT / "reproduce" / "outputs" / "metrics"


def select_day(df: pd.DataFrame, day_idx: int, freq: str = "15min") -> pd.DataFrame:
    steps_per_day = int(pd.Timedelta("1D") / pd.Timedelta(freq))
    start = day_idx * steps_per_day
    end = start + steps_per_day
    return df.iloc[start:end]


def numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: numpy_to_list(v) for k, v in obj.items()}
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simplified", "linear"], default="simplified")
    parser.add_argument("--day_idx", type=int, default=0, help="day index in selected split")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--solver", type=str, default="glpk")
    parser.add_argument("--tee", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Data manager (34-node time series)
    ts_csv = DATA_DIR / "time_series_data" / "34_node_time_series.csv"
    dm = TimeSeriesDataManager(DataSplitConfig(csv_path=ts_csv, test_days=30))
    df = dm.get_train_df() if args.split == "train" else dm.get_test_df()
    df_day = select_day(df, args.day_idx, freq="15min")

    # Network data
    nodes_csv = DATA_DIR / "network_data" / "node_34" / "Nodes_34.csv"
    lines_csv = DATA_DIR / "network_data" / "node_34" / "Lines_34.csv"
    nodes_df, lines_df = load_network_frames(NetworkFiles(nodes_csv=nodes_csv, lines_csv=lines_csv))

    # ESS config (paper default)
    ess_cfg = ESSConfig(nodes=[12, 16, 27, 34])

    load_cols = dm.column_groups["load"]
    res_cols = dm.column_groups["renewable"]
    price_col = dm.column_groups["price"][0]
    dt_hours = 0.25

    solver_cfg = NLPSolverConfig(solver=args.solver, tee=args.tee)
    if args.mode == "simplified":
        sol = run_simplified_nlp(
            df_day,
            ess_cfg,
            dt_hours,
            load_cols,
            res_cols,
            price_col,
            solver_cfg,
        )
    else:
        sol = run_linear_pf_nlp(
            df_day,
            nodes_df,
            lines_df,
            ess_cfg,
            dt_hours,
            load_cols,
            res_cols,
            price_col,
            solver_cfg,
        )

    out_path = OUTPUT_DIR / f"nlp_baseline_{args.mode}_day{args.day_idx}.json"
    payload = {
        "mode": args.mode,
        "day_idx": args.day_idx,
        "split": args.split,
        "objective": sol.get("objective"),
        "Pbatt_kw": numpy_to_list(sol.get("P_kw")),
        "SOC": numpy_to_list(sol.get("SOC")),
        "config": {
            "ess_nodes": ess_cfg.nodes,
            "p_min_kw": ess_cfg.p_min_kw,
            "p_max_kw": ess_cfg.p_max_kw,
            "soc_min": ess_cfg.soc_min,
            "soc_max": ess_cfg.soc_max,
            "dt_hours": dt_hours,
            "solver": args.solver,
        },
    }
    if args.mode == "linear":
        payload["Pline_kw"] = numpy_to_list(sol.get("Pline_kw"))
        payload["V_pu2"] = numpy_to_list(sol.get("V_pu2"))

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved baseline results to {out_path}")


if __name__ == "__main__":
    main()

