"""
Generate augmented time-series using GMC (GMM + Gaussian Copula).

Outputs:
- CSV at reproduce/outputs/augmented/augmented_{multiplier}x.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from reproduce.src.augment import GMCDataAugmentor
from reproduce.src.data_manager import DataSplitConfig, TimeSeriesDataManager


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "RL-ADN" / "rl_adn" / "data_sources"
OUT_DIR = ROOT / "reproduce" / "outputs" / "augmented"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", type=int, default=5, help="augmented rows = original * multiplier")
    parser.add_argument("--k_max", type=int, default=5, help="max GMM components")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ts_csv = DATA_DIR / "time_series_data" / "34_node_time_series.csv"
    dm = TimeSeriesDataManager(DataSplitConfig(csv_path=ts_csv))
    df = dm.raw_df.reset_index(drop=True)

    target_cols = dm.column_groups["load"] + dm.column_groups["renewable"] + dm.column_groups["price"]
    augmentor = GMCDataAugmentor(df[target_cols], target_cols, k_max=args.k_max).fit()
    n_new = len(df) * args.multiplier
    df_new = augmentor.sample(n_new)

    out_csv = OUT_DIR / f"augmented_{args.multiplier}x.csv"
    df_new.to_csv(out_csv, index=False)
    print(f"Saved augmented data to {out_csv}")


if __name__ == "__main__":
    main()

