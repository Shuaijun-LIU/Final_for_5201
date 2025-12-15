"""
Time-series data manager for RL-ADN reproduction.

Features:
- Load author-provided CSV time-series (15-min resolution).
- Fill missing timestamps, forward/backward fill missing values.
- Split into train/test by ratio or cut-off date or tail days.
- Provide day-level and timeslot-level selectors for the environment.
- Expose column groups (load, renewable, price) inferred from column names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


FREQ = "15min"


@dataclass
class DataSplitConfig:
    csv_path: Union[str, Path]
    freq: str = FREQ
    train_ratio: float = 0.8  # used if no split_date or test_days provided
    split_date: Optional[str] = None  # ISO date string; rows < split_date go to train
    test_days: Optional[int] = None  # if set, last N days go to test
    timezone: Optional[str] = None  # e.g., "UTC"; None keeps source tz


class TimeSeriesDataManager:
    def __init__(self, config: DataSplitConfig) -> None:
        self.config = config
        self.raw_df = self._load_and_clean(Path(config.csv_path))
        self.column_groups = self._infer_column_groups(self.raw_df.columns)
        self.train_df, self.test_df = self._split(self.raw_df, config)

    # --------------------
    # Public API
    # --------------------
    def get_train_df(self) -> pd.DataFrame:
        return self.train_df

    def get_test_df(self) -> pd.DataFrame:
        return self.test_df

    def select_day_data(self, split: str, day_idx: int) -> pd.DataFrame:
        df = self._pick_split(split)
        steps_per_day = self._steps_per_day(df.index.freq or pd.tseries.frequencies.to_offset(self.config.freq))
        start = day_idx * steps_per_day
        end = start + steps_per_day
        return df.iloc[start:end]

    def select_timeslot(self, split: str, idx: int) -> pd.Series:
        df = self._pick_split(split)
        return df.iloc[idx]

    def summary(self) -> Dict[str, Union[int, Dict[str, float]]]:
        return {
            "rows": len(self.raw_df),
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
            "columns": len(self.raw_df.columns),
            "time_range": {
                "start": str(self.raw_df.index.min()),
                "end": str(self.raw_df.index.max()),
                "freq": str(self.raw_df.index.freq),
            },
        }

    # --------------------
    # Internal helpers
    # --------------------
    def _load_and_clean(self, csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "date_time" not in df.columns:
            raise ValueError("Expected a 'date_time' column in the CSV.")

        df["date_time"] = pd.to_datetime(df["date_time"], utc=True, errors="coerce")
        df = df.dropna(subset=["date_time"]).sort_values("date_time")
        df = df.set_index("date_time")

        # Enforce regular frequency by reindexing
        freq = self.config.freq
        full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq=freq, tz=df.index.tz)
        df = df.reindex(full_idx)

        # Fill missing values: forward fill then back fill
        df = df.ffill().bfill()

        # Optionally convert timezone
        if self.config.timezone:
            df = df.tz_convert(self.config.timezone)

        # Ensure numeric columns are float
        numeric_cols = df.columns
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df = df.ffill().bfill()

        return df

    def _infer_column_groups(self, columns: List[str]) -> Dict[str, List[str]]:
        load_cols = [c for c in columns if c.startswith("active_power_node_")]
        renewable_cols = [c for c in columns if c.startswith("renewable_active_power_node_")]
        price_cols = [c for c in columns if c.lower() == "price"]
        return {
            "load": load_cols,
            "renewable": renewable_cols,
            "price": price_cols,
        }

    def _split(self, df: pd.DataFrame, config: DataSplitConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if config.test_days:
            steps_per_day = self._steps_per_day(pd.tseries.frequencies.to_offset(config.freq))
            test_steps = config.test_days * steps_per_day
            train_df = df.iloc[:-test_steps] if test_steps < len(df) else df.iloc[:0]
            test_df = df.iloc[-test_steps:]
            return train_df, test_df

        if config.split_date:
            split_ts = pd.to_datetime(config.split_date)
            train_df = df.loc[df.index < split_ts]
            test_df = df.loc[df.index >= split_ts]
            return train_df, test_df

        # default: ratio
        cut = int(len(df) * config.train_ratio)
        train_df = df.iloc[:cut]
        test_df = df.iloc[cut:]
        return train_df, test_df

    def _pick_split(self, split: str) -> pd.DataFrame:
        if split.lower() == "train":
            return self.train_df
        if split.lower() == "test":
            return self.test_df
        raise ValueError("split must be 'train' or 'test'")

    def _steps_per_day(self, offset: pd.tseries.offsets.DateOffset) -> int:
        return int(pd.Timedelta(days=1) / pd.Timedelta(offset))


__all__ = ["DataSplitConfig", "TimeSeriesDataManager"]

