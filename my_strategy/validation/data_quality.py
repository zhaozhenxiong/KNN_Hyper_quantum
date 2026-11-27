"""Data quality checks for OHLCV data."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_ohlcv(df: pd.DataFrame, *, date_col: str = "date") -> dict[str, Any]:
    issues: list[str] = []
    if df.empty:
        issues.append("empty_dataframe")
        return {"ok": False, "issues": issues, "rows": 0}
    if date_col not in df.columns:
        issues.append(f"missing_{date_col}")
    else:
        duplicated = int(pd.to_datetime(df[date_col]).duplicated().sum())
        if duplicated:
            issues.append(f"duplicate_dates:{duplicated}")
    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            issues.append(f"missing_{col}")
    if {"open", "high", "low", "close"}.issubset(df.columns):
        numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            issues.append("ohlc_nan")
        if (numeric <= 0).any().any():
            issues.append("ohlc_non_positive")
        if (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1)).any():
            issues.append("high_below_ohlc")
        if (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1)).any():
            issues.append("low_above_ohlc")
    if "volume" in df.columns:
        volume = pd.to_numeric(df["volume"], errors="coerce")
        if (volume < 0).any():
            issues.append("negative_volume")
    return {"ok": not issues, "issues": issues, "rows": int(len(df))}
