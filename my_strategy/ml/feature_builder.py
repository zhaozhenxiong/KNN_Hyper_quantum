#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature engineering for daily ML agents."""

from __future__ import annotations

import numpy as np
import pandas as pd

from my_strategy.indicators import arbr, boll, cci, expma, kdj, macd, obv, sar


LEGACY_FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "ma5_ratio",
    "ma10_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "ma20_ma60_ratio",
    "rsi14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_20d",
    "volume_ratio",
    "amount_change_5d",
    "high_low_range",
    "close_to_20d_high",
    "close_to_20d_low",
]

TECHNICAL_FEATURE_COLUMNS = [
    "boll_pct_b",
    "boll_bandwidth",
    "macd_5_10_6",
    "macd_5_10_6_signal",
    "macd_5_10_6_hist",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "macd_kdj_resonance",
    "ar_26",
    "br_26",
    "cci_14",
    "obv_change_20d",
    "sar_gap",
    "expma_5_10_ratio",
]

DEFAULT_FEATURE_COLUMNS = [*LEGACY_FEATURE_COLUMNS, *TECHNICAL_FEATURE_COLUMNS]
FEATURE_PROFILES = {
    "legacy": LEGACY_FEATURE_COLUMNS,
    "extended": DEFAULT_FEATURE_COLUMNS,
}


def feature_columns_for_profile(profile: str) -> list[str]:
    """Return a copy of the named, versioned training feature profile."""
    try:
        return list(FEATURE_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown feature profile: {profile}") from exc


def build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out.get("high", close), errors="coerce")
    low = pd.to_numeric(out.get("low", close), errors="coerce")
    open_ = pd.to_numeric(out.get("open", close), errors="coerce")
    volume_source = out["volume"] if "volume" in out.columns else pd.Series(0.0, index=out.index)
    volume = pd.to_numeric(volume_source, errors="coerce").fillna(0)
    amount_source = out["amount"] if "amount" in out.columns else pd.Series(0.0, index=out.index)
    amount = pd.to_numeric(amount_source, errors="coerce").fillna(0)

    out["return_3d"] = close.pct_change(3)
    out["return_10d"] = close.pct_change(10)
    for window in [5, 10, 20, 60]:
        ma_col = f"ma{window}"
        if ma_col in out.columns:
            out[f"{ma_col}_ratio"] = close / pd.to_numeric(out[ma_col], errors="coerce") - 1
    if "ma20" in out.columns and "ma60" in out.columns:
        out["ma20_ma60_ratio"] = pd.to_numeric(out["ma20"], errors="coerce") / pd.to_numeric(out["ma60"], errors="coerce") - 1

    out["amount_change_5d"] = amount.pct_change(5).replace([np.inf, -np.inf], np.nan)
    out["high_low_range"] = high / low.replace(0, np.nan) - 1
    rolling_high = high.rolling(20).max()
    rolling_low = low.rolling(20).min()
    out["close_to_20d_high"] = close / rolling_high.replace(0, np.nan) - 1
    out["close_to_20d_low"] = close / rolling_low.replace(0, np.nan) - 1

    indicator_frame = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    out["boll_upper"], out["boll_mid"], out["boll_lower"] = boll(close)
    boll_width = (out["boll_upper"] - out["boll_lower"]).replace(0, np.nan)
    out["boll_pct_b"] = ((close - out["boll_lower"]) / boll_width).clip(-1, 2)
    out["boll_bandwidth"] = boll_width / out["boll_mid"].replace(0, np.nan)
    out["macd_5_10_6"], out["macd_5_10_6_signal"], out["macd_5_10_6_hist"] = macd(
        close, fast_span=5, slow_span=10, signal_span=6
    )
    out["kdj_k"], out["kdj_d"], out["kdj_j"] = kdj(indicator_frame)
    out["macd_kdj_resonance"] = (
        (out["macd_5_10_6"] > out["macd_5_10_6_signal"])
        & (out["kdj_k"] > out["kdj_d"])
        & (out["kdj_j"] < 90)
    ).astype(float)
    out["ar_26"], out["br_26"] = arbr(indicator_frame, 26)
    out["cci_14"] = cci(indicator_frame, 14)
    out["obv"] = obv(close, volume)
    out["obv_change_20d"] = out["obv"].diff(20) / volume.rolling(20).sum().replace(0, np.nan)
    out["sar"] = sar(indicator_frame)
    out["sar_gap"] = close / out["sar"].replace(0, np.nan) - 1
    out["expma_5"], out["expma_10"] = expma(close, 5, 10)
    out["expma_5_10_ratio"] = out["expma_5"] / out["expma_10"].replace(0, np.nan) - 1

    for col in DEFAULT_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out[DEFAULT_FEATURE_COLUMNS] = out[DEFAULT_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out


def clean_feature_matrix(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    x = df[columns].copy()
    # SQLite may materialize dynamically added feature columns as TEXT even
    # when their logical values are numeric. Coerce at the model boundary.
    x = x.apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    return x.fillna(0.0)
