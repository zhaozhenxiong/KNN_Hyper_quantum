#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Feature schema shared by RL training and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd


ACTION_WEIGHTS = [0.0, 0.3, 0.5, 0.7, 0.9]

STATE_FEATURE_COLUMNS = [
    "final_score",
    "ml_score_signal",
    "dl_score_signal",
    "ml_confidence",
    "dl_confidence",
    "position_weight",
    "target_weight",
    "effective_target_weight",
    "unrealized_return",
    "return_1d",
    "return_5d",
    "return_20d",
    "volatility_20d",
    "drawdown",
    "rsi14",
    "volume_ratio",
    "ma20_ratio",
    "ma60_ratio",
    "current_drawdown",
]

Q_FEATURE_COLUMNS = STATE_FEATURE_COLUMNS + ["action_weight", "turnover"]


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def clean_feature_frame(df: pd.DataFrame, feature_columns: list[str] | None = None) -> pd.DataFrame:
    cols = feature_columns or Q_FEATURE_COLUMNS
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            out[col] = 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def state_from_engine_row(row, context: dict, base_final_score: float, base_target_weight: float) -> dict[str, float]:
    close = safe_float(row.get("close", 0), 0)
    ma20 = safe_float(row.get("ma20", close), close)
    ma60 = safe_float(row.get("ma60", close), close)
    entry_price = context.get("entry_price")
    unrealized = close / safe_float(entry_price, close) - 1 if entry_price and close > 0 else 0.0
    current_position = safe_float(context.get("current_position_weight", context.get("current_position", 0)), 0)
    current_drawdown = safe_float(context.get("current_drawdown", 0), 0)
    return {
        "final_score": float(base_final_score),
        "ml_score_signal": safe_float(row.get("ml_score_signal", row.get("ml_score", 0)), 0),
        "dl_score_signal": safe_float(row.get("dl_score_signal", row.get("dl_score", 0)), 0),
        "ml_confidence": safe_float(row.get("ml_confidence", 0), 0),
        "dl_confidence": safe_float(row.get("dl_confidence", 0), 0),
        "position_weight": current_position,
        "target_weight": float(base_target_weight),
        "effective_target_weight": float(base_target_weight),
        "unrealized_return": unrealized,
        "return_1d": safe_float(row.get("return_1d", 0), 0),
        "return_5d": safe_float(row.get("return_5d", 0), 0),
        "return_20d": safe_float(row.get("return_20d", 0), 0),
        "volatility_20d": safe_float(row.get("volatility_20d", 0), 0),
        "drawdown": current_drawdown,
        "rsi14": safe_float(row.get("rsi14", 50), 50),
        "volume_ratio": safe_float(row.get("volume_ratio", 1), 1),
        "ma20_ratio": close / ma20 - 1 if ma20 else 0.0,
        "ma60_ratio": close / ma60 - 1 if ma60 else 0.0,
        "current_drawdown": current_drawdown,
    }

