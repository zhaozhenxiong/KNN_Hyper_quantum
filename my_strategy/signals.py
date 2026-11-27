#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Signal helpers for the personal trend + RSI strategy."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from khquant_backtester import calc_ma, calc_rsi  # noqa: E402


def build_trend_rsi_signals(
    df: pd.DataFrame,
    short_ma: int = 5,
    long_ma: int = 20,
    trend_ma: int = 50,
    rsi_period: int = 14,
    oversold: int = 30,
    overbought: int = 70,
) -> pd.DataFrame:
    """Build research signals without simulating cash or positions."""
    out = df.copy()
    out["ma_short"] = calc_ma(out, short_ma)
    out["ma_long"] = calc_ma(out, long_ma)
    out["ma_trend"] = calc_ma(out, trend_ma)
    out["rsi"] = calc_rsi(out["close"], rsi_period)
    out["uptrend"] = out["close"] > out["ma_trend"]
    out["short_weak"] = out["ma_short"] < out["ma_long"]
    out["signal"] = 0

    buy = out["uptrend"] & (out["rsi"].shift(1) < oversold) & (out["rsi"] >= oversold)
    sell = (out["rsi"] > overbought) | (out["short_weak"] & ~out["short_weak"].shift(1).fillna(False))
    out.loc[buy, "signal"] = 1
    out.loc[sell, "signal"] = -1
    return out.dropna(subset=["ma_short", "ma_long", "ma_trend", "rsi"]).reset_index(drop=True)


def latest_signal(signal_df: pd.DataFrame) -> dict:
    """Return a compact latest-day signal dictionary."""
    row = signal_df.iloc[-1]
    signal = int(row.get("signal", 0))
    action = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
    return {
        "date": row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"]),
        "action": action,
        "signal": signal,
        "close": float(row["close"]),
        "rsi": float(row["rsi"]),
        "ma_short": float(row["ma_short"]),
        "ma_long": float(row["ma_long"]),
        "ma_trend": float(row["ma_trend"]),
        "uptrend": bool(row["uptrend"]),
    }

