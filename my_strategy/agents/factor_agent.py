#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Factor calculation agent."""

from __future__ import annotations

import numpy as np
import pandas as pd

from my_strategy.indicators import arbr, boll, cci, expma, kdj, macd, obv, sar


class FactorAgent:
    @staticmethod
    def calc_rsi(close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        return rsi.fillna(50)

    @staticmethod
    def add_factors(df: pd.DataFrame, volume_window: int = 20) -> pd.DataFrame:
        out = df.copy()
        out["ma5"] = out["close"].rolling(5).mean()
        out["ma10"] = out["close"].rolling(10).mean()
        out["ma20"] = out["close"].rolling(20).mean()
        out["ma60"] = out["close"].rolling(60).mean()
        out["rsi14"] = FactorAgent.calc_rsi(out["close"], 14)
        out["return_1d"] = out["close"].pct_change().fillna(0)
        out["return_5d"] = out["close"].pct_change(5)
        out["return_20d"] = out["close"].pct_change(20)
        out["vol_ma20"] = out["volume"].rolling(volume_window).mean()
        out["volume_ratio"] = (out["volume"] / out["vol_ma20"]).replace([np.inf, -np.inf], np.nan).fillna(1)

        out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
        out["macd_5_10_6"], out["macd_5_10_6_signal"], out["macd_5_10_6_hist"] = macd(
            out["close"], fast_span=5, slow_span=10, signal_span=6
        )
        out["boll_upper"], out["boll_mid"], out["boll_lower"] = boll(out["close"])
        boll_width = (out["boll_upper"] - out["boll_lower"]).replace(0, np.nan)
        out["boll_pct_b"] = ((out["close"] - out["boll_lower"]) / boll_width).clip(-1, 2)
        out["boll_bandwidth"] = boll_width / out["boll_mid"].replace(0, np.nan)
        out["kdj_k"], out["kdj_d"], out["kdj_j"] = kdj(out)
        out["macd_kdj_resonance"] = (
            (out["macd_5_10_6"] > out["macd_5_10_6_signal"])
            & (out["kdj_k"] > out["kdj_d"])
            & (out["kdj_j"] < 90)
        ).astype(float)
        out["ar_26"], out["br_26"] = arbr(out, 26)
        out["cci_14"] = cci(out, 14)
        out["obv"] = obv(out["close"], out["volume"])
        out["obv_change_20d"] = out["obv"].diff(20) / out["volume"].rolling(20).sum().replace(0, np.nan)
        out["sar"] = sar(out)
        out["sar_gap"] = out["close"] / out["sar"].replace(0, np.nan) - 1
        out["expma_5"], out["expma_10"] = expma(out["close"], 5, 10)
        out["expma_5_10_ratio"] = out["expma_5"] / out["expma_10"].replace(0, np.nan) - 1
        out["volatility_20d"] = out["return_1d"].rolling(20).std().fillna(0)
        return out.replace([np.inf, -np.inf], np.nan)
