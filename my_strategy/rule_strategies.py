#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qbot-inspired rule strategies normalized for screening."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import adx, aroon, arbr, boll, cci, ema, expma, kdj, macd, obv, sar, rsi, stoch_rsi
from .strategy_base import BaseLabStrategy, StrategySpec, make_output


def max_pos(config: dict) -> float:
    return float(config.get("backtest", {}).get("max_single_position", 0.30))


def indicator_config(config: dict, name: str) -> dict:
    return dict(config.get("technical_indicators", {}).get(name, {}))


class MACrossStrategy(BaseLabStrategy):
    spec = StrategySpec("ma_cross", "trend", "Qbot SMA/MA examples", "MA5/MA20 trend following")

    def generate(self, df, config):
        short = df["close"].rolling(5).mean()
        long = df["close"].rolling(20).mean()
        score = np.where(short > long, 0.65, -0.35)
        weight = pd.Series(np.where(short > long, max_pos(config), 0), index=df.index)
        return make_output(df, self.spec.name, pd.Series(score, index=df.index), weight, "MA5/MA20 趋势跟随")


class MACDStrategy(BaseLabStrategy):
    spec = StrategySpec("macd", "trend", "Qbot easy_macd_strategy", "MACD histogram crossover")

    def generate(self, df, config):
        line, sig, hist = macd(df["close"])
        score = (hist / df["close"].rolling(20).std().replace(0, np.nan)).clip(-1, 1).fillna(0)
        weight = pd.Series(np.where((line > sig) & (hist > 0), max_pos(config), 0), index=df.index)
        return make_output(df, self.spec.name, score, weight, "MACD 多空转换")


class MACD5106Strategy(BaseLabStrategy):
    spec = StrategySpec("macd_5_10_6", "trend", "local technical indicator", "MACD(5,10,6) histogram crossover")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        line, signal, hist = macd(
            df["close"],
            fast_span=int(params.get("fast_span", 5)),
            slow_span=int(params.get("slow_span", 10)),
            signal_span=int(params.get("signal_span", 6)),
        )
        cross_up = (line > signal) & (line.shift(1) <= signal.shift(1))
        exit_signal = (line < signal) & (line.shift(1) >= signal.shift(1))
        position = pd.Series(np.nan, index=df.index)
        position[cross_up] = max_pos(config)
        position[exit_signal] = 0
        weight = position.ffill().fillna(0)
        score = (hist / df["close"].rolling(20).std().replace(0, np.nan)).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "MACD(5,10,6) crossover")


class RSIReversalStrategy(BaseLabStrategy):
    spec = StrategySpec("rsi_reversal", "reversal", "Qbot RSI examples", "RSI oversold/overbought reversion")

    def generate(self, df, config):
        value = rsi(df["close"], 14)
        pos = pd.Series(np.nan, index=df.index)
        pos[value < 35] = max_pos(config)
        pos[value > 75] = 0
        weight = pos.ffill().fillna(0)
        score = ((50 - value) / 50).clip(-1, 1)
        return make_output(df, self.spec.name, score, weight, "RSI 超买超卖均值回归")


class BollReversionStrategy(BaseLabStrategy):
    spec = StrategySpec("boll_reversion", "reversal", "Qbot BOLL strategy", "Bollinger lower-band reversion")

    def generate(self, df, config):
        upper, mid, lower = boll(df["close"], 20, 2)
        pos = pd.Series(np.nan, index=df.index)
        pos[df["close"] < lower] = max_pos(config)
        pos[df["close"] > mid] = 0
        weight = pos.ffill().fillna(0)
        band_width = (upper - lower).replace(0, np.nan)
        score = ((mid - df["close"]) / band_width * 2).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "BOLL 下轨低吸，中轨止盈")


class KDJMACDStrategy(BaseLabStrategy):
    spec = StrategySpec("kdj_macd", "hybrid", "Qbot kdj_macd tutorial", "MACD entry, KDJ exit")

    def generate(self, df, config):
        line, sig, hist = macd(df["close"])
        _, d, j = kdj(df)
        macd_cross_up = (line > sig) & (line.shift(1) <= sig.shift(1))
        kdj_cross_down = (j < d) & (j.shift(1) >= d.shift(1))
        pos = pd.Series(np.nan, index=df.index)
        pos[macd_cross_up] = max_pos(config)
        pos[kdj_cross_down] = 0
        weight = pos.ffill().fillna(0)
        score = (0.6 * np.sign(hist.fillna(0)) + 0.4 * ((j - d) / 100).clip(-1, 1)).clip(-1, 1)
        return make_output(df, self.spec.name, pd.Series(score, index=df.index), weight, "MACD 金叉入场，KDJ 转弱离场")


class MACDKDJResonanceStrategy(BaseLabStrategy):
    spec = StrategySpec("macd_kdj_resonance", "hybrid", "local technical indicator", "MACD(5,10,6) and KDJ bullish resonance")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        line, signal, hist = macd(
            df["close"],
            fast_span=int(params.get("fast_span", 5)),
            slow_span=int(params.get("slow_span", 10)),
            signal_span=int(params.get("signal_span", 6)),
        )
        k, d, j = kdj(df, window=int(params.get("kdj_window", 9)))
        macd_cross_up = (line > signal) & (line.shift(1) <= signal.shift(1))
        resonance = macd_cross_up & (k > d) & (j < float(params.get("max_entry_j", 90)))
        exit_signal = (line < signal) | ((k < d) & (j > float(params.get("min_exit_j", 70))))
        position = pd.Series(np.nan, index=df.index)
        position[resonance] = max_pos(config)
        position[exit_signal] = 0
        weight = position.ffill().fillna(0)
        score = (0.55 * np.sign(hist.fillna(0)) + 0.45 * ((k - d) / 100).clip(-1, 1)).clip(-1, 1)
        return make_output(df, self.spec.name, score, weight, "MACD and KDJ resonance")


class ARBRStrategy(BaseLabStrategy):
    spec = StrategySpec("arbr_reversal", "reversal", "local technical indicator", "ARBR oversold reversal")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        ar, br = arbr(df, int(params.get("window", 26)))
        buy = (ar < float(params.get("ar_buy", 70))) & (br < float(params.get("br_buy", 70)))
        sell = (ar > float(params.get("ar_sell", 150))) | (br > float(params.get("br_sell", 300)))
        position = pd.Series(np.nan, index=df.index)
        position[buy] = max_pos(config)
        position[sell] = 0
        weight = position.ffill().fillna(0)
        score = ((100 - ar) / 100 * 0.4 + (100 - br) / 100 * 0.6).clip(-1, 1)
        return make_output(df, self.spec.name, score, weight, "ARBR oversold reversal")


class CCIStrategy(BaseLabStrategy):
    spec = StrategySpec("cci_reversal", "reversal", "local technical indicator", "CCI oversold recovery")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        value = cci(df, int(params.get("window", 14)))
        buy_level = float(params.get("buy_level", -100))
        sell_level = float(params.get("sell_level", 100))
        buy = (value > buy_level) & (value.shift(1) <= buy_level)
        sell = value > sell_level
        position = pd.Series(np.nan, index=df.index)
        position[buy] = max_pos(config)
        position[sell] = 0
        weight = position.ffill().fillna(0)
        return make_output(df, self.spec.name, (value / 200).clip(-1, 1), weight, "CCI reversal")


class OBVTrendStrategy(BaseLabStrategy):
    spec = StrategySpec("obv_trend", "trend", "local technical indicator", "OBV trend confirmation")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        value = obv(df["close"], df["volume"])
        window = int(params.get("window", 20))
        mean = value.rolling(window).mean()
        buy = (value > mean) & (value.diff() > 0)
        sell = (value < mean) & (value.diff() < 0)
        position = pd.Series(np.nan, index=df.index)
        position[buy] = max_pos(config)
        position[sell] = 0
        weight = position.ffill().fillna(0)
        score = ((value - mean) / df["volume"].rolling(window).sum().replace(0, np.nan)).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "OBV trend confirmation")


class SARStrategy(BaseLabStrategy):
    spec = StrategySpec("sar_trend", "trend", "local technical indicator", "Parabolic SAR trend following")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        value = sar(df, step=float(params.get("step", 0.02)), max_step=float(params.get("max_step", 0.20)))
        buy = (df["close"] > value) & (df["close"].shift(1) <= value.shift(1))
        sell = (df["close"] < value) & (df["close"].shift(1) >= value.shift(1))
        position = pd.Series(np.nan, index=df.index)
        position[buy] = max_pos(config)
        position[sell] = 0
        weight = position.ffill().fillna(0)
        score = (df["close"] / value.replace(0, np.nan) - 1).mul(10).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "SAR trend following")


class EXPMAStrategy(BaseLabStrategy):
    spec = StrategySpec("expma_cross", "trend", "local technical indicator", "EXPMA(5,10) crossover")

    def generate(self, df, config):
        params = indicator_config(config, self.spec.name)
        fast, slow = expma(df["close"], int(params.get("fast_span", 5)), int(params.get("slow_span", 10)))
        buy = (fast > slow) & (fast.shift(1) <= slow.shift(1))
        sell = (fast < slow) & (fast.shift(1) >= slow.shift(1))
        position = pd.Series(np.nan, index=df.index)
        position[buy] = max_pos(config)
        position[sell] = 0
        weight = position.ffill().fillna(0)
        score = (fast / slow.replace(0, np.nan) - 1).mul(20).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "EXPMA crossover")


class StochRSIStrategy(BaseLabStrategy):
    spec = StrategySpec("stoch_rsi", "reversal", "Qbot StochRSI", "Stochastic RSI reversion")

    def generate(self, df, config):
        value = stoch_rsi(df["close"])
        pos = pd.Series(np.nan, index=df.index)
        pos[value < 20] = max_pos(config)
        pos[value > 80] = 0
        weight = pos.ffill().fillna(0)
        score = ((50 - value) / 50).clip(-1, 1)
        return make_output(df, self.spec.name, score, weight, "StochRSI 均值回归")


class ADXMACDStrategy(BaseLabStrategy):
    spec = StrategySpec("adx_macd", "trend", "Qbot adx_strategy", "EMA stack + ADX + MACD trend filter")

    def generate(self, df, config):
        ema13 = ema(df["close"], 13)
        ema55 = ema(df["close"], 55)
        ema89 = ema(df["close"], 89)
        _, _, hist = macd(df["close"])
        adx_value = adx(df, 14)
        trend_ok = (ema13 > ema55) & (ema55 > ema89)
        buy = trend_ok & (adx_value <= 35) & (adx_value > adx_value.shift(1)) & (hist > hist.shift(1))
        sell = (~trend_ok) | (hist < hist.shift(1))
        pos = pd.Series(np.nan, index=df.index)
        pos[buy] = max_pos(config)
        pos[sell] = 0
        weight = pos.ffill().fillna(0)
        score = (0.5 * np.where(trend_ok, 1, -1) + 0.3 * np.sign(hist.fillna(0)) + 0.2 * (adx_value / 50)).clip(-1, 1)
        return make_output(df, self.spec.name, pd.Series(score, index=df.index), weight, "EMA 多头排列 + ADX/MACD 趋势确认")


class AroonStrategy(BaseLabStrategy):
    spec = StrategySpec("aroon", "trend", "Qbot Aroon strategy", "AroonUp/AroonDown trend strength")

    def generate(self, df, config):
        up, down = aroon(df, 25)
        energy = up + down
        diff = up - down
        pos = pd.Series(np.nan, index=df.index)
        pos[(energy > 50) & (diff > 0)] = max_pos(config)
        pos[diff < 0] = 0
        weight = pos.ffill().fillna(0)
        score = (diff / 100).clip(-1, 1)
        return make_output(df, self.spec.name, score, weight, "Aroon 趋势强弱")


class VolumeBreakoutStrategy(BaseLabStrategy):
    spec = StrategySpec("volume_breakout", "breakout", "Qbot volume/multi-factor ideas", "Price breakout confirmed by volume")

    def generate(self, df, config):
        high20 = df["high"].rolling(20).max().shift(1)
        low10 = df["low"].rolling(10).min().shift(1)
        vol_ratio = df["volume"] / df["volume"].rolling(20).mean()
        buy = (df["close"] > high20) & (vol_ratio > 1.3)
        sell = df["close"] < low10
        pos = pd.Series(np.nan, index=df.index)
        pos[buy] = max_pos(config)
        pos[sell] = 0
        weight = pos.ffill().fillna(0)
        score = (0.7 * ((df["close"] / high20) - 1) * 10 + 0.3 * (vol_ratio - 1)).clip(-1, 1).fillna(0)
        return make_output(df, self.spec.name, score, weight, "放量突破 20 日高点")


RULE_STRATEGIES = {
    cls.spec.name: cls
    for cls in [
        MACrossStrategy,
        MACDStrategy,
        MACD5106Strategy,
        RSIReversalStrategy,
        BollReversionStrategy,
        KDJMACDStrategy,
        MACDKDJResonanceStrategy,
        ARBRStrategy,
        CCIStrategy,
        OBVTrendStrategy,
        SARStrategy,
        EXPMAStrategy,
        StochRSIStrategy,
        ADXMACDStrategy,
        AroonStrategy,
        VolumeBreakoutStrategy,
    ]
}
