#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Indicator helpers used by the strategy screening lab."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def macd(
    close: pd.Series,
    fast_span: int = 12,
    slow_span: int = 26,
    signal_span: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return DIF, DEA, and histogram for a configurable MACD."""
    if fast_span <= 0 or slow_span <= 0 or signal_span <= 0:
        raise ValueError("MACD spans must be positive")
    fast = ema(close, fast_span)
    slow = ema(close, slow_span)
    line = fast - slow
    signal = ema(line, signal_span)
    hist = line - signal
    return line, signal, hist


def boll(close: pd.Series, window: int = 20, n_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def expma(close: pd.Series, fast_span: int = 5, slow_span: int = 10) -> tuple[pd.Series, pd.Series]:
    """Return the conventional fast and slow EXPMA lines."""
    if fast_span <= 0 or slow_span <= 0:
        raise ValueError("EXPMA spans must be positive")
    return ema(close, fast_span), ema(close, slow_span)


def arbr(df: pd.DataFrame, window: int = 26) -> tuple[pd.Series, pd.Series]:
    """Return AR and BR sentiment indicators using only current/past OHLC."""
    if window <= 0:
        raise ValueError("ARBR window must be positive")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    open_ = pd.to_numeric(df["open"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    ar = 100 * (high - open_).rolling(window).sum() / (open_ - low).rolling(window).sum().replace(0, np.nan)
    previous_close = close.shift(1)
    br_up = (high - previous_close).clip(lower=0)
    br_down = (previous_close - low).clip(lower=0)
    br = 100 * br_up.rolling(window).sum() / br_down.rolling(window).sum().replace(0, np.nan)
    return ar.replace([np.inf, -np.inf], np.nan).fillna(100), br.replace([np.inf, -np.inf], np.nan).fillna(100)


def cci(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Return Commodity Channel Index based on typical price."""
    if window <= 0:
        raise ValueError("CCI window must be positive")
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    mean = typical_price.rolling(window).mean()
    mean_deviation = typical_price.rolling(window).apply(
        lambda values: np.mean(np.abs(values - values.mean())), raw=True
    )
    return ((typical_price - mean) / (0.015 * mean_deviation.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Return On-Balance Volume with an initial value of zero."""
    direction = np.sign(close.diff()).fillna(0)
    signed_volume = direction * pd.to_numeric(volume, errors="coerce").fillna(0)
    return signed_volume.cumsum()


def sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.20) -> pd.Series:
    """Return Parabolic SAR without using data after each calculation date."""
    if not 0 < step <= max_step:
        raise ValueError("SAR requires 0 < step <= max_step")
    high = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    values = np.full(len(df), np.nan, dtype=float)
    if not len(df):
        return pd.Series(values, index=df.index, dtype=float)

    values[0] = low[0]
    is_uptrend = True
    extreme_point = high[0]
    acceleration = step
    for index in range(1, len(df)):
        previous_sar = values[index - 1]
        candidate = previous_sar + acceleration * (extreme_point - previous_sar)
        if is_uptrend:
            candidate = min(candidate, low[index - 1])
            if index > 1:
                candidate = min(candidate, low[index - 2])
            if low[index] < candidate:
                is_uptrend = False
                values[index] = extreme_point
                extreme_point = low[index]
                acceleration = step
            else:
                values[index] = candidate
                if high[index] > extreme_point:
                    extreme_point = high[index]
                    acceleration = min(acceleration + step, max_step)
        else:
            candidate = max(candidate, high[index - 1])
            if index > 1:
                candidate = max(candidate, high[index - 2])
            if high[index] > candidate:
                is_uptrend = True
                values[index] = extreme_point
                extreme_point = high[index]
                acceleration = step
            else:
                values[index] = candidate
                if low[index] < extreme_point:
                    extreme_point = low[index]
                    acceleration = min(acceleration + step, max_step)
    return pd.Series(values, index=df.index, dtype=float)


def kdj(df: pd.DataFrame, window: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    low_n = df["low"].rolling(window).min()
    high_n = df["high"].rolling(window).max()
    rsv = 100 * (df["close"] - low_n) / (high_n - low_n).replace(0, np.nan)
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k.fillna(50), d.fillna(50), j.fillna(50)


def stoch_rsi(close: pd.Series, rsi_window: int = 14, stoch_window: int = 14) -> pd.Series:
    r = rsi(close, rsi_window)
    low = r.rolling(stoch_window).min()
    high = r.rolling(stoch_window).max()
    return (100 * (r - low) / (high - low).replace(0, np.nan)).fillna(50)


def aroon(df: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    def periods_since_high(x):
        return window - 1 - int(np.argmax(x))

    def periods_since_low(x):
        return window - 1 - int(np.argmin(x))

    high_since = df["high"].rolling(window).apply(periods_since_high, raw=True)
    low_since = df["low"].rolling(window).apply(periods_since_low, raw=True)
    aroon_up = 100 * (window - high_since) / window
    aroon_down = 100 * (window - low_since) / window
    return aroon_up.fillna(0), aroon_down.fillna(0)


def adx(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(window).mean().fillna(0)
