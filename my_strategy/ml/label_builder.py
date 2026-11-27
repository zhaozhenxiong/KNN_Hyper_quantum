#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Label generation for ML training."""

from __future__ import annotations

import pandas as pd


def add_forward_return_labels(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    horizons = horizons or [5, 10, 20]
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    for horizon in horizons:
        ret_col = f"future_return_{horizon}d"
        up_col = f"future_up_{horizon}d"
        out[ret_col] = close.shift(-horizon) / close - 1
        out[up_col] = (out[ret_col] > 0).astype(float)
    return out
