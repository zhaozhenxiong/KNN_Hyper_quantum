#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sequence dataset helpers for deep learning style agents."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS, clean_feature_matrix


def build_sequence_matrix(
    dataset: pd.DataFrame,
    label_column: str,
    feature_columns: list[str] | None = None,
    lookback: int = 30,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Create flattened lookback windows grouped by stock.

    The label belongs to the last row of each lookback window, so every sample
    uses only information available up to that row.
    """

    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    lookback = int(lookback)
    if lookback <= 1:
        raise ValueError("lookback must be greater than 1")

    xs: list[np.ndarray] = []
    ys: list[float] = []
    metas: list[dict[str, Any]] = []
    required_meta = [col for col in ["date", "stock", "board", "neutral_group", "future_return_5d"] if col in dataset.columns]

    for stock, part in dataset.sort_values(["stock", "date"]).groupby("stock", sort=False):
        part = part.reset_index(drop=True)
        if len(part) < lookback:
            continue
        x_all = clean_feature_matrix(part, feature_columns).to_numpy(dtype=np.float32)
        y_all = pd.to_numeric(part[label_column], errors="coerce").to_numpy(dtype=np.float32)
        for idx in range(lookback - 1, len(part)):
            label = y_all[idx]
            if not np.isfinite(label):
                continue
            window = x_all[idx - lookback + 1 : idx + 1]
            if not np.isfinite(window).all():
                continue
            xs.append(window.reshape(-1))
            ys.append(float(label))
            meta = {col: part.loc[idx, col] for col in required_meta}
            meta["stock"] = stock
            metas.append(meta)

    if not xs:
        return np.empty((0, len(feature_columns) * lookback), dtype=np.float32), np.empty((0,), dtype=np.float32), pd.DataFrame()
    return np.vstack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(metas)


def sample_training_rows(
    x: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    max_samples: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y, meta
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(len(y), size=max_samples, replace=False))
    return x[idx], y[idx], meta.iloc[idx].reset_index(drop=True)

