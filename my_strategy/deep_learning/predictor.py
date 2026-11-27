#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load trained deep models and append DL prediction columns."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS, build_ml_features, clean_feature_matrix

from .registry import load_deep_model


@lru_cache(maxsize=4)
def _cached_load_deep_model(model_name: str, model_root: str | None):
    return load_deep_model(model_name, model_root)


def enrich_with_deep_predictions(df: pd.DataFrame, model_name: str, model_root: str | None = None) -> pd.DataFrame:
    bundle = _cached_load_deep_model(model_name, model_root)
    metadata = bundle.metadata
    feature_columns = metadata.get("feature_columns") or DEFAULT_FEATURE_COLUMNS
    lookback = int(metadata.get("lookback", 30))
    prediction_column = metadata.get("prediction_column", "dl_prediction")

    out = build_ml_features(df)
    x = clean_feature_matrix(out, feature_columns).to_numpy(dtype=np.float32)
    pred = np.full(len(out), np.nan, dtype=float)
    if len(out) >= lookback:
        windows = []
        target_idx = []
        for idx in range(lookback - 1, len(out)):
            window = x[idx - lookback + 1 : idx + 1]
            if np.isfinite(window).all():
                windows.append(window.reshape(-1))
                target_idx.append(idx)
        if windows:
            matrix = np.vstack(windows).astype(np.float32)
            matrix = bundle.scaler.transform(matrix)
            pred_values = np.asarray(bundle.model.predict(matrix), dtype=float)
            pred[np.asarray(target_idx, dtype=int)] = pred_values

    out[prediction_column] = pred
    out["dl_prediction"] = pred
    out["dl_score"] = np.clip(pred, -1.0, 1.0)
    out["dl_confidence"] = np.clip(np.abs(out["dl_score"]), 0.0, 1.0)
    out["dl_model_name"] = model_name
    out["dl_model_type"] = metadata.get("model_type", "")
    out["dl_prediction_kind"] = metadata.get("prediction_kind", "cross_sectional_rank_score")
    out.attrs["dl_metadata"] = metadata
    return out
