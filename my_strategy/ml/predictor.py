#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load trained ML models and append prediction columns."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .feature_builder import DEFAULT_FEATURE_COLUMNS, build_ml_features, clean_feature_matrix
from .model_registry import load_model


@lru_cache(maxsize=8)
def _cached_load_model(model_name: str, model_root: str | None):
    return load_model(model_name, model_root)


def enrich_with_ml_predictions(df: pd.DataFrame, model_name: str, model_root: str | None = None) -> pd.DataFrame:
    bundle = _cached_load_model(model_name, model_root)
    metadata = bundle.metadata
    feature_columns = metadata.get("feature_columns") or DEFAULT_FEATURE_COLUMNS
    task = metadata.get("task", "regression")
    prediction_column = metadata.get("prediction_column", "pred_return_5d")
    prediction_kind = metadata.get("prediction_kind", task)

    out = build_ml_features(df)
    out.attrs["ml_metadata"] = metadata
    x = clean_feature_matrix(out, feature_columns)
    if task == "classification" and hasattr(bundle.model, "predict_proba"):
        pred = bundle.model.predict_proba(x)[:, 1]
        out[prediction_column] = pred
        out["ml_pred_prob"] = pred
        out["ml_score"] = np.clip((pred - 0.5) * 2.0, -1.0, 1.0)
        out["ml_confidence"] = np.clip(np.abs(pred - 0.5) * 2.0, 0.0, 1.0)
    else:
        pred = np.asarray(bundle.model.predict(x), dtype=float)
        out[prediction_column] = pred
        if prediction_kind == "cross_sectional_rank_score":
            out["ml_pred_rank_score"] = pred
            out["ml_score"] = np.clip(pred, -1.0, 1.0)
            out["ml_confidence"] = np.clip(np.abs(pred), 0.0, 1.0)
        else:
            out["ml_pred_return"] = pred
            scale = float(metadata.get("score_scale", 0.05) or 0.05)
            out["ml_score"] = np.clip(pred / scale, -1.0, 1.0)
            out["ml_confidence"] = np.clip(np.abs(pred) / scale, 0.0, 1.0)
    out["ml_model_name"] = model_name
    return out
