#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inference helpers for offline RL policy models."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from .features import ACTION_WEIGHTS, Q_FEATURE_COLUMNS, clean_feature_frame
from .registry import load_rl_model


@lru_cache(maxsize=4)
def _cached_load_model(model_name: str, model_root: str | None):
    return load_rl_model(model_name, model_root)


def predict_action_values(state: dict[str, float], model_name: str, model_root: str | None = None) -> dict[str, object]:
    bundle = _cached_load_model(model_name, model_root)
    metadata = bundle.metadata
    actions = [float(x) for x in metadata.get("action_weights", ACTION_WEIGHTS)]
    feature_columns = metadata.get("feature_columns", Q_FEATURE_COLUMNS)

    rows = []
    current_weight = float(state.get("position_weight", 0.0) or 0.0)
    for weight in actions:
        row = dict(state)
        row["action_weight"] = weight
        row["turnover"] = abs(weight - current_weight)
        rows.append(row)
    x = clean_feature_frame(pd.DataFrame(rows), feature_columns)
    q_values = np.asarray(bundle.model.predict(x), dtype=float)
    best_idx = int(np.nanargmax(q_values)) if len(q_values) else 0
    best_weight = actions[best_idx] if actions else 0.0
    ordered = np.sort(q_values)
    spread = float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0
    confidence = float(np.clip(spread / max(float(np.nanstd(q_values)), 1e-6), 0.0, 1.0)) if len(q_values) else 0.0
    return {
        "model_name": model_name,
        "action_weights": actions,
        "q_values": {f"{weight:.2f}": float(value) for weight, value in zip(actions, q_values)},
        "best_weight": float(best_weight),
        "best_q": float(q_values[best_idx]) if len(q_values) else 0.0,
        "confidence": confidence,
        "metadata": metadata,
    }

