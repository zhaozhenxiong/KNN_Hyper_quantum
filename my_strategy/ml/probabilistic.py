#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Rolling Bayesian and Markov-state features for the AI Agent stack."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .feature_builder import build_ml_features


DEFAULT_PROB_MODEL_NAME = "rolling_bayes_markov_v1"


def enrich_with_probabilistic_predictions(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append no-lookahead Bayesian and Markov scores to a daily factor frame.

    The model is intentionally local and rolling:
    - Naive Bayes is fitted from rows whose future label would already be known.
    - Markov transition probabilities are fitted from historical regime states.
    """
    cfg = config or {}
    horizon = int(cfg.get("horizon", 5))
    min_history = int(cfg.get("min_history", 80))
    bayes_weight = float(cfg.get("bayes_weight", 0.60))
    markov_weight = float(cfg.get("markov_weight", 0.40))
    alpha = float(cfg.get("laplace_alpha", 1.0))
    model_name = str(cfg.get("model_name") or DEFAULT_PROB_MODEL_NAME)

    out = build_ml_features(df)
    out = out.sort_values("date").reset_index(drop=True)
    close = pd.to_numeric(out["close"], errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    labels = (future_return > 0).astype(float)

    feature_frame = _categorical_features(out)
    states = _market_states(out)

    rows: list[dict[str, Any]] = []
    for idx in range(len(out)):
        train_end = idx - horizon
        if train_end < min_history:
            rows.append(_neutral_row(model_name, states.iloc[idx]))
            continue

        train_features = feature_frame.iloc[:train_end].reset_index(drop=True)
        train_labels = labels.iloc[:train_end].reset_index(drop=True)
        valid_mask = train_labels.notna()
        train_features = train_features.loc[valid_mask].reset_index(drop=True)
        train_labels = train_labels.loc[valid_mask].astype(int).reset_index(drop=True)
        if len(train_labels) < min_history or train_labels.nunique() < 2:
            rows.append(_neutral_row(model_name, states.iloc[idx]))
            continue

        prob_up = _naive_bayes_prob_up(train_features, train_labels, feature_frame.iloc[idx], alpha=alpha)
        bayes_score = float(np.clip((prob_up - 0.5) * 2.0, -1.0, 1.0))

        transition_state, state_prob, bull_prob, bear_prob = _markov_next_state(states.iloc[:idx], states.iloc[idx], alpha=alpha)
        markov_score = float(np.clip(bull_prob - bear_prob, -1.0, 1.0))
        combined = float(np.clip(bayes_weight * bayes_score + markov_weight * markov_score, -1.0, 1.0))
        confidence = float(np.clip(max(abs(bayes_score), abs(markov_score), abs(combined)), 0.0, 1.0))

        rows.append(
            {
                "prob_model_name": model_name,
                "bayes_prob_up": round(float(prob_up), 8),
                "bayes_score": round(bayes_score, 8),
                "markov_state": str(states.iloc[idx]),
                "markov_next_state": str(transition_state),
                "markov_state_prob": round(float(state_prob), 8),
                "markov_bull_prob": round(float(bull_prob), 8),
                "markov_bear_prob": round(float(bear_prob), 8),
                "markov_score": round(markov_score, 8),
                "prob_score": round(combined, 8),
                "prob_confidence": round(confidence, 8),
            }
        )

    prob = pd.DataFrame(rows)
    for col in prob.columns:
        out[col] = prob[col].values
    out.attrs["probabilistic_metadata"] = {
        "model_name": model_name,
        "model_type": "rolling_naive_bayes_markov",
        "horizon": horizon,
        "min_history": min_history,
        "score_column": "prob_score",
        "confidence_column": "prob_confidence",
        "prediction_column": "bayes_prob_up",
    }
    return out


def _neutral_row(model_name: str, state: str) -> dict[str, Any]:
    return {
        "prob_model_name": model_name,
        "bayes_prob_up": 0.5,
        "bayes_score": 0.0,
        "markov_state": str(state),
        "markov_next_state": str(state),
        "markov_state_prob": 0.0,
        "markov_bull_prob": 0.0,
        "markov_bear_prob": 0.0,
        "markov_score": 0.0,
        "prob_score": 0.0,
        "prob_confidence": 0.0,
    }


def _categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["ret_5d"] = _bucket(pd.to_numeric(df.get("return_5d", 0), errors="coerce"), [-0.05, -0.02, 0.0, 0.02, 0.05])
    out["ret_20d"] = _bucket(pd.to_numeric(df.get("return_20d", 0), errors="coerce"), [-0.10, -0.04, 0.0, 0.04, 0.10])
    out["ma20"] = _bucket(pd.to_numeric(df.get("ma20_ratio", 0), errors="coerce"), [-0.08, -0.03, 0.0, 0.03, 0.08])
    out["ma60"] = _bucket(pd.to_numeric(df.get("ma60_ratio", 0), errors="coerce"), [-0.12, -0.05, 0.0, 0.05, 0.12])
    out["rsi"] = _bucket(pd.to_numeric(df.get("rsi14", 50), errors="coerce"), [30, 40, 50, 60, 70])
    out["vol"] = _bucket(pd.to_numeric(df.get("volatility_20d", 0), errors="coerce"), [0.015, 0.025, 0.04, 0.06])
    out["volume"] = _bucket(pd.to_numeric(df.get("volume_ratio", 1), errors="coerce"), [0.6, 0.9, 1.2, 1.8])
    return out.fillna("missing").astype(str)


def _bucket(series: pd.Series, bins: list[float]) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    bucketed = pd.cut(values, bins=[-math.inf, *bins, math.inf], labels=False)
    return bucketed.astype("Int64").astype(str).where(values.notna(), "missing")


def _market_states(df: pd.DataFrame) -> pd.Series:
    ret_20d = pd.to_numeric(df.get("return_20d", 0), errors="coerce").fillna(0)
    ma20_ratio = pd.to_numeric(df.get("ma20_ratio", 0), errors="coerce").fillna(0)
    ma60_ratio = pd.to_numeric(df.get("ma60_ratio", 0), errors="coerce").fillna(0)
    volatility = pd.to_numeric(df.get("volatility_20d", 0), errors="coerce").fillna(0)
    high_vol = volatility > max(float(volatility.quantile(0.75)), 0.04)
    bull = (ret_20d > 0.04) & (ma20_ratio > 0) & (ma60_ratio > -0.02)
    bear = (ret_20d < -0.04) & (ma20_ratio < 0) & (ma60_ratio < 0.02)
    states = pd.Series("sideways", index=df.index, dtype=object)
    states.loc[bull] = "bull"
    states.loc[bear] = "bear"
    states.loc[high_vol & bull] = "bull_high_vol"
    states.loc[high_vol & bear] = "bear_high_vol"
    states.loc[high_vol & ~(bull | bear)] = "volatile"
    return states


def _naive_bayes_prob_up(features: pd.DataFrame, labels: pd.Series, current: pd.Series, alpha: float = 1.0) -> float:
    classes = [0, 1]
    log_probs: dict[int, float] = {}
    n = len(labels)
    for cls in classes:
        cls_mask = labels.eq(cls)
        cls_count = int(cls_mask.sum())
        prior = (cls_count + alpha) / (n + alpha * len(classes))
        log_prob = math.log(max(prior, 1e-12))
        cls_features = features.loc[cls_mask]
        for col in features.columns:
            values = features[col].astype(str)
            unique_count = max(int(values.nunique()), 1)
            hit_count = int((cls_features[col].astype(str) == str(current[col])).sum())
            likelihood = (hit_count + alpha) / (cls_count + alpha * unique_count)
            log_prob += math.log(max(likelihood, 1e-12))
        log_probs[cls] = log_prob

    max_log = max(log_probs.values())
    down = math.exp(log_probs[0] - max_log)
    up = math.exp(log_probs[1] - max_log)
    return up / (up + down) if (up + down) else 0.5


def _markov_next_state(history: pd.Series, current_state: str, alpha: float = 1.0) -> tuple[str, float, float, float]:
    clean = history.dropna().astype(str).reset_index(drop=True)
    states = sorted(set(clean.tolist()) | {"bull", "bear", "sideways", "volatile", "bull_high_vol", "bear_high_vol"})
    if len(clean) < 3:
        return current_state, 0.0, 0.0, 0.0

    transitions = pd.DataFrame(alpha, index=states, columns=states, dtype=float)
    for left, right in zip(clean.iloc[:-1], clean.iloc[1:]):
        transitions.loc[left, right] += 1.0
    row = transitions.loc[current_state] if current_state in transitions.index else transitions.mean(axis=0)
    probs = row / row.sum()
    next_state = str(probs.idxmax())
    state_prob = float(probs.max())
    bull_prob = float(probs.get("bull", 0.0) + probs.get("bull_high_vol", 0.0))
    bear_prob = float(probs.get("bear", 0.0) + probs.get("bear_high_vol", 0.0) + 0.5 * probs.get("volatile", 0.0))
    return next_state, state_prob, bull_prob, bear_prob
