#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model factories and evaluation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def make_model(model_type: str, task: str = "regression", random_state: int = 42) -> Any:
    model_type = model_type.lower()
    if model_type == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0)
    if model_type == "elasticnet":
        from sklearn.linear_model import ElasticNet

        return ElasticNet(alpha=0.001, l1_ratio=0.3, max_iter=5000, random_state=random_state)
    if model_type == "logistic":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
        return cls(n_estimators=300, max_depth=5, min_samples_leaf=20, random_state=random_state, n_jobs=-1)
    if model_type == "hist_gb":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
        return cls(max_iter=300, learning_rate=0.03, max_leaf_nodes=31, random_state=random_state)
    if model_type == "lightgbm":
        import lightgbm as lgb

        cls = lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor
        common = dict(
            n_estimators=1200,
            learning_rate=0.02,
            num_leaves=15,
            max_depth=5,
            min_child_samples=80,
            min_split_gain=0.001,
            reg_alpha=0.2,
            reg_lambda=3.0,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        )
        if task == "classification":
            return cls(class_weight="balanced", **common)
        return cls(objective="regression_l1", **common)
    if model_type == "xgboost":
        import xgboost as xgb

        cls = xgb.XGBClassifier if task == "classification" else xgb.XGBRegressor
        common = dict(n_estimators=500, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=random_state)
        return cls(eval_metric="logloss", **common) if task == "classification" else cls(objective="reg:squarederror", **common)
    raise ValueError(f"Unsupported model_type: {model_type}")


def infer_task(label_column: str) -> str:
    return "classification" if label_column.startswith("future_up_") else "regression"


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    corr = pd.Series(left).corr(pd.Series(right), method="spearman")
    return float(corr) if pd.notna(corr) else 0.0


def _group_rank_ic(y_true: pd.Series, y_pred: np.ndarray, groups: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"y": y_true.astype(float).values, "pred": np.asarray(y_pred, dtype=float), "group": groups.values})
    ics: list[float] = []
    for _, part in frame.groupby("group"):
        if len(part) < 5 or part["y"].nunique() < 2 or part["pred"].nunique() < 2:
            continue
        ics.append(_safe_spearman(part["pred"], part["y"]))
    if not ics:
        return {"mean": 0.0, "std": 0.0, "ir": 0.0, "groups": 0.0}
    arr = np.asarray(ics, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ir": float(arr.mean() / std) if std > 1e-12 else 0.0,
        "groups": float(len(arr)),
    }


def evaluate_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    task: str,
    dates: pd.Series | None = None,
    stocks: pd.Series | None = None,
) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = np.asarray(y_pred, dtype=float)
    out: dict[str, float] = {"samples": float(len(y_true))}
    if len(y_true) == 0:
        return out
    if task == "classification":
        pred_label = (y_pred >= 0.5).astype(float)
        out["accuracy"] = float((pred_label == y_true.values).mean())
        out["positive_rate"] = float(pred_label.mean())
    else:
        out["rank_ic"] = _safe_spearman(pd.Series(y_pred), y_true)
        out["mae"] = float(np.mean(np.abs(y_pred - y_true.values)))
        order = np.argsort(y_pred)
        bucket_size = max(1, len(y_pred) // 5)
        top_idx = order[-bucket_size:]
        bottom_idx = order[:bucket_size]
        out["top20_mean_label"] = float(y_true.iloc[top_idx].mean())
        out["bottom20_mean_label"] = float(y_true.iloc[bottom_idx].mean())
        out["top_bottom_spread"] = float(out["top20_mean_label"] - out["bottom20_mean_label"])
        out["top20_positive_rate"] = float((y_true.iloc[top_idx] > 0).mean())
        if dates is not None:
            daily_ic = _group_rank_ic(y_true, y_pred, pd.to_datetime(dates).dt.strftime("%Y-%m-%d"))
            out["daily_rank_ic_mean"] = daily_ic["mean"]
            out["daily_rank_ic_std"] = daily_ic["std"]
            out["daily_rank_ic_ir"] = daily_ic["ir"]
            out["daily_rank_ic_days"] = daily_ic["groups"]
        if stocks is not None:
            stock_ic = _group_rank_ic(y_true, y_pred, pd.Series(stocks).astype(str))
            out["stock_rank_ic_mean"] = stock_ic["mean"]
            out["stock_rank_ic_std"] = stock_ic["std"]
            out["stock_rank_ic_ir"] = stock_ic["ir"]
            out["stock_rank_ic_count"] = stock_ic["groups"]
    return out
