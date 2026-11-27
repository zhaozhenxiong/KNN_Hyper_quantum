#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train deep learning style sequence models for KHQuant agents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.deep_learning.dataset import build_sequence_matrix, sample_training_rows  # noqa: E402
from my_strategy.deep_learning.models import make_deep_model  # noqa: E402
from my_strategy.deep_learning.registry import save_deep_model  # noqa: E402
from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS  # noqa: E402
from my_strategy.ml.trainers import evaluate_predictions  # noqa: E402
from my_strategy.train_ml_model import build_dataset, resolve_stocks  # noqa: E402


def _split_sequences(meta: pd.DataFrame, train_end: str, valid_end: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = pd.to_datetime(meta["date"])
    train_mask = dates <= pd.to_datetime(train_end)
    valid_mask = (dates > pd.to_datetime(train_end)) & (dates <= pd.to_datetime(valid_end))
    test_mask = dates > pd.to_datetime(valid_end)
    return train_mask.to_numpy(), valid_mask.to_numpy(), test_mask.to_numpy()


def _evaluate_split(name: str, y: np.ndarray, pred: np.ndarray, meta: pd.DataFrame) -> dict[str, Any]:
    if len(y) == 0:
        return {"samples": 0}
    metrics = evaluate_predictions(pd.Series(y), pred, "regression", dates=meta["date"], stocks=meta["stock"])
    metrics["split"] = name
    return metrics


def train_deep_model(
    dataset: pd.DataFrame,
    model_name: str,
    model_type: str,
    label_column: str,
    train_end: str,
    valid_end: str,
    lookback: int,
    max_samples: int,
    epochs: int = 25,
    batch_size: int = 1024,
    hidden_size: int = 64,
    feature_columns: list[str] | None = None,
) -> tuple[Any, StandardScaler, dict[str, Any]]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    x, y, meta = build_sequence_matrix(dataset, label_column, feature_columns=feature_columns, lookback=lookback)
    if len(y) == 0:
        raise RuntimeError("deep learning sequence dataset is empty")

    train_mask, valid_mask, test_mask = _split_sequences(meta, train_end, valid_end)
    if train_mask.sum() == 0:
        raise RuntimeError("deep learning training split is empty")

    x_train, y_train, train_meta = x[train_mask], y[train_mask], meta.loc[train_mask].reset_index(drop=True)
    x_valid, y_valid, valid_meta = x[valid_mask], y[valid_mask], meta.loc[valid_mask].reset_index(drop=True)
    x_test, y_test, test_meta = x[test_mask], y[test_mask], meta.loc[test_mask].reset_index(drop=True)

    x_fit, y_fit, fit_meta = sample_training_rows(x_train, y_train, train_meta, max_samples=max_samples)
    scaler = StandardScaler()
    x_fit_scaled = scaler.fit_transform(x_fit)
    model = make_deep_model(
        model_type,
        lookback=lookback,
        epochs=epochs,
        batch_size=batch_size,
        hidden_size=hidden_size,
    )
    model.fit(x_fit_scaled, y_fit)

    def predict(matrix: np.ndarray) -> np.ndarray:
        if len(matrix) == 0:
            return np.asarray([], dtype=float)
        return np.asarray(model.predict(scaler.transform(matrix)), dtype=float)

    pred_train = predict(x_train)
    pred_valid = predict(x_valid)
    pred_test = predict(x_test)

    report: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "task": "regression",
        "prediction_kind": "cross_sectional_rank_score",
        "label_column": label_column,
        "prediction_column": "dl_prediction",
        "score_column": "dl_score",
        "confidence_column": "dl_confidence",
        "feature_columns": feature_columns,
        "lookback": int(lookback),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "hidden_size": int(hidden_size),
        "device": str(getattr(model, "device", "")),
        "train_end": train_end,
        "valid_end": valid_end,
        "max_samples": int(max_samples),
        "samples": {
            "train": int(len(y_train)),
            "valid": int(len(y_valid)),
            "test": int(len(y_test)),
            "total": int(len(y)),
            "fit": int(len(y_fit)),
        },
        "stocks": {
            "train": int(train_meta["stock"].nunique()) if not train_meta.empty else 0,
            "valid": int(valid_meta["stock"].nunique()) if not valid_meta.empty else 0,
            "test": int(test_meta["stock"].nunique()) if not test_meta.empty else 0,
            "total": int(meta["stock"].nunique()) if not meta.empty else 0,
        },
        "train_metrics": _evaluate_split("train", y_train, pred_train, train_meta),
        "valid_metrics": _evaluate_split("valid", y_valid, pred_valid, valid_meta),
        "test_metrics": _evaluate_split("test", y_test, pred_test, test_meta),
        "training_loss": [float(v) for v in getattr(model, "loss_curve_", [])],
        "best_validation_score": float(getattr(model, "best_validation_score_", 0.0) or 0.0),
        "best_val_loss": float(getattr(model, "best_val_loss_", 0.0) or 0.0),
        "validation_loss": [float(v) for v in getattr(model, "val_loss_curve_", [])],
    }
    return model, scaler, report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train KHQuant deep learning style model.")
    parser.add_argument("--stocks", default="local")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--valid-end", default="2025-12-31")
    parser.add_argument("--model-type", default="lstm", choices=["mlp", "sequence_mlp", "lstm", "seq_lstm", "sequence_lstm"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--base-label", default="future_return_5d")
    parser.add_argument("--neutral-group", default="auto")
    parser.add_argument("--max-stocks", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=200000)
    parser.add_argument("--min-rows", type=int, default=240)
    parser.add_argument("--no-update", action="store_true")
    args = parser.parse_args()

    stocks, stock_source = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    if not stocks:
        raise RuntimeError("No stocks found for deep learning training")

    label_column = f"cs_rank_{args.base_label}"
    model_name = args.model_name or f"dl_{args.model_type}_cs_rank_5d_local_all"
    auto_update = not args.no_update and stock_source == "explicit"
    print(f"[deep-train] stocks={len(stocks)} source={stock_source} auto_update={auto_update} lookback={args.lookback}")
    dataset, dataset_summary = build_dataset(
        stocks,
        args.start,
        args.end,
        auto_update=auto_update,
        label_column=label_column,
        min_rows=args.min_rows,
        base_label_column=args.base_label,
        label_mode="cross_sectional_rank",
        neutral_group=args.neutral_group,
        min_neutral_group_size=5,
    )
    if dataset.empty:
        raise RuntimeError("Deep learning source dataset is empty")

    model, scaler, report = train_deep_model(
        dataset,
        model_name=model_name,
        model_type=args.model_type,
        label_column=label_column,
        train_end=args.train_end,
        valid_end=args.valid_end,
        lookback=args.lookback,
        max_samples=args.max_samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_size=args.hidden_size,
    )
    report["dataset_summary"] = {**dataset_summary, "stock_source": stock_source}
    model_path = save_deep_model(model_name, model, scaler, report)
    report_root = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "deep_learning" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"deep model saved: {model_path}")
    print(f"report saved: {report_path}")
    keys = ["samples", "stocks", "train_metrics", "valid_metrics", "test_metrics", "best_validation_score"]
    print(json.dumps({k: report[k] for k in keys if k in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
