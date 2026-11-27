#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train daily ML models for KHQuant AI Agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.agents.data_agent import DataAgent  # noqa: E402
from my_strategy.agents.factor_agent import FactorAgent  # noqa: E402
from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS, build_ml_features, clean_feature_matrix  # noqa: E402
from my_strategy.ml.label_builder import add_forward_return_labels  # noqa: E402
from my_strategy.ml.model_registry import save_model  # noqa: E402
from my_strategy.ml.trainers import evaluate_predictions, infer_task, make_model  # noqa: E402


LOCAL_DAILY_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "raw" / "stock_daily"
DEFAULT_NEUTRAL_COLUMNS = [
    "neutral_group",
    "industry",
    "sector",
    "sw_l1",
    "申万一级行业",
    "中信一级行业",
    "board",
]


def list_local_stocks(data_root: Path = LOCAL_DAILY_ROOT) -> list[str]:
    if not data_root.exists():
        return []
    return sorted(path.stem for path in data_root.glob("*.parquet"))


def resolve_stocks(value: str, max_stocks: int | None = None) -> tuple[list[str], str]:
    token = (value or "").strip().lower()
    if token in {"all", "local", "local_all", "*"}:
        stocks = list_local_stocks()
        source = "local_parquet_all"
    else:
        stocks = [item.strip() for item in value.split(",") if item.strip()]
        source = "explicit"
    if max_stocks and max_stocks > 0:
        stocks = stocks[:max_stocks]
    return stocks, source


def load_local_daily(stock: str, start: str, end: str) -> pd.DataFrame:
    path = LOCAL_DAILY_ROOT / f"{stock}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        raise ValueError(f"{stock} missing date column")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy()
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "trade_open",
        "trade_high",
        "trade_low",
        "trade_close",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def market_board(stock: str) -> str:
    code = stock.split(".")[0]
    if code.startswith(("300", "301")):
        return "chinext"
    if code.startswith("688"):
        return "star"
    if code.startswith(("600", "601", "603", "605")):
        return "sh_main"
    if code.startswith(("000", "001", "002", "003")):
        return "sz_main"
    if code.startswith(("8", "4")):
        return "bse"
    return "other"


def normalize_stock_code(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if "." in text:
        code, suffix = text.split(".", 1)
        return f"{code.zfill(6)}.{suffix.upper()}"
    code = text.zfill(6)
    suffix = "SH" if code.startswith(("6", "9")) else "SZ"
    return f"{code}.{suffix}"


def load_stock_metadata(metadata_path: str | Path | None = None) -> pd.DataFrame:
    candidates = []
    if metadata_path:
        candidates.append(Path(metadata_path))
    candidates.extend(
        [
            PROJECT_ROOT / "my_strategy" / "data" / "metadata" / "stock_metadata.csv",
            PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "reports" / "ranking.csv",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        try:
            meta = pd.read_csv(path)
        except Exception:
            continue
        if "stock" not in meta.columns:
            continue
        meta = meta.copy()
        meta["stock"] = meta["stock"].map(normalize_stock_code)
        return meta.drop_duplicates("stock", keep="last")
    return pd.DataFrame(columns=["stock"])


def attach_stock_metadata(dataset: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    if dataset.empty or metadata.empty:
        return dataset
    extra_cols = [col for col in metadata.columns if col != "stock" and col not in dataset.columns]
    if not extra_cols:
        return dataset
    return dataset.merge(metadata[["stock", *extra_cols]], on="stock", how="left")


def resolve_neutral_column(dataset: pd.DataFrame, neutral_group: str) -> str:
    if neutral_group and neutral_group.lower() not in {"auto", "none", "false"}:
        if neutral_group not in dataset.columns:
            raise ValueError(f"neutral group column not found: {neutral_group}")
        return neutral_group
    for col in DEFAULT_NEUTRAL_COLUMNS:
        if col in dataset.columns and dataset[col].notna().any():
            return col
    return "board"


def add_cross_sectional_rank_label(
    dataset: pd.DataFrame,
    base_label: str,
    neutral_col: str,
    rank_label: str,
    min_group_size: int = 5,
) -> pd.DataFrame:
    out = dataset.copy()
    out[neutral_col] = out[neutral_col].fillna("unknown").astype(str)
    group_keys = ["date", neutral_col]
    group_size = out.groupby(group_keys)[base_label].transform("count")
    pct_rank = out.groupby(group_keys)[base_label].rank(method="average", pct=True)
    out[rank_label] = (pct_rank - 0.5) * 2.0
    out.loc[group_size < min_group_size, rank_label] = np.nan
    out[f"{rank_label}_group_size"] = group_size
    return out


def build_dataset(
    stocks: list[str],
    start: str,
    end: str,
    auto_update: bool,
    label_column: str,
    min_rows: int = 240,
    base_label_column: str | None = None,
    label_mode: str = "return",
    neutral_group: str = "auto",
    min_neutral_group_size: int = 5,
    metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = []
    skipped: list[dict[str, str]] = []
    data_agent = DataAgent(auto_update=auto_update) if auto_update else None
    total = len(stocks)

    for idx, stock in enumerate(stocks, start=1):
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"[dataset] {idx}/{total} {stock}")
        try:
            raw = data_agent.load(stock, start, end) if data_agent else load_local_daily(stock, start, end)
            if len(raw) < min_rows:
                skipped.append({"stock": stock, "reason": f"too_few_rows:{len(raw)}"})
                continue
            factors = FactorAgent.add_factors(raw)
            features = build_ml_features(factors)
            labeled = add_forward_return_labels(features)
            labeled["stock"] = stock
            labeled["board"] = market_board(stock)
            rows.append(labeled)
        except Exception as exc:  # keep batch training moving
            skipped.append({"stock": stock, "reason": str(exc)[:240]})

    if not rows:
        return pd.DataFrame(), {"skipped": skipped, "used_stocks": 0}

    dataset = pd.concat(rows, ignore_index=True)
    dataset["date"] = pd.to_datetime(dataset["date"])
    metadata = load_stock_metadata(metadata_path)
    dataset = attach_stock_metadata(dataset, metadata)
    effective_label = label_column
    neutral_col = ""
    if label_mode == "cross_sectional_rank":
        base_label = base_label_column or label_column.replace("cs_rank_", "")
        if base_label not in dataset.columns:
            raise ValueError(f"base label for cross-sectional ranking not found: {base_label}")
        neutral_col = resolve_neutral_column(dataset, neutral_group)
        dataset = add_cross_sectional_rank_label(
            dataset,
            base_label=base_label,
            neutral_col=neutral_col,
            rank_label=label_column,
            min_group_size=min_neutral_group_size,
        )
        effective_label = label_column
    dataset = dataset.dropna(subset=[label_column]).reset_index(drop=True)
    summary = {
        "used_stocks": int(dataset["stock"].nunique()),
        "skipped_count": len(skipped),
        "skipped": skipped[:100],
        "date_min": str(dataset["date"].min().date()) if not dataset.empty else "",
        "date_max": str(dataset["date"].max().date()) if not dataset.empty else "",
        "label_column": effective_label,
        "label_mode": label_mode,
        "neutral_group_column": neutral_col,
        "min_neutral_group_size": min_neutral_group_size,
        "metadata_columns": list(metadata.columns) if not metadata.empty else [],
    }
    return dataset, summary


def _predict(model: Any, x: pd.DataFrame, task: str) -> np.ndarray:
    if task == "classification" and hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def _label_quantile_report(split: pd.DataFrame, pred: np.ndarray, label_column: str, bins: int = 5) -> list[dict[str, Any]]:
    if split.empty:
        return []
    frame = split[["date", "stock", label_column]].copy()
    frame["pred"] = pred
    try:
        frame["label_quantile"] = pd.qcut(frame[label_column], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return []
    out = []
    for quantile, part in frame.groupby("label_quantile", dropna=True):
        out.append(
            {
                "label_quantile": int(quantile),
                "samples": int(len(part)),
                "mean_label": float(part[label_column].mean()),
                "mean_pred": float(part["pred"].mean()),
                "positive_rate": float((part[label_column] > 0).mean()),
            }
        )
    return out


def _group_report(split: pd.DataFrame, pred: np.ndarray, label_column: str, group_col: str, task: str) -> list[dict[str, Any]]:
    if split.empty or group_col not in split.columns:
        return []
    frame = split[["date", "stock", label_column, group_col]].copy()
    frame["pred"] = pred
    out = []
    for group, part in frame.groupby(group_col):
        if len(part) < 50:
            continue
        metrics = evaluate_predictions(
            part[label_column],
            part["pred"].to_numpy(),
            task,
            dates=part["date"],
            stocks=part["stock"],
        )
        metrics[group_col] = str(group)
        out.append(metrics)
    return out


def _top_bottom_base_return_report(
    split: pd.DataFrame,
    pred: np.ndarray,
    base_label_column: str | None,
    neutral_group_column: str | None = None,
) -> dict[str, float]:
    if split.empty or not base_label_column or base_label_column not in split.columns:
        return {}
    frame = split[["date", base_label_column]].copy()
    frame["pred"] = np.asarray(pred, dtype=float)
    if neutral_group_column and neutral_group_column in split.columns:
        frame[neutral_group_column] = split[neutral_group_column].fillna("unknown").astype(str).values
        group_keys = ["date", neutral_group_column]
    else:
        group_keys = ["date"]
    frame[base_label_column] = pd.to_numeric(frame[base_label_column], errors="coerce")
    frame = frame.dropna(subset=[base_label_column, "pred"])
    if len(frame) < 10:
        return {}
    spreads = []
    top_returns = []
    bottom_returns = []
    top_positive_rates = []
    groups = 0
    for _, part in frame.groupby(group_keys):
        if len(part) < 10 or part["pred"].nunique() < 2:
            continue
        order = np.argsort(part["pred"].to_numpy())
        bucket_size = max(1, len(part) // 5)
        top = part.iloc[order[-bucket_size:]][base_label_column]
        bottom = part.iloc[order[:bucket_size]][base_label_column]
        top_mean = float(top.mean())
        bottom_mean = float(bottom.mean())
        top_returns.append(top_mean)
        bottom_returns.append(bottom_mean)
        spreads.append(top_mean - bottom_mean)
        top_positive_rates.append(float((top > 0).mean()))
        groups += 1
    if not spreads:
        return {}
    return {
        "base_label": base_label_column,
        "group_keys": ",".join(group_keys),
        "groups": float(groups),
        "top20_mean_return": float(np.mean(top_returns)),
        "bottom20_mean_return": float(np.mean(bottom_returns)),
        "top_bottom_return_spread": float(np.mean(spreads)),
        "top20_positive_rate": float(np.mean(top_positive_rates)),
    }


def fit_model(model: Any, model_type: str, task: str, x_train: pd.DataFrame, y_train: pd.Series, x_valid: pd.DataFrame | None, y_valid: pd.Series | None) -> None:
    if model_type.lower() == "lightgbm" and x_valid is not None and y_valid is not None and len(x_valid) > 0:
        import lightgbm as lgb

        model.fit(
            x_train,
            y_train,
            eval_set=[(x_valid, y_valid)],
            callbacks=[lgb.early_stopping(stopping_rounds=80, verbose=False), lgb.log_evaluation(period=0)],
        )
        return
    model.fit(x_train, y_train)


def train_model(
    dataset: pd.DataFrame,
    model_name: str,
    model_type: str,
    label_column: str,
    train_end: str,
    valid_end: str,
    feature_columns: list[str] | None = None,
    dataset_summary: dict[str, Any] | None = None,
    label_mode: str = "return",
    base_label_column: str | None = None,
    neutral_group_column: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS
    task = infer_task(label_column)
    train_end_ts = pd.to_datetime(train_end)
    valid_end_ts = pd.to_datetime(valid_end)

    train = dataset[dataset["date"] <= train_end_ts].copy()
    valid = dataset[(dataset["date"] > train_end_ts) & (dataset["date"] <= valid_end_ts)].copy()
    test = dataset[dataset["date"] > valid_end_ts].copy()
    if train.empty:
        raise RuntimeError("training set is empty; move start earlier or train_end later")

    x_train = clean_feature_matrix(train, feature_columns)
    y_train = train[label_column].astype(float)
    x_valid = clean_feature_matrix(valid, feature_columns) if not valid.empty else None
    y_valid = valid[label_column].astype(float) if not valid.empty else None
    model = make_model(model_type, task=task)
    fit_model(model, model_type, task, x_train, y_train, x_valid, y_valid)

    if task == "classification":
        prediction_column = "pred_prob_up_5d"
    elif label_mode == "cross_sectional_rank":
        prediction_column = label_column.replace("cs_rank_", "pred_rank_")
    else:
        prediction_column = label_column.replace("future_", "pred_")
    report: dict[str, Any] = {
        "model_name": model_name,
        "model_type": model_type,
        "task": task,
        "label_column": label_column,
        "label_mode": label_mode,
        "base_label_column": base_label_column,
        "neutral_group_column": neutral_group_column,
        "prediction_kind": "cross_sectional_rank_score" if label_mode == "cross_sectional_rank" else task,
        "prediction_column": prediction_column,
        "feature_columns": feature_columns,
        "train_end": train_end,
        "valid_end": valid_end,
        "samples": {
            "train": int(len(train)),
            "valid": int(len(valid)),
            "test": int(len(test)),
            "total": int(len(dataset)),
        },
        "stocks": {
            "train": int(train["stock"].nunique()),
            "valid": int(valid["stock"].nunique()),
            "test": int(test["stock"].nunique()),
            "total": int(dataset["stock"].nunique()),
        },
        "dataset_summary": dataset_summary or {},
        "score_scale": 0.05,
        "validation_style": "time_split_with_neutral_group_cross_sectional_reports",
    }

    split_predictions: dict[str, tuple[pd.DataFrame, np.ndarray]] = {}
    for split_name, split in [("train", train), ("valid", valid), ("test", test)]:
        if split.empty:
            report[f"{split_name}_metrics"] = {"samples": 0}
            continue
        x = clean_feature_matrix(split, feature_columns)
        pred = _predict(model, x, task)
        split_predictions[split_name] = (split, pred)
        report[f"{split_name}_metrics"] = evaluate_predictions(
            split[label_column],
            pred,
            task,
            dates=split["date"],
            stocks=split["stock"],
        )
        report[f"{split_name}_by_year"] = _group_report(
            split.assign(year=split["date"].dt.year.astype(str)),
            pred,
            label_column,
            "year",
            task,
        )
        report[f"{split_name}_by_board"] = _group_report(split, pred, label_column, "board", task)
        if neutral_group_column and neutral_group_column in split.columns and neutral_group_column != "board":
            report[f"{split_name}_by_neutral_group"] = _group_report(split, pred, label_column, neutral_group_column, task)
        report[f"{split_name}_label_quantiles"] = _label_quantile_report(split, pred, label_column)
        base_return = _top_bottom_base_return_report(split, pred, base_label_column, neutral_group_column)
        if base_return:
            report[f"{split_name}_base_return_spread"] = base_return

    if hasattr(model, "best_iteration_") and getattr(model, "best_iteration_", None):
        report["best_iteration"] = int(model.best_iteration_)
    if hasattr(model, "feature_importances_"):
        importance = pd.DataFrame({"feature": feature_columns, "importance": model.feature_importances_})
        report["feature_importance"] = importance.sort_values("importance", ascending=False).head(30).to_dict("records")
    elif hasattr(model, "coef_"):
        coef = np.ravel(model.coef_)
        report["feature_importance"] = (
            pd.DataFrame({"feature": feature_columns, "importance": np.abs(coef), "coef": coef})
            .sort_values("importance", ascending=False)
            .head(30)
            .to_dict("records")
        )

    return model, report


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train KHQuant daily ML model.")
    parser.add_argument("--stocks", default="local", help="Comma-separated stock codes, or local/all to use local parquet universe.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-07")
    parser.add_argument("--train-end", default="2024-12-31")
    parser.add_argument("--valid-end", default="2025-12-31")
    parser.add_argument("--model-type", default="lightgbm", choices=["ridge", "elasticnet", "logistic", "random_forest", "hist_gb", "lightgbm", "xgboost"])
    parser.add_argument("--model-name", default="")
    parser.add_argument("--label", default="future_return_5d")
    parser.add_argument("--label-mode", default="return", choices=["return", "cross_sectional_rank"])
    parser.add_argument("--base-label", default="", help="Raw future return label used before rank transform, e.g. future_return_5d.")
    parser.add_argument("--neutral-group", default="auto", help="Industry/sector column for neutral ranking; auto falls back to board.")
    parser.add_argument("--min-neutral-group-size", type=int, default=5)
    parser.add_argument("--metadata-path", default="", help="Optional stock metadata CSV with stock + industry/sector columns.")
    parser.add_argument("--max-stocks", type=int, default=0, help="Optional quick-test cap; 0 means no cap.")
    parser.add_argument("--min-rows", type=int, default=240)
    parser.add_argument("--no-update", action="store_true", help="Use only local cache and do not update online sources.")
    args = parser.parse_args()

    stocks, stock_source = resolve_stocks(args.stocks, max_stocks=args.max_stocks or None)
    if not stocks:
        raise RuntimeError("No stocks found. Check --stocks or local stock_daily parquet directory.")

    label_column = args.label
    base_label = args.base_label or ""
    if args.label_mode == "cross_sectional_rank":
        if not base_label:
            base_label = label_column if label_column.startswith("future_return_") else label_column.replace("cs_rank_", "")
        if not label_column.startswith("cs_rank_"):
            label_column = f"cs_rank_{base_label}"

    model_name = args.model_name or f"{args.model_type}_{label_column}_local_all"
    auto_update = not args.no_update and stock_source == "explicit"
    print(f"[train] stocks={len(stocks)} source={stock_source} auto_update={auto_update}")
    dataset, dataset_summary = build_dataset(
        stocks,
        args.start,
        args.end,
        auto_update=auto_update,
        label_column=label_column,
        min_rows=args.min_rows,
        base_label_column=base_label or None,
        label_mode=args.label_mode,
        neutral_group=args.neutral_group,
        min_neutral_group_size=args.min_neutral_group_size,
        metadata_path=args.metadata_path or None,
    )
    if dataset.empty:
        raise RuntimeError("ML dataset is empty")

    model, report = train_model(
        dataset,
        model_name,
        args.model_type,
        label_column,
        args.train_end,
        args.valid_end,
        dataset_summary={**dataset_summary, "stock_source": stock_source},
        label_mode=args.label_mode,
        base_label_column=base_label or None,
        neutral_group_column=dataset_summary.get("neutral_group_column") or None,
    )
    model_path = save_model(model_name, model, report)
    report_root = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ml" / "reports"
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / f"{model_name}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"model saved: {model_path}")
    print(f"report saved: {report_path}")
    keys = ["samples", "stocks", "best_iteration", "train_metrics", "valid_metrics", "test_metrics"]
    print(json.dumps({k: report[k] for k in keys if k in report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
