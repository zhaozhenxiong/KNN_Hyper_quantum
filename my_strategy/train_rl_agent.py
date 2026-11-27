#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Train an offline RL position-sizing policy from ai_agent_screen SQLite outputs."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.rl.experience_builder import DEFAULT_DB_PATH, build_rl_experiences  # noqa: E402
from my_strategy.rl.features import ACTION_WEIGHTS, Q_FEATURE_COLUMNS, clean_feature_frame  # noqa: E402
from my_strategy.rl.registry import save_report, save_rl_model, update_best_rl_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train offline RL policy from SQLite experiences.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--source-run-id", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--valid-start", default="2026-06-01")
    parser.add_argument("--max-samples", type=int, default=350000)
    parser.add_argument("--rebuild-experiences", action="store_true")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--cost-rate", type=float, default=0.001)
    parser.add_argument("--drawdown-penalty", type=float, default=0.20)
    parser.add_argument("--action-weights", default=",".join(str(x) for x in ACTION_WEIGHTS))
    parser.add_argument("--model-type", default="hist_gb", choices=["hist_gb", "random_forest", "lightgbm"])
    return parser.parse_args()


def make_model(model_type: str):
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=220, max_depth=8, min_samples_leaf=80, n_jobs=-1, random_state=42)
    if model_type == "lightgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=900,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=6,
            min_child_samples=120,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="regression_l1",
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        )
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(max_iter=420, learning_rate=0.035, max_leaf_nodes=31, l2_regularization=0.05, random_state=42)


def load_experiences(db_path: Path, source_run_id: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        if not source_run_id:
            row = conn.execute("select source_run_id from rl_experiences order by created_at desc limit 1").fetchone()
            source_run_id = str(row[0]) if row else ""
        if not source_run_id:
            raise RuntimeError("No rl_experiences found; run with --rebuild-experiences first.")
        return pd.read_sql("select * from rl_experiences where source_run_id = ?", conn, params=[source_run_id])


def evaluate_policy(model, valid: pd.DataFrame) -> dict[str, float]:
    if valid.empty:
        return {"valid_policy_mean_reward": 0.0, "best_action_weight_mean": 0.0}
    pred = np.asarray(model.predict(clean_feature_frame(valid, Q_FEATURE_COLUMNS)), dtype=float)
    frame = valid[["stock", "date", "action_weight", "reward"]].copy()
    frame["pred_q"] = pred
    idx = frame.groupby(["stock", "date"])["pred_q"].idxmax()
    chosen = frame.loc[idx]
    return {
        "valid_policy_mean_reward": float(chosen["reward"].mean()),
        "best_action_weight_mean": float(chosen["action_weight"].mean()),
        "best_action_zero_rate": float((chosen["action_weight"] <= 0).mean()),
        "best_action_high_rate": float((chosen["action_weight"] >= 0.7).mean()),
    }


def write_training_run(db_path: Path, report: dict[str, Any]) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            create table if not exists rl_training_runs (
                model_name text primary key,
                created_at text,
                source_run_id text,
                samples integer,
                train_samples integer,
                valid_samples integer,
                score real,
                valid_mean_reward real,
                valid_policy_mean_reward real,
                best_action_weight_mean real,
                model_path text,
                report_path text,
                metadata_json text
            )
            """
        )
        conn.execute(
            """
            insert or replace into rl_training_runs values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report["model_name"],
                report["created_at"],
                report["source_run_id"],
                int(report["samples"]),
                int(report["train_samples"]),
                int(report["valid_samples"]),
                float(report["score"]),
                float(report["valid_mean_reward"]),
                float(report["valid_policy_mean_reward"]),
                float(report["best_action_weight_mean"]),
                report["model_path"],
                report["report_path"],
                json.dumps(report, ensure_ascii=False),
            ),
        )
        conn.commit()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    db_path = Path(args.db_path)
    actions = [float(x.strip()) for x in args.action_weights.split(",") if x.strip()]
    if args.rebuild_experiences:
        build_info = build_rl_experiences(
            db_path=db_path,
            source_run_id=args.source_run_id or None,
            action_weights=actions,
            horizon=args.horizon,
            cost_rate=args.cost_rate,
            drawdown_penalty=args.drawdown_penalty,
            replace=True,
        )
        source_run_id = build_info["source_run_id"]
        print(json.dumps({"built_experiences": build_info}, ensure_ascii=False, indent=2))
    else:
        source_run_id = args.source_run_id

    data = load_experiences(db_path, source_run_id)
    if data.empty:
        raise RuntimeError("No RL experience rows loaded.")
    data["date"] = pd.to_datetime(data["date"])
    if args.max_samples and len(data) > args.max_samples:
        data = data.sample(args.max_samples, random_state=42).sort_values(["date", "stock", "action_weight"])

    valid_start = pd.to_datetime(args.valid_start)
    train = data[data["date"] < valid_start].copy()
    valid = data[data["date"] >= valid_start].copy()
    if train.empty or valid.empty:
        split = int(len(data) * 0.8)
        train = data.iloc[:split].copy()
        valid = data.iloc[split:].copy()

    model = make_model(args.model_type)
    x_train = clean_feature_frame(train, Q_FEATURE_COLUMNS)
    y_train = pd.to_numeric(train["reward"], errors="coerce").fillna(0.0)
    model.fit(x_train, y_train)

    x_valid = clean_feature_frame(valid, Q_FEATURE_COLUMNS)
    pred = np.asarray(model.predict(x_valid), dtype=float)
    y_valid = pd.to_numeric(valid["reward"], errors="coerce").fillna(0.0).to_numpy()
    mae = float(np.mean(np.abs(pred - y_valid))) if len(y_valid) else 0.0
    corr = float(pd.Series(pred).corr(pd.Series(y_valid), method="spearman")) if len(y_valid) > 5 else 0.0
    if not np.isfinite(corr):
        corr = 0.0
    policy_eval = evaluate_policy(model, valid)

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = args.model_name or f"rl_q_position_{created_at}"
    metadata = {
        "model_name": model_name,
        "model_type": args.model_type,
        "task": "offline_q_position_sizing",
        "source_run_id": str(train["source_run_id"].iloc[0]),
        "feature_columns": Q_FEATURE_COLUMNS,
        "action_weights": actions,
        "valid_start": args.valid_start,
        "horizon": args.horizon,
        "cost_rate": args.cost_rate,
        "drawdown_penalty": args.drawdown_penalty,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    model_path = save_rl_model(model_name, model, metadata)
    score = float(policy_eval["valid_policy_mean_reward"] * 100.0 + corr * 0.05 - mae)
    report = {
        **metadata,
        "samples": int(len(data)),
        "train_samples": int(len(train)),
        "valid_samples": int(len(valid)),
        "valid_mae": mae,
        "valid_spearman": corr,
        "valid_mean_reward": float(np.mean(y_valid)) if len(y_valid) else 0.0,
        **policy_eval,
        "score": score,
        "model_path": str(model_path),
    }
    report_path = save_report(model_name, report)
    report["report_path"] = str(report_path)
    save_report(model_name, report)
    is_best = update_best_rl_model(report)
    write_training_run(db_path, report)
    print(json.dumps({"report": report, "selected_as_best": is_best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

