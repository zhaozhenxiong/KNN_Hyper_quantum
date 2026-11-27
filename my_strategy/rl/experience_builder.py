#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build offline RL transition rows from AI-Agent SQLite outputs."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import ACTION_WEIGHTS, STATE_FEATURE_COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen" / "ai_agent_screen.db"


def build_rl_experiences(
    db_path: str | Path = DEFAULT_DB_PATH,
    source_run_id: str | None = None,
    action_weights: list[float] | None = None,
    horizon: int = 1,
    cost_rate: float = 0.001,
    drawdown_penalty: float = 0.20,
    replace: bool = True,
) -> dict[str, Any]:
    db = Path(db_path)
    actions = [float(x) for x in (action_weights or ACTION_WEIGHTS)]
    created_at = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(db) as conn:
        _ensure_rl_tables(conn)
        run_id = source_run_id or _latest_run_id(conn)
        if not run_id:
            raise RuntimeError("No screen run found in screen_runs.")
        daily = pd.read_sql(
            """
            select run_id, stock, date, close, equity, position_weight,
                   target_weight, effective_target_weight, unrealized_return,
                   final_score, final_action, risk_action,
                   ml_score_signal, dl_score_signal, ml_confidence, dl_confidence
            from stock_daily_results
            where run_id = ?
            order by stock, date
            """,
            conn,
            params=[run_id],
        )
        experiences = make_experience_frame(
            daily,
            run_id=run_id,
            action_weights=actions,
            horizon=horizon,
            cost_rate=cost_rate,
            drawdown_penalty=drawdown_penalty,
            created_at=created_at,
        )
        if replace:
            conn.execute("delete from rl_experiences where source_run_id = ?", (run_id,))
        experiences.to_sql("rl_experiences", conn, if_exists="append", index=False, chunksize=5000)
        conn.commit()
    return {
        "db_path": str(db),
        "source_run_id": run_id,
        "rows": int(len(experiences)),
        "stocks": int(experiences["stock"].nunique()) if not experiences.empty else 0,
        "created_at": created_at,
        "action_weights": actions,
    }


def make_experience_frame(
    daily: pd.DataFrame,
    run_id: str,
    action_weights: list[float],
    horizon: int,
    cost_rate: float,
    drawdown_penalty: float,
    created_at: str,
) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    numeric_cols = [c for c in data.columns if c not in {"run_id", "stock", "date", "final_action", "risk_action"}]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    frames: list[pd.DataFrame] = []
    for stock, part in data.groupby("stock", sort=False):
        part = part.sort_values("date").copy()
        close = part["close"].replace(0, np.nan)
        part["return_1d"] = close.pct_change().fillna(0.0)
        part["return_5d"] = close.pct_change(5).fillna(0.0)
        part["return_20d"] = close.pct_change(20).fillna(0.0)
        part["volatility_20d"] = part["return_1d"].rolling(20).std().fillna(0.0)
        peak = part["equity"].cummax().replace(0, np.nan)
        part["drawdown"] = (part["equity"] / peak - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        part["current_drawdown"] = part["drawdown"]
        part["rsi14"] = 50.0
        part["volume_ratio"] = 1.0
        part["ma20_ratio"] = 0.0
        part["ma60_ratio"] = 0.0
        part["future_return"] = close.shift(-horizon) / close - 1
        part["next_date"] = part["date"].shift(-horizon)
        part = part.dropna(subset=["future_return", "next_date"])
        if part.empty:
            continue
        for action_weight in action_weights:
            out = part.copy()
            out["source_run_id"] = run_id
            out["created_at"] = created_at
            out["action_weight"] = float(action_weight)
            out["turnover"] = (float(action_weight) - out["position_weight"].fillna(0.0)).abs()
            out["reward"] = (
                float(action_weight) * out["future_return"].fillna(0.0)
                - cost_rate * out["turnover"]
                - drawdown_penalty * out["drawdown"].abs().fillna(0.0) / 252.0
            )
            out["done"] = False
            out["stock"] = stock
            frames.append(out)
    if not frames:
        return pd.DataFrame()
    result = pd.concat(frames, ignore_index=True, sort=False)
    keep_cols = [
        "source_run_id",
        "created_at",
        "stock",
        "date",
        *STATE_FEATURE_COLUMNS,
        "action_weight",
        "turnover",
        "reward",
        "future_return",
        "next_date",
        "done",
        "final_action",
        "risk_action",
    ]
    for col in keep_cols:
        if col not in result.columns:
            result[col] = 0.0
    result["date"] = pd.to_datetime(result["date"]).dt.strftime("%Y-%m-%d")
    result["next_date"] = pd.to_datetime(result["next_date"]).dt.strftime("%Y-%m-%d")
    result["done"] = result["done"].astype(int)
    return result[keep_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _latest_run_id(conn: sqlite3.Connection) -> str:
    row = conn.execute("select run_id from screen_runs order by imported_at desc limit 1").fetchone()
    return str(row[0]) if row else ""


def _ensure_rl_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        create table if not exists rl_experiences (
            source_run_id text,
            created_at text,
            stock text,
            date text,
            final_score real,
            ml_score_signal real,
            dl_score_signal real,
            ml_confidence real,
            dl_confidence real,
            position_weight real,
            target_weight real,
            effective_target_weight real,
            unrealized_return real,
            return_1d real,
            return_5d real,
            return_20d real,
            volatility_20d real,
            drawdown real,
            rsi14 real,
            volume_ratio real,
            ma20_ratio real,
            ma60_ratio real,
            current_drawdown real,
            action_weight real,
            turnover real,
            reward real,
            future_return real,
            next_date text,
            done integer,
            final_action text,
            risk_action text
        )
        """
    )
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
    conn.execute("create index if not exists idx_rl_exp_run_stock_date on rl_experiences(source_run_id, stock, date)")
    conn.execute("create index if not exists idx_rl_exp_run_reward on rl_experiences(source_run_id, reward)")
    conn.execute("create index if not exists idx_rl_train_created on rl_training_runs(created_at)")

