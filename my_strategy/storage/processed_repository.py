#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DB-first repository for processed backtest and screening outputs."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from my_strategy.storage.artifact_store import ArtifactStore
from my_strategy.storage.db_paths import processed_db_path
from my_strategy.storage.persistence_guard import require_run_id, upsert_output_reference
from my_strategy.storage.warehouse_schema import create_views, init_warehouse


class ProcessedRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else processed_db_path()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        init_warehouse(conn)
        return conn

    def save_stock_backtest(
        self,
        *,
        run_id: str,
        stock: str,
        decisions: pd.DataFrame,
        daily: pd.DataFrame,
        trades: pd.DataFrame,
        report: dict[str, Any],
        source_root: str | Path,
    ) -> dict[str, int]:
        imported_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        run_key = require_run_id(run_id or str((report.get("run_context") or {}).get("run_id") or ""), operation="backtest persistence")
        source = str(source_root)
        counts: dict[str, int] = {}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, imported_at, task, source_root, artifact_root, start, end, stocks, success, failed, ml_model, dl_model, rl_model, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    imported_at=excluded.imported_at,
                    source_root=excluded.source_root,
                    artifact_root=excluded.artifact_root,
                    summary_json=excluded.summary_json
                """,
                (
                    run_key,
                    imported_at,
                    "multi-agent-backtest",
                    source,
                    source,
                    (report.get("run_context") or {}).get("as_of_date", ""),
                    (report.get("latest_decision") or {}).get("date", ""),
                    1,
                    1,
                    0,
                    (report.get("ml") or {}).get("model_name", ""),
                    (report.get("deep_learning") or {}).get("model_name", ""),
                    (report.get("reinforcement_learning") or {}).get("model_name", ""),
                    json.dumps(report, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
            for table in ["backtest_reports", "backtest_daily_ledger", "backtest_trades", "agent_signal_records"]:
                if _has_columns(conn, table, ["run_id", "stock"]):
                    conn.execute(f'DELETE FROM "{table}" WHERE run_id = ? AND stock = ?', (run_key, stock))
            report_row = pd.DataFrame(
                [
                    {
                        "run_id": run_key,
                        "imported_at": imported_at,
                        "stock": stock,
                        "stock_name": report.get("stock_name", ""),
                        "status": "ok",
                        "source_file": f"db://reports/{run_key}/{stock}",
                        "report_json": json.dumps(report, ensure_ascii=False, sort_keys=True, default=str),
                    }
                ]
            )
            counts["backtest_reports"] = _append_dataframe(conn, "backtest_reports", report_row)
            counts["backtest_daily_ledger"] = _append_detail(conn, "backtest_daily_ledger", daily, run_key, stock, imported_at)
            counts["backtest_trades"] = _append_detail(conn, "backtest_trades", trades, run_key, stock, imported_at)
            counts["agent_signal_records"] = _append_detail(conn, "agent_signal_records", decisions, run_key, stock, imported_at)
            create_views(conn)
        artifact_id = ArtifactStore(self.db_path).put_bytes(
            "backtest_report",
            "json",
            f"{run_key}:{stock}",
            json.dumps(report, ensure_ascii=False, indent=2, default=str).encode("utf-8"),
            metadata={"run_id": run_key, "stock": stock},
        )
        conn = self.connect()
        try:
            upsert_output_reference(
                conn,
                run_id=run_key,
                scope_key=stock,
                output_type="backtest_report",
                artifact_id=artifact_id,
                authority="processed_warehouse",
                metadata={"stock": stock, "source_root": source},
            )
        finally:
            conn.close()
        return counts

    def factor_snapshot_coverage(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        factor_version: str,
    ) -> dict[str, dict[str, Any]]:
        if not stocks:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for chunk_start in range(0, len(stocks), 500):
                chunk = [str(stock) for stock in stocks[chunk_start : chunk_start + 500]]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT stock, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                    FROM factor_snapshots
                    WHERE factor_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    GROUP BY stock
                    """,
                    [factor_version, *chunk, str(start_date), str(end_date)],
                ).fetchall()
                for row in rows:
                    out[str(row["stock"])] = {
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "rows": int(row["rows"] or 0),
                    }
        return out

    def read_factor_snapshots(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        factor_version: str,
        *,
        chunk_size: int = 500,
    ) -> pd.DataFrame:
        if not stocks:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        with self.connect() as conn:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")
            conn.execute("PRAGMA mmap_size=268435456")
            for chunk_start in range(0, len(stocks), chunk_size):
                chunk = [str(stock) for stock in stocks[chunk_start : chunk_start + chunk_size]]
                placeholders = ",".join("?" for _ in chunk)
                frame = pd.read_sql_query(
                    f"""
                    SELECT *
                    FROM factor_snapshots
                    WHERE factor_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    ORDER BY stock, date
                    """,
                    conn,
                    params=[factor_version, *chunk, str(start_date), str(end_date)],
                )
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values(["stock", "date"]).reset_index(drop=True)

    def replace_factor_snapshots(
        self,
        frame: pd.DataFrame,
        *,
        factor_version: str,
        source_start: str,
        source_end: str,
        volume_window: int,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        if frame is None or frame.empty:
            return {"rows": 0, "stocks": 0}
        imported_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        out = frame.copy()
        out["factor_version"] = factor_version
        out["volume_window"] = int(volume_window)
        out["computed_at"] = imported_at
        out["source_start"] = str(source_start)
        out["source_end"] = str(source_end)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["stock", "date"])
        stocks = sorted({str(stock) for stock in out["stock"].dropna().unique()})
        with self.connect() as conn:
            for chunk_start in range(0, len(stocks), 500):
                chunk = stocks[chunk_start : chunk_start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    DELETE FROM factor_snapshots
                    WHERE factor_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    """,
                    [factor_version, *chunk, str(source_start), str(source_end)],
                )
            rows = _append_dataframe(conn, "factor_snapshots", out)
            snapshot_id = f"{factor_version}:{source_start}:{source_end}:{imported_at}"
            payload = {
                "snapshot_id": snapshot_id,
                "factor_version": factor_version,
                "computed_at": imported_at,
                "source_start": str(source_start),
                "source_end": str(source_end),
                "stocks": len(stocks),
                "rows": rows,
                "volume_window": int(volume_window),
                "summary_json": json.dumps(summary or {}, ensure_ascii=False, sort_keys=True, default=str),
            }
            conn.execute(
                """
                INSERT INTO factor_snapshot_runs (
                    snapshot_id, factor_version, computed_at, source_start, source_end,
                    stocks, rows, volume_window, summary_json
                ) VALUES (
                    :snapshot_id, :factor_version, :computed_at, :source_start, :source_end,
                    :stocks, :rows, :volume_window, :summary_json
                )
                """,
                payload,
            )
            create_views(conn)
        return {"rows": int(len(out)), "stocks": int(len(stocks))}

    def model_feature_snapshot_coverage(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        feature_version: str,
    ) -> dict[str, dict[str, Any]]:
        if not stocks:
            return {}
        out: dict[str, dict[str, Any]] = {}
        with self.connect() as conn:
            for chunk_start in range(0, len(stocks), 500):
                chunk = [str(stock) for stock in stocks[chunk_start : chunk_start + 500]]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"""
                    SELECT stock, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                    FROM model_feature_snapshots
                    WHERE feature_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    GROUP BY stock
                    """,
                    [feature_version, *chunk, str(start_date), str(end_date)],
                ).fetchall()
                for row in rows:
                    out[str(row["stock"])] = {
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "rows": int(row["rows"] or 0),
                    }
        return out

    def read_model_feature_snapshots(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        feature_version: str,
        *,
        chunk_size: int = 500,
        include_labels: bool = False,
    ) -> pd.DataFrame:
        if not stocks:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        with self.connect() as conn:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")
            conn.execute("PRAGMA mmap_size=268435456")
            for chunk_start in range(0, len(stocks), chunk_size):
                chunk = [str(stock) for stock in stocks[chunk_start : chunk_start + chunk_size]]
                placeholders = ",".join("?" for _ in chunk)
                columns = """
                    features.stock AS stock, features.code AS code, features.date AS date,
                    features.return_1d AS return_1d, features.return_3d AS return_3d,
                    features.return_5d AS return_5d, features.return_10d AS return_10d,
                    features.return_20d AS return_20d, features.ma5_ratio AS ma5_ratio,
                    features.ma10_ratio AS ma10_ratio, features.ma20_ratio AS ma20_ratio,
                    features.ma60_ratio AS ma60_ratio, features.ma20_ma60_ratio AS ma20_ma60_ratio,
                    features.rsi14 AS rsi14, features.macd AS macd, features.macd_signal AS macd_signal,
                    features.macd_hist AS macd_hist, features.volatility_20d AS volatility_20d,
                    features.volume_ratio AS volume_ratio, features.amount_change_5d AS amount_change_5d,
                    features.high_low_range AS high_low_range, features.close_to_20d_high AS close_to_20d_high,
                    features.close_to_20d_low AS close_to_20d_low
                """
                available_columns = {
                    str(row[1]) for row in conn.execute('PRAGMA table_info("model_feature_snapshots")').fetchall()
                }
                technical_columns = [
                    "boll_pct_b", "boll_bandwidth", "macd_5_10_6", "macd_5_10_6_signal", "macd_5_10_6_hist",
                    "kdj_k", "kdj_d", "kdj_j", "macd_kdj_resonance", "ar_26", "br_26", "cci_14",
                    "obv_change_20d", "sar_gap", "expma_5_10_ratio",
                ]
                columns += "".join(
                    f", features.{name} AS {name}" for name in technical_columns if name in available_columns
                )
                label_columns = """
                    COALESCE(labels.future_return_5d, features.future_return_5d) AS future_return_5d,
                    COALESCE(labels.future_up_5d, features.future_up_5d) AS future_up_5d,
                    COALESCE(labels.future_return_10d, features.future_return_10d) AS future_return_10d,
                    COALESCE(labels.future_up_10d, features.future_up_10d) AS future_up_10d,
                    COALESCE(labels.future_return_20d, features.future_return_20d) AS future_return_20d,
                    COALESCE(labels.future_up_20d, features.future_up_20d) AS future_up_20d
                """ if include_labels else ""
                frame = pd.read_sql_query(
                    f"""
                    SELECT {columns}{',' if label_columns else ''}{label_columns}
                    FROM model_feature_snapshots AS features
                    LEFT JOIN model_feature_labels AS labels
                      ON labels.feature_version = features.feature_version
                     AND labels.stock = features.stock
                     AND labels.date = features.date
                    WHERE features.feature_version = ?
                      AND features.stock IN ({placeholders})
                      AND features.date >= ?
                      AND features.date <= ?
                    ORDER BY features.stock, features.date
                    """,
                    conn,
                    params=[feature_version, *chunk, str(start_date), str(end_date)],
                )
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        return out.sort_values(["stock", "date"]).reset_index(drop=True)

    def replace_model_feature_snapshots(
        self,
        frame: pd.DataFrame,
        *,
        feature_version: str,
        factor_version: str,
        source_start: str,
        source_end: str,
        summary: dict[str, Any] | None = None,
    ) -> dict[str, int]:
        if frame is None or frame.empty:
            return {"rows": 0, "stocks": 0}
        imported_at = datetime.now().astimezone().isoformat(timespec="microseconds")
        out = frame.copy()
        out["feature_version"] = feature_version
        out["factor_version"] = factor_version
        out["computed_at"] = imported_at
        out["source_start"] = str(source_start)
        out["source_end"] = str(source_end)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["stock", "date"])
        label_columns = [
            "future_return_5d", "future_up_5d", "future_return_10d", "future_up_10d", "future_return_20d", "future_up_20d",
        ]
        labels = out[["feature_version", "stock", "date", "computed_at", *[name for name in label_columns if name in out]]].copy()
        feature_columns = [name for name in out.columns if name not in label_columns]
        features = out[feature_columns].copy()
        stocks = sorted({str(stock) for stock in out["stock"].dropna().unique()})
        with self.connect() as conn:
            for chunk_start in range(0, len(stocks), 500):
                chunk = stocks[chunk_start : chunk_start + 500]
                placeholders = ",".join("?" for _ in chunk)
                conn.execute(
                    f"""
                    DELETE FROM model_feature_snapshots
                    WHERE feature_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    """,
                    [feature_version, *chunk, str(source_start), str(source_end)],
                )
                conn.execute(
                    f"""
                    DELETE FROM model_feature_labels
                    WHERE feature_version = ?
                      AND stock IN ({placeholders})
                      AND date >= ?
                      AND date <= ?
                    """,
                    [feature_version, *chunk, str(source_start), str(source_end)],
                )
            rows = _append_dataframe(conn, "model_feature_snapshots", features)
            _append_dataframe(conn, "model_feature_labels", labels)
            snapshot_id = f"{feature_version}:{source_start}:{source_end}:{imported_at}"
            conn.execute(
                """
                INSERT INTO model_feature_snapshot_runs (
                    snapshot_id, feature_version, factor_version, computed_at, source_start, source_end,
                    stocks, rows, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    feature_version,
                    factor_version,
                    imported_at,
                    str(source_start),
                    str(source_end),
                    len(stocks),
                    rows,
                    json.dumps(summary or {}, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
            create_views(conn)
        return {"rows": int(len(out)), "stocks": int(len(stocks))}


def _append_detail(conn: sqlite3.Connection, table: str, df: pd.DataFrame, run_id: str, stock: str, imported_at: str) -> int:
    if df is None or df.empty:
        return 0
    out = df.copy()
    if "symbol" in out.columns and "stock" not in out.columns:
        out["stock"] = out["symbol"]
    else:
        out["stock"] = stock
    out["run_id"] = run_id
    out["imported_at"] = imported_at
    out["source_file"] = f"db://{table}/{run_id}/{stock}"
    out["row_number"] = range(len(out))
    if table in {"backtest_daily_ledger", "agent_signal_records"}:
        from my_strategy.storage.persistence_guard import assert_unique_keys

        if "date" not in out.columns:
            raise ValueError(f"{table} requires a date column for run-scoped idempotency")
        assert_unique_keys(out, ["run_id", "stock", "date"], table=table)
    return _append_dataframe(conn, table, out)


def _append_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(_sqlite_value)
    clean = clean.where(pd.notna(clean), None)
    _ensure_columns(conn, table, clean)
    clean.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)
    return int(len(clean))


def _sqlite_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return value


def _ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    existing = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}
    for col in df.columns:
        if col in existing:
            continue
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" TEXT')
        existing.add(col)


def _has_columns(conn: sqlite3.Connection, table: str, cols: list[str]) -> bool:
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    return all(col in existing for col in cols)
