#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Import AI-Agent screen outputs into a local SQLite database."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.storage.persistence_guard import assert_unique_keys, require_run_id

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen"
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "model_registry" / "best_models.json"

RESULT_TABLES = [
    "screen_runs",
    "stock_screen_results",
    "buy_candidates",
    "recommend_buy_top",
    "focus_watchlist_results",
    "screen_failures",
    "backtest_reports",
    "stock_daily_results",
    "stock_trade_records",
    "agent_signal_records",
    "model_registry_history",
]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def default_db_path(output_root: str | Path) -> Path:
    return resolve_path(output_root) / "ai_agent_screen.db"


def import_screen_outputs(
    output_root: str | Path = DEFAULT_SCREEN_ROOT,
    db_path: str | Path | None = None,
    run_id: str | None = None,
    include_details: bool = True,
    model_registry_path: str | Path = DEFAULT_MODEL_REGISTRY,
    stocks: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Import compact screen results into SQLite; chart snapshots stay local."""
    root = resolve_path(output_root)
    db = resolve_path(db_path) if db_path else default_db_path(root)
    db.parent.mkdir(parents=True, exist_ok=True)

    imported_at = datetime.now().isoformat(timespec="seconds")
    summary = _read_json(root / "reports" / "summary.json")
    run_id = require_run_id(run_id, operation="screen persistence")
    stock_filter = _normalize_stock_filter(stocks)

    counts: dict[str, int] = {}
    with sqlite3.connect(db) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        init_database(conn)
        _delete_run(conn, run_id)

        run_row = _build_run_row(root, db, run_id, imported_at, summary, include_details)
        counts["screen_runs"] = _append_dataframe(conn, "screen_runs", pd.DataFrame([run_row]))

        report_dir = root / "reports"
        csv_specs = [
            ("stock_screen_results", report_dir / "ranking.csv"),
            ("buy_candidates", report_dir / "buy_candidates.csv"),
            ("recommend_buy_top", report_dir / "recommend_buy_top.csv"),
            ("focus_watchlist_results", report_dir / "focus_watchlist_backtest.csv"),
            ("screen_failures", report_dir / "failures.csv"),
        ]
        for table, path in csv_specs:
            df = _read_csv(path)
            if df is None:
                counts[table] = 0
                continue
            df = _filter_stock_dataframe(df, stock_filter)
            df = _with_import_columns(df, run_id, imported_at, path)
            counts[table] = _append_dataframe(conn, table, df)

        reports = _load_report_jsons(report_dir, run_id, imported_at, stock_filter=stock_filter)
        counts["backtest_reports"] = _append_dataframe(conn, "backtest_reports", reports) if not reports.empty else 0

        if include_details:
            detail_specs = [
                ("stock_daily_results", [(root / "daily", "*_daily.csv"), (root / "signals", "*_daily.csv")]),
                ("stock_trade_records", [(root / "trades", "*_trades.csv"), (root / "signals", "*_trades.csv")]),
                ("agent_signal_records", [(root / "signals", "*_signals.csv")]),
            ]
            for table, sources in detail_specs:
                detail = _load_detail_csv_sources(sources, run_id, imported_at, stock_filter=stock_filter)
                counts[table] = _append_dataframe(conn, table, detail) if not detail.empty else 0
        else:
            counts["stock_daily_results"] = 0
            counts["stock_trade_records"] = 0
            counts["agent_signal_records"] = 0

        # Result images remain local-only; the screen cache stores no chart
        # snapshots, image blobs, or chart-path rows.
        counts["chart_assets"] = 0

        registry = _load_model_registry(resolve_path(model_registry_path), run_id, imported_at)
        counts["model_registry_history"] = _append_dataframe(conn, "model_registry_history", registry) if not registry.empty else 0

        create_indexes(conn)
        conn.commit()

    return {
        "db_path": str(db),
        "output_root": str(root),
        "run_id": run_id,
        "imported_at": imported_at,
        "stocks_filter": sorted(stock_filter) if stock_filter else [],
        "counts": counts,
    }


def init_database(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS screen_runs (
            run_id TEXT PRIMARY KEY,
            imported_at TEXT,
            output_root TEXT,
            db_path TEXT,
            start TEXT,
            end TEXT,
            stocks INTEGER,
            success INTEGER,
            failed INTEGER,
            ml_model TEXT,
            dl_model TEXT,
            rl_model TEXT,
            elapsed_seconds REAL,
            include_details INTEGER,
            summary_json TEXT
        )
        """
    )
    for table in RESULT_TABLES:
        if table == "screen_runs":
            continue
        conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (run_id TEXT, imported_at TEXT)')


def create_indexes(conn: sqlite3.Connection) -> None:
    index_specs = [
        ("idx_screen_results_run_score", "stock_screen_results", ["run_id", "screen_score"]),
        ("idx_screen_results_stock_run", "stock_screen_results", ["stock", "run_id"]),
        ("idx_screen_results_action", "stock_screen_results", ["run_id", "latest_action", "latest_target_weight"]),
        ("idx_candidates_run_score", "buy_candidates", ["run_id", "screen_score"]),
        ("idx_recommend_run_score", "recommend_buy_top", ["run_id", "screen_score"]),
        ("idx_focus_stock_run", "focus_watchlist_results", ["stock", "run_id"]),
        ("idx_failures_run_stock", "screen_failures", ["run_id", "stock"]),
        ("idx_reports_stock_run", "backtest_reports", ["stock", "run_id"]),
        ("idx_daily_stock_date", "stock_daily_results", ["stock", "date"]),
        ("idx_trades_stock_date", "stock_trade_records", ["stock", "date"]),
        ("idx_signals_stock_date", "agent_signal_records", ["stock", "date"]),
        ("idx_model_registry_run_kind", "model_registry_history", ["run_id", "kind"]),
    ]
    for index_name, table, cols in index_specs:
        existing = _table_columns(conn, table)
        if all(col in existing for col in cols):
            quoted_cols = ", ".join(f'"{col}"' for col in cols)
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({quoted_cols})')


def _delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    for table in RESULT_TABLES:
        conn.execute(f'DELETE FROM "{table}" WHERE run_id = ?', (run_id,))


def _append_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    clean = df.copy()
    if table in {"stock_daily_results", "agent_signal_records"} and {"run_id", "stock", "date"}.issubset(clean.columns):
        assert_unique_keys(clean, ["run_id", "stock", "date"], table=table)
    clean = clean.where(pd.notna(clean), None)
    _ensure_columns(conn, table, clean)
    clean.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)
    return int(len(clean))


def _ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    existing = _table_columns(conn, table)
    for col in df.columns:
        if col in existing:
            continue
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {_sqlite_type(df[col])}')
        existing.add(col)


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _with_import_columns(df: pd.DataFrame, run_id: str, imported_at: str, path: Path) -> pd.DataFrame:
    out = df.copy()
    if "run_id" in out.columns:
        out["run_id"] = run_id
    else:
        out.insert(0, "run_id", run_id)
    if "imported_at" in out.columns:
        out["imported_at"] = imported_at
    else:
        out.insert(1, "imported_at", imported_at)
    out["source_file"] = str(path)
    return out


def _infer_run_id(root: Path, summary: dict[str, Any]) -> str:
    candidates = [
        root / "reports" / "summary.json",
        root / "reports" / "ranking.csv",
        root / "reports" / "recommend_buy_top.csv",
    ]
    newest = max((p.stat().st_mtime for p in candidates if p.exists()), default=root.stat().st_mtime if root.exists() else 0)
    stamp = datetime.fromtimestamp(newest).strftime("%Y%m%d_%H%M%S")
    start = str(summary.get("start", "")).replace("-", "")
    end = str(summary.get("end", "")).replace("-", "")
    prefix = "screen" if summary else "single"
    if start or end:
        return f"{prefix}_{start}_{end}_{stamp}"
    return f"{prefix}_{stamp}"


def _build_run_row(
    root: Path,
    db: Path,
    run_id: str,
    imported_at: str,
    summary: dict[str, Any],
    include_details: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "imported_at": imported_at,
        "output_root": str(root),
        "db_path": str(db),
        "start": summary.get("start", ""),
        "end": summary.get("end", ""),
        "stocks": summary.get("stocks", 0),
        "success": summary.get("success", 0),
        "failed": summary.get("failed", 0),
        "ml_model": summary.get("ml_model", ""),
        "dl_model": summary.get("dl_model", ""),
        "rl_model": summary.get("rl_model", ""),
        "elapsed_seconds": summary.get("elapsed_seconds", 0),
        "include_details": int(include_details),
        "summary_json": json.dumps(summary, ensure_ascii=False),
    }


def _load_report_jsons(
    report_dir: Path,
    run_id: str,
    imported_at: str,
    stock_filter: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not report_dir.exists():
        return pd.DataFrame()
    for path in sorted(report_dir.glob("*_report.json")):
        data = _read_json(path)
        if not data:
            continue
        row = _flatten_dict(data)
        stock = str(row.get("stock") or _stock_from_filename(path.name))
        if stock_filter and stock not in stock_filter:
            continue
        row.update(
            {
                "run_id": run_id,
                "imported_at": imported_at,
                "source_file": str(path),
                "report_type": "multi_agent" if "metrics" in data else "batch_stock",
            }
        )
        if "stock" not in row:
            row["stock"] = stock
        rows.append(row)
    return pd.DataFrame(rows)


def _load_detail_csvs(
    directory: Path,
    pattern: str,
    run_id: str,
    imported_at: str,
    stock_filter: set[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if not directory.exists():
        return pd.DataFrame()
    for path in sorted(directory.glob(pattern)):
        stock = _stock_from_filename(path.name)
        if stock_filter and stock not in stock_filter:
            continue
        df = _read_csv(path)
        if df is None or df.empty:
            continue
        df = df.copy()
        if "stock" not in df.columns:
            df.insert(0, "stock", stock)
        if "run_id" in df.columns:
            df["run_id"] = run_id
        else:
            df.insert(0, "run_id", run_id)
        if "imported_at" in df.columns:
            df["imported_at"] = imported_at
        else:
            df.insert(1, "imported_at", imported_at)
        df["source_file"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _load_detail_csv_sources(
    sources: list[tuple[Path, str]],
    run_id: str,
    imported_at: str,
    stock_filter: set[str] | None = None,
) -> pd.DataFrame:
    frames = [_load_detail_csvs(directory, pattern, run_id, imported_at, stock_filter=stock_filter) for directory, pattern in sources]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _scan_chart_assets(
    root: Path,
    run_id: str,
    imported_at: str,
    stock_filter: set[str] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for chart_type, directory in [("top_chart", root / "top_charts"), ("figure", root / "figures")]:
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.png")):
            row = _parse_chart_filename(path.name)
            if stock_filter and row.get("stock") not in stock_filter:
                continue
            row.update(
                {
                    "run_id": run_id,
                    "imported_at": imported_at,
                    "chart_type": chart_type,
                    "file_name": path.name,
                    "file_path": str(path),
                    "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _normalize_stock_filter(stocks: list[str] | set[str] | tuple[str, ...] | None) -> set[str] | None:
    if not stocks:
        return None
    normalized = {str(stock).strip().upper() for stock in stocks if str(stock).strip()}
    return normalized or None


def _filter_stock_dataframe(df: pd.DataFrame, stock_filter: set[str] | None) -> pd.DataFrame:
    if not stock_filter or df.empty:
        return df
    if "stock" not in df.columns:
        return df
    return df[df["stock"].astype(str).str.upper().isin(stock_filter)].reset_index(drop=True)


def _load_model_registry(path: Path, run_id: str, imported_at: str) -> pd.DataFrame:
    data = _read_json(path)
    source_file = str(path)
    if not data:
        try:
            from my_strategy.model_selector import load_best_models

            data = load_best_models(refresh_if_missing=False)
            source_file = "artifact://model_registry/json/best_models"
        except Exception:
            data = {}
    rows: list[dict[str, Any]] = []
    for kind in ["ml", "deep_learning"]:
        value = data.get(kind, {})
        if not isinstance(value, dict) or not value:
            continue
        row = _flatten_dict(value)
        row.update(
            {
                "run_id": run_id,
                "imported_at": imported_at,
                "registry_updated_at": data.get("updated_at", ""),
                "selection_rule": data.get("selection_rule", ""),
                "kind": kind,
                "source_file": source_file,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(_flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            row[name] = json.dumps(value, ensure_ascii=False)
        else:
            row[name] = value
    return row


def _stock_from_filename(name: str) -> str:
    match = re.search(r"(\d{6})_(SZ|SH|BJ)", name, flags=re.IGNORECASE)
    if not match:
        return ""
    return f"{match.group(1)}.{match.group(2).upper()}"


def _parse_chart_filename(name: str) -> dict[str, Any]:
    top = re.match(r"rank_(\d+)_(\d{6})_(SZ|SH|BJ)_(.+)_score_([0-9.]+)\.png$", name, re.IGNORECASE)
    if top:
        return {
            "rank": int(top.group(1)),
            "stock": f"{top.group(2)}.{top.group(3).upper()}",
            "stock_name": top.group(4),
            "screen_score": float(top.group(5)),
            "start": "",
            "end": "",
        }

    dated = re.match(r"(\d{6})_(SZ|SH|BJ)_(.+)_(\d{8})_(\d{8})_multi_agent\.png$", name, re.IGNORECASE)
    if dated:
        return {
            "rank": None,
            "stock": f"{dated.group(1)}.{dated.group(2).upper()}",
            "stock_name": dated.group(3),
            "screen_score": None,
            "start": _fmt_yyyymmdd(dated.group(4)),
            "end": _fmt_yyyymmdd(dated.group(5)),
        }

    return {
        "rank": None,
        "stock": _stock_from_filename(name),
        "stock_name": "",
        "screen_score": None,
        "start": "",
        "end": "",
    }


def _fmt_yyyymmdd(value: str) -> str:
    return f"{value[:4]}-{value[4:6]}-{value[6:]}" if len(value) == 8 else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import AI-Agent screen outputs into SQLite.")
    parser.add_argument("--output-root", default=str(DEFAULT_SCREEN_ROOT))
    parser.add_argument("--db-path", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-details", action="store_true", help="Skip daily/trade/signal CSV imports.")
    parser.add_argument("--model-registry", default=str(DEFAULT_MODEL_REGISTRY))
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    result = import_screen_outputs(
        output_root=args.output_root,
        db_path=args.db_path or None,
        run_id=args.run_id or None,
        include_details=not args.no_details,
        model_registry_path=args.model_registry,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
