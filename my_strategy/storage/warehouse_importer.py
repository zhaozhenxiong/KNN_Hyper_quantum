#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Import local KHQuant data files into the unified SQLite warehouse."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

import pandas as pd

from my_strategy.core.paths import AI_AGENT_SCREEN_ROOT, ARTIFACT_RUNS_ROOT, MY_STRATEGY_ROOT, project_path
from my_strategy.storage.access_layer import get_data_access
from my_strategy.storage.persistence_guard import assert_unique_keys
from my_strategy.storage.warehouse_schema import create_views, init_warehouse

DEFAULT_RAW_WAREHOUSE_DB = MY_STRATEGY_ROOT / "data" / "raw" / "khquant_raw.db"
DEFAULT_PROCESSED_WAREHOUSE_DB = MY_STRATEGY_ROOT / "data" / "processed" / "warehouse" / "khquant.db"
DEFAULT_WAREHOUSE_DB = DEFAULT_PROCESSED_WAREHOUSE_DB
RAW_STOCK_DAILY = MY_STRATEGY_ROOT / "data" / "raw" / "stock_daily"
MODEL_REPORT_ROOTS = [
    MY_STRATEGY_ROOT / "data" / "processed" / "ml" / "reports",
    MY_STRATEGY_ROOT / "data" / "processed" / "deep_learning" / "reports",
    MY_STRATEGY_ROOT / "data" / "processed" / "rl" / "reports",
]

RUN_TABLES = [
    "runs",
    "run_artifacts",
    "backtest_stock_results",
    "backtest_reports",
    "backtest_daily_ledger",
    "backtest_trades",
    "agent_signal_records",
    "backtest_failures",
]


@dataclass
class ImportStats:
    db_path: str
    dry_run: bool
    imported_at: str
    counts: dict[str, int] = field(default_factory=dict)
    issues: list[dict[str, Any]] = field(default_factory=list)
    sources: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, count: int) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(count)

    def issue(self, *, category: str, severity: str, entity_type: str, entity_id: str, source_file: str, message: str, details: dict[str, Any] | None = None) -> None:
        self.issues.append(
            {
                "category": category,
                "severity": severity,
                "entity_type": entity_type,
                "entity_id": entity_id,
                "source_file": source_file,
                "message": message,
                "details_json": json.dumps(details or {}, ensure_ascii=False, sort_keys=True, default=str),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "db_path": self.db_path,
            "dry_run": self.dry_run,
            "imported_at": self.imported_at,
            "counts": self.counts,
            "issues_count": len(self.issues),
            "issues_sample": self.issues[:20],
            "sources": self.sources,
        }


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def resolve(path: str | Path) -> Path:
    return project_path(path)


def import_warehouse(
    *,
    db_path: str | Path = DEFAULT_WAREHOUSE_DB,
    import_raw: bool = True,
    import_screen_runs: bool = True,
    import_models: bool = True,
    import_artifacts: bool = True,
    dry_run: bool = False,
    screen_roots: list[str | Path] | None = None,
    run_id: str | None = None,
    stocks: list[str] | set[str] | tuple[str, ...] | None = None,
) -> ImportStats:
    db = resolve(db_path)
    imported_at = now_iso()
    stats = ImportStats(db_path=str(db), dry_run=dry_run, imported_at=imported_at)
    stock_filter = _normalize_stock_filter(stocks)

    if dry_run:
        conn: sqlite3.Connection | None = None
    else:
        db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db)
        init_warehouse(conn)

    try:
        if import_raw:
            _import_raw_stock_daily(conn, stats)
        if import_screen_runs:
            roots = [resolve(p) for p in screen_roots] if screen_roots else discover_screen_roots()
            stats.sources["screen_roots"] = [str(p) for p in roots]
            for root in roots:
                _import_screen_root(conn, stats, root, run_id_override=run_id, stock_filter=stock_filter)
        if import_models:
            _import_model_reports(conn, stats)
        if import_artifacts:
            _import_artifact_catalog(conn, stats)
        _write_quality_issues(conn, stats)
        _record_import(conn, stats, source_root=str(MY_STRATEGY_ROOT))
        if conn is not None:
            create_views(conn)
        if conn is not None:
            conn.commit()
    finally:
        if conn is not None:
            conn.close()
    return stats


def import_backtest_run_to_warehouse(
    *,
    output_root: str | Path,
    db_path: str | Path = DEFAULT_WAREHOUSE_DB,
    run_id: str | None = None,
    stocks: list[str] | set[str] | tuple[str, ...] | None = None,
    include_models: bool = True,
    include_artifacts: bool = True,
    dry_run: bool = False,
) -> ImportStats:
    """Import one finished backtest output root without rescanning raw market data."""
    return import_warehouse(
        db_path=db_path,
        import_raw=False,
        import_screen_runs=True,
        import_models=include_models,
        import_artifacts=include_artifacts,
        dry_run=dry_run,
        screen_roots=[output_root],
        run_id=run_id,
        stocks=stocks,
    )


def discover_screen_roots() -> list[Path]:
    roots: list[Path] = []
    if (AI_AGENT_SCREEN_ROOT / "reports" / "summary.json").exists():
        roots.append(AI_AGENT_SCREEN_ROOT)
    compare_root = AI_AGENT_SCREEN_ROOT / "model_compare"
    if compare_root.exists():
        for child in sorted(compare_root.iterdir()):
            if child.is_dir() and (child / "reports" / "summary.json").exists():
                roots.append(child)
    return roots


def _import_raw_stock_daily(conn: sqlite3.Connection | None, stats: ImportStats) -> None:
    root = RAW_STOCK_DAILY
    files = sorted(root.glob("*.parquet"))
    stats.sources["raw_stock_daily"] = {"root": str(root), "files": len(files)}
    if conn is not None:
        for table in ["securities", "stock_daily_raw", "stock_daily_normalized"]:
            conn.execute(f'DELETE FROM "{table}"')
        conn.execute("DELETE FROM data_quality_issues WHERE entity_type = 'stock_daily_raw'")

    for path in files:
        stock = path.stem
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            stats.issue(
                category="raw_read_error",
                severity="error",
                entity_type="stock_daily_raw",
                entity_id=stock,
                source_file=str(path),
                message=str(exc),
            )
            continue
        if "date" not in df.columns:
            stats.issue(
                category="raw_schema",
                severity="error",
                entity_type="stock_daily_raw",
                entity_id=stock,
                source_file=str(path),
                message="missing date column",
            )
            continue
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])
        df["stock"] = df.get("code", stock).fillna(stock).astype(str)
        df.loc[df["stock"].eq("") | df["stock"].isna(), "stock"] = stock
        df["row_number"] = range(len(df))
        df["source_file"] = str(path)
        df["imported_at"] = stats.imported_at

        raw_cols = [
            "stock",
            "date",
            "row_number",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
            "source",
            "updated_at",
            "source_file",
            "imported_at",
        ]
        raw = _select_columns(df, raw_cols)
        if "updated_at" in raw.columns:
            raw["updated_at"] = pd.to_datetime(raw["updated_at"], errors="coerce").astype(str).replace("NaT", None)
        if conn is not None:
            _append_dataframe(conn, "stock_daily_raw", raw)
        stats.add("stock_daily_raw", len(raw))

        normalized = _normalize_daily(df, stock=stock, path=path, stats=stats)
        if conn is not None and not normalized.empty:
            _append_dataframe(conn, "stock_daily_normalized", normalized)
        stats.add("stock_daily_normalized", len(normalized))

        security = {
            "stock": stock,
            "exchange": stock.split(".")[-1] if "." in stock else "",
            "name": "",
            "first_date": normalized["date"].min() if not normalized.empty else None,
            "last_date": normalized["date"].max() if not normalized.empty else None,
            "raw_rows": len(raw),
            "source_file": str(path),
            "updated_at": _max_text(raw.get("updated_at")),
            "imported_at": stats.imported_at,
        }
        if conn is not None:
            _append_dataframe(conn, "securities", pd.DataFrame([security]))
        stats.add("securities", 1)


def _normalize_daily(df: pd.DataFrame, *, stock: str, path: Path, stats: ImportStats) -> pd.DataFrame:
    data = df.copy()
    trade_cols = ["trade_open", "trade_high", "trade_low", "trade_close"]
    has_trade_columns = all(col in data.columns for col in trade_cols)
    if not has_trade_columns:
        stats.issue(
            category="raw_schema",
            severity="warning",
            entity_type="stock_daily_raw",
            entity_id=stock,
            source_file=str(path),
            message="missing one or more trade OHLC columns; normalized price falls back to open/high/low/close",
            details={"missing": [col for col in trade_cols if col not in data.columns]},
        )
    date_counts = data.groupby("date").size()
    duplicate_dates = date_counts[date_counts > 1]
    if not duplicate_dates.empty:
        stats.issue(
            category="raw_duplicate_date",
            severity="warning",
            entity_type="stock_daily_raw",
            entity_id=stock,
            source_file=str(path),
            message=f"{len(duplicate_dates)} duplicate date(s) found; normalized table keeps preferred real-price row",
            details={"dates": duplicate_dates.head(20).to_dict()},
        )
    if has_trade_columns:
        data["_has_trade_price"] = pd.to_numeric(data["trade_close"], errors="coerce").gt(0).astype(int)
    else:
        data["_has_trade_price"] = 0
    data = data.sort_values(["date", "_has_trade_price", "row_number"], ascending=[True, False, False])
    picked = data.drop_duplicates("date", keep="first").copy()
    picked["duplicate_source_rows"] = picked["date"].map(date_counts).fillna(1).astype(int)

    out = pd.DataFrame(
        {
            "stock": stock,
            "date": picked["date"],
            "open": _price_with_fallback(picked, "trade_open", "open"),
            "high": _price_with_fallback(picked, "trade_high", "high"),
            "low": _price_with_fallback(picked, "trade_low", "low"),
            "close": _price_with_fallback(picked, "trade_close", "close"),
            "factor_open": pd.to_numeric(picked.get("open"), errors="coerce"),
            "factor_high": pd.to_numeric(picked.get("high"), errors="coerce"),
            "factor_low": pd.to_numeric(picked.get("low"), errors="coerce"),
            "factor_close": pd.to_numeric(picked.get("close"), errors="coerce"),
            "volume": pd.to_numeric(picked.get("volume"), errors="coerce"),
            "amount": pd.to_numeric(picked.get("amount"), errors="coerce"),
            "turnover": pd.to_numeric(picked.get("turnover"), errors="coerce") if "turnover" in picked else None,
            "pct_chg": pd.to_numeric(picked.get("pct_chg"), errors="coerce") if "pct_chg" in picked else None,
            "source": picked.get("source", ""),
            "source_file": str(path),
            "duplicate_source_rows": picked["duplicate_source_rows"],
            "has_trade_price": picked["_has_trade_price"],
            "imported_at": stats.imported_at,
        }
    )
    return out


def _import_screen_root(
    conn: sqlite3.Connection | None,
    stats: ImportStats,
    root: Path,
    *,
    run_id_override: str | None = None,
    stock_filter: set[str] | None = None,
) -> None:
    report_dir = root / "reports"
    summary = _read_json(report_dir / "summary.json")
    run_id = run_id_override or _infer_run_id(root, summary)
    single_run_mode = bool(run_id_override and stock_filter)
    if conn is not None:
        _delete_run(conn, run_id)

    reports = _load_json_reports(report_dir, run_id, stats.imported_at, stock_filter=stock_filter)
    if single_run_mode and not reports.empty:
        multi_agent_reports = reports[reports["source_file"].astype(str).str.contains("_multi_agent_report", regex=False)].copy()
        if not multi_agent_reports.empty:
            reports = multi_agent_reports
    summary_for_run = summary if not (run_id_override and stock_filter) else {}
    run_row = {
        "run_id": run_id,
        "imported_at": stats.imported_at,
        "task": _infer_task(root, run_id),
        "source_root": str(root),
        "artifact_root": str(_artifact_root_for_run(run_id)),
        "start": summary_for_run.get("start"),
        "end": summary_for_run.get("end"),
        "stocks": summary_for_run.get("stocks", len(stock_filter) if stock_filter else None),
        "success": summary_for_run.get("success", len(reports) if stock_filter else None),
        "failed": summary_for_run.get("failed"),
        "ml_model": summary_for_run.get("ml_model"),
        "dl_model": summary_for_run.get("dl_model"),
        "rl_model": summary_for_run.get("rl_model"),
        "elapsed_seconds": summary_for_run.get("elapsed_seconds"),
        "summary_json": json.dumps(summary_for_run, ensure_ascii=False, sort_keys=True, default=str),
    }
    if conn is not None:
        _append_dataframe(conn, "runs", pd.DataFrame([run_row]))
    stats.add("runs", 1)

    # A shared ai_agent_screen root may contain stale batch ranking files while
    # a single-stock run only updates *_multi_agent_* files. In that case avoid
    # relabeling stale aggregate CSV rows with the new single run_id.
    include_aggregate_csv = not single_run_mode
    if include_aggregate_csv:
        csv_specs = [
            ("backtest_stock_results", report_dir / "ranking.csv"),
            ("backtest_failures", report_dir / "failures.csv"),
        ]
        for table, path in csv_specs:
            df = _read_csv(path)
            if df is None:
                continue
            df = _filter_stock_dataframe(df, stock_filter)
            df = _with_run_columns(df, run_id, stats.imported_at, path)
            if conn is not None:
                _append_dataframe(conn, table, df)
            stats.add(table, len(df))

    if conn is not None and not reports.empty:
        _append_dataframe(conn, "backtest_reports", reports)
    stats.add("backtest_reports", len(reports))

    if single_run_mode:
        detail_specs = [
            ("backtest_daily_ledger", [(root / "signals", "*_multi_agent_daily.csv")]),
            ("backtest_trades", [(root / "signals", "*_multi_agent_trades.csv")]),
            ("agent_signal_records", [(root / "signals", "*_multi_agent_signals.csv")]),
        ]
    else:
        detail_specs = [
            ("backtest_daily_ledger", [(root / "daily", "*_daily.csv"), (root / "signals", "*_multi_agent_daily.csv")]),
            ("backtest_trades", [(root / "trades", "*_trades.csv"), (root / "signals", "*_multi_agent_trades.csv")]),
            ("agent_signal_records", [(root / "signals", "*_signals.csv"), (root / "signals", "*_multi_agent_signals.csv")]),
        ]
    for table, sources in detail_specs:
        rows = _load_detail_csv_sources(sources, run_id, stats.imported_at, stats=stats, stock_filter=stock_filter)
        if conn is not None and not rows.empty:
            _append_dataframe(conn, table, rows)
        stats.add(table, len(rows))

    artifacts = _scan_output_artifacts(root, run_id, stats.imported_at, stock_filter=stock_filter)
    if conn is not None and not artifacts.empty:
        _append_dataframe(conn, "run_artifacts", artifacts)
    stats.add("run_artifacts", len(artifacts))

    comparison = _load_comparison_summary(root, stats.imported_at)
    if comparison:
        if conn is not None:
            conn.execute("DELETE FROM comparison_reports WHERE comparison_id = ?", (comparison["comparison_id"],))
            _append_dataframe(conn, "comparison_reports", pd.DataFrame([comparison]))
        stats.add("comparison_reports", 1)


def _import_model_reports(conn: sqlite3.Connection | None, stats: ImportStats) -> None:
    reports: list[dict[str, Any]] = []
    reports.extend(_model_reports_from_artifacts(stats))
    for root in MODEL_REPORT_ROOTS:
        for path in sorted(root.glob("*.json")):
            data = _read_json(path)
            if not data:
                continue
            model_name = data.get("model_name") or data.get("model_id") or path.stem.replace("_report", "")
            kind = _model_kind_from_path(path)
            metrics = _model_metrics(data)
            row = {
                "model_name": model_name,
                "kind": kind,
                "run_id": (data.get("run_context") or {}).get("run_id") or data.get("run_id") or data.get("source_run_id"),
                "model_type": data.get("model_type"),
                "device": data.get("device"),
                "report_path": str(path),
                "model_path": data.get("model_path") or "",
                "created_at": data.get("created_at") or (data.get("run_context") or {}).get("created_at"),
                "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str),
                "samples_json": json.dumps(data.get("samples", {}), ensure_ascii=False, sort_keys=True, default=str),
                "stocks_json": json.dumps(data.get("stocks", {}), ensure_ascii=False, sort_keys=True, default=str),
                "feature_columns_json": json.dumps(data.get("feature_columns", []), ensure_ascii=False, default=str),
                "params_json": json.dumps(_model_params(data), ensure_ascii=False, sort_keys=True, default=str),
                "imported_at": stats.imported_at,
            }
            reports.append(row)
    if conn is not None:
        conn.execute("DELETE FROM model_training_runs")
        if reports:
            df = pd.DataFrame(reports).drop_duplicates(["kind", "model_name"], keep="last")
            _append_dataframe(conn, "model_training_runs", df)
    stats.add("model_training_runs", len(reports))


def _model_reports_from_artifacts(stats: ImportStats) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        storage = get_data_access()
        records = storage.list_artifacts(namespace="model_report", kind="json")
    except Exception:
        return rows
    for record in records:
        name = str(record["name"])
        item = storage.get_json_artifact("model_report", name)
        if item is None:
            continue
        data = item[0]
        if not isinstance(data, dict):
            continue
        kind = str((record.get("metadata") or {}).get("kind") or data.get("kind") or name.split(":")[0])
        model_name = str(data.get("model_name") or data.get("model_id") or (name.split(":")[1] if ":" in name else name))
        metrics = _model_metrics(data)
        rows.append(
            {
                "model_name": model_name,
                "kind": kind,
                "run_id": (data.get("run_context") or {}).get("run_id") or data.get("run_id") or data.get("source_run_id"),
                "model_type": data.get("model_type"),
                "device": data.get("device"),
                "report_path": f"artifact://model_report/json/{name}",
                "model_path": data.get("model_path") or f"artifact://{kind}/model/{model_name}",
                "created_at": data.get("created_at") or (data.get("run_context") or {}).get("created_at"),
                "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True, default=str),
                "samples_json": json.dumps(data.get("samples", {}), ensure_ascii=False, sort_keys=True, default=str),
                "stocks_json": json.dumps(data.get("stocks", {}), ensure_ascii=False, sort_keys=True, default=str),
                "feature_columns_json": json.dumps(data.get("feature_columns", []), ensure_ascii=False, default=str),
                "params_json": json.dumps(_model_params(data), ensure_ascii=False, sort_keys=True, default=str),
                "imported_at": stats.imported_at,
            }
        )
    return rows


def _import_artifact_catalog(conn: sqlite3.Connection | None, stats: ImportStats) -> None:
    if not ARTIFACT_RUNS_ROOT.exists():
        return
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in ARTIFACT_RUNS_ROOT.iterdir() if p.is_dir()):
        run_id = run_dir.name
        if conn is not None:
            conn.execute("DELETE FROM run_artifacts WHERE run_id = ? AND file_path LIKE ?", (run_id, f"{run_dir}%"))
        metadata = _read_json(run_dir / "metadata.json")
        if metadata and conn is not None:
            conn.execute(
                """
                INSERT INTO runs (run_id, imported_at, task, source_root, artifact_root, start, end, summary_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    artifact_root=excluded.artifact_root,
                    task=COALESCE(runs.task, excluded.task),
                    summary_json=CASE WHEN runs.summary_json = '{}' THEN excluded.summary_json ELSE runs.summary_json END
                """,
                (
                    run_id,
                    stats.imported_at,
                    metadata.get("task"),
                    str(run_dir),
                    str(run_dir),
                    metadata.get("as_of_date"),
                    metadata.get("as_of_date"),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True, default=str),
                ),
            )
        for file in run_dir.rglob("*"):
            if not file.is_file() or _is_result_image(file):
                continue
            rows.append(_artifact_row(run_id, file, stats.imported_at, base=run_dir))
    if conn is not None and rows:
        _append_dataframe(conn, "run_artifacts", pd.DataFrame(rows))
    stats.add("artifact_catalog_files", len(rows))


def _write_quality_issues(conn: sqlite3.Connection | None, stats: ImportStats) -> None:
    if conn is None or not stats.issues:
        return
    rows = []
    for issue in stats.issues:
        payload = dict(issue)
        payload["imported_at"] = stats.imported_at
        payload["issue_id"] = _stable_id(issue)
        rows.append(payload)
    _append_dataframe(conn, "data_quality_issues", pd.DataFrame(rows))


def _record_import(conn: sqlite3.Connection | None, stats: ImportStats, *, source_root: str) -> None:
    if conn is None:
        return
    payload = stats.to_dict()
    row = {
        "import_id": f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
        "imported_at": stats.imported_at,
        "mode": "warehouse_import",
        "source_root": source_root,
        "db_path": stats.db_path,
        "dry_run": int(stats.dry_run),
        "summary_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
    }
    _append_dataframe(conn, "import_runs", pd.DataFrame([row]))


def _delete_run(conn: sqlite3.Connection, run_id: str) -> None:
    for table in RUN_TABLES:
        conn.execute(f'DELETE FROM "{table}" WHERE run_id = ?', (run_id,))


def _append_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    clean = df.copy()
    clean = clean.where(pd.notna(clean), None)
    if table in {"backtest_daily_ledger", "agent_signal_records", "backtest_trades"}:
        keys = ["run_id", "stock", "date", "source_file", "row_number"]
        assert_unique_keys(clean, keys, table=table)
    _ensure_columns(conn, table, clean)
    clean.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)
    return int(len(clean))


def _ensure_columns(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> None:
    existing = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")').fetchall()}
    for col in df.columns:
        if col in existing:
            continue
        conn.execute(f'ALTER TABLE "{table}" ADD COLUMN "{col}" {_sqlite_type(df[col])}')
        existing.add(col)


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "REAL"
    return "TEXT"


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = df[col] if col in df.columns else None
    return out


def _price_with_fallback(df: pd.DataFrame, trade_col: str, factor_col: str) -> pd.Series:
    base = pd.to_numeric(df.get(factor_col), errors="coerce")
    if trade_col not in df.columns:
        return base
    trade = pd.to_numeric(df.get(trade_col), errors="coerce")
    return trade.where(trade.gt(0), base)


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
        return None


def _with_run_columns(df: pd.DataFrame, run_id: str, imported_at: str, path: Path) -> pd.DataFrame:
    out = df.copy()
    out["run_id"] = run_id
    out["imported_at"] = imported_at
    out["source_file"] = str(path)
    return out


def _infer_run_id(root: Path, summary: dict[str, Any]) -> str:
    for key in ["run_id", "source_run_id"]:
        if summary.get(key):
            return str(summary[key])
    ranking = _read_csv(root / "reports" / "ranking.csv")
    if ranking is not None and "run_id" in ranking.columns and not ranking["run_id"].dropna().empty:
        return str(ranking["run_id"].dropna().iloc[0])
    return root.name


def _infer_task(root: Path, run_id: str) -> str:
    if "model_compare" in str(root):
        return "model-compare-screen"
    if "multi-agent-backtest" in run_id:
        return "multi-agent-backtest"
    return "screen-run"


def _artifact_root_for_run(run_id: str) -> Path | str:
    path = ARTIFACT_RUNS_ROOT / run_id
    return path if path.exists() else ""


def _load_json_reports(report_dir: Path, run_id: str, imported_at: str, *, stock_filter: set[str] | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(report_dir.glob("*_report.json")):
        data = _read_json(path)
        stock = _stock_from_file(path)
        if stock_filter and stock not in stock_filter:
            continue
        rows.append(
            {
                "run_id": run_id,
                "imported_at": imported_at,
                "stock": data.get("stock") or stock,
                "stock_name": data.get("stock_name") or data.get("name"),
                "status": data.get("status", "ok"),
                "source_file": str(path),
                "report_json": json.dumps(data, ensure_ascii=False, sort_keys=True, default=str),
            }
        )
    return pd.DataFrame(rows)


def _load_detail_csv_sources(
    sources: list[tuple[Path, str]],
    run_id: str,
    imported_at: str,
    *,
    stats: ImportStats,
    stock_filter: set[str] | None = None,
) -> pd.DataFrame:
    frames = []
    seen: set[Path] = set()
    for directory, pattern in sources:
        for path in sorted(directory.glob(pattern)) if directory.exists() else []:
            if path in seen:
                continue
            seen.add(path)
            stock = _stock_from_file(path)
            if stock_filter and stock not in stock_filter:
                continue
            df = _load_one_detail_csv(path, run_id, imported_at, stock, stats=stats)
            if df is not None:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _load_one_detail_csv(path: Path, run_id: str, imported_at: str, stock: str, *, stats: ImportStats) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        stats.issue(
            category="empty_csv",
            severity="warning",
            entity_type="backtest_detail",
            entity_id=stock,
            source_file=str(path),
            message="empty CSV skipped",
        )
        return None
    except Exception as exc:
        stats.issue(
            category="csv_read_error",
            severity="error",
            entity_type="backtest_detail",
            entity_id=stock,
            source_file=str(path),
            message=str(exc),
        )
        return None
    if "symbol" in df.columns and "stock" not in df.columns:
        df["stock"] = df["symbol"]
    else:
        df["stock"] = stock
    df["run_id"] = run_id
    df["imported_at"] = imported_at
    df["source_file"] = str(path)
    df["row_number"] = range(len(df))
    frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def _scan_output_artifacts(root: Path, run_id: str, imported_at: str, *, stock_filter: set[str] | None = None) -> pd.DataFrame:
    rows = []
    for directory, artifact_type in [
        (root / "top_charts", "top_chart"),
        (root / "figures", "figure"),
        (root / "reports", "report"),
    ]:
        if not directory.exists():
            continue
        for file in directory.glob("*"):
            if file.is_file() and not _is_result_image(file):
                if stock_filter:
                    stock = _stock_from_file(file)
                    if stock not in stock_filter and not any(stock.replace(".", "_") in file.name for stock in stock_filter):
                        continue
                rows.append(_artifact_row(run_id, file, imported_at, artifact_type=artifact_type, base=root))
    return pd.DataFrame(rows)


def _is_result_image(path: Path) -> bool:
    """Result snapshots remain local files and are never catalogued in SQLite."""
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}


def _artifact_row(run_id: str, file: Path, imported_at: str, *, artifact_type: str | None = None, base: Path | None = None) -> dict[str, Any]:
    stat = file.stat()
    if artifact_type is None:
        artifact_type = file.parent.name
    return {
        "run_id": run_id,
        "artifact_type": artifact_type,
        "file_name": file.name,
        "file_path": str(file),
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
        "imported_at": imported_at,
        "relative_path": str(file.relative_to(base)) if base and file.is_relative_to(base) else file.name,
    }


def _load_comparison_summary(root: Path, imported_at: str) -> dict[str, Any] | None:
    path = root / "reports" / "model_diff_summary.json"
    data = _read_json(path)
    if not data:
        return None
    comparison_id = root.name
    return {
        "comparison_id": comparison_id,
        "imported_at": imported_at,
        "source_root": str(root),
        "baseline_run_id": (data.get("baseline", {}).get("summary") or {}).get("run_id", ""),
        "candidate_run_id": (data.get("today_models", {}).get("summary") or {}).get("run_id", ""),
        "summary_json": json.dumps(data, ensure_ascii=False, sort_keys=True, default=str),
        "diff_file": data.get("paths", {}).get("diff_csv", str(root / "reports" / "model_diff_vs_historical_best.csv")),
    }


def _model_kind_from_path(path: Path) -> str:
    text = str(path).replace("\\", "/")
    if "/deep_learning/" in text:
        return "deep_learning"
    if "/rl/" in text:
        return "rl"
    if "/ml/" in text:
        return "ml"
    return "unknown"


def _model_metrics(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": data.get("score"),
        "best_validation_score": data.get("best_validation_score"),
        "best_iteration": data.get("best_iteration"),
        "valid_metrics": data.get("valid_metrics"),
        "test_metrics": data.get("test_metrics"),
        "valid_mean_reward": data.get("valid_mean_reward"),
        "valid_policy_mean_reward": data.get("valid_policy_mean_reward"),
    }


def _model_params(data: dict[str, Any]) -> dict[str, Any]:
    keys = ["task", "label_column", "prediction_kind", "lookback", "epochs", "batch_size", "hidden_size", "max_samples", "train_end", "valid_end"]
    return {key: data.get(key) for key in keys if key in data}


def _stock_from_file(path: Path) -> str:
    stem = path.stem
    for suffix in [
        "_multi_agent_signals",
        "_multi_agent_daily",
        "_multi_agent_trades",
        "_multi_agent_report",
        "_signals",
        "_daily",
        "_trades",
        "_report",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.split("_")
    if len(parts) >= 2 and parts[-1] in {"SZ", "SH", "BJ"}:
        return ".".join(["_".join(parts[:-1]), parts[-1]])
    return stem.replace("_", ".")


def _stable_id(data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _max_text(series: pd.Series | None) -> str | None:
    if series is None:
        return None
    values = [str(v) for v in series.dropna().tolist() if str(v) not in {"NaT", "None", "nan"}]
    return max(values) if values else None


def _normalize_stock_filter(stocks: list[str] | set[str] | tuple[str, ...] | None) -> set[str] | None:
    if not stocks:
        return None
    normalized = set()
    for stock in stocks:
        value = str(stock).strip()
        if not value:
            continue
        if "." not in value and "_" in value:
            parts = value.split("_")
            if parts[-1] in {"SZ", "SH", "BJ"}:
                value = ".".join(["_".join(parts[:-1]), parts[-1]])
        normalized.add(value)
    return normalized or None


def _filter_stock_dataframe(df: pd.DataFrame, stock_filter: set[str] | None) -> pd.DataFrame:
    if not stock_filter or "stock" not in df.columns:
        return df
    return df[df["stock"].astype(str).isin(stock_filter)].copy()
