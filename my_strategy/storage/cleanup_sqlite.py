#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SQLite retention policy for AI-Agent screen results.

The cleanup is dry-run by default. Callers must pass execute=True before any
row deletion or VACUUM is performed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen"
DEFAULT_DB_PATH = DEFAULT_SCREEN_ROOT / "ai_agent_screen.db"
DEFAULT_BEST_RL_PATH = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "rl" / "best_rl_model.json"

DETAIL_TABLES = [
    "stock_daily_results",
    "stock_trade_records",
    "agent_signal_records",
    "backtest_reports",
    "chart_assets",
]

SUMMARY_TABLES = [
    "stock_screen_results",
    "buy_candidates",
    "recommend_buy_top",
    "focus_watchlist_results",
    "screen_failures",
    "model_registry_history",
]


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def cleanup_ai_agent_screen_db(
    db_path: str | Path = DEFAULT_DB_PATH,
    keep_latest_runs: int = 3,
    keep_detail_runs: int = 2,
    keep_summary_runs: int = 10,
    keep_rl_experience_runs: int = 1,
    best_rl_path: str | Path = DEFAULT_BEST_RL_PATH,
    vacuum: bool = False,
    execute: bool = False,
) -> dict[str, Any]:
    """Apply retention rules to the local AI-Agent SQLite database."""
    db = resolve_path(db_path)
    if not db.exists():
        return {"db_path": str(db), "execute": execute, "status": "missing_db", "actions": []}

    before_size = db.stat().st_size
    actions: list[dict[str, Any]] = []

    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        run_ids = _ordered_screen_runs(conn)
        latest_keep = set(run_ids[: max(0, keep_latest_runs)])
        detail_keep = set(run_ids[: max(1, keep_detail_runs)]) if run_ids else set()
        summary_keep = set(run_ids[: max(1, keep_summary_runs)]) if run_ids else set()

        for table in DETAIL_TABLES:
            actions.append(
                _plan_or_delete_run_table(
                    conn=conn,
                    table=table,
                    keep_run_ids=detail_keep,
                    reason=f"keep latest {len(detail_keep)} detail runs",
                    execute=execute,
                )
            )

        for table in SUMMARY_TABLES:
            actions.append(
                _plan_or_delete_run_table(
                    conn=conn,
                    table=table,
                    keep_run_ids=summary_keep,
                    reason=f"keep latest {len(summary_keep)} summary runs",
                    execute=execute,
                )
            )

        actions.append(
            _plan_or_delete_rl_experiences(
                conn=conn,
                keep_recent_source_runs=max(0, keep_rl_experience_runs),
                protected_source_runs=_protected_rl_source_runs(best_rl_path),
                execute=execute,
            )
        )

        if execute:
            conn.commit()
            if vacuum:
                conn.execute("VACUUM")
            conn.execute("ANALYZE")
            conn.commit()

    after_size = db.stat().st_size
    return {
        "db_path": str(db),
        "execute": execute,
        "vacuum_requested": vacuum,
        "database_size_before_bytes": before_size,
        "database_size_after_bytes": after_size,
        "screen_runs_total": len(run_ids),
        "kept_latest_runs": run_ids[: max(0, keep_latest_runs)],
        "kept_detail_runs": sorted(detail_keep),
        "kept_summary_runs": sorted(summary_keep),
        "actions": actions,
    }


def _ordered_screen_runs(conn: sqlite3.Connection) -> list[str]:
    if not _table_exists(conn, "screen_runs"):
        return []
    cols = _table_columns(conn, "screen_runs")
    if "imported_at" in cols:
        rows = conn.execute(
            'SELECT run_id FROM "screen_runs" ORDER BY COALESCE(imported_at, "") DESC, run_id DESC'
        ).fetchall()
    else:
        rows = conn.execute('SELECT run_id FROM "screen_runs" ORDER BY run_id DESC').fetchall()
    return [str(row["run_id"]) for row in rows if row["run_id"]]


def _plan_or_delete_run_table(
    conn: sqlite3.Connection,
    table: str,
    keep_run_ids: set[str],
    reason: str,
    execute: bool,
) -> dict[str, Any]:
    base = {"table": table, "reason": reason, "keep_run_ids": sorted(keep_run_ids)}
    if not _table_exists(conn, table):
        return {**base, "status": "missing_table", "rows": 0}
    if "run_id" not in _table_columns(conn, table):
        return {**base, "status": "missing_run_id", "rows": 0}

    where, params = _not_in_where("run_id", keep_run_ids)
    rows = _count_where(conn, table, where, params)
    if execute and rows:
        conn.execute(f'DELETE FROM "{table}" WHERE {where}', params)
    return {**base, "status": "deleted" if execute else "dry_run", "rows": rows}


def _plan_or_delete_rl_experiences(
    conn: sqlite3.Connection,
    keep_recent_source_runs: int,
    protected_source_runs: set[str],
    execute: bool,
) -> dict[str, Any]:
    table = "rl_experiences"
    base = {"table": table, "reason": "keep recent RL experience runs and current best RL source run"}
    if not _table_exists(conn, table):
        return {**base, "status": "missing_table", "rows": 0, "keep_source_run_ids": sorted(protected_source_runs)}
    cols = _table_columns(conn, table)
    if "source_run_id" not in cols:
        return {**base, "status": "missing_source_run_id", "rows": 0, "keep_source_run_ids": sorted(protected_source_runs)}

    recent = _recent_rl_source_runs(conn, keep_recent_source_runs)
    keep_source_runs = set(recent) | set(protected_source_runs)
    where, params = _not_in_where("source_run_id", keep_source_runs)
    rows = _count_where(conn, table, where, params)
    if execute and rows:
        conn.execute(f'DELETE FROM "{table}" WHERE {where}', params)
    return {
        **base,
        "status": "deleted" if execute else "dry_run",
        "rows": rows,
        "keep_source_run_ids": sorted(keep_source_runs),
    }


def _recent_rl_source_runs(conn: sqlite3.Connection, limit: int) -> list[str]:
    if limit <= 0:
        return []
    cols = _table_columns(conn, "rl_experiences")
    order_expr = "MAX(rowid)"
    if "imported_at" in cols:
        order_expr = 'MAX(COALESCE(imported_at, ""))'
    rows = conn.execute(
        f'''
        SELECT source_run_id
        FROM "rl_experiences"
        WHERE source_run_id IS NOT NULL AND source_run_id != ''
        GROUP BY source_run_id
        ORDER BY {order_expr} DESC
        LIMIT ?
        ''',
        (limit,),
    ).fetchall()
    return [str(row["source_run_id"]) for row in rows if row["source_run_id"]]


def _protected_rl_source_runs(best_rl_path: str | Path) -> set[str]:
    try:
        from my_strategy.storage.access_layer import get_data_access

        item = get_data_access().get_json_artifact("model_registry", "best_rl_model")
        if item is not None:
            source_run = str(item[0].get("source_run_id") or "")
            return {source_run} if source_run else set()
    except Exception:
        pass
    path = resolve_path(best_rl_path)
    if not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return set()
    source_run = str(data.get("source_run_id") or "")
    return {source_run} if source_run else set()


def _not_in_where(column: str, values: Iterable[str]) -> tuple[str, tuple[Any, ...]]:
    clean = [str(value) for value in values if str(value)]
    if not clean:
        return "1 = 1", tuple()
    placeholders = ", ".join("?" for _ in clean)
    return f'("{column}" IS NULL OR "{column}" NOT IN ({placeholders}))', tuple(clean)


def _count_where(conn: sqlite3.Connection, table: str, where: str, params: tuple[Any, ...]) -> int:
    row = conn.execute(f'SELECT COUNT(*) AS n FROM "{table}" WHERE {where}', params).fetchone()
    return int(row["n"] if row else 0)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return row is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    return {str(row[1]) for row in rows}
