"""Idempotency indexes and relationship audits for the processed warehouse.

SQLite cannot add foreign keys to an existing table without rebuilding it.  This
module therefore creates safe unique indexes where historic rows allow it and
provides a read-only audit for the logical run/artifact relationships.
"""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from typing import Any


RUN_DETAIL_KEYS = ("run_id", "stock", "date", "source_file", "row_number")
RUN_DETAIL_TABLES = ("backtest_daily_ledger", "agent_signal_records", "backtest_trades")


def ensure_integrity_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_integrity_audits (
            audit_id TEXT PRIMARY KEY,
            audited_at TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_detail_archives (
            archive_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            run_id TEXT NOT NULL,
            date_partition TEXT NOT NULL DEFAULT '',
            file_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            min_date TEXT,
            max_date TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (table_name, run_id, date_partition, file_path)
        )
        """
    )


def ensure_run_detail_unique_indexes(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    """Create physical unique indexes only after proving no historic duplicates."""
    result: dict[str, dict[str, Any]] = {}
    for table in RUN_DETAIL_TABLES:
        if not _table_has_columns(conn, table, RUN_DETAIL_KEYS):
            result[table] = {"status": "deferred_missing_columns", "keys": list(RUN_DETAIL_KEYS)}
            continue
        duplicate_rows = _duplicate_rows(conn, table, RUN_DETAIL_KEYS)
        if duplicate_rows:
            result[table] = {"status": "blocked_existing_duplicates", "keys": list(RUN_DETAIL_KEYS), "duplicate_rows": duplicate_rows}
            continue
        index_name = f"uq_{table}_run_stock_date_source_row"
        quoted = ", ".join(f'"{column}"' for column in RUN_DETAIL_KEYS)
        conn.execute(f'CREATE UNIQUE INDEX IF NOT EXISTS "{index_name}" ON "{table}" ({quoted})')
        result[table] = {"status": "created_or_present", "keys": list(RUN_DETAIL_KEYS), "index": index_name}
    return result


def audit_warehouse_integrity(conn: sqlite3.Connection) -> dict[str, Any]:
    """Check logical FKs and duplicate keys without changing source data."""
    checks: list[dict[str, Any]] = []
    for table in RUN_DETAIL_TABLES:
        if _table_has_columns(conn, table, RUN_DETAIL_KEYS):
            checks.append({"check": f"duplicate:{table}", "count": _duplicate_rows(conn, table, RUN_DETAIL_KEYS)})
    for table in ("backtest_reports", "backtest_stock_results", "run_artifacts", "run_output_references", "signal_snapshots"):
        if _table_has_columns(conn, table, ("run_id",)) and _table_has_columns(conn, "runs", ("run_id",)):
            missing = int(conn.execute(
                f'SELECT COUNT(*) FROM "{table}" child LEFT JOIN runs parent ON parent.run_id = child.run_id '
                "WHERE child.run_id IS NOT NULL AND child.run_id <> '' AND parent.run_id IS NULL"
            ).fetchone()[0])
            checks.append({"check": f"orphan_run:{table}", "count": missing})
    if _table_has_columns(conn, "run_output_references", ("artifact_id",)) and _table_has_columns(conn, "artifact_blobs", ("artifact_id",)):
        missing = int(conn.execute(
            "SELECT COUNT(*) FROM run_output_references ref LEFT JOIN artifact_blobs blob ON blob.artifact_id = ref.artifact_id "
            "WHERE blob.artifact_id IS NULL"
        ).fetchone()[0])
        checks.append({"check": "orphan_artifact:run_output_references", "count": missing})
    return {"status": "ok" if all(item["count"] == 0 for item in checks) else "issues", "checks": checks}


def record_integrity_audit(conn: sqlite3.Connection, report: dict[str, Any], *, audit_id: str) -> None:
    ensure_integrity_schema(conn)
    conn.execute(
        "INSERT INTO storage_integrity_audits (audit_id, audited_at, status, summary_json) VALUES (?, ?, ?, ?)",
        (audit_id, datetime.now().astimezone().isoformat(timespec="seconds"), str(report["status"]), json.dumps(report, ensure_ascii=False, sort_keys=True)),
    )


def _table_has_columns(conn: sqlite3.Connection, table: str, columns: tuple[str, ...]) -> bool:
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if exists is None:
        return False
    found = {str(row[1]) for row in conn.execute(f'PRAGMA table_info("{table}")')}
    return set(columns).issubset(found)


def _duplicate_rows(conn: sqlite3.Connection, table: str, keys: tuple[str, ...]) -> int:
    quoted = ", ".join(f'"{column}"' for column in keys)
    return int(conn.execute(
        f'SELECT COALESCE(SUM(n - 1), 0) FROM (SELECT COUNT(*) AS n FROM "{table}" GROUP BY {quoted} HAVING COUNT(*) > 1)'
    ).fetchone()[0])
