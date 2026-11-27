"""Copy-only, per-run storage_v2 migration and read-fallback primitives.

``storage_v2`` is an isolated target.  This module never mutates a source
database.  Its writer only creates/updates the v2 target, while its adapter is
legacy-first by default until an explicit observation-period decision enables
warehouse-first reads.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from typing import Any
from urllib.parse import quote

from my_strategy.storage.storage_contract import StorageContract


DEFAULT_RUN_TABLES = ("backtest_reports", "backtest_daily_ledger", "backtest_trades", "agent_signal_records")


def _uri(path: Path, *, read_only: bool) -> str:
    suffix = "?mode=ro" if read_only else ""
    return f"file:{quote(path.resolve().as_posix(), safe='/:')}{suffix}"


def connect_source(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(_uri(path, read_only=True), uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    connection.execute("PRAGMA busy_timeout=5000")
    return connection


def connect_target(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    return connection


def target_db_path(contract: StorageContract) -> Path:
    return contract.stage_path("warehouse_stage")


def _table_sql(source: sqlite3.Connection, table: str) -> str:
    row = source.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if row is None or not row[0]:
        raise KeyError(f"Source table missing: {table}")
    return str(row[0])


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    return [str(row[1]) for row in connection.execute(f'PRAGMA table_info("{table}")')]


def _ensure_table(source: sqlite3.Connection, target: sqlite3.Connection, table: str) -> list[str]:
    existing = target.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if existing is None:
        create_sql = _table_sql(source, table)
        create_sql = re.sub(r"^CREATE\s+TABLE\s+", "CREATE TABLE IF NOT EXISTS ", create_sql, count=1, flags=re.IGNORECASE)
        target.execute(create_sql)
    return _table_columns(source, table)


def _ensure_registry(target: sqlite3.Connection) -> None:
    target.execute(
        """
        CREATE TABLE IF NOT EXISTS storage_v2_partition_registry (
            migration_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            source_db TEXT NOT NULL,
            target_db TEXT NOT NULL,
            tables_json TEXT NOT NULL,
            verification_json TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            verified_at TEXT,
            PRIMARY KEY (migration_id, run_id)
        )
        """
    )
    target.execute(
        "CREATE INDEX IF NOT EXISTS idx_storage_v2_partition_run ON storage_v2_partition_registry(run_id, status)"
    )


def _key_columns(columns: list[str]) -> list[str]:
    available = set(columns)
    for candidate in (("run_id", "stock", "date"), ("run_id", "stock", "row_number"), ("run_id", "stock"), ("run_id",)):
        if all(item in available for item in candidate):
            return list(candidate)
    return []


def _row_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "length": len(value)}
    return value


def _metric_rows(connection: sqlite3.Connection, table: str, run_id: str) -> dict[str, Any]:
    columns = _table_columns(connection, table)
    key_columns = _key_columns(columns)
    order_columns = key_columns or columns
    query = f'SELECT * FROM "{table}" WHERE run_id=? ORDER BY ' + ", ".join(f'"{name}"' for name in order_columns)
    hasher = hashlib.sha256()
    key_hasher = hashlib.sha256()
    rows = 0
    keys: set[tuple[Any, ...]] = set()
    dates: list[str] = []
    for row in connection.execute(query, (run_id,)):
        values = {name: _row_value(row[name]) for name in columns}
        hasher.update(json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        if key_columns:
            key = tuple(row[name] for name in key_columns)
            keys.add(key)
            key_hasher.update(json.dumps(key, ensure_ascii=False, default=str).encode("utf-8"))
        if "date" in columns and row["date"] is not None:
            dates.append(str(row["date"]))
        rows += 1
    return {
        "row_count": rows,
        "key_columns": key_columns,
        "logical_key_count": len(keys) if key_columns else rows,
        "duplicate_key_count": (rows - len(keys)) if key_columns else 0,
        "row_sha256": hasher.hexdigest(),
        "logical_key_sha256": key_hasher.hexdigest() if key_columns else "",
        "date_start": min(dates) if dates else "",
        "date_end": max(dates) if dates else "",
    }


def _performance_summary(connection: sqlite3.Connection, run_id: str) -> dict[str, Any]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "backtest_reports" not in tables:
        return {"report_count": 0, "summary_sha256": hashlib.sha256(b"").hexdigest()}
    columns = _table_columns(connection, "backtest_reports")
    if "report_json" not in columns:
        return {"report_count": 0, "summary_sha256": hashlib.sha256(b"").hexdigest()}
    extracted: list[dict[str, Any]] = []
    for row in connection.execute('SELECT stock, report_json FROM "backtest_reports" WHERE run_id=? ORDER BY stock', (run_id,)):
        try:
            report = json.loads(str(row["report_json"] or "{}"))
        except json.JSONDecodeError:
            report = {}
        metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else report
        extracted.append(
            {
                "stock": str(row["stock"] or ""),
                "total_return": metrics.get("total_return"),
                "max_drawdown": metrics.get("max_drawdown"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "n_trades": metrics.get("n_trades"),
                "final_equity": metrics.get("final_equity"),
            }
        )
    encoded = json.dumps(extracted, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return {"report_count": len(extracted), "summary_sha256": hashlib.sha256(encoded).hexdigest(), "records": extracted}


class StorageV2Migrator:
    """Exports one complete run to the isolated v2 warehouse and verifies it."""

    def __init__(self, contract: StorageContract):
        self.contract = contract
        self.source_db = contract.processed_db
        self.target_db = target_db_path(contract)

    def dry_run(self, run_id: str, tables: Iterable[str] = DEFAULT_RUN_TABLES) -> dict[str, Any]:
        source = connect_source(self.source_db)
        try:
            stats = {table: _metric_rows(source, table, run_id) for table in tables}
            performance = _performance_summary(source, run_id)
        finally:
            source.close()
        return {
            "migration_id": "",
            "run_id": run_id,
            "mode": "dry_run",
            "source_db": str(self.source_db),
            "target_db": str(self.target_db),
            "tables": stats,
            "performance_summary": performance,
            "source_databases_modified": False,
            "runtime_mode_changed": False,
        }

    def copy_run(self, migration_id: str, run_id: str, tables: Iterable[str] = DEFAULT_RUN_TABLES) -> dict[str, Any]:
        selected = tuple(tables)
        source = connect_source(self.source_db)
        target = connect_target(self.target_db)
        try:
            _ensure_registry(target)
            target.execute("BEGIN")
            for table in selected:
                columns = _ensure_table(source, target, table)
                if "run_id" not in columns:
                    raise ValueError(f"Table is not run-partitioned: {table}")
                target.execute(f'DELETE FROM "{table}" WHERE run_id=?', (run_id,))
                placeholders = ", ".join("?" for _ in columns)
                insert = f'INSERT INTO "{table}" (' + ", ".join(f'"{name}"' for name in columns) + f") VALUES ({placeholders})"
                cursor = source.execute(f'SELECT * FROM "{table}" WHERE run_id=?', (run_id,))
                batch: list[tuple[Any, ...]] = []
                for row in cursor:
                    batch.append(tuple(row[name] for name in columns))
                    if len(batch) >= 500:
                        target.executemany(insert, batch)
                        batch.clear()
                if batch:
                    target.executemany(insert, batch)
            source_metrics = {table: _metric_rows(source, table, run_id) for table in selected}
            target_metrics = {table: _metric_rows(target, table, run_id) for table in selected}
            source_performance = _performance_summary(source, run_id)
            target_performance = _performance_summary(target, run_id)
            verified = source_metrics == target_metrics and source_performance == target_performance
            verification = {
                "valid": verified,
                "tables": {table: {"source": source_metrics[table], "target": target_metrics[table]} for table in selected},
                "performance": {"source": source_performance, "target": target_performance},
            }
            if not verified:
                target.rollback()
                raise RuntimeError("storage_v2 partition verification failed; target transaction rolled back")
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            target.execute(
                """
                INSERT INTO storage_v2_partition_registry (
                    migration_id, run_id, source_db, target_db, tables_json, verification_json, status, created_at, verified_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'verified', ?, ?)
                ON CONFLICT(migration_id, run_id) DO UPDATE SET
                    tables_json=excluded.tables_json, verification_json=excluded.verification_json,
                    status=excluded.status, verified_at=excluded.verified_at
                """,
                (
                    migration_id,
                    run_id,
                    str(self.source_db),
                    str(self.target_db),
                    json.dumps(list(selected), ensure_ascii=False),
                    json.dumps(verification, ensure_ascii=False, sort_keys=True, default=str),
                    now,
                    now,
                ),
            )
            target.commit()
        finally:
            source.close()
            target.close()
        return {
            "migration_id": migration_id,
            "run_id": run_id,
            "mode": "copy_only_execute",
            "source_db": str(self.source_db),
            "target_db": str(self.target_db),
            "verification": verification,
            "source_databases_modified": False,
            "runtime_mode_changed": False,
        }


class StorageV2ReadAdapter:
    """Warehouse-first optional reader with legacy fallback and no implicit cutover."""

    def __init__(self, contract: StorageContract):
        self.contract = contract
        self.legacy_db = contract.processed_db
        self.v2_db = target_db_path(contract)

    @staticmethod
    def read_mode() -> str:
        value = os.environ.get("KHQUANT_STORAGE_V2_READ_MODE", "legacy_only").strip().lower()
        return value if value in {"legacy_only", "warehouse_first"} else "legacy_only"

    @staticmethod
    def dual_write_mode() -> str:
        value = os.environ.get("KHQUANT_STORAGE_V2_DUAL_WRITE", "off").strip().lower()
        return value if value in {"off", "shadow"} else "off"

    def shadow_write_run(self, migration_id: str, run_id: str, tables: Iterable[str] = DEFAULT_RUN_TABLES) -> dict[str, Any]:
        """Mirror a completed legacy run only when the explicit shadow flag is on.

        This hook is intentionally not wired into production entrypoints yet.
        Operators can enable it for a bounded observation run with
        ``KHQUANT_STORAGE_V2_DUAL_WRITE=shadow``; it writes only storage_v2 and
        returns the same exhaustive verification used by manual migration.
        """
        if self.dual_write_mode() != "shadow":
            return {"status": "disabled", "mode": "off", "source_databases_modified": False}
        result = StorageV2Migrator(self.contract).copy_run(migration_id, run_id, tables)
        return {"status": "verified_shadow_copy", "mode": "shadow", **result}

    def read_run_rows(self, table: str, run_id: str) -> tuple[list[dict[str, Any]], str]:
        for path, label in self._read_candidates(run_id):
            if not path.exists():
                continue
            connection = connect_source(path)
            try:
                table_exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
                if table_exists is None:
                    continue
                rows = connection.execute(f'SELECT * FROM "{table}" WHERE run_id=?', (run_id,)).fetchall()
                if rows or label == "legacy":
                    return [dict(row) for row in rows], label
            finally:
                connection.close()
        return [], "legacy"

    def _read_candidates(self, run_id: str) -> list[tuple[Path, str]]:
        if self.read_mode() != "warehouse_first" or not self.v2_db.exists():
            return [(self.legacy_db, "legacy")]
        connection = connect_source(self.v2_db)
        try:
            row = connection.execute(
                "SELECT 1 FROM storage_v2_partition_registry WHERE run_id=? AND status='verified' LIMIT 1", (run_id,)
            ).fetchone()
        except sqlite3.DatabaseError:
            row = None
        finally:
            connection.close()
        return [(self.v2_db, "storage_v2"), (self.legacy_db, "legacy")] if row else [(self.legacy_db, "legacy")]
