"""Guards that keep run-scoped results idempotent and referenceable.

The guard is deliberately small and dependency-free so writers can apply it
before SQLite persistence.  It rejects ambiguous writes instead of silently
creating a second daily ledger or signal record for the same run and stock.
"""

from __future__ import annotations

from datetime import datetime
import json
import sqlite3
from typing import Iterable

import pandas as pd


def require_run_id(run_id: str | None, *, operation: str) -> str:
    """Return a normalized run id, rejecting untraceable persistence."""
    value = str(run_id or "").strip()
    if not value:
        raise ValueError(f"{operation} requires a non-empty run_id")
    return value


def assert_unique_keys(frame: pd.DataFrame | None, keys: Iterable[str], *, table: str) -> None:
    """Reject duplicate logical records within one persistence batch."""
    if frame is None or frame.empty:
        return
    key_list = list(keys)
    missing = [key for key in key_list if key not in frame.columns]
    if missing:
        raise ValueError(f"{table} is missing logical key columns: {missing}")
    duplicated = frame.duplicated(subset=key_list, keep=False)
    if bool(duplicated.any()):
        examples = frame.loc[duplicated, key_list].head(5).to_dict("records")
        raise ValueError(f"{table} contains duplicate logical keys {key_list}: {examples}")


def ensure_output_reference_schema(conn: sqlite3.Connection) -> None:
    """Create the compact authoritative-output reference registry."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_output_references (
            run_id TEXT NOT NULL,
            scope_key TEXT NOT NULL DEFAULT '',
            output_type TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            authority TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, scope_key, output_type)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_output_references_artifact "
        "ON run_output_references (artifact_id)"
    )


def upsert_output_reference(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    scope_key: str,
    output_type: str,
    artifact_id: str,
    authority: str,
    metadata: dict[str, object] | None = None,
) -> None:
    """Record the sole authoritative artifact for a run-scoped output."""
    ensure_output_reference_schema(conn)
    conn.execute(
        """
        INSERT INTO run_output_references (
            run_id, scope_key, output_type, artifact_id, authority, metadata_json, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(run_id, scope_key, output_type) DO UPDATE SET
            artifact_id=excluded.artifact_id,
            authority=excluded.authority,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        """,
        (
            require_run_id(run_id, operation="authoritative output reference"),
            str(scope_key or ""),
            str(output_type),
            str(artifact_id),
            str(authority),
            json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str),
            datetime.now().astimezone().isoformat(timespec="seconds"),
        ),
    )
