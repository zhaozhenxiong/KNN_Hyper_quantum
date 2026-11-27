#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SQLite-backed artifact storage for DB-first runtime assets."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from contextlib import closing
import hashlib
import json
from pathlib import Path
import pickle
import sqlite3
from typing import Any

from my_strategy.storage.db_paths import processed_db_path
from my_strategy.storage.storage_contract import load_storage_contract
from my_strategy.storage.warehouse_schema import init_warehouse


@dataclass
class ArtifactRecord:
    namespace: str
    kind: str
    name: str
    payload: bytes
    metadata: dict[str, Any]


class ArtifactStore:
    def __init__(self, db_path: str | Path | None = None, *, content_root: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else processed_db_path()
        self.content_root = Path(content_root) if content_root else _default_content_root(self.db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        init_warehouse(conn)
        ensure_artifact_schema(conn)
        return conn

    def put_bytes(self, namespace: str, kind: str, name: str, payload: bytes, metadata: dict[str, Any] | None = None) -> str:
        artifact_id = stable_artifact_id(namespace, kind, name)
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        sha = hashlib.sha256(payload).hexdigest()
        content_path = _write_content_addressed(self.content_root, payload, sha, kind)
        with closing(self.connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO artifact_blobs (
                        artifact_id, namespace, kind, name, payload, size_bytes, sha256,
                        metadata_json, created_at, updated_at, storage_backend, content_path, externalized_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'content_addressed_file', ?, ?)
                    ON CONFLICT(artifact_id) DO UPDATE SET
                        payload=excluded.payload,
                        size_bytes=excluded.size_bytes,
                        sha256=excluded.sha256,
                        metadata_json=excluded.metadata_json,
                        storage_backend=excluded.storage_backend,
                        content_path=excluded.content_path,
                        externalized_at=excluded.externalized_at,
                        updated_at=excluded.updated_at
                    """,
                    (
                        artifact_id, namespace, kind, name, sqlite3.Binary(b""), len(payload), sha,
                        json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str),
                        now, now, str(content_path.relative_to(self.content_root)), now,
                    ),
                )
        return artifact_id

    def get_bytes(self, namespace: str, kind: str, name: str) -> tuple[bytes, dict[str, Any]] | None:
        artifact_id = stable_artifact_id(namespace, kind, name)
        if not self.db_path.exists():
            return None
        with closing(self.connect()) as conn:
            row = conn.execute(
                "SELECT payload, metadata_json, sha256, size_bytes, storage_backend, content_path "
                "FROM artifact_blobs WHERE artifact_id = ?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        payload = _read_payload(row, self.content_root)
        return payload, json.loads(row["metadata_json"] or "{}")

    def list_records(self, namespace: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        if not self.db_path.exists():
            return []
        clauses: list[str] = []
        params: list[Any] = []
        if namespace:
            clauses.append("namespace = ?")
            params.append(namespace)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        sql = """
            SELECT artifact_id, namespace, kind, name, size_bytes, sha256,
                   metadata_json, created_at, updated_at, storage_backend, content_path
            FROM artifact_blobs
        """
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC, created_at DESC"
        with closing(self.connect()) as conn:
            rows = conn.execute(sql, params).fetchall()
        return [
            {
                "artifact_id": str(row["artifact_id"]),
                "namespace": str(row["namespace"]),
                "kind": str(row["kind"]),
                "name": str(row["name"]),
                "size_bytes": int(row["size_bytes"]),
                "sha256": str(row["sha256"]),
                "metadata": json.loads(row["metadata_json"] or "{}"),
                "created_at": str(row["created_at"]),
                "updated_at": str(row["updated_at"]),
                "storage_backend": str(row["storage_backend"] or "sqlite_blob"),
                "content_path": str(row["content_path"] or ""),
            }
            for row in rows
        ]

    def externalize_legacy_records(self, *, limit: int | None = None) -> dict[str, int]:
        """Move legacy BLOB payloads to verified files and replace them with empty BLOBs.

        The caller must explicitly opt in.  Metadata stays in SQLite and every
        file write is verified before the corresponding database row changes.
        """
        migrated = skipped = 0
        with closing(self.connect()) as conn:
            with conn:
                rows = conn.execute(
                    "SELECT artifact_id, kind, payload, size_bytes, sha256 FROM artifact_blobs "
                    "WHERE COALESCE(storage_backend, 'sqlite_blob') <> 'content_addressed_file' ORDER BY artifact_id"
                )
                for row in rows:
                    if limit is not None and migrated >= limit:
                        break
                    payload = bytes(row["payload"])
                    if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
                        raise ValueError(f"Legacy artifact checksum mismatch: {row['artifact_id']}")
                    target = _write_content_addressed(self.content_root, payload, str(row["sha256"]), str(row["kind"]))
                    conn.execute(
                        "UPDATE artifact_blobs SET payload=?, storage_backend='content_addressed_file', content_path=?, "
                        "externalized_at=? WHERE artifact_id=?",
                        (
                            sqlite3.Binary(b""), str(target.relative_to(self.content_root)),
                            datetime.now().astimezone().isoformat(timespec="seconds"), str(row["artifact_id"]),
                        ),
                    )
                    migrated += 1
                skipped = int(conn.execute(
                    "SELECT COUNT(*) FROM artifact_blobs WHERE COALESCE(storage_backend, 'sqlite_blob') <> 'content_addressed_file'"
                ).fetchone()[0])
        return {"migrated": migrated, "remaining_legacy": skipped}

    def put_pickle(self, namespace: str, kind: str, name: str, value: Any, metadata: dict[str, Any] | None = None) -> str:
        return self.put_bytes(namespace, kind, name, pickle.dumps(value), metadata=metadata)

    def get_pickle(self, namespace: str, kind: str, name: str) -> tuple[Any, dict[str, Any]] | None:
        item = self.get_bytes(namespace, kind, name)
        if item is None:
            return None
        payload, metadata = item
        return pickle.loads(payload), metadata


def ensure_artifact_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_blobs (
            artifact_id TEXT PRIMARY KEY,
            namespace TEXT NOT NULL,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            payload BLOB NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(namespace, kind, name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_artifact_blobs_namespace ON artifact_blobs(namespace, kind, name)")
    existing = {str(row[1]) for row in conn.execute('PRAGMA table_info("artifact_blobs")')}
    additions = {
        "storage_backend": "TEXT NOT NULL DEFAULT 'sqlite_blob'",
        "content_path": "TEXT",
        "externalized_at": "TEXT",
    }
    for name, definition in additions.items():
        if name not in existing:
            conn.execute(f'ALTER TABLE artifact_blobs ADD COLUMN "{name}" {definition}')


def _default_content_root(db_path: Path) -> Path:
    """Use the immutable artifact area for the canonical warehouse, locally otherwise."""
    try:
        contract = load_storage_contract()
        if db_path.resolve() == contract.processed_db.resolve():
            return contract.location("artifact_runs").path / "_content_store"
    except (FileNotFoundError, ValueError):
        pass
    return db_path.parent / "artifact_content"


def _extension(kind: str) -> str:
    return {"csv": ".csv", "json": ".json", "png": ".png", "joblib": ".joblib", "model": ".bin"}.get(kind, ".bin")


def _content_path(root: Path, sha256: str, kind: str) -> Path:
    if len(sha256) != 64 or any(char not in "0123456789abcdef" for char in sha256.lower()):
        raise ValueError(f"Invalid artifact SHA-256: {sha256}")
    return root / "sha256" / sha256[:2] / f"{sha256}{_extension(kind)}"


def _write_content_addressed(root: Path, payload: bytes, sha256: str, kind: str) -> Path:
    path = _content_path(root, sha256, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"Content-addressed path hash collision: {path}")
        return path
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"Artifact write checksum mismatch: {path}")
    temporary.replace(path)
    return path


def _read_payload(row: sqlite3.Row, content_root: Path) -> bytes:
    backend = str(row["storage_backend"] or "sqlite_blob")
    if backend != "content_addressed_file":
        return bytes(row["payload"])
    relative = str(row["content_path"] or "")
    path = Path(relative)
    if not path.is_absolute():
        path = content_root / path
    if not path.exists():
        raise FileNotFoundError(f"External artifact missing: {path}")
    payload = path.read_bytes()
    if len(payload) != int(row["size_bytes"]) or hashlib.sha256(payload).hexdigest() != str(row["sha256"]):
        raise ValueError(f"External artifact validation failed: {path}")
    return payload


def stable_artifact_id(namespace: str, kind: str, name: str) -> str:
    raw = json.dumps({"namespace": namespace, "kind": kind, "name": name}, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]
