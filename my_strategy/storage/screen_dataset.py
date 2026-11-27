#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset-library layer for AI-Agent screen outputs.

The legacy AI-Agent screen root contains many small CSV files. This module
adds a cataloged Parquet dataset layer beside the legacy files so callers can
gradually move to manifest-based reads without breaking existing scripts.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCREEN_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen"

SUMMARY_DATASETS = {"ranking", "buy_candidates", "failures", "reports", "chart_index"}
DETAIL_DATASETS = {"signals", "daily_ledger", "trades"}
VALID_DATASET_TYPES = SUMMARY_DATASETS | DETAIL_DATASETS


@dataclass
class DatasetFile:
    dataset_type: str
    path: str
    relative_path: str
    row_count: int
    size_bytes: int
    sha256: str
    partition_values: dict[str, str] = field(default_factory=dict)


@dataclass
class DatasetManifest:
    dataset_id: str
    dataset_type: str
    run_id: str
    as_of_date: str
    schema_version: int
    storage_format: str
    retention_tier: str
    partition_keys: list[str]
    row_count: int
    file_count: int
    content_hash: str
    created_at: str
    root_path: str
    manifest_path: str
    source_files: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    files: list[DatasetFile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["files"] = [asdict(item) for item in self.files]
        return payload


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_part(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    return text.strip("_") or "unknown"


def infer_as_of_date(summary: dict[str, Any] | None = None, fallback: str = "") -> str:
    summary = summary or {}
    for key in ["as_of_date", "end", "latest_date", "date"]:
        value = str(summary.get(key, "") or "").strip()
        if value:
            return value[:10]
    return fallback[:10] if fallback else datetime.now().strftime("%Y-%m-%d")


def infer_run_id(root: str | Path = DEFAULT_SCREEN_ROOT, summary: dict[str, Any] | None = None) -> str:
    root_path = resolve_path(root)
    summary = summary or {}
    for key in ["run_id", "source_run_id", "sqlite_run_id"]:
        value = str(summary.get(key, "") or "").strip()
        if value:
            return value
    ranking = root_path / "reports" / "ranking.csv"
    if ranking.exists():
        try:
            preview = pd.read_csv(ranking, nrows=1)
            if "run_id" in preview.columns and not preview.empty:
                value = str(preview.iloc[0].get("run_id", "") or "").strip()
                if value:
                    return value
        except Exception:
            pass
    stamp = datetime.fromtimestamp(root_path.stat().st_mtime if root_path.exists() else datetime.now().timestamp())
    return f"screen_{stamp.strftime('%Y%m%d_%H%M%S')}"


def init_dataset_catalog(conn: sqlite3.Connection) -> None:
    """Create dataset catalog tables inside ai_agent_screen.db."""
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_runs (
            dataset_id TEXT PRIMARY KEY,
            dataset_type TEXT NOT NULL,
            run_id TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            schema_version INTEGER NOT NULL,
            storage_format TEXT NOT NULL,
            retention_tier TEXT NOT NULL,
            partition_keys_json TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            file_count INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            root_path TEXT NOT NULL,
            manifest_path TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            source_files_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_files (
            dataset_id TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            run_id TEXT NOT NULL,
            file_path TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            partition_values_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            PRIMARY KEY (dataset_id, file_path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_aliases (
            alias TEXT PRIMARY KEY,
            dataset_id TEXT NOT NULL,
            dataset_type TEXT NOT NULL,
            run_id TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_runs_type_date ON dataset_runs (dataset_type, as_of_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_runs_run ON dataset_runs (run_id, dataset_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_files_run ON dataset_files (run_id, dataset_type)")


class ScreenDatasetLibrary:
    """Manifest and Parquet dataset facade for `ai_agent_screen`."""

    def __init__(self, root: str | Path = DEFAULT_SCREEN_ROOT, db_path: str | Path | None = None) -> None:
        self.root = resolve_path(root)
        self.db_path = resolve_path(db_path) if db_path else self.root / "ai_agent_screen.db"
        self.datasets_root = self.root / "datasets"
        self.manifest_root = self.datasets_root / "manifests"
        self.partition_root = self.datasets_root / "partitions"
        self.current_root = self.root / "current"

    def init(self) -> None:
        self.manifest_root.mkdir(parents=True, exist_ok=True)
        self.partition_root.mkdir(parents=True, exist_ok=True)
        self.current_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            init_dataset_catalog(conn)
            conn.commit()

    def write_table(
        self,
        *,
        dataset_type: str,
        frame: pd.DataFrame,
        run_id: str,
        as_of_date: str,
        dataset_id: str | None = None,
        retention_tier: str = "hot",
        metadata: dict[str, Any] | None = None,
        source_files: Iterable[str | Path] = (),
        partition_keys: list[str] | None = None,
    ) -> DatasetManifest:
        """Write one logical table as cataloged Parquet partitions."""
        if dataset_type not in VALID_DATASET_TYPES:
            raise ValueError(f"unsupported dataset_type: {dataset_type}")
        if frame.empty:
            raise ValueError(f"refuse to write empty dataset: {dataset_type}")
        self.init()

        clean = frame.copy()
        clean = clean.where(pd.notna(clean), None)
        run_id = str(run_id or "").strip()
        if not run_id:
            raise ValueError("run_id is required")
        as_of_date = str(as_of_date or "").strip()[:10]
        if not as_of_date:
            raise ValueError("as_of_date is required")
        dataset_id = dataset_id or self.dataset_id(dataset_type, run_id, as_of_date)
        partition_keys = partition_keys or default_partition_keys(dataset_type)

        base_dir = self._partition_base(dataset_type, run_id, as_of_date)
        files = self._write_partitions(
            dataset_type=dataset_type,
            frame=clean,
            base_dir=base_dir,
            partition_keys=partition_keys,
        )
        content_hash = hash_file_digests(item.sha256 for item in files)
        manifest_path = self.manifest_root / f"{safe_part(dataset_id)}.json"
        manifest = DatasetManifest(
            dataset_id=dataset_id,
            dataset_type=dataset_type,
            run_id=run_id,
            as_of_date=as_of_date,
            schema_version=1,
            storage_format="parquet",
            retention_tier=retention_tier,
            partition_keys=partition_keys,
            row_count=int(len(clean)),
            file_count=len(files),
            content_hash=content_hash,
            created_at=now_iso(),
            root_path=str(self.root),
            manifest_path=str(manifest_path),
            source_files=[str(resolve_path(path)) for path in source_files],
            metadata=metadata or {},
            files=files,
        )
        write_json_atomic(manifest_path, manifest.to_dict())
        self._register_manifest(manifest)
        return manifest

    def list_datasets(
        self,
        dataset_type: str | None = None,
        run_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.init()
        where = []
        params: list[Any] = []
        if dataset_type:
            where.append("dataset_type = ?")
            params.append(dataset_type)
        if run_id:
            where.append("run_id = ?")
            params.append(run_id)
        clause = "WHERE " + " AND ".join(where) if where else ""
        sql = f"""
            SELECT dataset_id, dataset_type, run_id, as_of_date, row_count, file_count,
                   retention_tier, content_hash, manifest_path, created_at
            FROM dataset_runs
            {clause}
            ORDER BY as_of_date DESC, created_at DESC
            LIMIT ?
        """
        params.append(max(1, int(limit)))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def load_manifest(self, dataset_id: str) -> DatasetManifest:
        self.init()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT manifest_path FROM dataset_runs WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
        if row is None:
            manifest_path = self.manifest_root / f"{safe_part(dataset_id)}.json"
        else:
            manifest_path = Path(str(row["manifest_path"]))
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = [DatasetFile(**item) for item in data.get("files", [])]
        data["files"] = files
        return DatasetManifest(**data)

    def read_dataset(self, dataset_id: str) -> pd.DataFrame:
        manifest = self.load_manifest(dataset_id)
        frames = []
        for item in manifest.files:
            path = Path(item.path)
            if not path.exists():
                raise FileNotFoundError(path)
            frames.append(pd.read_parquet(path))
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    def _partition_base(self, dataset_type: str, run_id: str, as_of_date: str) -> Path:
        if dataset_type in SUMMARY_DATASETS:
            return self.partition_root / dataset_type / f"as_of_date={safe_part(as_of_date)}" / f"run_id={safe_part(run_id)}"
        return self.partition_root / dataset_type / f"run_id={safe_part(run_id)}"

    def _write_partitions(
        self,
        *,
        dataset_type: str,
        frame: pd.DataFrame,
        base_dir: Path,
        partition_keys: list[str],
    ) -> list[DatasetFile]:
        files: list[DatasetFile] = []
        if "symbol_bucket" in partition_keys:
            stock_col = stock_column(frame)
            if stock_col is None:
                grouped = [("unknown", frame)]
            else:
                data = frame.copy()
                data["_symbol_bucket"] = data[stock_col].map(symbol_bucket)
                grouped = list(data.groupby("_symbol_bucket", dropna=False))
                grouped = [(str(bucket), part.drop(columns=["_symbol_bucket"])) for bucket, part in grouped]
            for index, (bucket, part) in enumerate(grouped):
                out_path = base_dir / f"symbol_bucket={safe_part(bucket)}" / f"part-{index:05d}.parquet"
                files.append(write_parquet_file(dataset_type, part, out_path, base_root=self.root, partition_values={"symbol_bucket": str(bucket)}))
        else:
            out_path = base_dir / "data.parquet"
            files.append(write_parquet_file(dataset_type, frame, out_path, base_root=self.root, partition_values={}))
        return files

    def _register_manifest(self, manifest: DatasetManifest) -> None:
        payload = manifest.to_dict()
        with sqlite3.connect(self.db_path) as conn:
            init_dataset_catalog(conn)
            conn.execute("DELETE FROM dataset_runs WHERE dataset_id = ?", (manifest.dataset_id,))
            conn.execute("DELETE FROM dataset_files WHERE dataset_id = ?", (manifest.dataset_id,))
            conn.execute(
                """
                INSERT INTO dataset_runs (
                    dataset_id, dataset_type, run_id, as_of_date, schema_version,
                    storage_format, retention_tier, partition_keys_json, row_count,
                    file_count, content_hash, root_path, manifest_path, metadata_json,
                    source_files_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    manifest.dataset_id,
                    manifest.dataset_type,
                    manifest.run_id,
                    manifest.as_of_date,
                    manifest.schema_version,
                    manifest.storage_format,
                    manifest.retention_tier,
                    json.dumps(manifest.partition_keys, ensure_ascii=False),
                    manifest.row_count,
                    manifest.file_count,
                    manifest.content_hash,
                    manifest.root_path,
                    manifest.manifest_path,
                    json.dumps(manifest.metadata, ensure_ascii=False, sort_keys=True, default=str),
                    json.dumps(manifest.source_files, ensure_ascii=False),
                    manifest.created_at,
                ),
            )
            for item in manifest.files:
                conn.execute(
                    """
                    INSERT INTO dataset_files (
                        dataset_id, dataset_type, run_id, file_path, relative_path,
                        row_count, size_bytes, sha256, partition_values_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        manifest.dataset_id,
                        item.dataset_type,
                        manifest.run_id,
                        item.path,
                        item.relative_path,
                        item.row_count,
                        item.size_bytes,
                        item.sha256,
                        json.dumps(item.partition_values, ensure_ascii=False, sort_keys=True),
                        manifest.created_at,
                    ),
                )
            for alias in [f"latest:{manifest.dataset_type}", f"latest:{manifest.dataset_type}:{manifest.run_id}"]:
                conn.execute(
                    """
                    INSERT INTO dataset_aliases (alias, dataset_id, dataset_type, run_id, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(alias) DO UPDATE SET
                        dataset_id=excluded.dataset_id,
                        dataset_type=excluded.dataset_type,
                        run_id=excluded.run_id,
                        updated_at=excluded.updated_at
                    """,
                    (alias, manifest.dataset_id, manifest.dataset_type, manifest.run_id, manifest.created_at),
                )
            conn.commit()
        write_json_atomic(self.current_root / "latest_manifest.json", payload)

    def register_manifest(self, manifest: DatasetManifest) -> None:
        """Register a manifest created by a streaming writer."""
        self._register_manifest(manifest)

    @staticmethod
    def dataset_id(dataset_type: str, run_id: str, as_of_date: str) -> str:
        return f"{dataset_type}-{safe_part(as_of_date)}-{safe_part(run_id)}-v1"


def default_partition_keys(dataset_type: str) -> list[str]:
    if dataset_type in DETAIL_DATASETS:
        return ["run_id", "symbol_bucket"]
    return ["as_of_date", "run_id"]


def stock_column(frame: pd.DataFrame) -> str | None:
    for name in ["stock", "symbol", "code"]:
        if name in frame.columns:
            return name
    return None


def symbol_bucket(value: Any) -> str:
    text = str(value or "").strip().upper()
    digits = re.sub(r"\D+", "", text)
    if len(digits) >= 2:
        return digits[:2]
    return (text[:2] or "unknown").lower()


def write_parquet_file(
    dataset_type: str,
    frame: pd.DataFrame,
    out_path: Path,
    *,
    base_root: Path,
    partition_values: dict[str, str],
) -> DatasetFile:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    frame.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)
    digest = sha256_file(out_path)
    return DatasetFile(
        dataset_type=dataset_type,
        path=str(out_path),
        relative_path=str(out_path.relative_to(base_root)),
        row_count=int(len(frame)),
        size_bytes=out_path.stat().st_size,
        sha256=digest,
        partition_values=partition_values,
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_file_digests(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
    return digest.hexdigest()
