"""Unified model registry backed by SQLite with a JSON compatibility snapshot."""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from typing import Any

from my_strategy.core.paths import PROCESSED_DATA_ROOT, PROJECT_ROOT
from my_strategy.core.run_context import current_git_commit, stable_hash
from my_strategy.model_registry.schema import ModelEntry, ModelStage
from my_strategy.storage.persistence_guard import require_run_id

DEFAULT_REGISTRY_PATH = Path(os.environ.get(
    "KHQUANT_MODEL_REGISTRY", str(PROCESSED_DATA_ROOT / "model_registry" / "unified_model_registry.sqlite")
))
DEFAULT_REGISTRY_SNAPSHOT_PATH = Path(os.environ.get(
    "KHQUANT_MODEL_REGISTRY_SNAPSHOT", str(PROCESSED_DATA_ROOT / "model_registry" / "unified_model_registry.json")
))


class ModelRegistry:
    def __init__(self, path: str | Path = DEFAULT_REGISTRY_PATH, *, snapshot_path: str | Path | None = DEFAULT_REGISTRY_SNAPSHOT_PATH):
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        if self.path.suffix.lower() == ".json":
            self.snapshot_path = self.path
            self.path = self.path.with_suffix(".sqlite")
        else:
            self.snapshot_path = Path(snapshot_path) if snapshot_path else None
            if self.snapshot_path is not None and not self.snapshot_path.is_absolute():
                self.snapshot_path = PROJECT_ROOT / self.snapshot_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._migrate_legacy_json()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                create table if not exists models (
                    model_id text primary key,
                    name text not null,
                    kind text not null,
                    stage text not null,
                    run_id text not null default '',
                    model_path text not null default '',
                    report_path text not null default '',
                    metrics_json text not null default '{}',
                    feature_columns_json text not null default '[]',
                    data_range_json text not null default '{}',
                    params_json text not null default '{}',
                    git_commit text not null default '',
                    created_at text not null,
                    updated_at text not null
                )
                """
            )
            conn.execute("create index if not exists idx_models_kind_stage_updated on models(kind, stage, updated_at)")

    def _migrate_legacy_json(self) -> None:
        if self.snapshot_path is None or not self.snapshot_path.exists():
            return
        if self.entries():
            return
        try:
            data = json.loads(self.snapshot_path.read_text(encoding="utf-8"))
            for row in data.get("models", []):
                self.upsert(ModelEntry.from_dict(row), write_snapshot=False)
        except Exception:
            return

    def _row_to_entry(self, row: sqlite3.Row) -> ModelEntry:
        return ModelEntry(
            model_id=str(row["model_id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            stage=str(row["stage"]),
            run_id=str(row["run_id"]),
            model_path=str(row["model_path"]),
            report_path=str(row["report_path"]),
            metrics=json.loads(row["metrics_json"] or "{}"),
            feature_columns=json.loads(row["feature_columns_json"] or "[]"),
            data_range=json.loads(row["data_range_json"] or "{}"),
            params=json.loads(row["params_json"] or "{}"),
            git_commit=str(row["git_commit"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _snapshot_payload(self) -> dict[str, Any]:
        return {
            "version": 2,
            "storage": "sqlite",
            "sqlite_path": str(self.path),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "models": [entry.to_dict() for entry in self.entries()],
        }

    def export_snapshot(self) -> Path | None:
        if self.snapshot_path is None:
            return None
        self.snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.snapshot_path.with_name(f"{self.snapshot_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self._snapshot_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.snapshot_path)
        return self.snapshot_path

    def load(self) -> dict[str, Any]:
        return self._snapshot_payload()

    def save(self, data: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute("delete from models")
        for row in data.get("models", []):
            self.upsert(ModelEntry.from_dict(row), write_snapshot=False)
        self.export_snapshot()

    def entries(self, *, kind: str | None = None, stage: ModelStage | None = None) -> list[ModelEntry]:
        sql = "select * from models"
        params: list[Any] = []
        clauses = []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if stage:
            clauses.append("stage = ?")
            params.append(stage)
        if clauses:
            sql += " where " + " and ".join(clauses)
        sql += " order by updated_at desc, created_at desc"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_entry(row) for row in rows]

    def upsert(self, entry: ModelEntry, *, write_snapshot: bool = True) -> ModelEntry:
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        if not entry.created_at:
            entry.created_at = now
        entry.updated_at = now
        with self._connect() as conn:
            conn.execute(
                """
                insert into models (
                    model_id, name, kind, stage, run_id, model_path, report_path,
                    metrics_json, feature_columns_json, data_range_json, params_json,
                    git_commit, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                on conflict(model_id) do update set
                    name=excluded.name,
                    kind=excluded.kind,
                    stage=excluded.stage,
                    run_id=excluded.run_id,
                    model_path=excluded.model_path,
                    report_path=excluded.report_path,
                    metrics_json=excluded.metrics_json,
                    feature_columns_json=excluded.feature_columns_json,
                    data_range_json=excluded.data_range_json,
                    params_json=excluded.params_json,
                    git_commit=excluded.git_commit,
                    updated_at=excluded.updated_at
                """,
                (
                    entry.model_id,
                    entry.name,
                    entry.kind,
                    entry.stage,
                    entry.run_id,
                    entry.model_path,
                    entry.report_path,
                    json.dumps(entry.metrics, ensure_ascii=False, sort_keys=True, default=str),
                    json.dumps(entry.feature_columns, ensure_ascii=False, default=str),
                    json.dumps(entry.data_range, ensure_ascii=False, sort_keys=True, default=str),
                    json.dumps(entry.params, ensure_ascii=False, sort_keys=True, default=str),
                    entry.git_commit,
                    entry.created_at,
                    entry.updated_at,
                ),
            )
        if write_snapshot:
            self.export_snapshot()
        return entry

    def register(
        self,
        *,
        name: str,
        kind: str,
        stage: ModelStage = "candidate",
        run_id: str = "",
        model_path: str | Path = "",
        report_path: str | Path = "",
        metrics: dict[str, Any] | None = None,
        feature_columns: list[str] | None = None,
        data_range: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        git_commit: str | None = None,
    ) -> ModelEntry:
        run_id = require_run_id(run_id, operation="model registry registration")
        model_id = stable_hash({"kind": kind, "name": name, "run_id": run_id, "model_path": str(model_path)})[:16]
        entry = ModelEntry(
            model_id=model_id,
            name=name,
            kind=kind,
            stage=stage,
            run_id=run_id,
            model_path=str(model_path),
            report_path=str(report_path),
            metrics=metrics or {},
            feature_columns=feature_columns or [],
            data_range=data_range or {},
            params=params or {},
            git_commit=git_commit or current_git_commit(),
        )
        return self.upsert(entry)

    def promote(self, model_id: str, stage: ModelStage = "champion", *, retire_existing: bool = True) -> ModelEntry:
        target = None
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        with self._connect() as conn:
            row = conn.execute("select * from models where model_id = ?", (model_id,)).fetchone()
            if row is not None:
                target = self._row_to_entry(row)
            if target is None:
                raise KeyError(f"model_id not found: {model_id}")
            if retire_existing and stage == "champion":
                conn.execute(
                    "update models set stage = 'retired', updated_at = ? where kind = ? and stage = 'champion' and model_id <> ?",
                    (now, target.kind, model_id),
                )
            conn.execute("update models set stage = ?, updated_at = ? where model_id = ?", (stage, now, model_id))
        self.export_snapshot()
        promoted = self.entries(kind=target.kind, stage=stage)
        for entry in promoted:
            if entry.model_id == model_id:
                return entry
        raise KeyError(f"model_id not found after promotion: {model_id}")

    def champion(self, kind: str) -> ModelEntry | None:
        champions = self.entries(kind=kind, stage="champion")
        return champions[0] if champions else None

    def find_by_name(self, kind: str, name: str, *, stage: ModelStage | None = None) -> ModelEntry | None:
        """Find the newest model with an exact name without changing lifecycle state."""

        matches = [entry for entry in self.entries(kind=kind, stage=stage) if entry.name == name]
        return matches[0] if matches else None


def default_registry() -> ModelRegistry:
    return ModelRegistry()


def register_candidate(**kwargs: Any) -> ModelEntry:
    kwargs.setdefault("stage", "candidate")
    return default_registry().register(**kwargs)


def promote_model(model_id: str, stage: ModelStage = "champion") -> ModelEntry:
    return default_registry().promote(model_id, stage=stage)
