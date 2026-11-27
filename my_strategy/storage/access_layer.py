#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified access layer for KHQuant DB-first storage.

Business entrypoints should depend on this facade instead of constructing raw,
processed, or artifact repositories directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from my_strategy.storage.artifact_store import ArtifactStore
from my_strategy.storage.db_paths import (
    processed_db_path,
    raw_db_path,
    read_db_first,
    storage_mode,
    write_db,
    write_files,
)
from my_strategy.storage.processed_repository import ProcessedRepository
from my_strategy.storage.persistence_guard import require_run_id, upsert_output_reference
from my_strategy.storage.raw_repository import RawMarketRepository


@dataclass(frozen=True)
class StorageSettings:
    mode: str
    raw_db: Path
    processed_db: Path
    read_db_first: bool
    write_db: bool
    write_files: bool


class DataAccessLayer:
    """Single facade for project storage reads and writes."""

    def __init__(self, raw_db: str | Path | None = None, processed_db: str | Path | None = None):
        self.raw = RawMarketRepository(raw_db)
        self.processed = ProcessedRepository(processed_db)
        self.artifacts = ArtifactStore(self.processed.db_path)

    @property
    def settings(self) -> StorageSettings:
        return StorageSettings(
            mode=storage_mode(),
            raw_db=raw_db_path(),
            processed_db=processed_db_path(),
            read_db_first=read_db_first(),
            write_db=write_db(),
            write_files=write_files(),
        )

    def list_stocks(self) -> list[str]:
        return self.raw.list_stocks()

    def has_stock(self, code: str) -> bool:
        return self.raw.has_stock(code)

    def read_stock_daily(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        return self.raw.read_stock_daily(code, start_date, end_date)

    def read_factor_source(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        """Read daily bars for factor, training, and backtest pipelines."""
        df = self.read_stock_daily(code, start_date, end_date)
        if not df.empty:
            df.attrs["khquant_data_source"] = "raw_sqlite"
            df.attrs["khquant_storage_mode"] = self.settings.mode
            df.attrs["khquant_raw_db"] = str(self.raw.db_path)
        return df

    def read_factor_source_batch(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Read daily bars for a stock universe in one warehouse pass."""
        df = self.raw.read_stock_daily_batch(stocks, start_date, end_date)
        if not df.empty:
            df.attrs["khquant_data_source"] = "raw_sqlite_batch"
            df.attrs["khquant_storage_mode"] = self.settings.mode
            df.attrs["khquant_raw_db"] = str(self.raw.db_path)
        return df

    def raw_daily_coverage(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        return self.raw.stock_daily_coverage(stocks, start_date, end_date)

    def factor_snapshot_coverage(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        factor_version: str,
    ) -> dict[str, dict[str, Any]]:
        return self.processed.factor_snapshot_coverage(stocks, start_date, end_date, factor_version)

    def read_factor_snapshots(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        factor_version: str,
    ) -> pd.DataFrame:
        df = self.processed.read_factor_snapshots(stocks, start_date, end_date, factor_version)
        if not df.empty:
            df.attrs["khquant_data_source"] = "processed_factor_snapshot"
            df.attrs["khquant_storage_mode"] = self.settings.mode
            df.attrs["khquant_processed_db"] = str(self.processed.db_path)
            df.attrs["khquant_factor_version"] = factor_version
        return df

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
        return self.processed.replace_factor_snapshots(
            frame,
            factor_version=factor_version,
            source_start=source_start,
            source_end=source_end,
            volume_window=volume_window,
            summary=summary,
        )

    def model_feature_snapshot_coverage(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        feature_version: str,
    ) -> dict[str, dict[str, Any]]:
        return self.processed.model_feature_snapshot_coverage(stocks, start_date, end_date, feature_version)

    def read_model_feature_snapshots(
        self,
        stocks: list[str],
        start_date: str,
        end_date: str,
        feature_version: str,
        *,
        include_labels: bool = False,
    ) -> pd.DataFrame:
        df = self.processed.read_model_feature_snapshots(
            stocks, start_date, end_date, feature_version, include_labels=include_labels
        )
        if not df.empty:
            df.attrs["khquant_data_source"] = "processed_model_feature_snapshot"
            df.attrs["khquant_storage_mode"] = self.settings.mode
            df.attrs["khquant_processed_db"] = str(self.processed.db_path)
            df.attrs["khquant_feature_version"] = feature_version
        return df

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
        return self.processed.replace_model_feature_snapshots(
            frame,
            feature_version=feature_version,
            factor_version=factor_version,
            source_start=source_start,
            source_end=source_end,
            summary=summary,
        )

    def save_stock_daily(self, code: str, df: pd.DataFrame, *, source: str = "unknown") -> int:
        return self.raw.upsert_stock_daily(code, df, source=source)

    def read_stock_basic(self) -> pd.DataFrame:
        return self.raw.read_stock_basic()

    def save_stock_basic(self, df: pd.DataFrame) -> int:
        return self.raw.save_stock_basic(df)

    def update_source_status(self, row: dict[str, Any]) -> None:
        self.raw.update_status(row)

    def append_update_log(self, row: dict[str, Any]) -> None:
        self.raw.append_log(row)

    def record_data_update(
        self,
        *,
        update_id: str,
        started_at: str,
        finished_at: str | None,
        mode: str,
        stocks_requested: str,
        end_date: str,
        report: pd.DataFrame,
    ) -> dict[str, Any]:
        finished = finished_at or datetime.now().astimezone().isoformat(timespec="seconds")
        status = report.get("status", pd.Series(dtype=str)).astype(str).str.lower() if not report.empty else pd.Series(dtype=str)
        success = int((status == "success").sum())
        failed = int((status == "failed").sum())
        skipped = int((status == "skipped").sum())
        settings = self.settings
        payload = {
            "update_id": update_id,
            "started_at": started_at,
            "finished_at": finished,
            "mode": mode,
            "stocks_requested": stocks_requested,
            "end_date": end_date,
            "storage_mode": settings.mode,
            "raw_db_path": str(settings.raw_db),
            "total": int(len(report)),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "report_json": json.dumps(report.to_dict("records"), ensure_ascii=False, sort_keys=True, default=str),
        }
        with self.raw.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_update_runs (
                    update_id, started_at, finished_at, mode, stocks_requested, end_date,
                    storage_mode, raw_db_path, total, success, failed, skipped, report_json
                ) VALUES (
                    :update_id, :started_at, :finished_at, :mode, :stocks_requested, :end_date,
                    :storage_mode, :raw_db_path, :total, :success, :failed, :skipped, :report_json
                )
                ON CONFLICT(update_id) DO UPDATE SET
                    finished_at=excluded.finished_at,
                    total=excluded.total,
                    success=excluded.success,
                    failed=excluded.failed,
                    skipped=excluded.skipped,
                    report_json=excluded.report_json
                """,
                payload,
            )
        return {key: payload[key] for key in ["update_id", "total", "success", "failed", "skipped", "storage_mode", "raw_db_path"]}

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
        return self.processed.save_stock_backtest(
            run_id=run_id,
            stock=stock,
            decisions=decisions,
            daily=daily,
            trades=trades,
            report=report,
            source_root=source_root,
        )

    def put_artifact_bytes(
        self,
        namespace: str,
        kind: str,
        name: str,
        payload: bytes,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.artifacts.put_bytes(namespace, kind, name, payload, metadata=metadata)

    def get_artifact_bytes(self, namespace: str, kind: str, name: str) -> tuple[bytes, dict[str, Any]] | None:
        return self.artifacts.get_bytes(namespace, kind, name)

    def get_json_artifact(self, namespace: str, name: str) -> tuple[Any, dict[str, Any]] | None:
        item = self.get_artifact_bytes(namespace, "json", name)
        if item is None:
            return None
        payload, metadata = item
        return json.loads(payload.decode("utf-8")), metadata

    def list_artifacts(self, namespace: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        return self.artifacts.list_records(namespace=namespace, kind=kind)

    def put_dataframe_csv_artifact(
        self,
        namespace: str,
        name: str,
        frame: pd.DataFrame,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        payload = frame.to_csv(index=False).encode("utf-8")
        meta = {"rows": int(len(frame)), "format": "csv", **(metadata or {})}
        return self.put_artifact_bytes(namespace, "csv", name, payload, metadata=meta)

    def put_json_artifact(
        self,
        namespace: str,
        name: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        payload = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
        meta = {"format": "json", **(metadata or {})}
        return self.put_artifact_bytes(namespace, "json", name, payload, metadata=meta)

    def register_authoritative_artifact(
        self,
        *,
        run_id: str,
        scope_key: str,
        output_type: str,
        artifact_id: str,
        authority: str = "processed_warehouse",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Point consumers to one authoritative artifact, without copying it."""
        if not self.settings.write_db:
            return
        conn = self.processed.connect()
        try:
            upsert_output_reference(
                conn,
                run_id=require_run_id(run_id, operation="artifact reference"),
                scope_key=scope_key,
                output_type=output_type,
                artifact_id=artifact_id,
                authority=authority,
                metadata=metadata,
            )
        finally:
            conn.close()

    def put_file_artifact(
        self,
        namespace: str,
        kind: str,
        name: str,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        file_path = Path(path)
        payload = file_path.read_bytes()
        meta = {
            "source_path": str(file_path),
            "suffix": file_path.suffix.lower(),
            **(metadata or {}),
        }
        return self.put_artifact_bytes(namespace, kind, name, payload, metadata=meta)

    def put_artifact_pickle(
        self,
        namespace: str,
        kind: str,
        name: str,
        value: Any,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        return self.artifacts.put_pickle(namespace, kind, name, value, metadata=metadata)

    def get_artifact_pickle(self, namespace: str, kind: str, name: str) -> tuple[Any, dict[str, Any]] | None:
        return self.artifacts.get_pickle(namespace, kind, name)

    def artifact_counts(self) -> dict[str, int]:
        if not self.processed.db_path.exists():
            return {}
        with self.artifacts.connect() as conn:
            rows = conn.execute(
                "SELECT namespace, kind, COUNT(*) AS count FROM artifact_blobs GROUP BY namespace, kind"
            ).fetchall()
        return {f"{row['namespace']}:{row['kind']}": int(row["count"]) for row in rows}

    def integrity_report(self) -> dict[str, dict[str, Any]]:
        return {
            "raw_db": _db_integrity(self.raw.db_path),
            "processed_db": _db_integrity(self.processed.db_path),
        }


def get_data_access(raw_db: str | Path | None = None, processed_db: str | Path | None = None) -> DataAccessLayer:
    """Create a fresh facade so temporary env-mode changes are respected."""
    return DataAccessLayer(raw_db=raw_db, processed_db=processed_db)


def _db_integrity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    conn = sqlite3.connect(path)
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        return {"path": str(path), "exists": True, "integrity_check": integrity}
    finally:
        conn.close()
