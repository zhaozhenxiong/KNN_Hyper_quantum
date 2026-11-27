#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Materialized factor snapshots for training and fast batch backtests."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from my_strategy.agents.factor_agent import FactorAgent
from my_strategy.ml.feature_builder import DEFAULT_FEATURE_COLUMNS, build_ml_features
from my_strategy.ml.label_builder import add_forward_return_labels
from my_strategy.storage.access_layer import DataAccessLayer, get_data_access

FACTOR_SCHEMA_VERSION = "factor_agent_v2"
MODEL_FEATURE_SCHEMA_VERSION = "model_features_v2"


def factor_snapshot_version(volume_window: int = 20) -> str:
    return f"{FACTOR_SCHEMA_VERSION}_vw{int(volume_window)}"


def model_feature_snapshot_version(volume_window: int = 20) -> str:
    return f"{MODEL_FEATURE_SCHEMA_VERSION}_{factor_snapshot_version(volume_window)}"


def build_factor_snapshot_frame(
    raw: pd.DataFrame,
    *,
    volume_window: int = 20,
    factor_version: str | None = None,
    source_start: str = "",
    source_end: str = "",
) -> pd.DataFrame:
    """Compute factors per stock while preserving raw OHLCV/trade columns."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    version = factor_version or factor_snapshot_version(volume_window)
    frame = raw.copy()
    if "stock" not in frame.columns:
        if "code" in frame.columns:
            frame["stock"] = frame["code"]
        else:
            raise ValueError("factor snapshot source frame must include stock or code")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["stock", "date"])
    computed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    parts: list[pd.DataFrame] = []
    for stock, group in frame.groupby("stock", sort=False):
        daily = group.sort_values("date").reset_index(drop=True)
        factors = FactorAgent.add_factors(daily, volume_window=volume_window)
        factors["stock"] = str(stock)
        factors["code"] = factors.get("code", str(stock))
        factors["factor_version"] = version
        factors["volume_window"] = int(volume_window)
        factors["computed_at"] = computed_at
        factors["source_start"] = str(source_start)
        factors["source_end"] = str(source_end)
        parts.append(factors)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    return out.sort_values(["stock", "date"]).reset_index(drop=True)


def read_or_build_factor_snapshots(
    stocks: list[str],
    start_date: str,
    end_date: str,
    *,
    volume_window: int = 20,
    storage: DataAccessLayer | None = None,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read materialized factors, building missing stock snapshots in one pass."""
    storage = storage or get_data_access()
    stock_list = [str(stock) for stock in stocks if str(stock).strip()]
    version = factor_snapshot_version(volume_window)
    summary: dict[str, Any] = {
        "factor_version": version,
        "requested_stocks": len(stock_list),
        "start": str(start_date),
        "end": str(end_date),
        "volume_window": int(volume_window),
        "cache_hit": False,
        "built_stocks": 0,
        "built_rows": 0,
    }
    if not stock_list:
        return pd.DataFrame(), summary

    raw_coverage = storage.raw_daily_coverage(stock_list, start_date, end_date)
    coverage = {} if force else storage.factor_snapshot_coverage(stock_list, start_date, end_date, version)
    covered = {
        stock
        for stock, item in coverage.items()
        if _covers_raw_range(item, raw_coverage.get(stock, {}))
    }
    missing = [stock for stock in stock_list if stock not in covered]
    if missing:
        raw = storage.read_factor_source_batch(missing, start_date, end_date)
        built = build_factor_snapshot_frame(
            raw,
            volume_window=volume_window,
            factor_version=version,
            source_start=start_date,
            source_end=end_date,
        )
        if not built.empty:
            result = storage.replace_factor_snapshots(
                built,
                factor_version=version,
                source_start=start_date,
                source_end=end_date,
                volume_window=volume_window,
                summary={"requested_stocks": len(stock_list), "missing_stocks": len(missing)},
            )
            summary["built_stocks"] = result["stocks"]
            summary["built_rows"] = result["rows"]
        summary["missing_stocks"] = missing[:100]

    snapshot = storage.read_factor_snapshots(stock_list, start_date, end_date, version)
    summary["rows"] = int(len(snapshot))
    summary["stocks"] = int(snapshot["stock"].nunique()) if not snapshot.empty else 0
    summary["cache_hit"] = bool(not force and not missing and not snapshot.empty)
    if not snapshot.empty:
        snapshot.attrs["khquant_data_source"] = "processed_factor_snapshot"
        snapshot.attrs["khquant_factor_version"] = version
    return snapshot, summary


def group_factor_snapshots(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame is None or frame.empty:
        return {}
    grouped: dict[str, pd.DataFrame] = {}
    for stock, group in frame.groupby("stock", sort=False):
        grouped[str(stock)] = group.sort_values("date").reset_index(drop=True)
    return grouped


def build_model_feature_snapshot_frame(
    factors: pd.DataFrame,
    *,
    volume_window: int = 20,
    source_start: str = "",
    source_end: str = "",
) -> pd.DataFrame:
    if factors is None or factors.empty:
        return pd.DataFrame()
    factor_version = factor_snapshot_version(volume_window)
    feature_version = model_feature_snapshot_version(volume_window)
    computed_at = datetime.now().astimezone().isoformat(timespec="seconds")
    parts: list[pd.DataFrame] = []
    for stock, group in factors.groupby("stock", sort=False):
        feature_frame = build_ml_features(group.sort_values("date").reset_index(drop=True))
        labeled = add_forward_return_labels(feature_frame)
        labeled["stock"] = str(stock)
        labeled["code"] = labeled.get("code", str(stock))
        labeled["feature_version"] = feature_version
        labeled["factor_version"] = factor_version
        labeled["computed_at"] = computed_at
        labeled["source_start"] = str(source_start)
        labeled["source_end"] = str(source_end)
        parts.append(labeled)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values(["stock", "date"]).reset_index(drop=True)


def read_or_build_model_feature_snapshots(
    stocks: list[str],
    start_date: str,
    end_date: str,
    *,
    volume_window: int = 20,
    storage: DataAccessLayer | None = None,
    force: bool = False,
    include_labels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    storage = storage or get_data_access()
    stock_list = [str(stock) for stock in stocks if str(stock).strip()]
    feature_version = model_feature_snapshot_version(volume_window)
    factor_version = factor_snapshot_version(volume_window)
    summary: dict[str, Any] = {
        "feature_version": feature_version,
        "factor_version": factor_version,
        "requested_stocks": len(stock_list),
        "start": str(start_date),
        "end": str(end_date),
        "volume_window": int(volume_window),
        "cache_hit": False,
        "built_stocks": 0,
        "built_rows": 0,
    }
    if not stock_list:
        return pd.DataFrame(), summary

    raw_coverage = storage.raw_daily_coverage(stock_list, start_date, end_date)
    coverage = {} if force else storage.model_feature_snapshot_coverage(stock_list, start_date, end_date, feature_version)
    covered = {
        stock
        for stock, item in coverage.items()
        if _covers_raw_range(item, raw_coverage.get(stock, {}))
    }
    missing = [stock for stock in stock_list if stock not in covered]
    if missing:
        factors, factor_summary = read_or_build_factor_snapshots(
            missing,
            start_date,
            end_date,
            volume_window=volume_window,
            storage=storage,
            force=force,
        )
        built = build_model_feature_snapshot_frame(
            factors,
            volume_window=volume_window,
            source_start=start_date,
            source_end=end_date,
        )
        if not built.empty:
            result = storage.replace_model_feature_snapshots(
                built,
                feature_version=feature_version,
                factor_version=factor_version,
                source_start=start_date,
                source_end=end_date,
                summary={"factor_summary": factor_summary, "missing_stocks": len(missing)},
            )
            summary["built_stocks"] = result["stocks"]
            summary["built_rows"] = result["rows"]
        summary["missing_stocks"] = missing[:100]

    snapshot = storage.read_model_feature_snapshots(
        stock_list, start_date, end_date, feature_version, include_labels=include_labels
    )
    missing_feature_columns = [name for name in DEFAULT_FEATURE_COLUMNS if name not in snapshot.columns]
    if missing_feature_columns and not snapshot.empty:
        # A versioned cache created before a storage-column migration must not
        # masquerade as an extended feature snapshot. Rebuild in place.
        factors, factor_summary = read_or_build_factor_snapshots(
            stock_list,
            start_date,
            end_date,
            volume_window=volume_window,
            storage=storage,
            force=False,
        )
        rebuilt = build_model_feature_snapshot_frame(
            factors,
            volume_window=volume_window,
            source_start=start_date,
            source_end=end_date,
        )
        if not rebuilt.empty:
            result = storage.replace_model_feature_snapshots(
                rebuilt,
                feature_version=feature_version,
                factor_version=factor_version,
                source_start=start_date,
                source_end=end_date,
                summary={"factor_summary": factor_summary, "reason": "missing_feature_columns"},
            )
            summary["built_stocks"] += result["stocks"]
            summary["built_rows"] += result["rows"]
            summary["rebuilt_missing_feature_columns"] = missing_feature_columns
        snapshot = storage.read_model_feature_snapshots(
            stock_list, start_date, end_date, feature_version, include_labels=include_labels
        )
    summary["rows"] = int(len(snapshot))
    summary["stocks"] = int(snapshot["stock"].nunique()) if not snapshot.empty else 0
    summary["cache_hit"] = bool(not force and not missing and not snapshot.empty)
    if not snapshot.empty:
        snapshot.attrs["khquant_data_source"] = "processed_model_feature_snapshot"
        snapshot.attrs["khquant_feature_version"] = feature_version
    return snapshot, summary


def _covers_raw_range(item: dict[str, Any], raw_item: dict[str, Any]) -> bool:
    if int(item.get("rows") or 0) <= 0:
        return False
    raw_rows = int(raw_item.get("rows") or 0)
    if raw_rows <= 0:
        return False
    first_date = str(item.get("first_date") or "")
    last_date = str(item.get("last_date") or "")
    raw_first_date = str(raw_item.get("first_date") or "")
    raw_last_date = str(raw_item.get("last_date") or "")
    if not first_date or not last_date:
        return False
    first = pd.to_datetime(first_date, errors="coerce")
    last = pd.to_datetime(last_date, errors="coerce")
    raw_first = pd.to_datetime(raw_first_date, errors="coerce")
    raw_last = pd.to_datetime(raw_last_date, errors="coerce")
    if pd.isna(first) or pd.isna(last) or pd.isna(raw_first) or pd.isna(raw_last):
        return False
    return int(item.get("rows") or 0) >= raw_rows and first <= raw_first and last >= raw_last
