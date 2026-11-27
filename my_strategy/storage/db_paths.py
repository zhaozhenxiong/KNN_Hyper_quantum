#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Central DB-first storage paths and mode switches."""

from __future__ import annotations

import os
from pathlib import Path

from my_strategy.core.paths import MY_STRATEGY_ROOT, project_path
from my_strategy.storage.storage_contract import load_storage_contract


def _contract_defaults() -> tuple[Path, Path]:
    """Use the storage contract while retaining stable legacy fallbacks."""
    try:
        contract = load_storage_contract()
        return contract.raw_db, contract.processed_db
    except (FileNotFoundError, ValueError):
        return (
            MY_STRATEGY_ROOT / "data" / "raw" / "khquant_raw.db",
            MY_STRATEGY_ROOT / "data" / "processed" / "warehouse" / "khquant.db",
        )


DEFAULT_RAW_DB, DEFAULT_PROCESSED_DB = _contract_defaults()


def storage_mode() -> str:
    """Return db, dual or file. DB-first is the project target."""
    value = os.environ.get("KHQUANT_STORAGE_MODE", "db").strip().lower()
    return value if value in {"db", "dual", "file"} else "db"


def raw_db_path() -> Path:
    return project_path(os.environ.get("KHQUANT_RAW_DB", str(DEFAULT_RAW_DB)))


def processed_db_path() -> Path:
    return project_path(os.environ.get("KHQUANT_PROCESSED_DB", str(DEFAULT_PROCESSED_DB)))


def read_db_first() -> bool:
    return storage_mode() in {"db", "dual"}


def write_db() -> bool:
    return storage_mode() in {"db", "dual"}


def write_files() -> bool:
    return storage_mode() in {"file", "dual"}
