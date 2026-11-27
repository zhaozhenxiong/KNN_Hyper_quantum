"""Configuration helpers for the local data center."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "my_strategy" / "configs" / "data_config.yaml"


def _data_root_child(data_root: str, *parts: str) -> str:
    if data_root.startswith("/"):
        return str(PurePosixPath(data_root, *parts))
    return str(Path(data_root, *parts))


def load_data_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"Data config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config.setdefault("data_root", str(PROJECT_ROOT / "my_strategy" / "data"))
    data_root_override = os.getenv("KHQUANT_DATA_ROOT", "").strip()
    if data_root_override:
        config["data_root"] = data_root_override
        stock_pool = config.setdefault("stock_pool", {})
        stock_pool["custom_pool_path"] = _data_root_child(data_root_override, "raw", "custom_pool", "my_pool.csv")
        stock_pool["all_a_cache_path"] = _data_root_child(
            data_root_override,
            "raw",
            "stock_basic",
            "all_a_stock_list.parquet",
        )
    return config


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_project_on_path() -> None:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
