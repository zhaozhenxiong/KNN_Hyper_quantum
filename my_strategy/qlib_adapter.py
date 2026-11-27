#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qlib integration notes and dependency checks for the strategy lab."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def qlib_available() -> bool:
    return importlib.util.find_spec("qlib") is not None


def available_qlib_candidates(project_root: Path) -> list[dict]:
    examples = project_root / "qlib-main" / "examples" / "benchmarks"
    return [
        {
            "name": "qlib_lightgbm_alpha158",
            "config": str(examples / "LightGBM" / "workflow_config_lightgbm_Alpha158.yaml"),
            "status": "ready" if qlib_available() else "requires qlib install",
        },
        {
            "name": "qlib_lightgbm_alpha360",
            "config": str(examples / "LightGBM" / "workflow_config_lightgbm_Alpha360.yaml"),
            "status": "ready" if qlib_available() else "requires qlib install",
        },
        {
            "name": "qlib_linear_alpha158",
            "config": str(examples / "Linear" / "workflow_config_linear_Alpha158.yaml"),
            "status": "ready" if qlib_available() else "requires qlib install",
        },
    ]

