"""Central path helpers for KHQuant runtime outputs."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MY_STRATEGY_ROOT = PROJECT_ROOT / "my_strategy"
DATA_ROOT = Path(os.environ.get("KHQUANT_DATA_ROOT", str(MY_STRATEGY_ROOT / "data")))
if not DATA_ROOT.is_absolute():
    DATA_ROOT = PROJECT_ROOT / DATA_ROOT
PROCESSED_DATA_ROOT = DATA_ROOT / "processed"
ARTIFACT_ROOT = Path(os.environ.get("KHQUANT_ARTIFACT_ROOT", str(MY_STRATEGY_ROOT / "artifacts")))
if not ARTIFACT_ROOT.is_absolute():
    ARTIFACT_ROOT = PROJECT_ROOT / ARTIFACT_ROOT
ARTIFACT_RUNS_ROOT = ARTIFACT_ROOT / "runs"
LEGACY_RESULTS_ROOT = PROJECT_ROOT / "results"
AI_AGENT_SCREEN_ROOT = Path(
    os.environ.get("KHQUANT_OUTPUT_ROOT", str(MY_STRATEGY_ROOT / "data" / "processed" / "ai_agent_screen"))
)
if not AI_AGENT_SCREEN_ROOT.is_absolute():
    AI_AGENT_SCREEN_ROOT = PROJECT_ROOT / AI_AGENT_SCREEN_ROOT


def project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def artifact_run_dir(run_id: str, *, create: bool = True) -> Path:
    if not run_id:
        raise ValueError("run_id is required")
    path = ARTIFACT_RUNS_ROOT / run_id
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_subdir(run_id: str, *parts: str, create: bool = True) -> Path:
    path = artifact_run_dir(run_id, create=create).joinpath(*parts)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
