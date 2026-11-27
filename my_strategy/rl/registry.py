#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persist and load offline RL policy models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "rl" / "models"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "rl" / "reports"
DEFAULT_BEST_PATH = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "rl" / "best_rl_model.json"


@dataclass
class RLModelBundle:
    model: Any
    metadata: dict[str, Any]


def model_dir(model_name: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_MODEL_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / model_name


def save_rl_model(model_name: str, model: Any, metadata: dict[str, Any], root: str | Path | None = None) -> Path:
    path = model_dir(model_name, root)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_rl_model(model_name: str, root: str | Path | None = None) -> RLModelBundle:
    path = model_dir(model_name, root)
    with open(path / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return RLModelBundle(model=model, metadata=metadata)


def save_report(model_name: str, report: dict[str, Any], root: str | Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_REPORT_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{model_name}_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def best_rl_model_name(best_path: str | Path | None = None) -> str:
    path = Path(best_path) if best_path else DEFAULT_BEST_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("model_name", "") or "")
    except Exception:
        return ""


def update_best_rl_model(report: dict[str, Any], best_path: str | Path | None = None) -> bool:
    path = Path(best_path) if best_path else DEFAULT_BEST_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    old_score = -1e18
    if path.exists():
        try:
            old_score = float(json.loads(path.read_text(encoding="utf-8")).get("score", -1e18))
        except Exception:
            old_score = -1e18
    new_score = float(report.get("score", -1e18))
    if new_score >= old_score:
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return True
    return False

