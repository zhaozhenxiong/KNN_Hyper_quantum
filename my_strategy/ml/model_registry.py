#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persist and load ML models with metadata."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ml" / "models"


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]


def model_dir(model_name: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_MODEL_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / model_name


def save_model(model_name: str, model: Any, metadata: dict[str, Any], root: str | Path | None = None) -> Path:
    path = model_dir(model_name, root)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_model(model_name: str, root: str | Path | None = None) -> ModelBundle:
    path = model_dir(model_name, root)
    with open(path / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return ModelBundle(model=model, metadata=metadata)
