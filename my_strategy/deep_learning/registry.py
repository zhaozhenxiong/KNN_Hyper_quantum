#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persist and load deep learning model bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "deep_learning" / "models"


@dataclass
class DeepModelBundle:
    model: Any
    scaler: Any
    metadata: dict[str, Any]


def model_dir(model_name: str, root: str | Path | None = None) -> Path:
    base = Path(root) if root else DEFAULT_MODEL_ROOT
    if not base.is_absolute():
        base = PROJECT_ROOT / base
    return base / model_name


def save_deep_model(model_name: str, model: Any, scaler: Any, metadata: dict[str, Any], root: str | Path | None = None) -> Path:
    path = model_dir(model_name, root)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "model.pkl", "wb") as fh:
        pickle.dump(model, fh)
    with open(path / "scaler.pkl", "wb") as fh:
        pickle.dump(scaler, fh)
    (path / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_deep_model(model_name: str, root: str | Path | None = None) -> DeepModelBundle:
    path = model_dir(model_name, root)
    with open(path / "model.pkl", "rb") as fh:
        model = pickle.load(fh)
    with open(path / "scaler.pkl", "rb") as fh:
        scaler = pickle.load(fh)
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return DeepModelBundle(model=model, scaler=scaler, metadata=metadata)

