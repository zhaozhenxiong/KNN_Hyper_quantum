#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Keep the best model artifacts and prune obsolete training outputs."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

try:
    from my_strategy.model_selector import model_score
except Exception:  # pragma: no cover - script fallback
    model_score = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed"
BEST_MODELS_PATH = PROCESSED_ROOT / "model_registry" / "best_models.json"
BEST_RL_PATH = PROCESSED_ROOT / "rl" / "best_rl_model.json"
KEEP_MODELS_PATH = PROCESSED_ROOT / "model_registry" / "keep_models.json"

MODEL_SPECS = {
    "ml": (PROCESSED_ROOT / "ml" / "models", PROCESSED_ROOT / "ml" / "reports"),
    "deep_learning": (PROCESSED_ROOT / "deep_learning" / "models", PROCESSED_ROOT / "deep_learning" / "reports"),
    "rl": (PROCESSED_ROOT / "rl" / "models", PROCESSED_ROOT / "rl" / "reports"),
}


def cleanup_model_artifacts(
    keep_recent: int = 2,
    keep_top: int = 3,
    keep_models_path: str | Path = KEEP_MODELS_PATH,
    execute: bool = False,
) -> dict[str, Any]:
    """Plan or delete non-kept model directories and reports."""
    protected = _protected_model_names(keep_models_path)
    by_kind: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []

    for kind, (model_root, report_root) in MODEL_SPECS.items():
        rows = _scan_kind(kind, model_root, report_root)
        keep = set(protected.get(kind, set()))
        keep.update(row["model_name"] for row in _latest_rows(rows, keep_recent))
        keep.update(row["model_name"] for row in _top_rows(rows, keep_top))

        candidates = []
        for row in rows:
            if row["model_name"] in keep:
                continue
            candidates.extend([row["model_path"], row["report_path"]])

        for path in _dedupe_existing(candidates):
            size = _path_size(path)
            artifact_type = "directory" if path.is_dir() else "file"
            if execute:
                _remove_path(path)
            actions.append(
                {
                    "kind": kind,
                    "model_name": _model_name_from_path(path),
                    "path": str(path),
                    "artifact_type": artifact_type,
                    "size_bytes": size,
                    "status": "deleted" if execute else "dry_run",
                }
            )

        by_kind[kind] = {
            "models_found": len(rows),
            "kept_models": sorted(keep),
            "candidate_models": sorted({row["model_name"] for row in rows if row["model_name"] not in keep}),
        }

    return {
        "execute": execute,
        "keep_recent": keep_recent,
        "keep_top": keep_top,
        "protected_models": {kind: sorted(names) for kind, names in protected.items()},
        "by_kind": by_kind,
        "candidate_count": len(actions),
        "candidate_size_bytes": sum(int(item["size_bytes"]) for item in actions),
        "actions": actions,
    }


def _protected_model_names(keep_models_path: str | Path) -> dict[str, set[str]]:
    protected: dict[str, set[str]] = {"ml": set(), "deep_learning": set(), "rl": set()}
    best = _read_json(BEST_MODELS_PATH)
    for kind in ["ml", "deep_learning"]:
        entry = best.get(kind) or {}
        if isinstance(entry, dict) and entry.get("model_name"):
            protected[kind].add(str(entry["model_name"]))

    best_rl = _read_json(BEST_RL_PATH)
    if best_rl.get("model_name"):
        protected["rl"].add(str(best_rl["model_name"]))

    manual = _read_json(Path(keep_models_path))
    for kind, value in manual.items():
        if kind not in protected:
            continue
        if isinstance(value, list):
            protected[kind].update(str(item) for item in value if str(item))
        elif isinstance(value, dict):
            protected[kind].update(str(item) for item in value.get("models", []) if str(item))
    return protected


def _scan_kind(kind: str, model_root: Path, report_root: Path) -> list[dict[str, Any]]:
    if not model_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for model_path in sorted([p for p in model_root.iterdir() if p.is_dir()]):
        model_name = model_path.name
        report_path = report_root / f"{model_name}_report.json"
        report = _read_json(report_path)
        score = _score_report(kind, report)
        rows.append(
            {
                "kind": kind,
                "model_name": model_name,
                "model_path": model_path,
                "report_path": report_path,
                "score": score,
                "mtime": max(
                    model_path.stat().st_mtime,
                    report_path.stat().st_mtime if report_path.exists() else 0,
                ),
            }
        )
    return rows


def _score_report(kind: str, report: dict[str, Any]) -> float:
    if not report:
        return -1e18
    if kind == "rl":
        return _safe_float(report.get("score"), -1e18)
    if model_score is not None:
        try:
            return float(model_score(report))
        except Exception:
            pass
    return _safe_float(report.get("score"), -1e18)


def _latest_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["mtime"], reverse=True)[: max(0, limit)]


def _top_rows(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row["score"], reverse=True)[: max(0, limit)]


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if not p.exists() or p.stat().st_size == 0:
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dedupe_existing(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _model_name_from_path(path: Path) -> str:
    if path.is_dir():
        return path.name
    name = path.stem
    return name[:-7] if name.endswith("_report") else name
