#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Select and persist the best trained ML/DL models."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from my_strategy.core.paths import PROCESSED_DATA_ROOT
from my_strategy.model_registry import default_registry
from my_strategy.storage.access_layer import get_data_access
from my_strategy.storage.db_paths import read_db_first, write_db, write_files

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_REPORT_ROOT = PROCESSED_DATA_ROOT / "ml" / "reports"
ML_MODEL_ROOT = PROCESSED_DATA_ROOT / "ml" / "models"
DL_REPORT_ROOT = PROCESSED_DATA_ROOT / "deep_learning" / "reports"
DL_MODEL_ROOT = PROCESSED_DATA_ROOT / "deep_learning" / "models"
REGISTRY_ROOT = PROCESSED_DATA_ROOT / "model_registry"
BEST_MODELS_PATH = REGISTRY_ROOT / "best_models.json"
MIN_SELECTION_TEST_STOCKS = 500
MIN_SELECTION_TEST_DAYS = 60


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _nested(report: dict[str, Any], section: str, key: str, default: float = 0.0) -> float:
    section_data = report.get(section) or {}
    return _safe_float(section_data.get(key), default)


def model_score(report: dict[str, Any]) -> float:
    """Score reports by out-of-sample rank quality with a light overfit penalty."""
    valid_ic = _nested(report, "valid_metrics", "daily_rank_ic_mean")
    test_ic = _nested(report, "test_metrics", "daily_rank_ic_mean")
    valid_spread = _nested(report, "valid_metrics", "top_bottom_spread")
    test_spread = _nested(report, "test_metrics", "top_bottom_spread")
    valid_stock_ir = _nested(report, "valid_metrics", "stock_rank_ic_ir")
    test_stock_ir = _nested(report, "test_metrics", "stock_rank_ic_ir")
    train_ic = _nested(report, "train_metrics", "daily_rank_ic_mean")
    test_days = _nested(report, "test_metrics", "daily_rank_ic_days")
    test_stocks = _nested(report, "stocks", "test")

    overfit_gap = max(0.0, train_ic - test_ic - 0.08)
    sample_penalty = 0.0
    if test_days and test_days < 60:
        sample_penalty += (60 - test_days) / 60 * 0.02
    if test_stocks and test_stocks < 100:
        sample_penalty += (100 - test_stocks) / 100 * 0.02

    return float(
        0.32 * test_ic
        + 0.24 * valid_ic
        + 0.18 * test_spread
        + 0.12 * valid_spread
        + 0.08 * min(max(test_stock_ir, -1.0), 2.0) / 2.0
        + 0.06 * min(max(valid_stock_ir, -1.0), 2.0) / 2.0
        - 0.35 * overfit_gap
        - sample_penalty
    )


def _kind_paths(kind: str) -> tuple[Path, Path]:
    if kind == "ml":
        return ML_REPORT_ROOT, ML_MODEL_ROOT
    if kind in {"deep_learning", "dl"}:
        return DL_REPORT_ROOT, DL_MODEL_ROOT
    raise ValueError(f"unknown model kind: {kind}")


def _model_name_from_report_path(path: Path) -> str:
    name = path.stem
    return name[:-7] if name.endswith("_report") else name


def _model_artifact_names(kind: str) -> set[str]:
    storage = get_data_access()
    records = storage.list_artifacts(namespace=kind)
    by_name: dict[str, set[str]] = {}
    for record in records:
        by_name.setdefault(str(record["name"]), set()).add(str(record["kind"]))
    required = {"model", "metadata"} if kind == "ml" else {"model", "metadata", "scaler"}
    return {name for name, kinds in by_name.items() if required.issubset(kinds)}


def _model_uri(kind: str, model_name: str) -> str:
    return f"artifact://{kind}/model/{model_name}"


def _report_uri(name: str) -> str:
    return f"artifact://model_report/json/{name}"


def _row_from_report(kind: str, report: dict[str, Any], *, report_path: str, model_path: str, modified_at: str) -> dict[str, Any]:
    normalized_kind = "deep_learning" if kind in {"deep_learning", "dl"} else "ml"
    model_name = str(report.get("model_name") or report.get("requested_model_name") or "final")
    return {
        "kind": normalized_kind,
        "model_name": model_name,
        "score": model_score(report),
        "report_path": report_path,
        "model_path": model_path,
        "run_id": str((report.get("run_context") or {}).get("run_id") or report.get("run_id") or ""),
        "model_type": report.get("model_type", ""),
        "prediction_kind": report.get("prediction_kind", report.get("task", "")),
        "valid_daily_rank_ic": _nested(report, "valid_metrics", "daily_rank_ic_mean"),
        "test_daily_rank_ic": _nested(report, "test_metrics", "daily_rank_ic_mean"),
        "test_daily_rank_ic_days": _nested(report, "test_metrics", "daily_rank_ic_days"),
        "test_top_bottom_spread": _nested(report, "test_metrics", "top_bottom_spread"),
        "test_stock_rank_ic_ir": _nested(report, "test_metrics", "stock_rank_ic_ir"),
        "test_stocks": _nested(report, "stocks", "test"),
        "modified_at": modified_at,
    }


def _scan_model_report_artifacts(kind: str) -> list[dict[str, Any]]:
    normalized_kind = "deep_learning" if kind in {"deep_learning", "dl"} else "ml"
    storage = get_data_access()
    available_models = _model_artifact_names(normalized_kind)
    if not available_models:
        return []
    rows: list[dict[str, Any]] = []
    for record in storage.list_artifacts(namespace="model_report", kind="json"):
        name = str(record["name"])
        if not name.startswith(f"{normalized_kind}:"):
            continue
        item = storage.get_json_artifact("model_report", name)
        if item is None:
            continue
        try:
            report = item[0]
        except Exception:
            continue
        name_parts = name.split(":")
        model_name = str(report.get("model_name") or (name_parts[1] if len(name_parts) >= 2 else "final"))
        if model_name not in available_models:
            continue
        rows.append(
            _row_from_report(
                normalized_kind,
                report,
                report_path=_report_uri(name),
                model_path=_model_uri(normalized_kind, model_name),
                modified_at=str(record["updated_at"]),
            )
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def scan_model_reports(kind: str) -> list[dict[str, Any]]:
    if read_db_first():
        rows = _scan_model_report_artifacts(kind)
        if rows or not write_files():
            return rows

    report_root, model_root = _kind_paths(kind)
    rows: list[dict[str, Any]] = []
    if not report_root.exists():
        return rows

    for path in sorted(report_root.glob("*_report.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_name = str(report.get("model_name") or _model_name_from_report_path(path))
        model_path = model_root / model_name
        if not model_path.exists():
            continue
        rows.append(
            _row_from_report(
                kind,
                report,
                report_path=str(path),
                model_path=str(model_path),
                modified_at=datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
            )
        )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows


def select_best_model(kind: str) -> dict[str, Any] | None:
    rows = scan_model_reports(kind)
    eligible = [
        row
        for row in rows
        if _safe_float(row.get("test_stocks")) >= MIN_SELECTION_TEST_STOCKS
        and _safe_float(row.get("test_daily_rank_ic")) != 0.0
    ]
    if eligible:
        return eligible[0]
    return rows[0] if rows else None


def refresh_best_models(*, bootstrap_if_missing: bool = True) -> dict[str, Any]:
    """Refresh pointers from champion models without auto-promoting new candidates.

    Training scripts register freshly trained models as ``candidate``.  A
    candidate must pass the dedicated performance and signal-stability gate
    before it can become champion.  The one exception is bootstrapping an
    empty legacy registry, where no incumbent exists to protect.
    """

    ml_candidate = select_best_model("ml")
    dl_candidate = select_best_model("deep_learning")
    service = default_registry()
    ml_best = _champion_or_bootstrap(service, "ml", ml_candidate, bootstrap_if_missing)
    dl_best = _champion_or_bootstrap(service, "deep_learning", dl_candidate, bootstrap_if_missing)
    registry: dict[str, Any] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": "sqlite" if write_db() else "file",
        "selection_rule": "0.32*test_daily_ic + 0.24*valid_daily_ic + spreads/IR - overfit penalty",
        "ml": ml_best,
        "deep_learning": dl_best,
        "candidate_best": {"ml": ml_candidate, "deep_learning": dl_candidate},
    }
    if write_db():
        get_data_access().put_json_artifact(
            "model_registry",
            "best_models",
            registry,
            metadata={"kind": "ml_deep_learning", "updated_at": registry["updated_at"]},
        )
    if write_files():
        REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)
        BEST_MODELS_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    return registry


def _champion_or_bootstrap(
    service: Any,
    kind: str,
    candidate_row: dict[str, Any] | None,
    bootstrap_if_missing: bool,
) -> dict[str, Any] | None:
    champion = service.champion(kind)
    if champion is not None:
        return _entry_to_row(champion)
    if not bootstrap_if_missing:
        return None
    # A real candidate already in the lifecycle must go through the promotion
    # gate.  Bootstrap is reserved for legacy report files predating the
    # registry, not the first newly trained candidate.
    if service.entries(kind=kind):
        return None
    return _bootstrap_champion(service, kind, candidate_row)


def _entry_to_row(entry: Any) -> dict[str, Any]:
    metrics = entry.metrics or {}
    return {
        "kind": entry.kind,
        "model_name": entry.name,
        "score": metrics.get("score", 0.0),
        "report_path": entry.report_path,
        "model_path": entry.model_path,
        "run_id": entry.run_id,
        "model_type": entry.params.get("model_type", ""),
        "prediction_kind": entry.params.get("prediction_kind", ""),
        "valid_daily_rank_ic": metrics.get("valid_daily_rank_ic", 0.0),
        "test_daily_rank_ic": metrics.get("test_daily_rank_ic", metrics.get("daily_rank_ic_mean", 0.0)),
        "test_daily_rank_ic_days": metrics.get("test_daily_rank_ic_days", metrics.get("daily_rank_ic_days", 0)),
        "test_top_bottom_spread": metrics.get("test_top_bottom_spread", metrics.get("top_bottom_spread", 0.0)),
        "test_stock_rank_ic_ir": metrics.get("test_stock_rank_ic_ir", metrics.get("stock_rank_ic_ir", 0.0)),
        "test_stocks": metrics.get("test_stocks", metrics.get("stocks", 0)),
        "modified_at": entry.updated_at,
    }


def _bootstrap_champion(service: Any, kind: str, row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return
    entry = service.register(
        name=str(row.get("model_name", "")),
        kind=kind,
        stage="validated",
        run_id=str(row.get("run_id") or "legacy-selection"),
        model_path=str(row.get("model_path", "")),
        report_path=str(row.get("report_path", "")),
        metrics={
            "score": row.get("score"),
            "valid_daily_rank_ic": row.get("valid_daily_rank_ic"),
            "test_daily_rank_ic": row.get("test_daily_rank_ic"),
            "test_daily_rank_ic_days": row.get("test_daily_rank_ic_days"),
            "test_top_bottom_spread": row.get("test_top_bottom_spread"),
            "test_stock_rank_ic_ir": row.get("test_stock_rank_ic_ir"),
            "test_stocks": row.get("test_stocks"),
        },
        params={"model_type": row.get("model_type", ""), "prediction_kind": row.get("prediction_kind", "")},
    )
    return _entry_to_row(service.promote(entry.model_id, "champion"))


def load_best_models(refresh_if_missing: bool = True) -> dict[str, Any]:
    if read_db_first():
        artifact = get_data_access().get_json_artifact("model_registry", "best_models")
        if artifact is not None:
            return artifact[0]
        champions = _champion_best_models()
        if champions.get("ml") or champions.get("deep_learning"):
            return champions
        if refresh_if_missing:
            return refresh_best_models()
    if BEST_MODELS_PATH.exists():
        try:
            return json.loads(BEST_MODELS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    if refresh_if_missing:
        return refresh_best_models()
    return {}


def _champion_best_models() -> dict[str, Any]:
    service = default_registry()
    payload: dict[str, Any] = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "storage": "sqlite",
        "selection_rule": "unified_model_registry champion fallback",
        "ml": None,
        "deep_learning": None,
    }
    for kind in ["ml", "deep_learning"]:
        entry = service.champion(kind)
        if entry is None:
            continue
        payload[kind] = {
            "kind": kind,
            "model_name": entry.name,
            "score": entry.metrics.get("score", 0.0),
            "report_path": entry.report_path,
            "model_path": entry.model_path,
            "run_id": entry.run_id,
            "model_type": entry.params.get("model_type", ""),
            "prediction_kind": entry.params.get("prediction_kind", ""),
            "valid_daily_rank_ic": entry.metrics.get("valid_daily_rank_ic", 0.0),
            "test_daily_rank_ic": entry.metrics.get("test_daily_rank_ic", 0.0),
            "test_top_bottom_spread": entry.metrics.get("test_top_bottom_spread", 0.0),
            "test_stock_rank_ic_ir": entry.metrics.get("test_stock_rank_ic_ir", 0.0),
            "test_stocks": entry.metrics.get("test_stocks", 0.0),
            "modified_at": entry.updated_at,
        }
    return payload


def best_model_name(kind: str, refresh_if_missing: bool = True) -> str:
    registry = load_best_models(refresh_if_missing=refresh_if_missing)
    key = "deep_learning" if kind in {"deep_learning", "dl"} else "ml"
    entry = registry.get(key) or {}
    return str(entry.get("model_name") or "")


def main() -> None:
    registry = refresh_best_models()
    print(json.dumps(registry, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
