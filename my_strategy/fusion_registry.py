#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Registry helpers for the best nonlinear Agent fusion policy."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "model_registry"
BEST_FUSION_POLICY_PATH = REGISTRY_ROOT / "best_fusion_policy.json"


def load_best_fusion_policy(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path) if path else BEST_FUSION_POLICY_PATH
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return {}
    policy = data.get("policy", data)
    return policy if isinstance(policy, dict) else {}


def save_best_fusion_policy(
    policy: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    path: str | Path | None = None,
) -> Path:
    target = Path(path) if path else BEST_FUSION_POLICY_PATH
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "policy": policy,
        "metrics": metrics or {},
    }
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def apply_best_fusion_policy(config: dict[str, Any], path: str | Path | None = None) -> bool:
    policy = load_best_fusion_policy(path)
    if not policy:
        return False
    decision = config.setdefault("decision", {})
    fusion = decision.setdefault("fusion", {})
    fusion.update(policy)
    fusion["enabled"] = bool(fusion.get("enabled", True))
    return True
