"""Return-only automatic replacement of the non-champion production fallback."""

from __future__ import annotations

from datetime import datetime
import json
from math import isfinite
from pathlib import Path

from my_strategy.model_registry import expert_fallback, expert_release
from my_strategy.model_registry.registry import ModelRegistry

RULE = "higher_compound_development_double_cost_total_return"
DEVELOPMENT = expert_fallback.DEVELOPMENT


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效的生产回退证据 {path.name}")
    return value


def _performance(report: dict, config: dict, coverage: dict, arm: str) -> dict:
    folds = {fold.get("period"): fold for fold in report.get("folds", [])}
    returns, windows = {}, []
    compound = 1.0
    for period in DEVELOPMENT:
        fold = folds.get(period, {})
        first, last = fold.get("evaluation_start", "")[:10], fold.get("evaluation_end", "")[:10]
        execution = fold.get("arms", {}).get(arm, {}).get("execution", {}).get("double", {})
        value = execution.get("total_return")
        checks = execution.get("verification", {})
        if (not first or not last or isinstance(value, bool) or not isinstance(value, (int, float))
                or not isfinite(value) or not checks or not all(checks.values())):
            raise ValueError(f"{period} 缺少完整的双倍成本收益或账本核验。")
        returns[period] = float(value)
        windows.append({"period": period, "start": first, "end": last})
        compound *= 1 + float(value)
    initial_cash = config.get("initial_cash")
    expense = config.get("expense_model")
    pool = coverage.get("pool")
    if (isinstance(initial_cash, bool) or not isinstance(initial_cash, (int, float))
            or not isfinite(initial_cash) or initial_cash <= 0):
        raise ValueError("研究缺少有效的初始资金口径。")
    if not isinstance(expense, dict) or config.get("cost_multipliers", {}).get("double") != 2:
        raise ValueError("研究缺少可比较的双倍成本口径。")
    if not isinstance(pool, str) or not pool:
        raise ValueError("研究缺少可比较的股票池口径。")
    return {
        "rule": RULE, "development_periods": list(DEVELOPMENT), "fold_returns": returns,
        "total_return": compound - 1,
        "scope": {"pool": pool, "windows": windows, "initial_cash": float(initial_cash), "expense_model": expense},
    }


def _standard_performance(root: Path) -> tuple[dict, dict, dict]:
    report = _json(expert_release._within(root, root / expert_release.REPORT))
    config = _json(expert_release._within(root, root / "config/study.json"))
    arm = report.get("final_portfolio", {}).get("arm")
    return _performance(report, config, report.get("input_coverage", {}), arm), report, config


def _release_performance(release: dict) -> dict:
    root = expert_release._within(
        expert_release.ARTIFACT_RUNS_ROOT, expert_release.ARTIFACT_RUNS_ROOT / release["source_run_id"],
    )
    if release.get("production_source") == "research_candidate":
        return _standard_performance(root)[0]
    proof = expert_fallback.load(root)
    report = _json(expert_release._within(root, root / proof["research_report"]))
    if proof.get("direct_archive"):
        config = _json(expert_release._within(root, root / "config/study.json"))
        return _performance(report, config, report.get("input_coverage", {}), proof["arm"])
    source = expert_release._within(root.parent, root.parent / proof["data_source_run_id"])
    source_report = _json(expert_release._within(source, source / expert_release.REPORT))
    source_config = _json(expert_release._within(source, source / "config/study.json"))
    return _performance(report, source_config, source_report.get("input_coverage", {}), proof["arm"])


def _retain(entry, registry: ModelRegistry, reason: str, candidate: dict | None = None):
    comparison = {"state": "not_replaced", "reason": reason, "rule": RULE}
    if candidate is not None:
        comparison["candidate_return"] = candidate["total_return"]
        comparison["scope"] = candidate["scope"]
    entry.params = entry.params | {"production_comparison": comparison}
    registry.upsert(entry)
    return entry


def consider(entry, registry: ModelRegistry):
    """Mark a completed candidate only when its comparable test return is strictly higher."""
    root = expert_release._within(
        expert_release.ARTIFACT_RUNS_ROOT, expert_release.ARTIFACT_RUNS_ROOT / entry.run_id,
    )
    try:
        candidate, report, _ = _standard_performance(root)
        status = _json(expert_release._within(root, root / "status.json"))
        if status.get("status") != "completed" or report.get("decision", {}).get("state") != "research_only":
            raise ValueError("训练研究尚未完成或状态无效。")
        incumbent_release = expert_release.current_strategy(registry)
        if incumbent_release.get("status") != "production_fallback":
            return _retain(entry, registry, "当前没有可按收益比较替换的生产回退。", candidate)
        incumbent = _release_performance(incumbent_release)
        if candidate["scope"] != incumbent["scope"]:
            return _retain(entry, registry, "测试区间、股票池、初始资金或成本口径不同，未自动替换。", candidate)
        if candidate["total_return"] <= incumbent["total_return"]:
            return _retain(entry, registry, "双倍成本复合测试收益未严格高于当前生产模型。", candidate)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return _retain(entry, registry, f"自动替换证据不完整：{error}")

    evidence = candidate | {
        "previous_release_id": incumbent_release["release_id"],
        "previous_run_id": incumbent_release["run_id"],
        "previous_return": incumbent["total_return"],
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "independent_forward_evidence": False,
    }
    for previous in registry.entries(kind=expert_release.KIND, stage="candidate"):
        if previous.params.get("production_fallback") is True:
            previous.params = {key: value for key, value in previous.params.items()
                               if key not in {"production_fallback", "production_selection"}}
            registry.upsert(previous)
    entry.params = entry.params | {"production_fallback": True, "production_selection": evidence}
    registry.upsert(entry)
    return entry


def current(registry: ModelRegistry) -> dict | None:
    """Load the unique automatic fallback and reject changed or incomplete evidence."""
    selected = [entry for entry in registry.entries(kind=expert_release.KIND, stage="candidate")
                if entry.params.get("production_fallback") is True]
    if not selected:
        return None
    if len(selected) != 1:
        raise ValueError("自动生产回退存在多个有效标记。")
    entry = selected[0]
    root = expert_release._within(
        expert_release.ARTIFACT_RUNS_ROOT, expert_release.ARTIFACT_RUNS_ROOT / entry.run_id,
    )
    performance, report, config = _standard_performance(root)
    status = _json(expert_release._within(root, root / "status.json"))
    arm = report.get("final_portfolio", {}).get("arm")
    digest = expert_release._sha(root / expert_release.REPORT)
    evidence = entry.params.get("production_selection", {})
    if (status.get("status") != "completed" or report.get("run", {}).get("run_id") != entry.run_id
            or report.get("decision", {}).get("state") != "research_only" or arm not in expert_release.RELEASE_ARMS
            or digest != entry.params.get("research_report_sha256")
            or any(evidence.get(key) != performance[key] for key in (
                "rule", "development_periods", "fold_returns", "total_return", "scope",
            )) or not isinstance(evidence.get("selected_at"), str)):
        raise ValueError("自动生产回退与训练报告或选择证据不一致。")
    windows = performance["scope"]["windows"]
    return {
        "status": "production_fallback", "production_source": "research_candidate",
        "run_id": entry.run_id, "source_run_id": entry.run_id,
        "release_id": entry.model_id, "model_id": entry.model_id,
        "model_version": report.get("unified", {}).get("model_version"),
        "arm": arm, "label": "多模态专家 · 较高收益自动生产回退",
        "analysis_start": windows[0]["start"], "as_of": windows[-1]["end"],
        "research_report_sha256": digest, "fallback_manifest_sha256": None, "manifest_sha256": None,
        "portfolio": config["portfolio"], "selection_evidence": evidence,
        "approval_note": "按用户决定，仅因同口径双倍成本开发折复合收益高于前生产模型而自动替换；不是正式冠军，也没有独立前瞻收益证据。",
    }
