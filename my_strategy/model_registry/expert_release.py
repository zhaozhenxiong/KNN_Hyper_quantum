"""Candidate registration and verified release lookup for the unified expert."""

from __future__ import annotations

from datetime import date
from hashlib import file_digest
import json
from pathlib import Path
import sqlite3

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT
from my_strategy.model_registry.registry import ModelRegistry, default_registry

KIND = "multimodal_expert"
REPORT = "reports/expert_research.json"
MANIFEST = "reports/expert_release.json"
RELEASE_ARMS = frozenset({"fusion", "domestic_gpu"})


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效的专家证据文件 {path.name}")
    return value


def _sha(path: Path) -> str:
    with path.open("rb") as content:
        return file_digest(content, "sha256").hexdigest()


def _within(root: Path, file: Path) -> Path:
    result = file.resolve(strict=True)
    if not result.is_relative_to(root.resolve(strict=True)):
        raise ValueError("专家证据路径超出归档目录。")
    return result


def record_candidate(run_id: str, report: Path, registry: ModelRegistry | None = None):
    """Register research and compare its test return without promoting a formal champion."""
    expected = _within(ARTIFACT_RUNS_ROOT, ARTIFACT_RUNS_ROOT / run_id / REPORT)
    if report.resolve(strict=True) != expected:
        raise ValueError("候选报告不属于该研究运行。")
    result = _json(expected)
    arm = result.get("final_portfolio", {}).get("arm")
    if result.get("run", {}).get("run_id") != run_id or arm not in RELEASE_ARMS:
        raise ValueError("研究报告缺少统一组合或运行编号。")
    registry = registry or default_registry()
    name = f"multimodal-expert:{run_id}"
    digest = _sha(expected)
    existing = registry.find_by_name(KIND, name)
    if existing:
        if existing.params.get("arm") != arm or existing.params.get("research_report_sha256") != digest:
            raise ValueError("已登记候选的策略或证据摘要发生变化。")
        return existing
    entry = registry.register(
        name=name, kind=KIND, stage="candidate", run_id=run_id,
        model_path=expected.parent.parent / "models", report_path=expected,
        metrics=result.get("decision", {}), data_range=result.get("input_coverage", {}),
        params={"arm": arm, "research_report_sha256": digest},
    )
    from my_strategy.model_registry import expert_auto_fallback

    return expert_auto_fallback.consider(entry, registry)


def _verify_manifest(entry) -> tuple[dict, str]:
    root = ARTIFACT_RUNS_ROOT / entry.run_id
    report_path = _within(ARTIFACT_RUNS_ROOT, root / REPORT)
    manifest_path = _within(root, root / MANIFEST)
    report, manifest = _json(report_path), _json(manifest_path)
    decision = report.get("decision", {})
    if not (decision.get("historical_numerical_gate") is True and decision.get("point_in_time_verified") is True):
        raise ValueError("研究候选尚未通过历史样本外与点时验收。")
    arm = report.get("final_portfolio", {}).get("arm")
    if arm not in RELEASE_ARMS or manifest.get("run_id") != entry.run_id or manifest.get("arm") != arm or entry.params.get("arm") != arm:
        raise ValueError("发布归档与登记模型不符。")
    if _sha(report_path) != manifest.get("research_report_sha256") or _sha(report_path) != entry.params.get("research_report_sha256"):
        raise ValueError("研究报告摘要与登记记录不符。")
    forward = manifest.get("forward", {})
    evidence = _within(root, root / "reports/expert_forward_acceptance.json")
    checked = _json(evidence)
    dates = [date.fromisoformat(day) for day in checked.get("sessions", [])]
    if (checked.get("run_id") != entry.run_id or checked.get("qualified_forward_results") is not True
            or len(dates) != 60 or sorted(set(dates)) != dates or _sha(evidence) != forward.get("sha256")
            or checked.get("double_cost_and_risk_gate") is not True
            or checked.get("capture_chain_verified") is not True):
        raise ValueError("发布要求完整且通过验收的 60 日前瞻证据。")
    versions = manifest.get("model_versions", [])
    if not versions:
        raise ValueError("发布模型缺少按历史日期生效的版本清单。")
    from my_strategy.research_workbench.expert_forward import verify_release_evidence

    verify_release_evidence(root, checked, manifest)
    previous = None
    for version in versions:
        start, end = (date.fromisoformat(version[key]) for key in ("valid_from", "valid_through"))
        trained = date.fromisoformat(version["training_max_label_date"])
        available = date.fromisoformat(version["available_at"][:10])
        if (version["run_id"] != entry.run_id or not version["fold"] or not trained <= available < start <= end
                or (previous and start <= previous)):
            raise ValueError("模型版本的训练、生效或区间顺序无效。")
        previous = end
    return manifest, _sha(manifest_path)


def promote_verified(run_id: str, registry: ModelRegistry | None = None):
    """Advance a registered expert only after its frozen evidence passes all gates."""
    registry = registry or default_registry()
    candidate = registry.find_by_name(KIND, f"multimodal-expert:{run_id}")
    if candidate is None:
        raise ValueError("找不到已登记的多模态专家候选。")
    _, digest = _verify_manifest(candidate)
    candidate.params = candidate.params | {"release_manifest_sha256": digest}
    registry.upsert(candidate)
    return registry.promote(candidate.model_id)


def chart_release(registry: ModelRegistry | None = None) -> dict:
    """Resolve only the approved expert active in the shared registry."""
    reason = "暂无通过滚动与前瞻验收的最终模型。"
    try:
        active = (registry or default_registry()).champion(KIND)
        if active:
            manifest, digest = _verify_manifest(active)
            if active.params.get("release_manifest_sha256") != digest:
                raise ValueError("发布摘要与注册记录不符。")
            return {"status": "approved", "release_id": active.model_id, "source_run_id": active.run_id,
                    "arm": manifest["arm"], "label": "多模态专家", "manifest_sha256": digest,
                    "model_versions": manifest["model_versions"]}
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as error:
        reason = f"最终版本证据待核验。{error}"
    return {"status": "unavailable", "label": "多模态专家待发布", "reason": reason}


def current_strategy(registry: ModelRegistry | None = None) -> dict:
    """Resolve the verified release or the most recent registered full-pool research evidence."""
    registry = registry or default_registry()
    published = chart_release(registry)
    if published["status"] == "approved":
        versions = published["model_versions"]
        return published | {
            "run_id": published["source_run_id"], "model_id": published["release_id"],
            "analysis_start": versions[0]["valid_from"], "as_of": versions[-1]["valid_through"],
            "publication": published,
        }

    from my_strategy.model_registry import expert_auto_fallback, expert_fallback

    try:
        hybrid = expert_fallback.active_hybrid_fallback()
    except (OSError, ValueError, TypeError, KeyError) as error:
        return published | {"status": "unavailable", "label": "混合生产证据待核验",
                            "reason": f"混合生产证据无效：{error}", "publication": published}
    if hybrid is not None:
        return hybrid | {"publication": published}

    try:
        pinned = expert_fallback.active_fallback()
    except (OSError, ValueError, TypeError, KeyError) as error:
        return published | {"status": "unavailable", "label": "纯 KNN 生产证据待核验",
                            "reason": f"纯 KNN 生产证据无效：{error}", "publication": published}
    if pinned is not None:
        return pinned | {"publication": published}

    try:
        automatic = expert_auto_fallback.current(registry)
    except (OSError, ValueError, TypeError, KeyError) as error:
        return published | {"status": "unavailable", "label": "自动生产回退证据待核验",
                            "reason": f"自动生产回退证据无效：{error}", "publication": published}
    if automatic is not None:
        return automatic | {"publication": published}

    try:
        fallback = expert_fallback.default_fallback()
    except (OSError, ValueError, TypeError, KeyError) as error:
        return published | {"status": "unavailable", "label": "生产回退证据待核验",
                            "reason": f"在线 KNN 生产回退证据无效：{error}", "publication": published}
    if fallback is not None:
        return fallback | {"publication": published}

    current = None
    for entry in registry.entries(kind=KIND, stage="candidate"):
        if entry.data_range.get("pool") != "full":
            continue
        try:
            root = _within(ARTIFACT_RUNS_ROOT, ARTIFACT_RUNS_ROOT / entry.run_id)
            path = _within(root, root / REPORT)
            report = _json(path)
            coverage = report["input_coverage"]
            run = report["run"]
            arm = report["final_portfolio"]["arm"]
            digest = _sha(path)
            status = _json(_within(root, root / "status.json"))
            start = date.fromisoformat(coverage["first_evaluation_session"])
            end = date.fromisoformat(coverage["last_evaluation_session"])
            if (status.get("status") != "completed" or run["run_id"] != entry.run_id
                    or coverage["pool"] != "full" or start > end or arm not in RELEASE_ARMS
                    or report["decision"]["state"] != "research_only"
                    or digest != entry.params["research_report_sha256"] or arm != entry.params["arm"]):
                continue
        except (OSError, ValueError, TypeError, KeyError):
            continue
        # Registry order resolves equal coverage by latest registered run.
        if current is None or end.isoformat() > current["as_of"]:
            current = {
                "status": "research_only", "run_id": entry.run_id, "model_id": entry.model_id,
                "source_run_id": entry.run_id, "release_id": entry.run_id, "arm": arm,
                "label": "多模态专家 · 全池研究候选", "analysis_start": start.isoformat(), "as_of": end.isoformat(),
                "research_report_sha256": digest, "publication": published,
            }
    if current is not None:
        try:
            from my_strategy.research_workbench.expert_score_health import inspect_archive

            root = ARTIFACT_RUNS_ROOT / current["run_id"]
            current["score_health"] = inspect_archive(root, _json(_within(root, root / REPORT)))
        except (OSError, ValueError, TypeError, KeyError) as error:
            current["score_health"] = {"status": "unavailable", "reason": f"分数健康证据暂不可读：{error}"}
    return current or (published | {"publication": published})
