"""Verified, explicitly non-champion fallback to the frozen online KNN strategy."""

from __future__ import annotations

import argparse
from datetime import datetime
from hashlib import file_digest
import json
from math import fsum, isfinite
import os
from pathlib import Path

from my_strategy.core.paths import ARTIFACT_RUNS_ROOT

RUN_ID = "20261126-expert-multiview-full-v1"
RELEASE_ID = "online-knn-multiview-20261126-v1"
ARM = "online_knn_volume_trend"
REPORT = "reports/expert_multiview_study.json"
STANDARD_REPORT = "reports/expert_research.json"
MANIFEST = "reports/expert_production_fallback.json"
DEVELOPMENT = ("oos_1", "oos_2", "oos_3")
DIRECT_ENGINE = "online_knn_volume_trend"
HYBRID_ENGINE = "knn60_multimodal40"
HYBRID_ARM = HYBRID_ENGINE
HYBRID_POINTER = "knn60_multimodal40_production.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"无效的回退证据文件 {path.name}")
    return value


def _sha(path: Path) -> str:
    with path.open("rb") as content:
        return file_digest(content, "sha256").hexdigest()


def _within(root: Path, path: Path) -> Path:
    result = path.resolve(strict=True)
    if not result.is_relative_to(root.resolve(strict=True)):
        raise ValueError("生产回退证据路径超出归档目录。")
    return result


def _reported_folds(report: dict) -> list[dict]:
    review = report.get("historical_review")
    return [*report.get("folds", []), *([review] if isinstance(review, dict) and review else [])]


def _fit_end(fold: dict) -> str:
    stages = ("train", "validation", "calibration") if "validation" in fold else ("train", "calibration")
    stamps = [fold.get(stage, {}).get("last_label_end") for stage in stages]
    if not all(isinstance(stamp, str) for stamp in stamps):
        raise ValueError(f"{fold.get('period')} 缺少训练、验证或校准标签截止证据。")
    return max(stamp[:10] for stamp in stamps)


def _selection(report: dict, config: dict) -> dict:
    hybrid = config.get("training_engine") == HYBRID_ENGINE
    selected = HYBRID_ARM if hybrid else ARM
    development = tuple(config.get("development_periods", ()))
    if development != DEVELOPMENT:
        raise ValueError("生产回退要求固定的三个开发折。")
    folds = {fold.get("period"): fold for fold in _reported_folds(report)}
    if set(folds) != {*DEVELOPMENT, "historical_replay"}:
        raise ValueError("KNN 归档折集合与回退清单不一致。")
    candidates = {}
    for candidate_arm in folds[DEVELOPMENT[0]].get("arms", {}):
        returns = []
        for period in DEVELOPMENT:
            execution = folds[period]["arms"].get(candidate_arm, {}).get("execution", {}).get("double", {})
            value = execution.get("total_return")
            if not isinstance(value, (int, float)) or not isfinite(value):
                break
            returns.append(float(value))
        if len(returns) == len(DEVELOPMENT):
            compound = 1.0
            for value in returns:
                compound *= 1 + value
            candidates[candidate_arm] = {
                "fold_returns": dict(zip(DEVELOPMENT, returns, strict=True)),
                "mean_return": fsum(returns) / len(returns), "compound_return": compound - 1,
            }
    best = max((row["mean_return"] for row in candidates.values()), default=None)
    winners = [candidate_arm for candidate_arm, row in candidates.items() if row["mean_return"] == best]
    if best is None or selected not in candidates or (not hybrid and winners != [ARM]):
        raise ValueError("当前测试缺少预定策略的唯一有效收益证据。")
    return {
        "rule": "fixed_knn60_multimodal40_development_double_cost_compound" if hybrid else "unique_highest_arithmetic_mean_development_double_cost_total_return",
        "development_periods": list(DEVELOPMENT), "candidate_count": len(candidates),
        **candidates[selected], "independent_forward_evidence": False,
    }


def _build(root: Path, created_at: str) -> dict:
    root = root.resolve(strict=True)
    config_path = _within(root, root / "config/study.json")
    config = _json(config_path)
    engine = config.get("training_engine")
    direct = engine in {DIRECT_ENGINE, HYBRID_ENGINE} and config.get("model_spec") is not None
    arm = HYBRID_ARM if engine == HYBRID_ENGINE else ARM
    report_name = STANDARD_REPORT if direct else REPORT
    report_path = _within(root, root / report_name)
    report = _json(report_path)
    status = _json(_within(root, root / "status.json"))
    decision = report.get("decision", {})
    if (status.get("status") != "completed" or report.get("run", {}).get("run_id") != root.name
            or decision.get("status", decision.get("state")) != "research_only"
            or decision.get("registry_written") is not False):
        raise ValueError("KNN 研究运行身份或历史研究状态无效。")
    if direct and report.get("training_engine") != engine:
        raise ValueError("标准研究报告与训练引擎不一致。")
    verification = report.get("artifact_verification", {})
    if (verification.get("all_daily_dates_unique") is not True
            or verification.get("all_trade_counts_match") is not True
            or verification.get("ledger_tables") != verification.get("matrix_rows")):
        raise ValueError("KNN 研究账本完整性尚未核验。")
    policy = config.get("rank_policy", config.get("portfolio", {}))
    if policy.get("selection_mode") != "cross_sectional_rank" or policy.get("min_score") is not None:
        raise ValueError("在线 KNN 回退必须使用显式横截面排名配置。")
    if direct:
        data_source_id = config.get("source_run_id")
        data_config_sha = data_report_sha = None
    else:
        data_source_id = config.get("source_run_id")
        source = _within(root.parent, root.parent / data_source_id)
        source_config_path = _within(source, source / "config/study.json")
        source_report_path = _within(source, source / STANDARD_REPORT)
        if _json(source_config_path).get("source_run_id") is None:
            raise ValueError("回退数据来源缺少冻结输入身份。")
        data_config_sha, data_report_sha = _sha(source_config_path), _sha(source_report_path)
    selection = _selection(report, config)
    versions, prediction_hashes = [], {}
    for fold in _reported_folds(report):
        period = fold["period"]
        first, last = fold["evaluation_start"][:10], fold["evaluation_end"][:10]
        fit_end = _fit_end(fold)
        arm_evidence = fold["arms"][arm]
        checks = arm_evidence["execution"]["double"].get("verification", {})
        if (arm_evidence.get("strategy") != policy or not fit_end < first <= last
                or not checks or not all(checks.values())):
            raise ValueError(f"{period} 的策略、标签时点或账本检查无效。")
        prediction = _within(root, root / f"backtests/{period}/{arm}_predictions.parquet")
        prediction_hashes[period] = _sha(prediction)
        versions.append({
            "run_id": root.name, "fold": period, "valid_from": first, "valid_through": last,
            "training_max_label_date": fit_end, "available_at": None,
        })
    release_id = f"hybrid-knn60-multimodal40-{root.name}" if engine == HYBRID_ENGINE else f"online-knn-{root.name}" if direct else RELEASE_ID
    return {
        "schema_version": 1, "release_id": release_id, "run_id": root.name,
        "data_source_run_id": data_source_id, "direct_archive": direct, "arm": arm, "portfolio": policy,
        "research_report": report_name, "research_report_sha256": _sha(report_path),
        "research_config_sha256": _sha(config_path),
        "data_config_sha256": data_config_sha or _sha(config_path),
        "data_report_sha256": data_report_sha or _sha(report_path),
        "prediction_sha256": prediction_hashes, "archived_code_sha256": config.get("code_sha256", {}),
        "selection": selection, "model_versions": versions, "created_at": created_at,
    }


def write_manifest(root: Path) -> Path:
    """Seal the selected frozen predictions without changing research or registry state."""
    path = _within(root, root / MANIFEST) if (root / MANIFEST).exists() else root / MANIFEST
    if path.exists():
        raise FileExistsError(f"生产回退清单已存在：{path}")
    manifest = _build(root, datetime.now().astimezone().isoformat(timespec="seconds"))
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load(root: Path) -> dict:
    """Recompute and verify every pinned property before exposing the fallback."""
    root = root.resolve(strict=True)
    path = _within(root, root / MANIFEST)
    manifest = _json(path)
    created_at = manifest.get("created_at")
    if not isinstance(created_at, str):
        raise ValueError("生产回退清单缺少封存时间。")
    if manifest != _build(root, created_at):
        raise ValueError("生产回退清单与当前冻结研究证据不一致。")
    hybrid = manifest["arm"] == HYBRID_ARM
    return {
        "status": "production_fallback", "production_source": "hybrid" if hybrid else "online_knn",
        "run_id": root.name, "source_run_id": root.name,
        "release_id": manifest["release_id"], "model_id": manifest["release_id"], "model_version": None,
        "arm": manifest["arm"], "label": "KNN 60% + 多模态 40% · 已验证生产回退" if hybrid else "纯在线 KNN · 已验证生产回退",
        "analysis_start": manifest["model_versions"][0]["valid_from"],
        "as_of": manifest["model_versions"][-1]["valid_through"],
        "research_report": manifest["research_report"],
        "research_report_sha256": manifest["research_report_sha256"],
        "fallback_manifest_sha256": _sha(path), "manifest_sha256": None,
        "data_source_run_id": manifest["data_source_run_id"], "direct_archive": manifest["direct_archive"],
        "portfolio": manifest["portfolio"], "selection_evidence": manifest["selection"],
        "model_versions": manifest["model_versions"],
        "approval_note": "按同口径双倍成本复合收益选择的混合生产回退；不是正式冠军，也没有独立前瞻收益证据。" if hybrid else "按用户明确决定全面切回纯在线 KNN；不是正式冠军，也没有独立前瞻收益证据。",
    }


def replay_config(root: Path) -> tuple[dict, dict]:
    """Adapt either KNN archive generation to the common replay contract."""
    proof = load(root)
    if proof["direct_archive"]:
        config = _json(_within(root, root / "config/study.json"))
    else:
        source = _within(root.parent, root.parent / proof["data_source_run_id"])
        config = _json(_within(source, source / "config/study.json"))
        config = config | {"data_source_run_id": source.name}
    config = config | {
        "portfolio": proof["portfolio"], "fallback_release_id": proof["release_id"], "_fallback_release": proof,
    }
    return config, proof


def _active_path() -> Path:
    return ARTIFACT_RUNS_ROOT / "online_knn_production.json"


def active_fallback() -> dict | None:
    path = _active_path()
    if not path.is_file():
        return None
    pointer = _json(path)
    run_id = pointer.get("run_id")
    if pointer.get("schema_version") != 1 or not isinstance(run_id, str) or not run_id:
        raise ValueError("当前 KNN 生产指针无效。")
    proof = load(ARTIFACT_RUNS_ROOT / run_id)
    if pointer.get("release_id") != proof["release_id"]:
        raise ValueError("当前 KNN 生产指针与封存版本不一致。")
    return proof


def active_hybrid_fallback() -> dict | None:
    path = ARTIFACT_RUNS_ROOT / HYBRID_POINTER
    if not path.is_file():
        return None
    pointer = _json(path)
    run_id = pointer.get("run_id")
    if pointer.get("schema_version") != 1 or not isinstance(run_id, str) or not run_id:
        raise ValueError("当前混合生产指针无效。")
    proof = load(ARTIFACT_RUNS_ROOT / run_id)
    if proof["arm"] != HYBRID_ARM or pointer.get("release_id") != proof["release_id"]:
        raise ValueError("当前混合生产指针与封存版本不一致。")
    return proof


def default_fallback() -> dict | None:
    active = active_fallback()
    if active is not None:
        return active
    root = ARTIFACT_RUNS_ROOT / RUN_ID
    return load(root) if (root / MANIFEST).is_file() else None


def _performance(root: Path) -> dict:
    from my_strategy.model_registry.expert_auto_fallback import _performance as performance

    proof = load(root)
    config = _json(_within(root, root / "config/study.json"))
    report = _json(_within(root, root / proof["research_report"]))
    if proof["direct_archive"]:
        return performance(report, config, report.get("input_coverage", {}), proof["arm"])
    source = _within(root.parent, root.parent / proof["data_source_run_id"])
    source_config = _json(_within(source, source / "config/study.json"))
    source_report = _json(_within(source, source / STANDARD_REPORT))
    return performance(report, source_config, source_report.get("input_coverage", {}), ARM)


def activate(root: Path, *, force: bool = False) -> dict:
    """Activate a sealed KNN run; later automatic switches still require higher return."""
    root = root.resolve(strict=True)
    if not (root / MANIFEST).is_file():
        write_manifest(root)
    proof = load(root)
    candidate = _performance(root)
    incumbent_release = None
    incumbent = None
    if not force:
        from my_strategy.model_registry import expert_release

        incumbent_release = expert_release.current_strategy()
        if incumbent_release.get("run_id") == root.name and incumbent_release.get("arm") == proof["arm"]:
            return proof | {"activated": True, "comparison": {"state": "already_active"}}
        if incumbent_release.get("status") == "production_fallback":
            from my_strategy.model_registry.expert_auto_fallback import _release_performance

            incumbent = _release_performance(incumbent_release)
            if candidate["scope"] != incumbent["scope"] or candidate["total_return"] <= incumbent["total_return"]:
                return proof | {
                    "activated": False,
                    "comparison": {
                        "state": "not_replaced", "reason": "测试口径不同或双倍成本复合收益未严格高于当前生产。",
                        "candidate_return": candidate["total_return"],
                        "incumbent_return": incumbent["total_return"], "incumbent_run_id": incumbent_release["run_id"],
                    },
                }
    pointer = {
        "schema_version": 1, "run_id": root.name, "release_id": proof["release_id"],
        "selected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "selection": {
            "mode": ("explicit_user_hybrid_switch" if proof["arm"] == HYBRID_ARM else "explicit_user_full_knn_switch") if force else
                    ("higher_comparable_hybrid_return" if proof["arm"] == HYBRID_ARM else "higher_comparable_knn_return"),
            "candidate": candidate, "previous_run_id": incumbent_release and incumbent_release.get("run_id"),
            "previous_return": incumbent and incumbent["total_return"],
        },
    }
    path = ARTIFACT_RUNS_ROOT / HYBRID_POINTER if proof["arm"] == HYBRID_ARM else _active_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(pointer, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return proof | {"activated": True, "comparison": pointer["selection"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--activate", action="store_true")
    parser.add_argument("--force", action="store_true", help="用户明确要求全面切换时使用；账本证据仍不可省略")
    args = parser.parse_args()
    root = ARTIFACT_RUNS_ROOT / args.run_id
    if args.activate or args.force:
        print(activate(root, force=args.force))
    else:
        print(write_manifest(root))


if __name__ == "__main__":
    main()
