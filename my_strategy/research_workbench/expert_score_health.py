"""Audit score health without refitting or changing an archived expert."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.research_workbench.runner import write_json


REPORT = "expert_score_health.json"
Q50_INDEX = models.QUANTILE_COLUMNS.index("q50")


def _number(value):
    return float(value) if value is not None and math.isfinite(float(value)) else None


def _distribution(values) -> dict:
    series = pd.to_numeric(pd.Series(values), errors="coerce")
    finite = series[np.isfinite(series)].to_numpy(dtype=float)
    if not len(finite):
        return {"rows": 0, "missing": int(series.size), "unique": 0, "positive_rate": None,
                "mean": None, "std": None, "min": None, "p01": None, "p10": None, "median": None,
                "p90": None, "p99": None, "max": None}
    quantiles = np.quantile(finite, [0.01, 0.1, 0.5, 0.9, 0.99])
    return {
        "rows": int(len(finite)), "missing": int(series.size - len(finite)), "unique": int(np.unique(finite).size),
        "positive_rate": float(np.mean(finite > 0)), "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
        "min": float(np.min(finite)), "p01": float(quantiles[0]), "p10": float(quantiles[1]),
        "median": float(quantiles[2]), "p90": float(quantiles[3]), "p99": float(quantiles[4]),
        "max": float(np.max(finite)),
    }


def _learned_median_health(tree_count: int | None, values) -> dict:
    scores = values if isinstance(values, dict) and "unique" in values else _distribution(values)
    early = isinstance(tree_count, int) and tree_count <= 1
    collapsed = early and 0 < scores["unique"] <= 16
    return {
        "q50_tree_count": tree_count, "early_stopped_median": early, "collapsed": collapsed,
        "evaluation_q50": scores,
        "reason": "q50 仅保留一棵树且实际分数不超过 16 个不同取值。" if collapsed else
                  "q50 仅保留一棵树，需要结合分数离散度复核。" if early else "",
    }


def _reported_folds(report: dict) -> list[dict]:
    review = report.get("historical_review")
    return [*report.get("folds", []), *([review] if isinstance(review, dict) and review else [])]


def _inspect_knn(root: Path, report: dict) -> dict:
    folds = []
    for fold in _reported_folds(report):
        period = fold["period"]
        arm = fold["arms"]["online_knn_volume_trend"]
        evidence = arm["score_evidence"]
        table = pd.read_parquet(
            root / "backtests" / period / "online_knn_volume_trend_predictions.parquet", columns=["date", "q50"],
        )
        calibrated = table["q50"].to_numpy(dtype=float)
        observed = _distribution(calibrated)
        if observed != evidence["calibrated_evaluation_q50"]:
            raise ValueError(f"{period} 的 KNN 分数分布与冻结预测不一致")
        concerns = [] if observed["unique"] > 1 else ["KNN q50 整折只有一个不同取值。"]
        folds.append({
            "period": period, "retrospective_review": period == "historical_replay",
            "status": "attention" if concerns else "ok", "concerns": concerns,
            "entry": evidence["entry"], "contributing_models": [],
            "raw_q50": evidence["raw_evaluation_q50"], "calibrated_q50": observed,
            "calibration_offset_q50": float(arm["calibration"]["offsets"][Q50_INDEX]),
            "raw_recovery": "纯在线 KNN 原生预测；校准仅做折内残差分位数调整，选股使用横截面排名。",
            "qualified_rows": evidence["qualified_rows"], "raw_qualified_rows": None,
            "target_evidence": {stage: fold.get(stage, {}).get("target") for stage in ("train", "validation", "calibration")},
        })
    attention = any(fold["status"] == "attention" for fold in folds)
    return {
        "schema_version": 1, "source_run_id": report.get("run", {}).get("run_id"),
        "status": "attention" if attention else "ok", "selection_scope": "historical_diagnostic",
        "release_eligible": False, "registry_mutated": False,
        "summary": "纯在线 KNN 使用横截面排名，不设置 q50 大于零的绝对入场门槛。",
        "folds": folds,
    }


def _inspect_hybrid(root: Path, report: dict) -> dict:
    folds = []
    for fold in _reported_folds(report):
        period = fold["period"]
        knn = fold["arms"]["online_knn_volume_trend"]
        evidence = fold["arms"]["knn60_multimodal40"]["score_evidence"]
        table = pd.read_parquet(root / "backtests" / period / "knn60_multimodal40_predictions.parquet")
        observed = {
            "hybrid_score": _distribution(table["hybrid_score"]),
            "knn_rank": _distribution(table["knn_rank"]),
            "multimodal_rank": _distribution(table["multimodal_rank"]),
        }
        full = int(table["effective_multimodal_weight"].eq(0.4).sum())
        fallback = int(table["effective_knn_weight"].eq(1.0).sum())
        invalid = int(table["effective_knn_weight"].eq(0.0).sum())
        reasons = {key: int(value) for key, value in table["fallback_reason"].value_counts().items() if key}
        calibrated = _distribution(table["knn_q50"])
        if (len(table) != evidence["evaluation_rows"] or full != evidence["full_60_40_rows"]
                or fallback != evidence["knn_fallback_rows"] or invalid != evidence["invalid_knn_rows"]
                or reasons != evidence["fallback_reasons"] or calibrated != knn["score_evidence"]["calibrated_evaluation_q50"]
                or any(observed[name] != evidence[name] for name in observed)):
            raise ValueError(f"{period} 的混合权重、覆盖率或分数分布与冻结预测不一致")
        concerns = [f"{invalid} 个股票日缺少有效 KNN 主分数，未进入组合。"] if invalid else []
        backend = fold["hybrid"]["knn_backend"]
        folds.append({
            "period": period, "retrospective_review": period == "historical_replay",
            "status": "attention" if concerns else "ok", "concerns": concerns,
            "entry": evidence["entry"], "contributing_models": [],
            "raw_q50": knn["score_evidence"]["raw_evaluation_q50"], "calibrated_q50": calibrated,
            "multimodal_q50": _distribution(table["multimodal_q50"]),
            "hybrid_score": observed["hybrid_score"], "knn_rank": observed["knn_rank"],
            "multimodal_rank": observed["multimodal_rank"], "coverage": evidence,
            "qualified_rows": int(table["hybrid_score"].notna().sum()), "raw_qualified_rows": None,
            "calibration_offset_q50": float(knn["calibration"]["offsets"][Q50_INDEX]),
            "raw_recovery": "双方原生及校准分位数分别封存；混合列仅为决策排名，不是收益分位数。",
            "knn_backend": backend, "fusion_weights": fold["hybrid"].get("fusion_weights", fold.get("fusion", {}).get("winner", {}).get("weights", {})),
            "multimodal_validation_health": fold["hybrid"].get("multimodal_validation_health"),
            "target_evidence": {stage: fold.get(stage, {}).get("target") for stage in ("train", "validation", "calibration")},
        })
    attention = any(fold["status"] == "attention" for fold in folds)
    return {
        "schema_version": 1, "source_run_id": report.get("run", {}).get("run_id"),
        "status": "attention" if attention else "ok", "selection_scope": "historical_diagnostic",
        "release_eligible": False, "registry_mutated": False, "mode": "knn60_multimodal40",
        "summary": "完整覆盖行采用 KNN 60% + 多模态 40%；多模态无效时由 KNN 100% 兜底，混合分不作为收益分位数。",
        "folds": folds,
    }


def inspect_archive(root: Path, report: dict) -> dict:
    """Read score and model metadata only; target/panel audits are a separate explicit task."""
    if report.get("training_engine") == "knn60_multimodal40":
        return _inspect_hybrid(Path(root), report)
    if report.get("training_engine") == "online_knn_volume_trend":
        return _inspect_knn(Path(root), report)
    root = Path(root)
    threshold = float(report.get("portfolio", {}).get("min_score", 0.0))
    folds = []
    for fold in _reported_folds(report):
        period = fold["period"]
        method = fold.get("fusion", {}).get("method")
        conditional = method in {"knn_conditional_residual_quantiles", "online_knn_conditional_residual_quantiles"}
        if conditional:
            weights = fold["fusion"]["source_fusion_weights"]
            offsets = None
        else:
            fusion = json.loads((root / "models" / period / "fusion.json").read_text(encoding="utf-8"))
            calibration = fusion["calibration"]["fusion"]
            offsets = np.asarray(calibration["offsets"], dtype=float)
            weights = fusion.get("winner", {}).get("weights", {})
            recoverable = calibration.get("raw_crossing_rate") == 0 and calibration.get("crossing_rate_after_offsets") == 0
        table = pd.read_parquet(root / "backtests" / period / "fusion_predictions.parquet", columns=["date", "q50"])
        calibrated = table["q50"].to_numpy(dtype=float)
        raw = calibrated - offsets[Q50_INDEX] if offsets is not None and recoverable else None
        if conditional:
            score_evidence = fold["arms"]["fusion"]["score_evidence"]
            raw_distribution = score_evidence["raw_evaluation_q50"]
            raw_qualified = score_evidence["raw_qualified_rows"]
            raw_recovery = "条件 KNN 没有固定偏移；原始分布来自同折原生模型复算，不从校准分数反推。"
        else:
            raw_distribution = _distribution(raw) if raw is not None else None
            raw_qualified = int(np.sum(raw > threshold)) if raw is not None else None
            raw_recovery = "inverse_no_crossing" if recoverable else "分位数曾交叉，需要原生推理，未反推原始分数。"
        contributing = []
        concerns = []
        for arm, weight in weights.items():
            evidence = fold.get("models", {}).get("arms", {}).get(arm)
            if not evidence:
                continue
            trees = evidence.get("winner", {}).get("tree_counts", [])
            tree_count = trees[Q50_INDEX] if len(trees) > Q50_INDEX else None
            path = root / "backtests" / period / f"{arm}_predictions.parquet"
            values = pd.read_parquet(path, columns=["q50"])["q50"] if path.is_file() else []
            item = {"arm": arm, "weight": weight, **_learned_median_health(tree_count, values)}
            if item["collapsed"]:
                concerns.append(f"{arm} q50 仅 {item['evaluation_q50']['unique']} 个不同取值，融合权重 {weight:.0%}。")
            contributing.append(item)
        if 0 < table["q50"].nunique() <= 64:
            concerns.append(f"融合 q50 整折仅 {table['q50'].nunique()} 个不同取值。")
        entry = {"min_score": threshold, "comparison": "strictly_greater", "score": "calibrated_q50"}
        folds.append({
            "period": period, "retrospective_review": period == "historical_replay", "status": "attention" if concerns else "ok",
            "concerns": concerns, "entry": entry, "contributing_models": contributing,
            "raw_q50": raw_distribution,
            "calibrated_q50": _distribution(calibrated),
            "calibration_offset_q50": float(offsets[Q50_INDEX]) if offsets is not None else None,
            "raw_recovery": raw_recovery,
            "qualified_rows": int(np.sum(calibrated > threshold)),
            "raw_qualified_rows": raw_qualified,
            "target_evidence": {stage: fold.get(stage, {}).get("target") for stage in ("train", "validation", "calibration")},
        })
    attention = any(fold["status"] == "attention" for fold in folds)
    return {
        "schema_version": 1, "source_run_id": report.get("run", {}).get("run_id"),
        "status": "attention" if attention else "ok", "selection_scope": "historical_diagnostic",
        "release_eligible": False, "registry_mutated": False,
        "summary": "发现中位模型或融合分数塌缩，需保持研究状态并复核。" if attention else "未发现预设的分数塌缩条件。",
        "folds": folds,
    }


def _boundary(frame: pd.DataFrame, sessions: pd.DatetimeIndex, fold: dict, features: list[str]) -> dict:
    start = pd.Timestamp(fold["evaluation_start"])
    position = sessions.searchsorted(start)
    before_days = sessions[max(0, position - 40):position]
    after_days = sessions[position:position + 20]
    eligible = frame["signal_eligible"].eq(True)
    before = frame.loc[eligible & frame["date"].isin(before_days)]
    after = frame.loc[eligible & frame["date"].isin(after_days)]
    before_daily, after_daily = before.groupby("date")[features].median(), after.groupby("date")[features].median()
    shifts = {}
    for feature in features:
        scale = before_daily[feature].std()
        shift = (after_daily[feature].median() - before_daily[feature].median()) / scale if pd.notna(scale) and scale > 0 else np.nan
        shifts[feature] = {"before_median": _number(before_daily[feature].median()),
                           "after_median": _number(after_daily[feature].median()), "standardized_shift": _number(shift)}
    def mismatch(rows):
        if not {"trade_close", "prev_trade_close", "return_1d"} <= set(rows):
            return None
        raw_return = rows["trade_close"] / rows["prev_trade_close"] - 1
        difference = (rows["return_1d"] - raw_return).abs().replace([np.inf, -np.inf], np.nan).dropna()
        return {"rows": len(difference), "median": _number(difference.median()), "p99": _number(difference.quantile(.99)),
                "over_one_pct": int(difference.gt(.01).sum())}
    return {"before_sessions": len(before_days), "after_sessions": len(after_days), "feature_shifts": shifts,
            "adjusted_vs_trade_return": {"before": mismatch(before), "after": mismatch(after)},
            "selection_use": "retrospective_diagnostic_only"}


def _verified_target(rows: pd.DataFrame, proof: dict) -> dict:
    observed = _distribution(rows["target"])
    if "target" in proof and observed != proof["target"]:
        raise ValueError("归档目标分布与固定面板复算不一致")
    return observed


def run(*, source_run_id: str, run_id: str) -> Path:
    from . import expert_calibration as calibration

    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号无效")
    archive = calibration.open_archive(source_run_id)
    health = inspect_archive(archive.root, archive.report)
    by_period = {fold["period"]: fold for fold in health["folds"]}
    for fold in calibration.research._archived_folds(archive.report):
        target = {}
        for stage in ("train", "validation", "calibration"):
            rows = calibration._settled(archive.data.frame, fold[stage])
            target[stage] = _verified_target(rows, fold[stage])
        evidence = by_period[fold["period"]]
        evidence["target_evidence"] = target
        arm = fold.get("models", {}).get("arms", {}).get("domestic_gpu", {})
        features = list(arm.get("features", []))
        evidence["evaluation_boundary"] = _boundary(archive.data.frame, archive.data.sessions, fold, features)
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    status = root / "status.json"
    write_json(status, {"status": "running", "phase": "score_health"})
    try:
        configuration = {"source_run_id": source_run_id, "prediction_evidence_sha256": archive.report["prediction_evidence_sha256"],
                         "panel_audit": True, "registry_mutated": False}
        context = create_run_context(run_id=run_id, task="expert-score-health", as_of_date=archive.config["end"],
                                     config=configuration, data_version=stable_hash(archive.config["sources"]), seed=archive.config["seed"])
        write_json(context.subdir("config") / "study.json", configuration)
        destination = context.subdir("reports") / REPORT
        write_json(destination, health | {"run": context.to_dict(), "configuration": configuration})
        write_json(status, {"status": "completed", "phase": "complete"})
        return destination
    except Exception as error:
        write_json(status, {"status": "failed", "phase": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(source_run_id=args.source_run_id, run_id=args.run_id or make_run_id("expert-score-health")))


if __name__ == "__main__":
    main()
