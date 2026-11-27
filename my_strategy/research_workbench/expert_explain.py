"""Explain sealed expert q50 models on common eligible evaluation inputs."""

from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.latent_quant.models import _matrix
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from .expert_research import MACRO_FEATURES


MAX_SESSIONS = 96
SHAP_ABSOLUTE_TOLERANCE = 1e-10
_RUN = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,119}")
_STOCK = re.compile(r"[0-9]{6}\.(SH|SZ|BJ)")
_PERIOD = re.compile(r"oos_[1-9][0-9]*")
_EXPERTS = {"domestic_gpu", "overseas_gpu"}
_KEYS = ["date", "stock"]


def _file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"冻结证据文件缺失或路径无效：{relative}")
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _source(store: ExpertArtifactStore, files: dict, config_sources: dict, key: str) -> Path:
    references = [ref for role, ref in files.items() if role.startswith(f"frozen_inputs/{key}/")]
    if len(references) != 1 or references[0] != config_sources.get(key):
        raise ValueError(f"冻结输入 {key} 的哈希与运行配置不一致")
    return store.verify(references[0])


def _eligible_forecasts(root: Path, period: str, arms: dict, symbol: str | None) -> pd.DataFrame:
    filters = [("stock", "=", symbol)] if symbol is not None else None
    common = None
    for arm in ("fusion", *arms):
        path = _file(root, f"backtests/{period}/{arm}_predictions.parquet")
        predicted = pd.read_parquet(path, columns=[*_KEYS, "q50"], filters=filters)
        if predicted.duplicated(_KEYS).any():
            raise ValueError(f"{period}/{arm} 冻结预测包含重复股票日期")
        keys = predicted.loc[np.isfinite(predicted["q50"]), _KEYS]
        common = keys if common is None else common.merge(keys, on=_KEYS, validate="one_to_one")
    return common


def _verified_macro(path: Path) -> pd.DataFrame:
    fields = ["date", *MACRO_FEATURES, "fomc_availability_basis", "fomc_available_at", "cutoff_at"]
    macro = pd.read_parquet(path, columns=fields)
    if macro["date"].duplicated().any():
        raise ValueError("冻结 FOMC 输入存在重复交易日")
    available = pd.to_datetime(macro["fomc_available_at"], utc=True)
    cutoff = pd.to_datetime(macro["cutoff_at"], utc=True)
    allowed = macro["fomc_availability_basis"].eq("official_release_header") & available.le(cutoff)
    allowed &= np.isfinite(macro[list(MACRO_FEATURES)]).all(axis=1)
    return macro.loc[allowed, ["date", *MACRO_FEATURES]]


def _sample(eligible: pd.DataFrame) -> pd.DataFrame:
    """Take one stable stock per date and spread the capped dates across the fold."""
    ordered = eligible.sort_values(_KEYS).assign(_key_order=pd.util.hash_pandas_object(eligible[_KEYS], index=False))
    one_per_day = ordered.sort_values(["date", "_key_order"]).drop_duplicates("date")
    indices = np.linspace(0, len(one_per_day) - 1, min(MAX_SESSIONS, len(one_per_day)), dtype=int)
    return one_per_day.iloc[indices].drop(columns="_key_order")


def _validate_fold(fold: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = pd.Timestamp(fold["evaluation_start"]), pd.Timestamp(fold["evaluation_end"])
    previous = (fold["models"][name] for name in ("train", "validation", "calibration"))
    train, validation, calibration = previous
    chronology = all(pd.Timestamp(left["last_label_end"]) < pd.Timestamp(right) for left, right in (
        (train, validation["start"]), (validation, calibration["start"]), (calibration, start),
    ))
    if not chronology or start > end:
        raise ValueError("冻结训练、验证、校准及解释日期不符合时点边界")
    return start, end


def _explain_model(root: Path, period: str, arm: str, record: dict, sample: pd.DataFrame) -> dict:
    from catboost import CatBoostRegressor, Pool

    relative = f"models/{period}/{arm}/q50.cbm"
    if relative != record["model_files"].get("q50"):
        raise ValueError(f"{period}/{arm} 没有可信的 q50 模型文件引用")
    path = _file(root, relative)
    digest = _sha(path)
    if digest != record["model_file_sha256"].get("q50"):
        raise ValueError(f"{period}/{arm} q50 模型哈希不符")
    features = record["features"]
    model = CatBoostRegressor()
    model.load_model(str(path))
    if features != list(model.feature_names_) or model.get_param("loss_function") != "Quantile:alpha=0.5":
        raise ValueError(f"{period}/{arm} 模型特征顺序或目标与冻结 q50 不符")
    values = Pool(_matrix(sample, features))
    contribution = np.asarray(model.get_feature_importance(values, type="ShapValues", thread_count=4))
    raw = np.asarray(model.predict(values, task_type="CPU", thread_count=4))
    if contribution.shape != (len(sample), len(features) + 1) or raw.shape != (len(sample),):
        raise ValueError(f"{period}/{arm} SHAP 或模型输出维度无效")
    error = float(np.max(np.abs(contribution.sum(axis=1) - raw)))
    if not np.isfinite(error) or error > SHAP_ABSOLUTE_TOLERANCE:
        raise ValueError(f"{period}/{arm} SHAP 未通过对原始 q50 的加和验证")
    means = contribution[:, :-1].mean(axis=0)
    absolute = np.abs(contribution[:, :-1]).mean(axis=0)
    by_feature = sorted(zip(features, means, absolute, strict=True), key=lambda item: (-item[2], item[0]))
    return {
        "status": "ok", "features": features, "model_sha256": digest, "max_additivity_error": error,
        "sample_rows": len(sample), "device": "CPU", "mean_expected_value": float(contribution[:, -1].mean()),
        "top5": [{"feature": feat, "mean_abs_shap": float(magnitude), "mean_shap": float(mean)}
                 for feat, mean, magnitude in by_feature[:5]],
    }


def explain(run_id: str, *, symbol: str | None = None) -> dict:
    """Read a completed run, verify source references, and explain bounded shared samples."""
    if not _RUN.fullmatch(run_id) or (symbol is not None and not _STOCK.fullmatch(symbol)):
        raise ValueError("运行编号或股票代码不符合格式")
    root = artifact_run_dir(run_id, create=False)
    status = _json(_file(root, "status.json"))
    if status.get("status") != "completed" or status.get("run_id") != run_id:
        raise ValueError("只有已完成的专家研究可读取 SHAP")
    report = _json(_file(root, "reports/expert_research.json"))
    config = _json(_file(root, "config/study.json"))
    store = ExpertArtifactStore()
    references = store.read_manifest(run_id)["files"]
    roster = pd.read_parquet(_source(store, references, config["sources"], "roster"), columns=["stock"])
    result = {
        "run_id": run_id, "symbol": symbol, "status": "ok", "reason": None, "output": "raw_q50",
        "sample_policy": "各折共同合格预测股票日；按固定顺序每交易日一只股票，等距抽取至多 96 日。",
        "interpretation": "单专家原始 q50 的 SHAP。最终交易分数另经过分位排序、校准及融合；专家之间的原始 SHAP 不作排名。",
        "point_in_time_verified": bool(report.get("decision", {}).get("point_in_time_verified", False)),
        "additivity_tolerance": SHAP_ABSOLUTE_TOLERANCE, "folds": [],
    }
    if symbol is not None and not roster["stock"].eq(symbol).any():
        return {**result, "status": "not_covered", "reason": "该股票不在本次专家研究的冻结股票池内。"}

    features = {name for fold in report["folds"] for model in fold["models"]["arms"].values() for name in model["features"]}
    columns = [*_KEYS, "signal_eligible", *sorted(features.difference(MACRO_FEATURES))]
    filter_rows = [("stock", "=", symbol)] if symbol is not None else None
    panel = pd.read_parquet(_source(store, references, config["sources"], "panel"), columns=columns, filters=filter_rows)
    panel = panel.loc[panel["signal_eligible"].eq(True)].drop(columns="signal_eligible")
    macro = _verified_macro(_source(store, references, config["sources"], "overseas")) if features.intersection(MACRO_FEATURES) else None
    for fold in report["folds"]:
        period, arms = fold["period"], fold["models"]["arms"]
        if not _PERIOD.fullmatch(period) or not arms or set(arms) - _EXPERTS:
            raise ValueError("该运行的冻结专家或样本外窗口结构无效")
        start, end = _validate_fold(fold)
        evidence = _eligible_forecasts(root, period, arms, symbol)
        if not evidence["date"].between(start, end).all():
            raise ValueError(f"{period} 冻结预测包含样本外窗口之外的日期")
        eligible = evidence.merge(panel, on=_KEYS, validate="one_to_one")
        if "overseas_gpu" in arms:
            if macro is None:
                raise ValueError("海外专家缺少冻结时点数据")
            eligible = eligible.merge(macro, on="date", validate="many_to_one")
        sample = _sample(eligible)
        weights = fold["fusion"]["winner"]["weights"]
        result["folds"].append({
            "period": period, "from": start.date().isoformat(), "to": end.date().isoformat(),
            "status": "ok" if len(sample) else "missing_evidence", "eligible_rows": len(eligible),
            "sample_rows": len(sample), "sample_dates": sample["date"].nunique(),
            "fusion": {"weights": weights, "active_experts": [arm for arm, weight in weights.items() if weight > 0]},
            "experts": {arm: _explain_model(root, period, arm, record, sample) for arm, record in arms.items()} if len(sample) else {},
        })
    ready = sum(fold["status"] == "ok" for fold in result["folds"])
    if not ready:
        result.update(status="missing_evidence", reason="所选股票没有共同的样本外解释样本。")
    elif ready < len(result["folds"]):
        result.update(status="partial", reason="部分窗口缺少共同的样本外解释样本。")
    return result
