"""GPU quantile experts and nested chronological fusion on frozen A-share data."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, replace
from decimal import Decimal, InvalidOperation
from itertools import product
import json
import logging
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.data import BASE_FEATURES
from my_strategy.latent_quant.domestic_runner import (
    BASELINE_ID, MAX_CALIBRATION_DAILY_TURNOVER, MAX_CALIBRATION_DRAWDOWN, FrozenInputs, _load_frozen, _sha,
)
from my_strategy.latent_quant.evaluation import equity_returns, paired_block_interval, score_quantiles
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, execution_evidence_identity, scaled_study_cost, simulate
from my_strategy.latent_quant.sector_features import MODEL_FEATURES as SECTOR_MODEL_FEATURES
from my_strategy.latent_quant.sector_features import attach_model_features, build_native_sector_daily, build_sector_daily
from my_strategy.latent_quant.runner import (
    COSTS, HISTORICAL_END, INITIAL_CASH, PORTFOLIO, PROSPECTIVE_START,
    _features_validation, _summary_table, _verify_ledger, plan_windows,
)
from my_strategy.research_workbench.runner import Window, _write_table, write_json
from my_strategy.storage.expert_artifacts import ExpertArtifactStore

from . import expert_gpu, expert_score_health


LOG = logging.getLogger(__name__)
REPORT = "expert_research.json"
OVERSEAS_ID = "latent-quant-overseas-20261123-v2"
GPU_GRID = ((4, 0.05), (6, 0.07))
GPU_ITERATIONS = 72
MACRO_FEATURES = ("fed_target_lower", "fed_target_upper", "fed_target_change_bps", "fomc_new_statement", "fomc_age_days")
CONTRASTS = {"fusion_vs_historical": "historical_quantiles", "fusion_vs_domestic": "domestic_gpu"}
FULL_CONTRASTS = {
    "domestic_vs_historical": ("domestic_gpu", "historical_quantiles"),
    "domestic_vs_fixed": ("domestic_gpu", "domestic_fixed"),
    "fusion_vs_fixed": ("fusion", "domestic_fixed"),
}
BOOTSTRAP_MAIN = 20
BOOTSTRAP_SENSITIVITY = (10, 30)
SEED = 42
GPU_INFERENCE_TOLERANCE = 1e-7
RISK_ACCEPTANCE = {
    "reference": "domestic_gpu", "maximum_drawdown": MAX_CALIBRATION_DRAWDOWN,
    "maximum_mean_daily_turnover": MAX_CALIBRATION_DAILY_TURNOVER,
}
EXECUTION_FIELDS = {"commission_bps", "rebalance_days", "top_n", "max_weight", "max_industry_weight"}
PREDICTION_EVIDENCE = "prediction_evidence.json"
MATCHED_PROTOCOL = "matched_inner_gpu_v1"


def _normalized_model_spec(spec: dict | None) -> dict:
    from .expert_config import normalize_expert_spec

    model_spec = {key: value for key, value in (spec or {}).items() if key not in EXECUTION_FIELDS}
    return normalize_expert_spec(model_spec or None)


def _execution_spec(spec: dict | None) -> tuple[str, Strategy, StudyCostModel]:
    spec = spec or {}
    try:
        bps = Decimal(str(spec.get("commission_bps", "3")))
    except InvalidOperation as error:
        raise ValueError("佣金的万分比必须为有效数字") from error
    if not bps.is_finite() or bps < 0 or bps >= 10000:
        raise ValueError("佣金的万分比超出研究范围")
    overrides = {key: value for key, value in spec.items() if key in EXECUTION_FIELDS - {"commission_bps"}}
    policy = replace(PORTFOLIO, **overrides)
    if not (1 <= policy.top_n <= 120 and 1 <= policy.rebalance_days <= 60
            and 0 < policy.max_weight <= 1 and 0 < policy.max_industry_weight <= 1):
        raise ValueError("持仓上限、调仓天数或集中度无效")
    return format(bps.normalize(), "f"), policy, replace(StudyCostModel(), commission=float(bps / 10_000))


def _sources() -> dict[str, Path]:
    baseline = artifact_run_dir(BASELINE_ID, create=False)
    overseas = artifact_run_dir(OVERSEAS_ID, create=False)
    paths = {
        "baseline_config": baseline / "config/study.json",
        **{name: baseline / "data_snapshots" / f"{name}.parquet" for name in ("panel", "benchmark", "market_status", "roster")},
    }
    macro = overseas / "data_snapshots/overseas.parquet"
    if macro.is_file():
        paths["overseas"] = macro
    return paths


def research_inputs(*, include_final_fit: bool = False) -> dict:
    """Expose deterministic source and code identity without opening the registry.

    Historical prediction identity uses the default code set.  Final fitting is
    a separate, later contract so producer/inference changes do not make valid
    archived fold predictions look like a different historical experiment.
    """
    code = [Path(__file__), *(Path(models.__file__).with_name(name) for name in (
        "models.py", "runner.py", "execution.py", "evaluation.py", "domestic_runner.py", "data.py", "sector_features.py",
    )), *(Path(__file__).with_name(name) for name in ("expert_config.py", "stock_replay_data.py", "universe.py"))]
    if include_final_fit:
        code.extend(Path(__file__).with_name(name) for name in ("expert_final_fit.py", "expert_forward.py"))
    return {
        "baseline_run_id": BASELINE_ID, "overseas_run_id": OVERSEAS_ID,
        "sources": {name: {"sha256": _sha(path), "size_bytes": path.stat().st_size} for name, path in _sources().items()},
        "code_sha256": {path.name: _sha(path) for path in code},
    }


def _full_freeze(spec: dict) -> tuple[Path, dict]:
    from .expert_full_data import full_freeze_destination, read_expert_full_manifest

    location = Path(os.environ.get("KHQUANT_EXPERT_FULL_FREEZE") or full_freeze_destination(spec))
    if not (location / "manifest.json").is_file():
        raise ValueError(f"全量历史面板尚未封存：{location}")
    return location, read_expert_full_manifest(spec, location)


def prediction_identity(spec: dict | None = None) -> dict:
    """Describe the model-producing inputs; deliberately exclude execution settings."""
    from .expert_config import expert_input_identity

    model_spec = _normalized_model_spec(spec)
    research = research_inputs()
    if model_spec.get("sector_provider") == "ths_fuyao":
        from .fuyao_sector import load_latest_snapshot

        snapshot = load_latest_snapshot()
        research = {**research, "code_sha256": {
            **research["code_sha256"], "fuyao_sector.py": _sha(Path(__file__).with_name("fuyao_sector.py")),
        }, "ths_fuyao_snapshot": {
            "observed_at": snapshot["manifest"]["observed_at"],
            "manifest_sha256": _sha(snapshot["root"] / "manifest.json"),
        }}
    if model_spec["pool"] == "full":
        _, manifest = _full_freeze(model_spec)
        for module in (expert_gpu.__file__, Path(__file__).with_name("expert_full_data.py")):
            file = Path(module)
            research["code_sha256"][file.name] = _sha(file)
        return {"spec": model_spec, "inputs": manifest["input_identity"], "full_freeze": manifest, "research": research}
    return {"spec": model_spec, "inputs": expert_input_identity(model_spec), "research": research}


def prediction_key(spec: dict | None = None) -> str:
    return stable_hash(prediction_identity(spec))


def _verified_sector(data: FrozenInputs, enabled: bool, provider: str = "capco") -> tuple[FrozenInputs, dict]:
    if not enabled:
        return data, {"enabled": False, "candidate_features_used": False}
    if data.benchmark_close is None:
        raise ValueError("板块数值视图缺少同一冻结输入的沪深300收盘价")
    if provider == "ths_fuyao":
        from .fuyao_sector import apply_membership_snapshots, load_latest_snapshot

        snapshot = load_latest_snapshot()
        classified = apply_membership_snapshots(data.frame, snapshot["memberships"])
        daily = build_native_sector_daily(snapshot["daily"], data.benchmark_close, sessions=data.sessions)
        basis = "same_session_ths_native_index_with_observed_membership"
    elif provider == "capco":
        classified = data.frame
        daily = build_sector_daily(data.frame, data.benchmark_close, sessions=data.sessions)
        basis = "same_session_point_in_time_capco_adjusted_equal_weight"
    else:
        raise ValueError("未知板块数据提供方")
    frame = attach_model_features(classified, daily)
    columns = [f"sector_{name}" for name in SECTOR_MODEL_FEATURES]
    if not np.isfinite(frame[columns].to_numpy(dtype=float)).any():
        raise ValueError("板块数值视图没有任何有效的点时行业特征；同花顺当前成分快照不能回填历史。")
    return replace(data, frame=frame, sector_daily=daily), {
        "enabled": True, "provider": provider, "features": columns, "rows": len(daily),
        "industries": int(daily["industry"].nunique()),
        "basis": basis,
        "candidate_features_used": False, "standardization": "none; native CatBoost numeric inputs",
        "unknown_industry": "coverage_only; model values remain missing",
    }


def _verified_macro(data: FrozenInputs, path: Path | None) -> tuple[FrozenInputs, dict]:
    if path is None:
        return data, {"verified_sessions": 0, "sessions": len(data.sessions), "reason": "没有冻结的海外输入"}
    fields = ["date", *MACRO_FEATURES, "fomc_availability_basis", "fomc_available_at", "cutoff_at"]
    macro = pd.read_parquet(path, columns=fields).set_index("date")
    if not macro.index.is_unique:
        raise ValueError("冻结海外数据含重复交易日")
    macro = macro.reindex(data.sessions)
    cutoff = pd.to_datetime(macro["cutoff_at"], utc=True)
    approved = (macro["fomc_availability_basis"].eq("official_release_header")
                & pd.to_datetime(macro["fomc_available_at"], utc=True).le(cutoff)
                & np.isfinite(macro[list(MACRO_FEATURES)]).all(axis=1))
    frame = data.frame.join(macro.loc[approved, list(MACRO_FEATURES)], on="date", validate="many_to_one")
    return replace(data, frame=frame), {
        "verified_sessions": int(approved.sum()), "sessions": len(macro),
        "features": list(MACRO_FEATURES), "basis": "official_release_header",
        "nominal_index_history_used": False,
    }


def plan_expert_calendar(sessions: pd.DatetimeIndex, spec: dict | None) -> list[Window]:
    """Build the requested rolling windows before checking feature and label coverage."""
    if spec is None or spec.get("preset") == "frozen_v2":
        return [window for window in plan_windows(sessions) if window.period.startswith("oos_")]
    if spec.get("pool") == "full":
        if spec.get("start", "2025-03-03") != "2025-03-03" or spec.get("end", HISTORICAL_END) != HISTORICAL_END:
            raise ValueError("本轮全池冠军研究使用锁定的 2025-03-03 至 2026-11-20 历史边界")
        if not len(sessions) or sessions[-1] != pd.Timestamp(HISTORICAL_END):
            raise ValueError("全量封存日历必须在 2026-11-20 截止")
        return plan_windows(sessions)
    evaluate = sessions[(sessions >= pd.Timestamp(spec["start"])) & (sessions <= pd.Timestamp(spec["end"]))]
    if len(evaluate) < 3 * max(BOOTSTRAP_SENSITIVITY):
        raise ValueError("自选窗口须有至少 90 个实际交易日，供三个不交叠外折及 30 日分块检验")
    windows = []
    for number, block in enumerate(np.array_split(evaluate.to_numpy(), 3), start=1):
        dates = pd.DatetimeIndex(block)
        offset = sessions.get_loc(dates[0])
        if offset < 70:
            raise ValueError("自选窗口前缺少独立训练、验证及校准交易日")
        windows.append(Window(f"oos_{number}", sessions[max(0, offset - 315)], sessions[offset - 70], sessions[offset - 35], dates))
    return windows


def plan_expert_windows(data: FrozenInputs, spec: dict | None) -> list[Window]:
    """Check actual mature stock days against the calendar-only plan."""
    windows = plan_expert_calendar(data.sessions, spec)
    if spec is None or spec.get("preset") == "frozen_v2":
        return windows
    settled = data.frame["signal_eligible"].eq(True) & np.isfinite(data.frame["target"]) & data.frame["label_end"].gt(data.frame["date"])
    for window in windows:
        train, valid, cal = [data.frame.loc[mask & settled, "date"] for mask in window.masks(data.frame)]
        if train.nunique() < 100 or valid.empty or cal.empty:
            raise ValueError(f"{window.period} 缺少 100 个成熟训练日，或缺独立验证/校准样本")
        mature = data.frame["date"].isin(window.evaluate) & data.frame["label_end"].le(window.evaluate[-1]) & settled
        if not mature.any():
            raise ValueError(f"{window.period} 缺少窗口内成熟标签")
    return windows


def _fit_gpu(train: pd.DataFrame, valid: pd.DataFrame | None, features: tuple[str, ...], params: tuple[int, float], seed: int):
    from catboost import CatBoostRegressor, Pool

    train_x = models._matrix(train, features)
    validation = None
    if valid is not None:
        weights = 1.0 / valid.groupby("date")["date"].transform("size")
        validation = Pool(models._matrix(valid, features), valid["target"], weight=weights)
    fitted = []
    for alpha in models.QUANTILE_LEVELS:
        model = CatBoostRegressor(
            loss_function=f"Quantile:alpha={alpha}", iterations=GPU_ITERATIONS, depth=params[0], learning_rate=params[1],
            task_type="GPU", devices="0", random_seed=seed, allow_writing_files=False, verbose=False,
        )
        kwargs = {"eval_set": validation, "use_best_model": True, "early_stopping_rounds": 12} if validation is not None else {}
        model.fit(train_x, train["target"], **kwargs)
        fitted.append(model)
    return fitted


def _inference_audit() -> dict:
    return {"requested_device": "GPU", "calls": {"GPU": 0, "CPU": 0}, "fallback_reasons": {},
            "max_absolute_error": None, "absolute_tolerance": GPU_INFERENCE_TOLERANCE}


def _predict(fitted: list, rows: pd.DataFrame, features: tuple[str, ...], audit: dict) -> np.ndarray:
    from catboost import CatBoostError

    matrix = models._matrix(rows, features)
    columns = []
    for model in fitted:
        fallback = None
        try:
            gpu = model.predict(matrix, task_type="GPU", thread_count=4)
        except CatBoostError as error:
            gpu = None
            fallback = str(error).strip()
        cpu = model.predict(matrix, task_type="CPU", thread_count=4)
        if gpu is not None:
            difference = float(np.max(np.abs(gpu - cpu)))
            if np.isfinite(difference):
                audit["max_absolute_error"] = max(difference, audit["max_absolute_error"] or 0)
            if not np.isfinite(difference) or difference > GPU_INFERENCE_TOLERANCE or not np.array_equal(gpu > 0, cpu > 0):
                fallback = "GPU/CPU 数值或零阈值方向不一致"
        audit["calls"]["CPU" if fallback else "GPU"] += 1
        if fallback:
            audit["fallback_reasons"][fallback] = audit["fallback_reasons"].get(fallback, 0) + 1
        columns.append(cpu if fallback else gpu)
    raw = np.column_stack(columns)
    if not np.isfinite(raw).all():
        raise ValueError("GPU 训练模型返回非有限分位数")
    return raw


def _pinball(rows: pd.DataFrame, predicted: np.ndarray) -> float:
    table = pd.DataFrame(predicted, index=rows.index, columns=models.QUANTILE_COLUMNS)
    return float(score_quantiles(rows[["date", "target"]].join(table))["pinball"])


def _native_trial_key(metrics: dict) -> tuple[bool, float]:
    """Validation health is an eligibility criterion, not a post-selection warning."""
    median = expert_score_health._learned_median_health(
        metrics["tree_counts"][1], metrics.get("validation_q50", []),
    )
    return median["collapsed"], metrics["validation_pinball"]


def _tune(train: pd.DataFrame, valid: pd.DataFrame, features: tuple[str, ...], seed: int, audit: dict) -> tuple[list, dict]:
    trials = []
    best, winner = None, None
    for params in GPU_GRID:
        fitted = _fit_gpu(train, valid, features, params, seed)
        predicted = _predict(fitted, valid, features, audit)
        metrics = {
            "depth": params[0], "learning_rate": params[1], "iterations_limit": GPU_ITERATIONS,
            "tree_counts": [int(model.tree_count_) for model in fitted],
            "validation_pinball": _pinball(valid, predicted),
            "validation_q50": expert_score_health._distribution(predicted[:, 1]),
        }
        trials.append(metrics)
        if winner is None or _native_trial_key(metrics) < _native_trial_key(winner):
            best, winner = fitted, metrics
    return best, {"features": list(features), "candidates": trials, "winner": winner, "training_device": "GPU:0"}


def _tune_full(train, valid, features, seed, audit, scheduler, *, fixed=False, repeat_winner=True) -> tuple[list, dict]:
    trials = []
    best, winner = None, None
    grid = (expert_gpu.FIXED,) if fixed else expert_gpu.GRID
    iterations, patience = (GPU_ITERATIONS, 12) if fixed else (expert_gpu.ITERATIONS, expert_gpu.PATIENCE)
    for params in grid:
        fitted, jobs = scheduler.fit(train, valid, features, params, seed, iterations=iterations, patience=patience)
        predicted = _predict(fitted, valid, features, audit)
        metrics = {
            "depth": params[0], "learning_rate": params[1], "l2_leaf_reg": params[2],
            "iterations_limit": iterations, "early_stopping_rounds": patience, "seed": seed,
            "tree_counts": [entry["trees"] for entry in jobs], "gpu_jobs": jobs,
            "validation_pinball": _pinball(valid, predicted),
            "validation_q50": expert_score_health._distribution(predicted[:, 1]),
        }
        trials.append(metrics)
        LOG.info("%s features=%s seed=%s parameters=%s validation_pinball=%.8f",
                 "固定参数" if fixed else "全池调参", len(features), seed, params, metrics["validation_pinball"])
        if winner is None or _native_trial_key(metrics) < _native_trial_key(winner):
            best, winner = fitted, metrics
    evidence = {"features": list(features), "candidates": trials, "winner": winner, "training_device": "GPU:0,1"}
    if not fixed and repeat_winner:
        params = (winner["depth"], winner["learning_rate"], winner["l2_leaf_reg"])
        repeat_seed = (seed + 1) % (2**32 - 1)
        check, jobs = scheduler.fit(train, valid, features, params, repeat_seed, iterations=iterations, patience=patience)
        repeated = _pinball(valid, _predict(check, valid, features, audit))
        evidence["seed_stability"] = {"seed": repeat_seed, "validation_pinball": repeated,
                                      "difference": repeated - winner["validation_pinball"], "gpu_jobs": jobs,
                                      "used_for_selection": False}
    return best, evidence


def _rules(train: pd.DataFrame, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    pooled = train["target"].quantile(list(models.QUANTILE_LEVELS)).to_numpy()
    return {
        "historical_quantiles": models._historical_quantiles(train, rows, pooled)[0],
        "supertrend_atr": models._trend_quantiles(train, rows, pooled)[0],
    }


def _inner_windows(train: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    dates = pd.DatetimeIndex(sorted(train["date"].unique()))
    if len(dates) < 100:
        raise ValueError("联合折内 OOF 需要至少 100 个可训练交易日")
    blocks = (dates[-80:-40], dates[-40:])
    return [
        (train.loc[train["date"].lt(block[0]) & train["label_end"].lt(block[0])], train.loc[train["date"].isin(block)])
        for block in blocks
    ]


def _inner_fit_split(past: pd.DataFrame, unseen: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(sorted(past["date"].unique()))
    if len(dates) < 70:
        raise ValueError("匹配拟合缺少外部训练前史和独立早停窗口")
    split = dates[-29]
    start = unseen["date"].min()
    fit = past.loc[past["date"].lt(split) & past["label_end"].lt(split)]
    valid = past.loc[past["date"].ge(split) & past["date"].lt(start) & past["label_end"].lt(start)]
    if fit.empty or valid["date"].nunique() != 29:
        raise ValueError("匹配拟合需要独立的 29 个成熟验证交易日")
    return fit, valid


def _expert_raw(fitted: list, rows: pd.DataFrame, features: tuple[str, ...], audit: dict, *, fallback: np.ndarray | None = None) -> np.ndarray:
    if fallback is None:
        return _predict(fitted, rows, features, audit)
    present = rows[list(MACRO_FEATURES)].notna().all(axis=1).to_numpy()
    result = fallback.copy()
    if present.any():
        result[present] = _predict(fitted, rows.loc[present], features, audit)
    return result


def _fuse(weights: dict[str, float], forecasts: dict[str, np.ndarray]) -> np.ndarray:
    return sum(weight * np.sort(forecasts[arm], axis=1) for arm, weight in weights.items())


def _inner_fusion(train: pd.DataFrame, features: dict, seed: int, devices: dict, scheduler=None, *, matched_fit=False) -> tuple[dict, dict, pd.DataFrame]:
    collected, boundaries, labels = {}, [], []
    for past, unseen in _inner_windows(train):
        if past.empty or unseen.empty:
            raise ValueError("折内 OOF 子窗没有足够的历史或预测股票日")
        fit, valid = _inner_fit_split(past, unseen) if matched_fit else (past, None)
        raw = _rules(fit, unseen)
        gpu_jobs = {}
        for arm, columns in features.items():
            source = fit if arm == "domestic_gpu" else fit.dropna(subset=list(MACRO_FEATURES))
            if matched_fit:
                validation = valid if arm == "domestic_gpu" else valid.dropna(subset=list(MACRO_FEATURES))
                model, gpu_jobs[arm] = _tune_full(source, validation, columns, seed, devices[arm], scheduler, repeat_winner=False)
            elif scheduler is None:
                model = _fit_gpu(source, None, columns, GPU_GRID[0], seed)
            else:
                model, gpu_jobs[arm] = scheduler.fit(source, None, columns, expert_gpu.GRID[0], seed, iterations=GPU_ITERATIONS)
            raw[arm] = _expert_raw(model, unseen, columns, devices[arm], fallback=raw.get("domestic_gpu") if arm == "overseas_gpu" else None)
        for arm, values in raw.items():
            collected.setdefault(arm, []).append(values)
        labels.append(unseen[["date", "stock", "label_end", "target"]])
        boundaries.append({"trained": models._sample_evidence(fit), "out_of_fold": models._sample_evidence(unseen),
                           **({"validation": models._sample_evidence(valid)} if matched_fit else {}),
                           **({"gpu_jobs": gpu_jobs} if scheduler else {})})
    observed = pd.concat(labels)
    forecasts = {arm: np.vstack(group) for arm, group in collected.items()}
    expert_names = ("supertrend_atr", *features)
    grids = [parts for parts in product(range(5), repeat=len(expert_names)) if sum(parts) == 4 and parts[1] >= 2]
    trials = []
    for points in grids:
        weights = {arm: weight / 4 for arm, weight in zip(expert_names, points, strict=True) if weight}
        trials.append({"weights": weights, "inner_oof_pinball": _pinball(observed, _fuse(weights, forecasts))})
    winner = min(trials, key=lambda row: (row["inner_oof_pinball"], len(row["weights"])))
    export = observed.copy()
    for arm, values in forecasts.items():
        for index, column in enumerate(models.QUANTILE_COLUMNS):
            export[f"{arm}_{column}"] = values[:, index]
    protocol = ({"fit_protocol": MATCHED_PROTOCOL, "expert_grid": expert_gpu.GRID,
                 "iterations": expert_gpu.ITERATIONS, "early_stopping_rounds": expert_gpu.PATIENCE}
                if matched_fit else {"expert_grid_fixed_at": expert_gpu.GRID[0] if scheduler else GPU_GRID[0]})
    return winner["weights"], {"subfolds": boundaries, "candidates": trials, "winner": winner, **protocol}, export


def _paired(folds: list[dict], block_size: int, seed: int = SEED) -> dict:
    pairs = {name: ("fusion", reference) for name, reference in CONTRASTS.items()}
    if all("domestic_fixed" in fold["double"] for fold in folds):
        pairs.update(FULL_CONTRASTS)
    return {name: {cost: paired_block_interval(
        [(fold[cost][candidate], fold[cost][reference]) for fold in folds], seed=seed, block_size=block_size,
    ) for cost in COSTS} for name, (candidate, reference) in pairs.items()}


def _ablations(weights: dict, features: dict) -> dict[str, dict]:
    available = ("supertrend_atr", *features)
    orders = {"fusion_equal": {arm: 1 / len(available) for arm in available}}
    removals = {"fusion_no_trend": "supertrend_atr", "fusion_no_overseas": "overseas_gpu", "fusion_no_sector": "sector_gpu"}
    for arm, removed in removals.items():
        if removed in available:
            remaining = {name: value for name, value in weights.items() if name != removed}
            orders[arm] = {name: value / sum(remaining.values()) for name, value in remaining.items()}
    return {arm: {"weights": spec, "changes_weights": spec != weights, "extra_models_fitted": False} for arm, spec in orders.items()}


def _paired_ablations(folds: list[dict], seed: int = SEED) -> dict:
    arms = sorted(set.intersection(*(set(fold["double"]) for fold in folds)))
    return {arm: {cost: paired_block_interval(
        [(fold[cost]["fusion"], fold[cost][arm]) for fold in folds], seed=seed, block_size=BOOTSTRAP_MAIN,
    ) for cost in COSTS} for arm in arms if arm.startswith("fusion_")}


def _execution(
    data: FrozenInputs, prediction: pd.Series, dates: pd.DatetimeIndex, context, backtests: Path, arm: str,
    policy: Strategy, cost_model: StudyCostModel | None,
    target_exposure: pd.Series | None = None,
) -> tuple[dict, dict]:
    execution, returns = {}, {}
    for cost, multiplier in COSTS.items():
        outcome = simulate(
            data.frame, prediction, dates, data.benchmark_returns, run_id=context.run_id,
            cost_multiplier=multiplier, initial_cash=INITIAL_CASH, strategy=policy,
            market_status=data.market_status, roster=data.roster, base_cost_model=cost_model,
            target_exposure=target_exposure,
        )
        verified = _verify_ledger(outcome, dates)
        execution[cost] = dict(outcome["metrics"], verification=verified)
        returns[cost] = equity_returns(outcome["daily"], INITIAL_CASH)
        for name in ("daily", "orders", "trades", "positions"):
            _write_table(backtests / f"{arm}_{cost}_{name}.parquet", outcome[name])
        _write_table(backtests / f"{arm}_{cost}_selection_daily.parquet", outcome["selection_daily"])
        _write_table(backtests / f"{arm}_{cost}_selection_candidates.parquet", outcome["selection_candidates"])
        _write_table(backtests / f"{arm}_{cost}_sector_candidate_daily.parquet", outcome["sector_candidate_daily"])
    return execution, returns


def _fold(
    data: FrozenInputs, window: Window, context, features: dict, *,
    seed: int = SEED, policy: Strategy = PORTFOLIO, cost_model: StudyCostModel | None = None, scheduler=None, matched_fit=False,
    include_raw_quantiles: bool = False,
) -> tuple[dict, dict]:
    frame = data.frame
    eligible = frame["signal_eligible"].eq(True)
    settled = eligible & np.isfinite(frame["target"]) & frame["label_end"].gt(frame["date"])
    train, valid, cal = [frame.loc[mask & settled] for mask in window.masks(frame)]
    evaluation = frame.loc[eligible & frame["date"].isin(window.evaluate)]
    if any(rows.empty for rows in (train, valid, cal, evaluation)):
        raise ValueError(f"{window.period} 在训练、验证、校准或评估段缺少合格样本")
    backtests = context.subdir("backtests", window.period)
    devices = {arm: _inference_audit() for arm in features}
    weights, joint, oof = (_inner_fusion(train, features, seed, devices) if scheduler is None
                           else _inner_fusion(train, features, seed, devices, scheduler, matched_fit=matched_fit))
    _write_table(backtests / "inner_oof.parquet", oof)
    rows = pd.concat([cal, evaluation])
    raw = _rules(train, rows)
    model_evidence, calibration = {}, {}
    fitted_features = dict(features)
    if scheduler is not None:
        fitted_features["domestic_fixed"] = features["domestic_gpu"]
        devices["domestic_fixed"] = _inference_audit()
    for arm, columns in fitted_features.items():
        source = train.dropna(subset=list(MACRO_FEATURES)) if arm == "overseas_gpu" else train
        validation = valid.dropna(subset=list(MACRO_FEATURES)) if arm == "overseas_gpu" else valid
        fitted, details = (_tune(source, validation, columns, seed, devices[arm]) if scheduler is None
                           else _tune_full(source, validation, columns, seed, devices[arm], scheduler, fixed=arm == "domestic_fixed"))
        raw[arm] = _expert_raw(fitted, rows, columns, devices[arm], fallback=raw.get("domestic_gpu") if arm == "overseas_gpu" else None)
        folder = context.subdir("models", window.period, arm)
        for name, model in zip(models.QUANTILE_COLUMNS, fitted, strict=True):
            model.save_model(str(folder / f"{name}.cbm"), format="cbm")
        median = expert_score_health._learned_median_health(
            details["winner"]["tree_counts"][1], details["winner"].get("validation_q50", []),
        )
        model_evidence[arm] = dict(
            details, model_files={name: f"models/{window.period}/{arm}/{name}.cbm" for name in models.QUANTILE_COLUMNS},
            model_file_sha256={name: _sha(folder / f"{name}.cbm") for name in models.QUANTILE_COLUMNS},
            inference=devices[arm], validation_target=expert_score_health._distribution(validation["target"]),
            validation_median_collapsed=median["collapsed"],
        )
    raw["fusion"] = _fuse(weights, raw)
    ablations = _ablations(weights, features)
    for arm, specification in ablations.items():
        raw[arm] = _fuse(specification["weights"], raw)
    common = evaluation[["date", "stock", "trade_date", "label_end", "target"]].copy()
    common["scored"] = common["target"].notna() & common["label_end"].le(window.evaluate[-1])
    common["target"] = common["target"].where(common["scored"])
    if not common["scored"].any():
        raise ValueError(f"{window.period} 没有成熟的共同股票日标签")
    arms, returns = {}, {cost: {} for cost in COSTS}
    for arm, values in raw.items():
        quantiles, calibration[arm] = models._calibrate(values, cal["target"].to_numpy())
        prediction = models._prediction_table(quantiles, evaluation.index)
        if include_raw_quantiles:
            prediction = prediction.join(pd.DataFrame(
                values[len(cal):], index=evaluation.index,
                columns=[f"raw_{column}" for column in models.QUANTILE_COLUMNS],
            ))
        if arm == "supertrend_atr":
            prediction["score"] = models._trend_score(evaluation)
        joined = common.join(prediction)
        measures = score_quantiles(joined)
        measures["all_signal_rows"] = len(joined)
        raw_evaluation_q50 = values[len(cal):, 1]
        if policy.min_score is None:
            raw_qualified = int(np.isfinite(raw_evaluation_q50).sum())
            qualified = int(np.isfinite(quantiles[:, 1]).sum())
            comparison = "finite_cross_sectional_rank"
        else:
            raw_qualified = int(np.sum(raw_evaluation_q50 > policy.min_score))
            qualified = int(np.sum(quantiles[:, 1] > policy.min_score))
            comparison = "strictly_greater"
        score_evidence = {
            "raw_calibration_q50": expert_score_health._distribution(values[:len(cal), 1]),
            "raw_evaluation_q50": expert_score_health._distribution(raw_evaluation_q50),
            "calibrated_evaluation_q50": expert_score_health._distribution(quantiles[:, 1]),
            "raw_qualified_rows": raw_qualified,
            "qualified_rows": qualified,
            "entry": {"score": "calibrated_q50", "min_score": policy.min_score, "comparison": comparison,
                      "selection_mode": policy.selection_mode},
        }
        _write_table(backtests / f"{arm}_predictions.parquet", joined)
        execution, timeseries = _execution(data, prediction["score"], window.evaluate, context, backtests, arm, policy, cost_model)
        arms[arm] = {"prediction": measures, "score_evidence": score_evidence, "execution": execution}
        for cost in COSTS:
            returns[cost][arm] = timeseries[cost]
    for arm, details in model_evidence.items():
        write_json(context.subdir("models", window.period, arm) / "model.json", dict(details, calibration=calibration[arm]))
    write_json(context.subdir("models", window.period) / "fusion.json", dict(joint, calibration=calibration, ablations=ablations))
    timing = {
        stage: {**models._sample_evidence(rows), "target": expert_score_health._distribution(rows["target"])}
        for stage, rows in (("train", train), ("validation", valid), ("calibration", cal))
    }
    failed_medians = sorted(arm for arm, details in model_evidence.items()
                            if weights.get(arm, 0) > 0 and details.get("validation_median_collapsed"))
    model_health = {"passed": not failed_medians, "failed_experts": failed_medians,
                    "basis": "validation_only", "rule": "weighted_single_tree_median_with_at_most_16_unique_scores"}
    return {
        "period": window.period, "evaluation_start": window.evaluate[0], "evaluation_end": window.evaluate[-1],
        "evaluation_sessions": len(window.evaluate), **timing,
        "models": {**timing, "arms": model_evidence}, "fusion": joint, "arms": arms, "contrasts": _paired([returns], BOOTSTRAP_MAIN, seed),
        "ablations": ablations, "inference": devices, "model_health": model_health,
    }, returns


def _risk_checks(folds: list[dict], candidate: str = "fusion", reference: str = "domestic_gpu", *, explicit_evidence=False) -> dict:
    checks = {}
    for fold in folds:
        arms = fold["arms"]
        checks[fold["period"]] = {}
        for cost in COSTS:
            fused, main = (arms[arm]["execution"][cost] for arm in (candidate, reference))
            measured = all(arm["order_count"] > 0 and arm["reject_rate"] is not None for arm in (fused, main))
            measure = {
                "drawdown_pass": fused["max_drawdown"] + 1e-12 >= max(main["max_drawdown"], -RISK_ACCEPTANCE["maximum_drawdown"]),
                "turnover_pass": fused["mean_daily_turnover"] <= 1e-12 + min(main["mean_daily_turnover"], RISK_ACCEPTANCE["maximum_mean_daily_turnover"]),
                "reject_rate_pass": (None if explicit_evidence and not measured
                                     else (fused["reject_rate"] or 0) <= (main["reject_rate"] or 0) + 1e-12),
            }
            checks[fold["period"]][cost] = {**measure, "passed": all(value for value in measure.values() if value is not None),
                                           "fusion_submitted": fused["order_count"], "domestic_submitted": main["order_count"]}
            if explicit_evidence:
                checks[fold["period"]][cost]["execution_evidence"] = "measured" if measured else "insufficient"
    result = {"passed": all(cost["passed"] for fold in checks.values() for cost in fold.values()),
              "thresholds": {**RISK_ACCEPTANCE, "reference": reference}, "folds": checks}
    if explicit_evidence:
        support = {cost: sum(fold[cost]["execution_evidence"] == "measured" for fold in checks.values()) for cost in COSTS}
        result.update(execution_evidence_folds=support, minimum_execution_evidence_folds=2)
        result["passed"] = result["passed"] and min(support.values()) >= 2
    return result


def _decision(folds: list[dict], pooled: dict, point_in_time: bool) -> dict:
    checks = {}
    for contrast, reference in CONTRASTS.items():
        checks[contrast] = {
            "pinball_wins": sum(fold["arms"]["fusion"]["prediction"]["pinball"] < fold["arms"][reference]["prediction"]["pinball"] for fold in folds),
            "double_cost_wins": sum(fold["contrasts"][contrast]["double"]["net_return_difference"] > 0 for fold in folds),
            "double_cost_ci_above_zero": pooled[contrast]["double"]["ci95_low"] > 0,
        }
    evidence_pass = len(folds) == 3 and all(x["pinball_wins"] >= 2 and x["double_cost_wins"] >= 2 and x["double_cost_ci_above_zero"] for x in checks.values())
    risk = _risk_checks(folds)
    median_health = all(fold.get("model_health", {}).get("passed", True) for fold in folds)
    reasons = ["一键研究尚无独立封存的 60 日多专家前瞻留出。"]
    if not point_in_time:
        reasons.append("国内历史归档的首次可用时间与供应商修订尚未完整验证。")
    if not median_health:
        reasons.append("参与融合的中位模型在验证段退化为单树且分数塌缩，历史数值验收不予通过。")
    if not evidence_pass:
        reasons.append("融合尚未同时通过相对历史分位数和国内单专家的开发折预测及双倍成本门槛。")
    if not risk["passed"]:
        reasons.append("至少一个窗口或成本档的回撤、均日换手或拒单率超出预定风险门槛。")
    return {
        "state": "research_only", "reasons": reasons, "historical_numerical_gate": evidence_pass and risk["passed"] and median_health,
        "median_model_health_gate": median_health,
        "point_in_time_verified": point_in_time, "qualified_forward_results": False, "advance_tcn": False,
        "comparisons": checks, "risk_acceptance": risk, "paper_model": {"state": "blocked", "artifact": None},
    }


def _single_champion(folds: list[dict], pooled: dict, *, explicit_evidence=False) -> dict:
    ranking = []
    for arm, pairs in (
        ("domestic_gpu", ("domestic_vs_historical", "domestic_vs_fixed")),
        ("fusion", ("fusion_vs_historical", "fusion_vs_fixed")),
    ):
        references = {**{name: pair[1] for name, pair in FULL_CONTRASTS.items()}, **CONTRASTS}
        comparisons = {}
        for name in pairs:
            reference = references[name]
            comparisons[name] = {
                "pinball_wins": sum(fold["arms"][arm]["prediction"]["pinball"] < fold["arms"][reference]["prediction"]["pinball"] for fold in folds),
                "double_cost_wins": sum(fold["contrasts"][name]["double"]["net_return_difference"] > 0 for fold in folds),
                "double_cost_ci_lower": pooled[name]["double"]["ci95_low"],
            }
        risk = _risk_checks(folds, arm, "domestic_fixed", explicit_evidence=explicit_evidence)
        lower = min(item["double_cost_ci_lower"] for item in comparisons.values())
        ranking.append({
            "arm": arm, "historical_gate": len(folds) == 3 and risk["passed"] and lower > 0 and all(
                item["pinball_wins"] >= 2 and item["double_cost_wins"] >= 2 for item in comparisons.values()),
            "risk_pass": risk["passed"], "risk": risk, "minimum_double_cost_ci_lower": lower,
            "mean_pinball": float(np.mean([fold["arms"][arm]["prediction"]["pinball"] for fold in folds])),
            "mean_daily_turnover": float(np.mean([fold["arms"][arm]["execution"]["double"]["mean_daily_turnover"] for fold in folds])),
            "comparisons": comparisons,
        })
    ranking.sort(key=lambda row: (not row["historical_gate"], -row["minimum_double_cost_ci_lower"], row["mean_pinball"],
                                  row["mean_daily_turnover"], row["arm"] != "domestic_gpu"))
    winner = ranking[0]
    return {"selected_arm": winner["arm"], "state": "historical_candidate" if winner["historical_gate"] else "best_research_only",
            "release_eligible": False, "ranking": ranking, "scope": "oos_1_to_oos_3", "prospective_ready": False}


def _final_portfolio(root: Path, folds: list[dict], latest: str, arm: str = "fusion") -> dict:
    artifacts = {name: f"backtests/{latest}/{arm}_normal_{name}.parquet" for name in ("daily", "orders", "trades", "positions")}
    artifacts["predictions"] = f"backtests/{latest}/{arm}_predictions.parquet"
    submitted = pd.read_parquet(root / artifacts["orders"])
    submitted = submitted.loc[submitted["requested_shares"].gt(0)]
    last_signal = submitted["signal_date"].max() if not submitted.empty else None
    latest_actions = [{
        "symbol": row.stock, "signal_date": row.signal_date, "execution_date": row.date,
        "action": row.side, "target_weight": float(row.target_weight), "reason": row.reason or "",
        "requested_shares": int(row.requested_shares), "filled_shares": int(row.filled_shares), "status": row.status,
    } for row in submitted.loc[submitted["signal_date"].eq(last_signal)].head(50).itertuples(index=False)]
    metrics = {}
    for cost in COSTS:
        pieces = [fold["arms"][arm]["execution"][cost] for fold in folds]
        orders = sum(piece["order_count"] for piece in pieces)
        metrics[cost] = {
            "total_return": float(np.prod([1 + piece["total_return"] for piece in pieces]) - 1),
            "max_drawdown": min(piece["max_drawdown"] for piece in pieces), "turnover": sum(piece["turnover"] for piece in pieces),
            "reject_rate": sum(piece["rejected_order_count"] for piece in pieces) / orders if orders else None,
        }
    return {
        "arm": arm, "fold": latest, "cost": "normal", "channel": "research", "artifacts": artifacts,
        "latest_action_date": last_signal, "latest_actions": latest_actions, "order_count": len(submitted), "metrics": metrics,
        "action_scope": "最近开发折已模拟的最后申报日，非当天信号。",
        "metric_scope": "三折从现金出发并复合净收益；回撤取最差折，换手相加。",
    }


def _fixed_references(store: ExpertArtifactStore, identity: dict) -> dict:
    references = {f"frozen_inputs/{name}/{path.name}": store.put_file(path) for name, path in _sources().items()}
    for role, ref in references.items():
        name = role.split("/")[1]
        if identity["sources"][name] != asdict(ref):
            raise ValueError(f"冻结输入在研究运行中变化：{name}")
    return references


def _research_sector_artifacts(root: Path, data: FrozenInputs, folds: list[dict], arm: str, provider: str = "capco") -> dict:
    from .expert_sector import write_sector_artifacts

    selections, candidates = [], []
    for fold in folds:
        folder = root / "backtests" / fold["period"]
        selection_path = folder / f"{arm}_normal_selection_candidates.parquet"
        candidate_path = folder / f"{arm}_normal_sector_candidate_daily.parquet"
        if not selection_path.is_file() or not candidate_path.is_file():
            return {"status": "unavailable", "reason": "该研究折未封存板块候选明细；不重写预测或交易账本。"}
        selections.append(pd.read_parquet(selection_path))
        candidates.append(pd.read_parquet(candidate_path))
    return write_sector_artifacts(
        root, data.frame, data.benchmark_close,
        pd.concat(selections, ignore_index=True), pd.concat(candidates, ignore_index=True), provider=provider,
    )


def _custom_references(store: ExpertArtifactStore, data: FrozenInputs, identity: dict) -> dict:
    shared = {name: path for name, path in _sources().items() if name in {"baseline_config", "overseas"}}
    references = {f"frozen_inputs/{name}/{path.name}": store.put_file(path) for name, path in shared.items()}
    for role, ref in references.items():
        if identity["sources"][role.split("/")[1]] != asdict(ref):
            raise ValueError("自定义研究的共享数据在任务中变化")
    benchmark = pd.read_parquet(_sources()["benchmark"])
    benchmark = benchmark.loc[benchmark["date"].isin(data.sessions)]
    if not pd.DatetimeIndex(benchmark["date"]).equals(data.sessions):
        raise ValueError("自定义面板缺少对应的冻结基准")
    with TemporaryDirectory(prefix="khquant-expert-inputs-") as directory:
        for name, table in {
            "panel": data.frame, "benchmark": benchmark, "roster": data.roster.reset_index(), "market_status": data.market_status,
        }.items():
            path = Path(directory) / f"{name}.parquet"
            _write_table(path, table)
            references[f"frozen_inputs/{name}/{path.name}"] = store.put_file(path)
    return references


def _full_references(store: ExpertArtifactStore, root: Path, manifest: dict, identity: dict) -> dict:
    shared = {name: path for name, path in _sources().items() if name in {"baseline_config", "overseas"}}
    paths = {**shared, **{name: root / f"{name}.parquet" for name in ("panel", "benchmark", "roster", "market_status")}}
    references = {}
    for name, path in paths.items():
        expected = identity["sources"][name] if name in shared else manifest["files"][path.name]
        reference = store.put_file(path)
        if asdict(reference) != expected:
            raise ValueError(f"全池输入的封存指纹在运行中改变：{name}")
        references[f"frozen_inputs/{name}/{path.name}"] = reference
    return references


def _reference_values(references: dict) -> dict:
    return {role.split("/")[1]: (ref if isinstance(ref, dict) else asdict(ref)) for role, ref in references.items()}


def _data_by_reference(store: ExpertArtifactStore, references: dict, coverage: dict) -> FrozenInputs:
    by_name = _reference_values(references)
    panel, benchmark, roster, market_status = (pd.read_parquet(store.verify(by_name[name])) for name in (
        "panel", "benchmark", "roster", "market_status",
    ))
    benchmark = benchmark.set_index("date")
    return FrozenInputs(panel, pd.DatetimeIndex(benchmark.index), benchmark["daily_return"], market_status,
                        roster.set_index("stock"), coverage, benchmark["close"])


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_evidence(root: Path, folds: list[dict]) -> dict:
    files = []
    for fold in folds:
        period = fold["period"]
        files.extend((root / "models" / period).rglob("*"))
        files.extend((root / "backtests" / period / f"{arm}_predictions.parquet") for arm in fold["arms"])
        files.append(root / "backtests" / period / "inner_oof.parquet")
    return {path.relative_to(root).as_posix(): {"sha256": _sha(path), "size_bytes": path.stat().st_size}
            for path in sorted(files) if path.is_file()}


def _archived_folds(report: dict) -> list[dict]:
    return [*report["folds"], *([report["historical_review"]] if report.get("historical_review") else [])]


def _reuse_source(run_id: str, key: str) -> tuple[Path, dict, dict, dict]:
    if run_id != slugify(run_id, limit=120):
        raise ValueError("预测来源运行编号无效")
    source = artifact_run_dir(run_id, create=False)
    status = _json(source / "status.json")
    report_path = source / "reports" / REPORT
    report = _json(report_path)
    config = _json(source / "config" / "study.json")
    sealed_after_final_fit_failure = status.get("status") == "failed" and report_path.is_file()
    if ((status.get("status") != "completed" and not sealed_after_final_fit_failure)
            or report.get("prediction_key") != key or config.get("prediction_key") != key):
        raise ValueError("没有匹配该股票池、窗口及模型输入的完整封存预测")
    evidence = _json(source / PREDICTION_EVIDENCE)
    if evidence.get("prediction_key") != key or stable_hash(evidence["files"]) != report.get("prediction_evidence_sha256"):
        raise ValueError("锁定预测证据与完成报告不符")
    if evidence["files"] != _prediction_evidence(source, _archived_folds(report)):
        raise ValueError("来源模型或逐折预测在归档后发生变化")
    return source, report, config, evidence


def _copy_predictions(source: Path, root: Path, evidence: dict) -> None:
    for relative, info in evidence["files"].items():
        origin, destination = (source / relative).resolve(), (root / relative).resolve()
        if not origin.is_relative_to(source.resolve()) or not destination.is_relative_to(root.resolve()):
            raise ValueError("归档预测证据路径超出运行目录")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination)
        if _sha(destination) != info["sha256"]:
            raise ValueError("复用后的模型或预测哈希不符")


def _reprice_fold(
    data: FrozenInputs, window: Window, context, archived: dict, *, policy: Strategy, cost_model: StudyCostModel, seed: int,
) -> tuple[dict, dict]:
    if any(pd.Timestamp(archived[key]) != value for key, value in (
        ("evaluation_start", window.evaluate[0]), ("evaluation_end", window.evaluate[-1]),
    )):
        raise ValueError("复用预测的折边界与当前请求不一致")
    frame = data.frame
    eligible = frame.loc[frame["signal_eligible"].eq(True) & frame["date"].isin(window.evaluate)]
    keys = pd.MultiIndex.from_frame(eligible[["date", "stock"]])
    folder = context.subdir("backtests", window.period)
    arms, returns = {}, {cost: {} for cost in COSTS}
    for arm, prior in archived["arms"].items():
        pred = pd.read_parquet(folder / f"{arm}_predictions.parquet").set_index(["date", "stock"])["score"]
        if not pred.index.is_unique or len(pred) != len(keys) or not keys.isin(pred.index).all() or not np.isfinite(pred).all():
            raise ValueError("复用预测与冻结输入股票日期集合不一致")
        scores = pd.Series(pred.reindex(keys).to_numpy(), index=eligible.index)
        execution, timeseries = _execution(data, scores, window.evaluate, context, folder, arm, policy, cost_model)
        arms[arm] = {**prior, "execution": execution}
        for cost in COSTS:
            returns[cost][arm] = timeseries[cost]
    return {**archived, "arms": arms, "contrasts": _paired([returns], BOOTSTRAP_MAIN, seed)}, returns


def run(
    *, run_id: str, spec: dict | None = None, reuse_run_id: str | None = None, matched_fit: bool = False,
    archival_reprice: bool = False,
) -> Path:
    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号必须为最多 120 个字符的单级目录名")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)

    def update(status: str, phase: str, **extra) -> None:
        write_json(root / "status.json", {"run_id": run_id, "status": status, "phase": phase, **extra})

    update("running", "loading_inputs")
    try:
        from .expert_config import prepare_expert_inputs
        from .expert_full_data import read_expert_full_freeze

        identity = prediction_identity(spec)
        model_spec = identity["spec"]
        full = model_spec["pool"] == "full"
        if matched_fit:
            if not full:
                raise ValueError("匹配折内拟合仅用于已封存全量股票池")
            identity["fit_protocol"] = MATCHED_PROTOCOL
        key = stable_hash(identity)
        if archival_reprice:
            if not reuse_run_id or reuse_run_id != slugify(reuse_run_id, limit=120):
                raise ValueError("重算成交需要有效的冻结预测来源编号")
            archived = _json(artifact_run_dir(reuse_run_id, create=False) / "config/study.json")
            if archived["model_spec"] != model_spec or archived.get("fit_protocol") != identity.get("fit_protocol"):
                raise ValueError("重算成交的股票池、窗口或拟合协议与归档不一致")
            key = archived["prediction_key"]
        commission_bps, policy, cost_model = _execution_spec(spec)
        seed = model_spec.get("seed", SEED)
        source_identity = identity["research"]
        store = ExpertArtifactStore()
        reused = _reuse_source(reuse_run_id, key) if reuse_run_id else None
        if reused:
            previous, prior_report, prior_config, prior_evidence = reused
            if archival_reprice:
                source_identity = prior_report["input_coverage"]["source_identity"]
            references = store.read_manifest(reuse_run_id)["files"]
            if _reference_values(references) != prior_config["sources"]:
                raise ValueError("复用来源的输入清单不匹配")
            coverage = prior_report["input_coverage"]
            data = _data_by_reference(store, references, coverage)
            data, sector = _verified_sector(data, model_spec.get("sector_view", False), model_spec.get("sector_provider", "capco"))
            macro = coverage["overseas"]
            features = {arm: tuple(columns) for arm, columns in prior_config["features"].items()}
            version = prior_config["catboost_version"]
        else:
            import catboost
            from catboost.utils import get_gpu_device_count

            if get_gpu_device_count() < 1:
                raise RuntimeError("CatBoost GPU 训练需要可用 CUDA 设备")
            if full:
                stage, manifest = _full_freeze(model_spec)
                if manifest != identity["full_freeze"]:
                    raise ValueError("全量研究的封存输入清单在提交后变化")
                raw_data = read_expert_full_freeze(model_spec, stage)
                references = _full_references(store, stage, manifest, source_identity)
            elif model_spec["preset"] == "frozen_v2":
                raw_data = _load_frozen()
                references = _fixed_references(store, source_identity)
            else:
                raw_data = prepare_expert_inputs(model_spec)
                references = _custom_references(store, raw_data, source_identity)
            if model_spec["preset"] != "frozen_v2" and raw_data.coverage["input_identity"] != identity["inputs"]:
                raise ValueError("自定义面板在准备前后输入身份不一致")
            data, macro = _verified_macro(raw_data, _sources().get("overseas"))
            data, sector = _verified_sector(data, model_spec.get("sector_view", False), model_spec.get("sector_provider", "capco"))
            del raw_data
            features = {"domestic_gpu": tuple(BASE_FEATURES)}
            if sector["enabled"]:
                features["sector_gpu"] = (*BASE_FEATURES, *sector["features"])
            if macro["verified_sessions"]:
                features["overseas_gpu"] = (*BASE_FEATURES, *MACRO_FEATURES)
            coverage = {**data.coverage, "overseas": macro, "sector_view": sector, "source_identity": source_identity}
            version = catboost.__version__
        windows = plan_expert_windows(data, model_spec)
        if sum(window.period.startswith("oos_") for window in windows) != 3:
            raise ValueError("冻结计划必须有三个历史开发折")
        checks = prior_report["validation"] if reused else _features_validation(data.frame, data.sessions)
        actual_costs = {cost: asdict(scaled_study_cost(multiplier, cost_model)) for cost, multiplier in COSTS.items()}
        actual_end = data.sessions[-1].date().isoformat()
        config = {
            **source_identity, "sources": _reference_values(references), "source_run_id": BASELINE_ID,
            "end": actual_end, "seed": seed, "catboost_version": version, "stock_cap": len(data.roster), "model_spec": model_spec,
            "model_grid": expert_gpu.GRID if full else GPU_GRID, "iterations": expert_gpu.ITERATIONS if full else GPU_ITERATIONS,
            "quantiles": models.QUANTILE_LEVELS,
            "features": features, "portfolio": asdict(policy), "initial_cash": INITIAL_CASH,
            "expense_model": asdict(cost_model), "cost_multipliers": COSTS, "execution_costs": actual_costs,
            "execution_evidence": execution_evidence_identity(),
            "prediction_key": key, "commission_bps": commission_bps, "reused_predictions_from": reuse_run_id,
            "forward_isolation": PROSPECTIVE_START, "primary_block_sessions": BOOTSTRAP_MAIN,
            "risk_acceptance": RISK_ACCEPTANCE, "inference_priority": ["GPU", "CPU"],
            "final_fit_code_sha256": research_inputs(include_final_fit=True)["code_sha256"],
        }
        if full:
            config.update(early_stopping_rounds=expert_gpu.PATIENCE, fixed_domestic_parameters=expert_gpu.FIXED,
                          full_input_manifest_sha256=stable_hash(identity["full_freeze"]),
                          risk_acceptance={**RISK_ACCEPTANCE, "reference": "domestic_fixed"})
        if matched_fit:
            config["fit_protocol"] = MATCHED_PROTOCOL
        context = create_run_context(task="latent-expert-research", as_of_date=actual_end, seed=seed,
                                     config=config, data_version=stable_hash(config["sources"]), run_id=run_id)
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", coverage)
        write_json(context.subdir("validation") / "feature_label_timing.json", checks)
        if reused:
            _copy_predictions(previous, root, prior_evidence)
            past_folds = {entry["period"]: entry for entry in _archived_folds(prior_report)}
        folds, all_returns, review, pilot = [], [], None, None
        with expert_gpu.DualGPU() if full and not reused else nullcontext() as scheduler:
            if scheduler is not None:
                update("running", "gpu_resource_pilot")
                frame = data.frame
                train_mask = windows[0].masks(frame)[0] & frame["signal_eligible"].eq(True) & np.isfinite(frame["target"])
                train = frame.loc[train_mask, ["stock", "date", "target", *BASE_FEATURES]]
                pilot = scheduler.pilot(train, BASE_FEATURES, seed)
                del train, train_mask
                write_json(context.subdir("reports") / "gpu_resource_pilot.json", pilot)
            elif full:
                pilot = prior_report["gpu_resource_pilot"]
                write_json(context.subdir("reports") / "gpu_resource_pilot.json", pilot)
            for window in windows:
                LOG.info("%s %s", "费用/仓位重算" if reused else "GPU/OOF", window.period)
                update("running", window.period)
                with scheduler.measure() if scheduler is not None else nullcontext() as resource:
                    if reused:
                        fold, returns = _reprice_fold(data, window, context, past_folds[window.period], policy=policy, cost_model=cost_model, seed=seed)
                    else:
                        fold, returns = _fold(data, window, context, features, seed=seed, policy=policy, cost_model=cost_model,
                                              scheduler=scheduler, matched_fit=matched_fit)
                if resource is not None:
                    fold["gpu_resources"] = resource
                if window.period == "historical_replay":
                    review = fold | {"scope": "retrospective_review", "included_in_champion_selection": False}
                    fold = review
                else:
                    folds.append(fold)
                    all_returns.append(returns)
                write_json(context.subdir("reports") / f"{window.period}.json", fold)
        all_folds = [*folds, *([review] if review else [])]
        predicted_files = _prediction_evidence(root, all_folds)
        if reused and predicted_files != prior_evidence["files"]:
            raise ValueError("重算费用后预测证据发生变化")
        if execution_evidence_identity() != config["execution_evidence"]:
            raise ValueError("试验期间历史撮合规则发生变化，请重新提交。")
        write_json(root / PREDICTION_EVIDENCE, {"prediction_key": key, "files": predicted_files})
        manifest = store.write_manifest(run_id, references, metadata={
            "source_run_id": BASELINE_ID, "overseas_run_id": OVERSEAS_ID, "task": "latent-expert-research",
        })
        pooled = _paired(all_returns, BOOTSTRAP_MAIN, seed)
        decision = _decision(folds, pooled, bool(data.coverage["point_in_time"]["verified"]))
        champion = _single_champion(folds, pooled, explicit_evidence=matched_fit) if full else None
        if full:
            selected = champion["ranking"][0]
            reasons = decision["reasons"][:1]
            if not decision["point_in_time_verified"]:
                reasons.append("国内历史归档的首次可用时间与供应商修订尚未完整验证。")
            if not selected["historical_gate"]:
                reasons.append("领先候选尚未通过相对历史分位数及固定国内单专家的历史验收。")
            decision.update(historical_numerical_gate=selected["historical_gate"], comparisons=selected["comparisons"],
                            risk_acceptance=selected["risk"], reasons=reasons)
        final_portfolio = _final_portfolio(root, folds, folds[-1]["period"], champion["selected_arm"] if champion else "fusion")
        sector_report = _research_sector_artifacts(
            root, data, folds, final_portfolio["arm"], model_spec.get("sector_provider", "capco"),
        )
        final_portfolio["sector_resonance"] = "reports/sector_resonance.json"
        experts = {
            "historical_quantiles": {"status": "active", "role": "reference"},
            "supertrend_atr": {"status": "active", "role": "rule"},
            "domestic_gpu": {"status": "active", "role": "main", "training_device": "GPU:0,1" if full else "GPU:0"},
            "sector_gpu": {"status": "active" if "sector_gpu" in features else "abstain", "role": "sector_numeric",
                           "coverage": coverage.get("sector_view", {"enabled": False})},
            "overseas_gpu": {"status": "active" if "overseas_gpu" in features else "abstain", "role": "auxiliary", "coverage": macro},
            **{name: {"status": "abstain", "reason": "没有与当前冻结面板对齐的折内 OOF 和独立时点证据"}
               for name in ("SeqLSTM", "TCN", "announcement_text", "chart_image")},
        }
        report = {
            "run": context.to_dict(), "input_coverage": coverage, "validation": checks, "experts": experts,
            "model_spec": model_spec, "portfolio": asdict(policy), "commission_bps": commission_bps, "execution_costs": actual_costs,
            "execution_evidence": config["execution_evidence"],
            "prediction_key": key, "prediction_evidence_sha256": stable_hash(predicted_files),
            "reused_predictions_from": reuse_run_id, "fit_executed": not reused,
            "folds": folds, "oos_block_bootstrap": pooled, "decision": decision, "final_portfolio": final_portfolio,
            "ablation_block_bootstrap": _paired_ablations(all_returns, seed), "sector_resonance": sector_report,
            "block_sensitivity": {str(size): _paired(all_returns, size, seed) for size in BOOTSTRAP_SENSITIVITY},
            "artifact_manifest": str(manifest),
            "interpretation": {"execution": "同折各臂使用共同股票池，以先前收盘确定计划、次日开盘模拟，费用双档，逐折从现金开始。",
                               "scope": "历史开发研究。前瞻及原有 v2 模型锁与纸面冻结不参与此试验。"},
        }
        if full:
            report.update(single_champion=champion, historical_review=review, gpu_resource_pilot=pilot)
            experts["domestic_fixed"] = {"status": "active", "role": "reference", "training_device": "GPU:0,1"}
        if matched_fit:
            report.update(fit_protocol=MATCHED_PROTOCOL, cash_reference={cost: {"initial_cash": INITIAL_CASH,
                          "total_return": 0.0, "max_drawdown": 0.0, "order_count": 0, "trade_count": 0,
                          "reject_rate": None, "scope": "all_evaluation_sessions"} for cost in COSTS})
        report_dir = context.subdir("reports")
        write_json(report_dir / "selected_hyperparameters.json", {fold["period"]: {"models": fold["models"], "fusion": fold["fusion"]} for fold in all_folds})
        _write_table(report_dir / "cost_and_prediction.csv", _summary_table(all_folds))
        path = report_dir / REPORT
        write_json(path, report)
        update("running", "final_fit", release_state=decision["state"])
        from . import expert_final_fit

        expert_final_fit.fit_final_fit(root, data.frame)
        update("completed", "complete", release_state=decision["state"])
        return path
    except Exception as error:
        update("failed", "failed", error=str(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    parser.add_argument("--full", action="store_true", help="使用已经封存的全量历史股票池")
    parser.add_argument("--matched-fit", action="store_true", help="采用外折同口径的折内 GPU 网格和早停")
    parser.add_argument("--reprice-run-id", help="校验并复用同范围归档预测，只重算成交和验收")
    args = parser.parse_args()
    if args.matched_fit and not args.full:
        parser.error("--matched-fit requires --full")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    task = "latent-expert-matched-fit" if args.matched_fit else "latent-expert-research"
    print(run(run_id=args.run_id or make_run_id(task, HISTORICAL_END), spec={"pool": "full"} if args.full else None,
              matched_fit=args.matched_fit, reuse_run_id=args.reprice_run_id, archival_reprice=bool(args.reprice_run_id)))


if __name__ == "__main__":
    main()
