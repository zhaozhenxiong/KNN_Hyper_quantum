"""Chronological factor, model, and strategy research using the local market warehouse."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context
from my_strategy.model_registry.registry import register_candidate
from my_strategy.research_workbench.backtest import Strategy, simulate, target_weights
from my_strategy.research_workbench.data import load_research_data
from my_strategy.research_workbench.models import fit_predict, sample_training_rows, sequence_ready
from my_strategy.validation.leakage import validate_feature_columns_no_lookahead


REPORT = "workbench.json"
TASK = "research-workbench"
PRIOR_HOLDOUT_END = pd.Timestamp("2026-07-28")
SEQLSTM_LOOKBACK = 30
SEQLSTM_MAX_TRAIN_SAMPLES = 200_000


@dataclass(frozen=True)
class Window:
    period: str
    train_start: pd.Timestamp
    valid_start: pd.Timestamp
    calibration_start: pd.Timestamp
    evaluate: pd.DatetimeIndex

    def masks(self, frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        date, end = frame["date"], frame["label_end"]
        labeled = frame["target"].notna()
        train = labeled & date.ge(self.train_start) & date.lt(self.valid_start) & end.lt(self.valid_start)
        valid = labeled & date.ge(self.valid_start) & date.lt(self.calibration_start) & end.lt(self.calibration_start)
        calibration = (
            labeled & date.ge(self.calibration_start) & date.lt(self.evaluate[0]) & end.lt(self.evaluate[0])
        )
        return train, valid, calibration


@dataclass(frozen=True)
class Study:
    name: str
    model: str
    static_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class StudyCohort:
    eligible: pd.Series
    train: pd.Series
    valid: pd.Series
    calibration: pd.Series
    uncapped_train_rows: int


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a JSON artifact."""
    def serialize(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, (datetime, Path)):
            return str(item)
        raise TypeError(f"Cannot serialize {type(item).__name__}")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=serialize, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def _write_table(path: Path, table: pd.DataFrame) -> None:
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    if path.suffix == ".csv":
        table.to_csv(temp, index=False)
    else:
        table.to_parquet(temp, index=False)
    temp.replace(path)


def _windows(sessions: pd.DatetimeIndex, holdout_start: str | None = None) -> list[Window]:
    size = len(sessions)
    if size < 125:
        raise ValueError("至少需要 125 个交易日进行独立校准、两段滚动测试和最终留出。")
    if holdout_start:
        requested = pd.Timestamp(holdout_start)
        if requested <= PRIOR_HOLDOUT_END:
            raise ValueError("SeqLSTM 指定留出开始日须晚于上一轮已查看的 2026-07-28。")
        holdout_offset = int(sessions.searchsorted(requested))
        if holdout_offset < 100 or size - holdout_offset < 8:
            raise ValueError("指定留出前至少需 100 个交易日，留出段至少需 8 个交易日完成 5 日收益标签。")
    else:
        holdout_offset = size - min(63, max(18, size // 8))
    planning_size = holdout_offset if holdout_start else size
    block = min(63, max(18, planning_size // 8))
    inner = min(48, max(16, planning_size // 11))
    offsets = (holdout_offset - 2 * block, holdout_offset - block, holdout_offset, size)
    return [
        Window(
            "holdout" if index == 2 else f"oos_{index + 1}",
            sessions[max(0, offset - 315)],
            sessions[offset - 2 * inner],
            sessions[offset - inner],
            sessions[offset:end],
        )
        for index, (offset, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True))
    ]


def _rank_evidence(rows: pd.DataFrame, scores: pd.Series) -> dict[str, Any]:
    comparison = rows[["date", "target"]].assign(score=scores.reindex(rows.index)).dropna()
    counts = comparison.groupby("date")["target"].transform("size")
    comparison = comparison.loc[counts >= 5]
    if comparison.empty:
        return {"rank_ic": None, "icir": None, "ic_days": 0, "samples": 0, "top_bottom": None}
    ranks = comparison.groupby("date")[["target", "score"]].rank()
    normalized = ranks - ranks.groupby(comparison["date"]).transform("mean")
    product = (normalized["target"] * normalized["score"]).groupby(comparison["date"]).sum()
    squares = normalized.pow(2).groupby(comparison["date"]).sum()
    denominator = (squares["target"] * squares["score"]).pow(0.5).replace(0, np.nan)
    ic = (product / denominator).dropna()
    quantiles = ranks["score"] / comparison.groupby("date")["score"].transform("size")
    upside = comparison.loc[quantiles.ge(0.8)].groupby("date")["target"].mean()
    downside = comparison.loc[quantiles.le(0.2)].groupby("date")["target"].mean()
    spread = (upside - downside).dropna()
    mean = float(ic.mean()) if not ic.empty else None
    deviation = float(ic.std()) if len(ic) > 1 else 0.0
    return {
        "rank_ic": mean, "icir": mean / deviation if mean is not None and deviation > 0 else None,
        "ic_days": len(ic), "samples": len(comparison), "top_bottom": float(spread.mean()) if not spread.empty else None,
    }


def _rank_ic(rows: pd.DataFrame, scores: pd.Series) -> tuple[float | None, int, float | None]:
    evidence = _rank_evidence(rows, scores)
    return evidence["rank_ic"], evidence["samples"], evidence["top_bottom"]


def _rank_by_classification(rows: pd.DataFrame, scores: pd.Series) -> dict[str, list[dict[str, Any]]]:
    evidence = {}
    for column in ("board", "industry"):
        groups = rows[["date", "stock", "target"]].groupby(rows[column].fillna("unknown"), sort=True)
        evidence[column] = [
            {
                "name": name, "labeled_rows": len(group), "stocks": group["stock"].nunique(),
                "prediction_rows": int(scores.reindex(group.index).notna().sum()), **_rank_evidence(group, scores),
            }
            for name, group in groups
        ]
    return evidence


def _group(name: str) -> str:
    if name.startswith(("volume", "amount", "amihud", "obv", "liquidity")):
        return "流动性"
    if name.startswith(("atr", "volatility", "boll_band", "high_low")):
        return "波动"
    if name.startswith(("rsi", "mean_reversion", "boll_pct", "kdj", "cci")):
        return "反转"
    if name.startswith(("return", "roc", "momentum")):
        return "动量"
    return "趋势"


FACTOR_DESCRIPTIONS = {
    "return_20d": "近 20 个有效行情日的价格变化",
    "ma20_ratio": "收盘价相对 20 日均线的距离",
    "mean_reversion_zscore": "短期价格偏离历史均值的标准分",
    "atr_pct": "平均真实波幅占价格的比例",
    "amihud_illiquidity": "单位成交金额对应的绝对收益变化",
    "volume_ratio": "成交量相对历史均值的比例",
}


def factor_catalog(names: list[str] | tuple[str, ...]) -> list[dict[str, Any]]:
    return [{"name": name, "group": _group(name), "description": FACTOR_DESCRIPTIONS.get(name, "")} for name in names]


def _factor_evidence(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    correlations = frame[columns].rank().corr().abs()
    records = factor_catalog(columns)
    for item in records:
        name = item["name"]
        ic, count, spread = _rank_ic(frame, frame[name])
        redundant = correlations.loc[name].drop(name).dropna()
        ordered = frame[["date", "target", name]].dropna()
        counts = ordered.groupby("date")[name].transform("size")
        ordered = ordered.loc[counts.ge(5)].copy()
        ordinal = ordered.groupby("date")[name].rank(method="first")
        quantile = ((ordinal - 1) * 5 / counts.reindex(ordered.index)).astype(int)
        means = ordered.groupby(["date", quantile])["target"].mean().groupby(level=1).mean()
        item.update(
            coverage=float(frame[name].notna().mean()),
            rank_ic=ic,
            samples=count,
            top_bottom=spread,
            quantile_returns=[float(means[band]) if band in means else None for band in range(5)],
            max_abs_correlation=float(redundant.max()) if not redundant.empty else None,
        )
    return records


def _data_summary(coverage: dict, frame: pd.DataFrame) -> dict:
    return {
        **coverage,
        "latest_date": str(frame["date"].max().date()),
        "data_start": str(frame["date"].min().date()),
        "total_stocks": int(frame["stock"].nunique()),
        "latest_stocks": int(frame.loc[frame["date"].eq(frame["date"].max()), "stock"].nunique()),
    }


def _study_cohorts(frame: pd.DataFrame, windows: list[Window], config: dict[str, Any]) -> dict[str, StudyCohort]:
    lookback = config.get("lookback", SEQLSTM_LOOKBACK)
    max_samples = config.get("max_train_samples", SEQLSTM_MAX_TRAIN_SAMPLES)
    eligible = frame["signal_eligible"].fillna(False).astype(bool) & sequence_ready(frame, lookback)
    cohorts = {}
    for window in windows:
        train, valid, calibration = (mask & eligible for mask in window.masks(frame))
        rows = np.flatnonzero(train.to_numpy())
        sampled = sample_training_rows(frame, rows, max_samples, config["seed"])
        train_sample = pd.Series(False, index=frame.index)
        train_sample.iloc[sampled] = True
        cohorts[window.period] = StudyCohort(eligible, train_sample, valid, calibration, len(rows))
    return cohorts


def _window_evidence(frame: pd.DataFrame, window: Window, cohort: StudyCohort | None) -> dict[str, Any]:
    train, valid, calibration = window.masks(frame) if cohort is None else (cohort.train, cohort.valid, cohort.calibration)
    evidence = {
        "period": window.period, "train_start": window.train_start, "valid_start": window.valid_start,
        "calibration_start": window.calibration_start, "evaluation_start": window.evaluate[0],
        "evaluation_end": window.evaluate[-1], "train_rows": int(train.sum()), "valid_rows": int(valid.sum()),
        "calibration_rows": int(calibration.sum()),
    }
    if cohort is not None:
        keys = frame.loc[train, ["stock", "date"]].sort_values(["stock", "date"])
        evidence["train_keys_sha256"] = hashlib.sha256(pd.util.hash_pandas_object(keys, index=False).to_numpy().tobytes()).hexdigest()
        evidence["uncapped_train_rows"] = cohort.uncapped_train_rows
    return evidence


def _studies(config: dict[str, Any], factors_window: pd.DataFrame) -> list[Study]:
    board = ("exchange", "board")
    industry = bool(config.get("classification_path") and factors_window["industry"].ne("unknown").any())
    if config["mode"] == "seqlstm":
        static = (*board, "industry") if industry else board
        studies = [
            Study("LightGBM · 同口径因子与分类", "lightgbm", static),
            Study("SeqLSTM · 时序", "seqlstm"),
            Study("SeqLSTM · 时序与板别", "seqlstm", board),
        ]
        if industry:
            studies.append(Study("SeqLSTM · 时序、板别与行业", "seqlstm", static))
        return studies
    studies = [
        Study("等权持有", "rules"),
        Study("趋势规则", "rules"),
        Study("反转规则", "rules"),
        Study("LightGBM · 因子", "lightgbm"),
        Study("LightGBM · 因子与板别", "lightgbm", board),
    ]
    static = board
    if industry:
        static = (*static, "industry")
        studies.append(Study("LightGBM · 加入行业", "lightgbm", static))
    if config.get("classification_path") and factors_window["concepts"].map(bool).any():
        static = (*static, "concepts")
        studies.append(Study("LightGBM · 加入概念", "lightgbm", static))
    if config["mode"] == "full":
        studies.extend([Study("LSTM · 时序", "lstm"), Study("LSTM · 时序与分类", "lstm", static)])
    return studies


def _fold(
    data: Any,
    window: Window,
    study: Study,
    config: dict,
    bundle: Path | None = None,
    cohort: StudyCohort | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    frame = data.frame
    if cohort is None:
        train, valid, calibration = window.masks(frame)
        eligible = frame["signal_eligible"].fillna(False).astype(bool)
    else:
        eligible, train, valid, calibration = cohort.eligible, cohort.train, cohort.valid, cohort.calibration
    predict = eligible & (frame["date"].isin(window.evaluate) | calibration)
    if bundle is not None:
        predict |= eligible & frame["date"].eq(frame["date"].max())
    metadata: dict[str, Any] = {}
    if study.model == "rules":
        factor = {
            "等权持有": None,
            "趋势规则": "return_20d",
            "反转规则": "mean_reversion_zscore",
        }[study.name]
        if factor is None:
            predictions = pd.Series(1.0, index=frame.index[predict])
        else:
            if factor not in frame:
                factor = "return_5d"
            predictions = frame.loc[predict, factor].copy()
            if study.name == "反转规则":
                predictions = -predictions
    else:
        params: dict[str, Any] = {"n_jobs": 4}
        if study.model == "lstm":
            params = {"lookback": 30, "epochs": config["epochs"]}
        elif study.model == "seqlstm":
            params = {
                "lookback": config.get("lookback", SEQLSTM_LOOKBACK), "epochs": config["epochs"],
                "max_train_samples": config.get("max_train_samples", SEQLSTM_MAX_TRAIN_SAMPLES),
            }
        model_frame = frame
        if cohort is not None:
            inputs = ["stock", "date", "label_end", "target", *data.feature_columns, *study.static_columns]
            model_frame = frame.loc[frame["date"].le(window.evaluate[-1]), inputs]
            model_frame["target"] = model_frame["target"].where(train | valid)
        output = fit_predict(
            model_frame,
            feature_columns=data.feature_columns,
            train_mask=(train & eligible).reindex(model_frame.index),
            valid_mask=(valid & eligible).reindex(model_frame.index),
            predict_mask=predict.reindex(model_frame.index),
            model=study.model,
            params=params,
            static_columns=study.static_columns,
            seed=config["seed"],
            device=config["device"],
            bundle_path=bundle,
        )
        predictions, metadata = output.predictions, output.metadata

    evaluation = frame["date"].isin(window.evaluate) & frame["label_end"].le(window.evaluate[-1])
    if cohort is not None:
        evaluation &= eligible
    settled = frame.loc[evaluation, ["date", "target", "stock", "board", "industry"]].dropna(subset=["target"])
    metadata = {
        **metadata,
        "period": window.period,
        "start": str(window.evaluate[0].date()),
        "end": str(window.evaluate[-1].date()),
        "train_end": str(frame.loc[train, "date"].max().date()) if train.any() else None,
        "valid_end": str(frame.loc[valid, "date"].max().date()) if valid.any() else None,
        **_rank_evidence(settled, predictions),
    }
    if cohort is not None:
        metadata["uncapped_train_rows"] = cohort.uncapped_train_rows
        metadata["evaluation_labeled_rows"] = len(settled)
        metadata["by_classification"] = _rank_by_classification(settled, predictions)
    return predictions, calibration, metadata


def _objective(results: list[dict]) -> float | None:
    scores = []
    for result in results:
        metrics = result["metrics"]
        excess = metrics["excess_return"]
        if excess is None or not metrics["trade_count"]:
            return None
        scores.append(excess - 0.5 * abs(metrics["max_drawdown"]) - 0.005 * metrics["turnover"])
    return float(np.mean(scores))


def _simulate(data: Any, window: Window, scores: pd.Series, strategy: Strategy, run_id: str, cost: float = 1) -> dict:
    return simulate(data.frame, scores, window.evaluate, data.benchmark_returns, strategy, run_id=run_id, cost_multiplier=cost)


def _calibration(frame: pd.DataFrame, scores: pd.Series, calibration: pd.Series, evaluation: Window) -> dict:
    sample = frame.loc[calibration, "target"].sub(scores).dropna().abs()
    if len(sample) < 40:
        return {"samples": len(sample), "half_width": None, "holdout_coverage": None}
    alpha = 0.10
    rank = min(len(sample), math.ceil((len(sample) + 1) * (1 - alpha)))
    width = float(np.partition(sample.to_numpy(), rank - 1)[rank - 1])
    test = frame["date"].isin(evaluation.evaluate) & frame["label_end"].le(evaluation.evaluate[-1])
    errors = frame.loc[test, "target"].sub(scores).dropna().abs()
    return {
        "samples": len(sample),
        "half_width": width,
        "nominal_coverage": 1 - alpha,
        "holdout_coverage": float(errors.le(width).mean()) if not errors.empty else None,
        "holdout_samples": len(errors),
    }


def _signals(data: Any, scores: pd.Series, interval: dict, strategy: Strategy) -> list[dict]:
    frame = data.frame
    day = frame["date"].max()
    latest = frame.loc[frame["date"].eq(day)].assign(score=scores).dropna(subset=["score"])
    latest = latest.sort_values(["score", "stock"], ascending=[False, True])
    targets = target_weights(latest, strategy)
    width = interval["half_width"]
    next_session = data.coverage.get("next_session")
    records = []
    for index, row in enumerate(latest.head(max(50, strategy.top_n)).itertuples(index=False), 1):
        reasons = [f"当日预测排序 {index}/{len(latest)}"]
        for factor, label in (("return_20d", "20 日动量"), ("volume_ratio", "量比"), ("atr_pct", "ATR 波动比")):
            value = getattr(row, factor, None)
            if pd.notna(value):
                reasons.append(f"{label} {value:.3f}")
        if row.industry == "unknown":
            reasons.append("历史行业归属缺失")
        target = targets.get(row.stock, 0.0)
        records.append(
            {
                "stock": row.stock, "date": str(day.date()), "trade_date": next_session,
                "score": float(row.score), "rank": index,
                "action": "watch" if target else "observe", "target_weight": target,
                "board": row.board, "industry": row.industry, "reasons": reasons,
                "prediction_low": float(row.score - width) if width is not None else None,
                "prediction_high": float(row.score + width) if width is not None else None,
            }
        )
    return records


def run(config: dict[str, Any], run_id: str, progress: Any) -> dict:
    progress("loading", "加载行情、因子和历史分类")
    seq_profile = config["mode"] == "seqlstm"
    universe = {key: config[key] for key in ("universe_policy", "universe_path") if config.get(key)} if seq_profile else {}
    data = load_research_data(
        config["start"], config["end"], max_stocks=config["max_stocks"],
        stocks=[stock.strip() for stock in config["stocks"].split(",") if stock.strip()] if config.get("stocks") else None,
        classification_csv=config.get("classification_path"), benchmark_path=config.get("benchmark_path"),
        price_basis=config["price_basis"], adjusted_path=config.get("adjusted_path"),
        execution_reference_path=config.get("execution_reference_path"),
        feature_progress=lambda completed, total: progress("loading", f"计算历史量价因子 {completed}/{total}"),
        **universe,
    )
    progress("loading", "固定共同样本与时间切分")
    frame = data.frame
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        raise ValueError("Research dataset must have an aligned range index.")
    columns = list(data.feature_columns)
    check = validate_feature_columns_no_lookahead(columns)
    if not check["ok"] or frame.duplicated(["date", "stock"]).any():
        raise ValueError(f"特征时点校验未通过或存在重复股票日期。{check['issues']}")
    known_labels = frame["target"].notna()
    if not (frame.loc[known_labels, "label_end"] > frame.loc[known_labels, "date"]).all():
        raise ValueError("发现缺失标签结束日或与信号时点冲突。")
    dates = pd.DatetimeIndex(data.sessions)
    if not dates.is_monotonic_increasing or not dates.is_unique:
        raise ValueError("交易日日历必须严格递增且无重复。")
    requested_holdout = config.get("holdout_start") if seq_profile else None
    windows = _windows(dates, requested_holdout)
    cohorts = _study_cohorts(frame, windows, config) if seq_profile else {}
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
    context = create_run_context(task=TASK, as_of_date=str(dates[-1].date()), config=config, data_version=digest, seed=config["seed"], run_id=run_id)
    report_path = context.subdir("reports") / REPORT
    write_json(context.subdir("config") / "workbench.json", config)
    write_json(context.subdir("data_snapshots") / "coverage.json", {**data.coverage, "frame_sha256": digest})
    _write_table(context.subdir("data_snapshots") / "features.parquet", frame)
    write_json(
        context.subdir("validation") / "timing.json",
        {"feature_names": check, "label_rows": int(known_labels.sum()), "duplicate_stock_dates": 0,
         "folds": [_window_evidence(frame, window, cohorts.get(window.period)) for window in windows]},
    )
    first = windows[0]
    factors_window = frame.loc[cohorts[first.period].train if seq_profile else first.masks(frame)[0]]
    if factors_window.empty:
        raise ValueError("首段训练窗缺少已结算标签。请检查行情、复权和基准的交集。")
    factors = _factor_evidence(factors_window, columns)
    base = Strategy(**{key: config[key] for key in asdict(Strategy())})
    studies = _studies(config, factors_window)

    research, predictions = [], {}
    for study in studies:
        progress("training", f"滚动验证 {study.name}")
        policy = _study_policy(study, base, frame)
        periods = []
        for window in windows[:-1]:
            scores, _, metrics = _fold(data, window, study, config, cohort=cohorts.get(window.period))
            simulation = _simulate(data, window, scores, policy, run_id)
            periods.append({**metrics, "metrics": simulation["metrics"]})
            predictions[study.name, window.period] = scores
        values = [fold["rank_ic"] for fold in periods if fold["rank_ic"] is not None]
        research.append(
            {
                **asdict(study), "folds": periods,
                "mean_rank_ic": float(np.mean(values)) if values else None,
                "strategy_score": _objective(periods),
            }
        )

    ranked = [
        item for item in research
        if item["model"] != "rules" and item["mean_rank_ic"] is not None and item["strategy_score"] is not None
    ]
    if not ranked:
        raise RuntimeError("滚动窗口没有同时具备排名指标和完整基准的有效模型。")
    selected = max(ranked, key=lambda item: (item["strategy_score"], item["mean_rank_ic"]))
    progress("tuning", "在滚动样本外窗口对比策略参数")
    trials = _tune(data, windows[:-1], [predictions[selected["name"], w.period] for w in windows[:-1]], base, run_id)
    tuned = max(trials, key=lambda trial: trial["score"] if trial["score"] is not None else -math.inf)
    strategy = Strategy(**{key: tuned[key] for key in asdict(base)})

    holdout = windows[-1]
    selection, final_scores, interval, bundle = {}, None, {}, None
    holdout_predictions = {}
    candidate_bundles: dict[str, Path] = {}
    for index, (study, research_item) in enumerate(zip(studies, research, strict=True), 1):
        progress("holdout", f"评估冻结的留出窗口 · {study.name}")
        final_path = context.subdir("models") / f"study-{index}" if study.model != "rules" else None
        scores, cal_mask, evidence = _fold(data, holdout, study, config, final_path, cohorts.get(holdout.period))
        holdout_predictions[f"study_{index}"] = scores
        if final_path is not None:
            candidate_bundles[study.name] = final_path
        policy = _study_policy(study, base, frame)
        simulation = _simulate(data, holdout, scores, policy, run_id)
        research_item["holdout"] = {**evidence, "metrics": simulation["metrics"]}
        if study.name == selected["name"]:
            selection = _simulate(data, holdout, scores, strategy, run_id)
            final_scores, bundle = scores, final_path
            interval = _calibration(frame, scores, cal_mask, holdout)
    if final_scores is None or bundle is None:
        raise RuntimeError("选中的候选模型缺少留出期产物。")
    stress = _simulate(data, holdout, final_scores, strategy, run_id, cost=2)
    selection["cost_stress"] = {"multiplier": 2, "metrics": stress["metrics"]}
    signals = _signals(data, final_scores, interval, strategy)
    holdout_scores = frame.loc[frame["date"].isin(holdout.evaluate), ["date", "stock", "label_end", "target"]]
    _write_table(context.subdir("backtests") / "holdout_predictions.parquet", holdout_scores.assign(**holdout_predictions))
    _write_table(context.subdir("backtests") / "daily.csv", pd.DataFrame(selection["equity"]))
    _write_table(context.subdir("backtests") / "trades.csv", pd.DataFrame(selection["trades"]))
    write_json(context.subdir("signals") / "workbench.json", {"run_id": run_id, "signals": signals})
    warnings = [
        "研究候选。日线撮合使用近似涨跌限制和上一交易日的成交额，缺少历史 ST、特殊成交和公司行动流水。",
        "收益区间以更早校准窗估计，存在市场状态变化导致的覆盖偏差。",
    ]
    if config["price_basis"] == "raw_research":
        warnings.append("本次使用未复权行情研究口径。公司行动可能影响收益标签和净值，结果保留为候选证据。")
    counts = data.coverage["classification_counts"]
    if not counts["industry"]:
        warnings.append("本次样本无点时行业覆盖。组合行业约束仅作用于已知行业。")
    elif counts["industry"] < len(frame):
        warnings.append("历史行业资料仅覆盖部分股票日期，未覆盖归属标为未知。")
    if not counts["concepts"]:
        warnings.append("本次样本未提供带可用日期的历史概念归属。")
    if seq_profile and len(studies) < 4:
        warnings.append("首段共同训练样本没有可用的点时行业，未运行加入行业的消融对照。")
    if seq_profile and holdout.evaluate[0] <= PRIOR_HOLDOUT_END:
        warnings.append("本次留出与上轮已查看行情重叠，只作为历史对照。")
    if (pd.Timestamp.now(tz="Asia/Shanghai").date() - dates[-1].date()).days > 7:
        warnings.append("行情距当前日期超过七天，盘后信号仅用于历史复盘。")
    if not data.coverage.get("next_session"):
        warnings.append("数据源未覆盖下一交易日，候选信号尚无已验证的拟执行日期。")
    if selection["metrics"]["benchmark_return"] is None:
        warnings.append("留出期的基准行情有缺口，组合超额收益留空。")
    if selected["holdout"]["rank_ic"] is None or selected["holdout"]["rank_ic"] <= 0:
        warnings.append("留出期排序指标未提供正向证据，信号只作复盘观察。")
    if stress["metrics"]["excess_return"] is not None and stress["metrics"]["excess_return"] <= 0:
        warnings.append("提高佣金和滑点后，留出期策略未跑赢基准。")
    if selection["metrics"]["trade_count"] == 0:
        warnings.append("留出期没有成交，组合指标无法用于验证策略可执行性。")
    result = {
        "run_id": run_id, "status": "completed", "run_context": context.to_dict(), "config": config,
        "data": _data_summary(data.coverage, frame),
        "factors": factors, "studies": research, "tuning": {"selected": asdict(strategy), "trials": trials},
        "backtest": selection, "calibration": interval, "signals": signals, "warnings": warnings,
        "summary": {
            "selected_model": selected["name"], "rank_ic": selected["holdout"]["rank_ic"],
            "total_return": selection["metrics"]["total_return"], "max_drawdown": selection["metrics"]["max_drawdown"],
            "trade_count": selection["metrics"]["trade_count"], "stage": "candidate",
        },
    }
    if seq_profile:
        result["comparison"] = {
            "lookback": config.get("lookback", SEQLSTM_LOOKBACK),
            "max_train_samples": config.get("max_train_samples", SEQLSTM_MAX_TRAIN_SAMPLES),
            "training_sampling": "seeded_stock_balanced",
            "holdout_start_requested": requested_holdout,
            "holdout_start_actual": holdout.evaluate[0].date().isoformat(),
            "holdout_end_actual": holdout.evaluate[-1].date().isoformat(),
            "after_previous_holdout": bool(holdout.evaluate[0] > PRIOR_HOLDOUT_END),
            "used_for_selection": False,
            "selection_folds": [window.period for window in windows[:-1]],
        }
    write_json(report_path, result)
    registered = []
    for index, item in enumerate(research, 1) if seq_profile else [(None, selected)]:
        study_bundle = candidate_bundles.get(item["name"])
        if study_bundle is None:
            continue
        is_selected = item is selected
        candidate_name = f"{item['model']}-{run_id}" if index is None else f"{item['model']}-{run_id}-study-{index}"
        entry = register_candidate(
            name=candidate_name, kind=f"workbench_{item['model']}", run_id=run_id, model_path=study_bundle,
            report_path=report_path, feature_columns=[*columns, *item["static_columns"]],
            metrics={"rank_ic": item["holdout"]["rank_ic"], **(selection["metrics"] if is_selected else item["holdout"]["metrics"])},
            data_range={"start": config["start"], "end": config["end"], "data_version": digest},
            params={"strategy": asdict(strategy if is_selected else base), "price_basis": config["price_basis"],
                    **({"study": item["name"], "model": item["holdout"]["params"]} if seq_profile else {})},
            git_commit=context.git_commit,
        )
        if is_selected:
            result["model_id"] = entry.model_id
        if seq_profile:
            registered.append({"study": item["name"], "model_id": entry.model_id, "stage": "candidate"})
    if seq_profile:
        result["candidate_models"] = registered
    write_json(report_path, result)
    return result


def _study_policy(study: Study, base: Strategy, frame: pd.DataFrame) -> Strategy:
    if study.name != "等权持有":
        return base
    return replace(base, top_n=int(frame.loc[frame["signal_eligible"], "stock"].nunique()), score_threshold=0, max_weight=1, max_industry_weight=1)


def _tune(data: Any, windows: list[Window], scores: list[pd.Series], base: Strategy, run_id: str) -> list[dict]:
    # Keep a small bounded search and feed all trials the same held-out fold predictions.
    alternatives = [
        base, replace(base, top_n=max(2, base.top_n // 2)),
        replace(base, top_n=min(100, base.top_n * 2)),
        replace(base, rebalance_days=max(1, base.rebalance_days // 2)),
        replace(base, rebalance_days=min(30, base.rebalance_days * 2)),
        replace(base, score_threshold=min(0.98, base.score_threshold + 0.10)),
        replace(base, score_threshold=max(0.0, base.score_threshold - 0.10)),
        replace(base, max_weight=max(0.01, base.max_weight * 0.75)),
        replace(base, max_industry_weight=max(0.01, base.max_industry_weight * 0.75)),
    ]
    trials = []
    for strategy in dict.fromkeys(alternatives):
        results = [_simulate(data, window, score, strategy, run_id) for window, score in zip(windows, scores, strict=True)]
        metrics: dict[str, Any] = {}
        for name in ("total_return", "sharpe", "turnover", "excess_return"):
            values = [result["metrics"][name] for result in results]
            metrics[name] = float(np.mean(values)) if all(value is not None for value in values) else None
        metrics["max_drawdown"] = min(result["metrics"]["max_drawdown"] for result in results)
        metrics["trade_count"] = sum(result["metrics"]["trade_count"] for result in results)
        trials.append({**asdict(strategy), "score": _objective(results), "metrics": metrics})
    return trials
