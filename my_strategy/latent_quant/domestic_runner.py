"""Tune domestic quantiles and portfolio rules on separate historical windows."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from hashlib import file_digest
from itertools import product
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.research_workbench.runner import Window, _write_table, write_json

from .data import BASE_FEATURES
from .evaluation import equity_returns, paired_block_interval, score_quantiles
from .execution import StudyCostModel, Strategy, simulate
from .models import _matrix, replay_catboost
from .runner import COSTS, HISTORICAL_END, INITIAL_CASH, PORTFOLIO, PROSPECTIVE_START, _features_validation, _summary_table, _verify_ledger, plan_windows
from .tuning import EARLY_STOPPING_ROUNDS, GRID, ITERATIONS, tune_fold


LOG = logging.getLogger(__name__)
BASELINE_ID = "latent-quant-20261122-cpu-v2"
REPORT = "domestic_research.json"
PORTFOLIO_GRID = tuple(product((5, 8, 10), (5, 10)))
MAX_CALIBRATION_DRAWDOWN = 0.15
MAX_CALIBRATION_DAILY_TURNOVER = 0.20
ARMS = ("domestic_fixed", "domestic_tuned")
CONTRAST = "tuned_vs_fixed"


@dataclass(frozen=True)
class FrozenInputs:
    frame: pd.DataFrame
    sessions: pd.DatetimeIndex
    benchmark_returns: pd.Series
    market_status: pd.DataFrame
    roster: pd.DataFrame
    coverage: dict
    benchmark_close: pd.Series | None = None
    sector_daily: pd.DataFrame | None = None


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _load_frozen() -> FrozenInputs:
    root = artifact_run_dir(BASELINE_ID, create=False)
    stage = root / "data_snapshots"
    config_file = root / "config" / "study.json"
    config = _read(config_file)
    portfolio = asdict(PORTFOLIO)
    configured_portfolio = config["portfolio"]
    portfolio_matches = all(configured_portfolio.get(key, value) == value for key, value in portfolio.items())
    if (_read(root / "status.json")["status"] != "completed" or config["end"] != HISTORICAL_END
            or config["max_stocks"] != 120 or config["base_features"] != list(BASE_FEATURES)
            or not portfolio_matches or config["expense_model"] != asdict(StudyCostModel())
            or config["cost_multipliers"] != COSTS or config["initial_cash"] != INITIAL_CASH):
        raise ValueError("国内回顾性试验要求冻结的 v2 数据、仓位和费用配置")
    frame = pd.read_parquet(stage / "panel.parquet").sort_values(["date", "stock"]).reset_index(drop=True)
    benchmark = pd.read_parquet(stage / "benchmark.parquet").set_index("date")
    sessions = pd.DatetimeIndex(benchmark.index)
    cutoff = pd.Timestamp(HISTORICAL_END)
    labeled = frame["target"].notna()
    approved = frame["date"].isin(sessions).all() and frame.loc[labeled, "label_end"].le(cutoff).all()
    if (sessions[-1] != cutoff or cutoff >= pd.Timestamp(PROSPECTIVE_START) or not approved
            or not sessions.is_unique or not sessions.is_monotonic_increasing
            or not np.isfinite(benchmark[["open", "close"]].to_numpy()).all()):
        raise ValueError("冻结快照日期或标签越界，或基准覆盖不完整")
    roster = pd.read_parquet(stage / "roster.parquet").set_index("stock")
    if len(roster) != 120 or not roster.index.is_unique or not frame["stock"].isin(roster.index).all():
        raise ValueError("股票必须属于固定的 v2 120 只股票池")
    market = pd.read_parquet(stage / "market_status.parquet")
    if not market["date"].isin(sessions).all():
        raise ValueError("冻结的执行状态日期越界")
    files = [config_file, *(stage / name for name in (
        "panel.parquet", "benchmark.parquet", "roster.parquet", "market_status.parquet", "coverage.json",
    ))]
    coverage = _read(stage / "coverage.json")
    coverage.update(source_run_id=BASELINE_ID, frozen_input_sha256={file.name: _sha(file) for file in files})
    return FrozenInputs(frame, sessions, benchmark["daily_return"], market, roster, coverage, benchmark["close"])


def _simulate(data: FrozenInputs, scores: pd.Series, sessions: pd.DatetimeIndex, strategy: Strategy, multiplier: float, run_id: str) -> dict:
    return simulate(
        data.frame, scores, sessions, data.benchmark_returns, run_id=run_id, cost_multiplier=multiplier,
        initial_cash=INITIAL_CASH, strategy=strategy, market_status=data.market_status, roster=data.roster,
    )


def _selection_sessions(sessions: pd.DatetimeIndex, window: Window) -> pd.DatetimeIndex:
    return sessions[(sessions >= window.calibration_start) & (sessions < window.evaluate[0])]


def _portfolio(data: FrozenInputs, window: Window, scores: pd.Series, run_id: str) -> tuple[Strategy, dict]:
    sessions = _selection_sessions(data.sessions, window)
    if len(sessions) < 2 or not data.frame.loc[scores.index, "date"].isin(sessions).all():
        raise ValueError("组合调优的分数必须完全落在外层测试前的独立窗口")
    candidates = []
    for top_n, rebalance_days in PORTFOLIO_GRID:
        strategy = replace(PORTFOLIO, top_n=top_n, rebalance_days=rebalance_days)
        outcome = _simulate(data, scores, sessions, strategy, COSTS["double"], run_id)
        metrics = outcome["metrics"]
        drawdown_ok = metrics["max_drawdown"] >= -MAX_CALIBRATION_DRAWDOWN
        turnover_ok = metrics["mean_daily_turnover"] <= MAX_CALIBRATION_DAILY_TURNOVER
        candidates.append({
            "strategy": asdict(strategy), "metrics": metrics,
            "drawdown_gate_pass": drawdown_ok, "turnover_gate_pass": turnover_ok,
            "risk_gate_pass": drawdown_ok and turnover_ok,
        })
    accepted = [item for item in candidates if item["risk_gate_pass"]]
    if accepted:
        winner = max(accepted, key=lambda item: (
            item["metrics"]["total_return"], item["metrics"]["max_drawdown"], -item["metrics"]["mean_daily_turnover"],
            item["strategy"] == asdict(PORTFOLIO),
        ))
        selected = Strategy(**winner["strategy"])
        rule = "在通过风险门槛的候选中按双倍成本净收益、回撤、换手排序；完全持平优先固定规则。"
    else:
        winner = None
        selected = PORTFOLIO
        rule = "所有候选未过风险门槛；研究对照沿用固定规则。"
    return selected, {
        "selection_start": sessions[0], "selection_end": sessions[-1], "selection_sessions": len(sessions),
        "selection_score": "reordered_raw_q50", "selection_cost": "double", "rule": rule,
        "maximum_drawdown": MAX_CALIBRATION_DRAWDOWN, "maximum_mean_daily_turnover": MAX_CALIBRATION_DAILY_TURNOVER,
        "candidates": candidates, "winner": winner, "selected_strategy": asdict(selected),
    }


def _paired(returns: list[dict], seed: int) -> dict:
    return {CONTRAST: {
        cost: paired_block_interval([(fold[cost]["domestic_tuned"], fold[cost]["domestic_fixed"]) for fold in returns], seed=seed)
        for cost in COSTS
    }}


def _fold(data: FrozenInputs, window: Window, context, seed: int) -> tuple[dict, dict]:
    from catboost import CatBoostRegressor

    model_dir = context.subdir("models", window.period)
    folder = context.subdir("backtests", window.period)
    predictions, evidence = tune_fold(data.frame, window, {"domestic": BASE_FEATURES}, seed=seed, model_dir=model_dir)
    prediction = predictions["domestic"]
    model_path = model_dir / evidence["arms"]["domestic"]["model_file"]
    pd.testing.assert_frame_equal(prediction, replay_catboost(data.frame.loc[prediction.index], model_path))
    model = CatBoostRegressor()
    model.load_model(str(model_path))
    sessions = _selection_sessions(data.sessions, window)
    choose = data.frame.loc[data.frame["date"].isin(sessions) & data.frame["signal_eligible"].eq(True)]
    raw = model.predict(_matrix(choose, BASE_FEATURES), thread_count=4)
    if not np.isfinite(raw).all():
        raise ValueError("组合调优窗口存在非有限预测")
    raw_scores = pd.Series(np.sort(raw, axis=1)[:, 1], index=choose.index, name="score")
    tuned, selection = _portfolio(data, window, raw_scores, context.run_id)
    _write_table(folder / "portfolio_selection_scores.parquet", choose[["date", "stock"]].join(raw_scores))
    write_json(folder / "portfolio_selection.json", selection)

    joined = data.frame.loc[prediction.index, ["date", "stock", "trade_date", "label_end", "target"]].join(prediction)
    joined["scored"] = joined["target"].notna() & joined["label_end"].le(window.evaluate[-1])
    joined["target"] = joined["target"].where(joined["scored"])
    measures = score_quantiles(joined)
    if not measures["rows"]:
        raise ValueError("评估窗口没有已结算标签")
    measures["all_signal_rows"] = len(joined)
    _write_table(folder / "domestic_predictions.parquet", joined)
    returns = {cost: {} for cost in COSTS}
    arms = {}
    for arm, strategy in zip(ARMS, (PORTFOLIO, tuned), strict=True):
        execution = {}
        for cost, multiplier in COSTS.items():
            outcome = _simulate(data, prediction["score"], window.evaluate, strategy, multiplier, context.run_id)
            verified = _verify_ledger(outcome, window.evaluate)
            returns[cost][arm] = equity_returns(outcome["daily"], INITIAL_CASH)
            execution[cost] = dict(
                outcome["metrics"], verification=verified,
                mean_cash_weight=float((outcome["daily"]["cash"] / outcome["daily"]["equity"]).mean()),
            )
            for key in ("daily", "trades", "orders", "positions"):
                _write_table(folder / f"{arm}_{cost}_{key}.parquet", outcome[key])
        arms[arm] = {"prediction": measures, "execution": execution, "strategy": asdict(strategy)}
    return {
        "period": window.period, "train_start": window.train_start, "valid_start": window.valid_start,
        "calibration_start": window.calibration_start, "evaluation_start": window.evaluate[0],
        "evaluation_end": window.evaluate[-1], "evaluation_sessions": len(window.evaluate),
        "models": evidence, "portfolio": selection, "arms": arms, "contrasts": _paired([returns], seed),
    }, returns


def run(*, run_id: str) -> Path:
    if run_id != slugify(run_id, limit=120):
        raise ValueError("运行编号必须为最多 120 个字符的单级目录名")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"run_id": run_id, "status": "running", "phase": "loading_inputs"})
    try:
        import catboost

        data = _load_frozen()
        checks = _features_validation(data.frame, data.sessions)
        config = {
            "source_run_id": BASELINE_ID, "end": HISTORICAL_END, "seed": 42, "stock_cap": 120,
            "domestic_features": list(BASE_FEATURES), "model_grid": GRID, "iterations": ITERATIONS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS, "portfolio_grid": PORTFOLIO_GRID,
            "maximum_calibration_drawdown": MAX_CALIBRATION_DRAWDOWN,
            "maximum_calibration_daily_turnover": MAX_CALIBRATION_DAILY_TURNOVER,
            "initial_cash": INITIAL_CASH, "fixed_portfolio": asdict(PORTFOLIO),
            "cost_multipliers": COSTS, "expense_model": asdict(StudyCostModel()), "catboost_version": catboost.__version__,
            "holdout_excluded_from": PROSPECTIVE_START,
            "code_sha256": {name: _sha(Path(__file__).with_name(name)) for name in (
                "domestic_runner.py", "tuning.py", "models.py", "runner.py", "evaluation.py", "execution.py",
            )},
        }
        seed = config["seed"]
        context = create_run_context(
            task="latent-quant-domestic-tuning", as_of_date=HISTORICAL_END, config=config,
            data_version=stable_hash(data.coverage), seed=seed, run_id=run_id,
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", data.coverage)
        write_json(context.subdir("validation") / "feature_label_timing.json", checks)
        folds, development = [], []
        for window in plan_windows(data.sessions):
            LOG.info("国内折内调优 %s", window.period)
            write_json(root / "status.json", {"run_id": run_id, "status": "running", "phase": window.period})
            result, returns = _fold(data, window, context, seed)
            write_json(context.subdir("reports") / f"{window.period}.json", result)
            folds.append(result)
            if window.period.startswith("oos_"):
                development.append(returns)
        contrasts = _paired(development, seed)
        decision = {
            "double_cost_wins": sum(fold["contrasts"][CONTRAST]["double"]["net_return_difference"] > 0
                                    for fold in folds if fold["period"].startswith("oos_")),
            "double_cost_ci_above_zero": contrasts[CONTRAST]["double"]["ci95_low"] > 0,
            "qualified_forward_results": False, "advance_tcn": False,
        }
        report = {
            "run": context.to_dict(), "input_coverage": data.coverage, "validation": checks, "folds": folds,
            "oos_block_bootstrap": contrasts, "decision": decision,
            "interpretation": {
                "comparison": "固定和调优仓位使用完全相同的模型与样本外预测。配对差衡量组合规则变化。",
                "portfolio_selection": "前置组合选择窗只使用已选模型的未校准分数和双倍成本行情。样本外统一使用独立校准分数。",
                "historical_replay": "既有复核期单列，排除于滚动合并统计。",
                "qualification": "历史研究，与 v2 前瞻封存隔离。",
            },
        }
        path = context.subdir("reports") / REPORT
        write_json(path, report)
        _write_table(context.subdir("reports") / "cost_and_prediction.csv", _summary_table(folds))
        write_json(context.subdir("reports") / "selected_hyperparameters.json", {
            fold["period"]: {"model": fold["models"]["arms"]["domestic"], "portfolio": fold["portfolio"]} for fold in folds
        })
        write_json(root / "status.json", {"run_id": run_id, "status": "completed", "phase": "complete", "advance_tcn": False})
        return path
    except Exception as error:
        write_json(root / "status.json", {"run_id": run_id, "status": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(run_id=args.run_id or make_run_id("latent-quant-domestic-tuning", HISTORICAL_END)))


if __name__ == "__main__":
    main()
