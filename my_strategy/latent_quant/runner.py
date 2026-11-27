"""Run fixed-policy Latent-Quant baselines against frozen market inputs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from hashlib import sha256
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.core.paths import MY_STRATEGY_ROOT, artifact_run_dir
from my_strategy.core.run_context import create_run_context, make_run_id, slugify, stable_hash
from my_strategy.research_workbench.runner import Window, _write_table, write_json
from my_strategy.validation.leakage import validate_feature_columns_no_lookahead, validate_feature_label_timing
from my_strategy.validation.signal_timing import validate_signal_execution_timing

from .data import BASE_FEATURES, EXTRA_FEATURES, load_panel
from .evaluation import equity_returns, paired_block_interval, score_quantiles, summarize_regime_returns
from .execution import Strategy, StudyCostModel, simulate
from .models import ARMS, predict_fold, replay_catboost

LOG = logging.getLogger(__name__)
START = "2024-01-02"
HISTORICAL_END = "2026-11-20"
PROSPECTIVE_START = "2026-11-23"
PROSPECTIVE_SESSIONS = 60
OOS_STARTS = ("2025-03-03", "2025-10-09", "2026-04-01", "2026-07-29")
INITIAL_CASH = 1_000_000.0
PORTFOLIO = Strategy()
COSTS = {"normal": 1.0, "double": 2.0}
COMPARISONS = {
    "catboost_base_vs_historical": ("catboost_base", "historical_quantiles"),
    "catboost_extended_vs_historical": ("catboost_supertrend_atr", "historical_quantiles"),
    "catboost_extended_vs_base": ("catboost_supertrend_atr", "catboost_base"),
    "supertrend_vs_historical": ("supertrend_atr", "historical_quantiles"),
}
INPUT_ARGS = (
    "raw_db", "roster_path", "calendar_path", "benchmark_path", "adjusted_path", "execution_reference_path",
    "classification_path", "status_paths",
)


def plan_windows(sessions: pd.DatetimeIndex) -> list[Window]:
    """Build frozen date windows with optional evaluation after 60 fresh sessions."""
    if sessions[0] > pd.Timestamp(START) or sessions[-1] < pd.Timestamp(HISTORICAL_END):
        raise ValueError("Inputs must cover the frozen historical study span")
    bounds = [pd.Timestamp(value) for value in (*OOS_STARTS, PROSPECTIVE_START)]
    periods = [
        (name, sessions[(sessions >= first) & (sessions < last)])
        for name, first, last in zip(
            ("oos_1", "oos_2", "oos_3", "historical_replay"), bounds[:-1], bounds[1:], strict=True
        )
    ]
    prospective = sessions[sessions >= bounds[-1]]
    if len(prospective) >= PROSPECTIVE_SESSIONS:
        periods.append(("prospective_holdout", prospective[:PROSPECTIVE_SESSIONS]))
    result = []
    for name, dates in periods:
        offset = sessions.get_loc(dates[0])
        result.append(Window(
            name, sessions[max(0, offset - 315)], sessions[offset - 70], sessions[offset - 35], dates,
        ))
    return result


def _features_validation(frame: pd.DataFrame, sessions: pd.DatetimeIndex) -> dict:
    calendar = pd.Series(sessions, index=sessions)
    for column, offset in (("trade_date", 1), ("label_end", 6)):
        expected = frame["date"].map(calendar.shift(-offset))
        actual = frame[column]
        paired = expected.notna() | actual.notna()
        if not actual[paired].eq(expected[paired]).all():
            raise ValueError(f"{column} is not aligned to the market calendar")
    if frame.loc[frame["target"].notna(), "label_end"].isna().any():
        raise ValueError("A label is present without a known exit session")
    timing = validate_feature_label_timing(frame, label_end_col="label_end", label_horizon_days=5)
    columns = validate_feature_columns_no_lookahead([*BASE_FEATURES, *EXTRA_FEATURES])
    if not timing["ok"] or not columns["ok"]:
        raise ValueError(f"Future-function checks failed: {timing}, {columns}")
    return {"label_calendar_verified": True, "feature_timing": timing, "feature_names": columns}


def _regimes(frame: pd.DataFrame, window: Window) -> tuple[pd.DataFrame, dict]:
    regime = frame.groupby("date").agg(
        trend=("breadth", "median"), volatility=("volatility_20d", "median"), liquidity=("amount", "median")
    )
    training = regime.index.to_series().between(window.train_start, window.valid_start, inclusive="left")
    thresholds = regime.loc[training].median()
    if thresholds.isna().any():
        raise ValueError("Regime thresholds require complete training-period summaries")
    groups = regime.ge(thresholds).replace({False: "low", True: "high"}).where(regime.notna())
    return groups, thresholds.to_dict()


def _verify_ledger(simulation: dict, sessions: pd.DatetimeIndex, *, initial_cash: float = INITIAL_CASH) -> dict:
    daily, trades, orders, positions = (simulation[key] for key in ("daily", "trades", "orders", "positions"))
    date = daily["date"]
    if not pd.DatetimeIndex(pd.to_datetime(date)).equals(sessions):
        raise ValueError("Daily ledger does not match the evaluation calendar")
    stock_value = positions["shares"].astype(float) * positions["mark_price"].astype(float)
    cash = initial_cash + date.map(trades.groupby("date")["cash_flow"].sum()).fillna(0).cumsum()
    equity = cash + date.map(stock_value.groupby(positions["date"]).sum()).fillna(0)
    np.testing.assert_allclose(stock_value, positions["market_value"].astype(float), rtol=0, atol=1e-7)
    tolerance = max(1e-6, 32 * float(np.spacing(initial_cash)))
    np.testing.assert_allclose(cash, daily["cash"], rtol=0, atol=tolerance)
    np.testing.assert_allclose(equity, daily["equity"], rtol=0, atol=tolerance)
    timing = validate_signal_execution_timing(orders, signal_time_col="signal_date", execution_time_col="date")
    expected = pd.Series(sessions[1:], index=sessions[:-1])
    execution_date = pd.to_datetime(orders["date"])
    next_session = (True if orders.empty else
                    pd.to_datetime(orders["signal_date"]).map(expected).eq(execution_date).all())
    if not timing["ok"] or not next_session:
        raise ValueError("An order was not scheduled for the next evaluation session")
    return {"cash_verified": True, "equity_verified": True, "next_session_verified": True}


def _evaluate_fold(data, window: Window, context, *, iterations: int, seed: int) -> tuple[dict, dict]:
    frame = data.frame
    model_dir = context.subdir("models", window.period)
    backtest_dir = context.subdir("backtests", window.period)
    groups, thresholds = _regimes(frame, window)
    predictions, evidence = predict_fold(
        frame, window, BASE_FEATURES, EXTRA_FEATURES, seed=seed, model_dir=model_dir, iterations=iterations
    )
    fold = {
        "period": window.period, "train_start": window.train_start, "valid_start": window.valid_start,
        "calibration_start": window.calibration_start, "evaluation_start": window.evaluate[0],
        "evaluation_end": window.evaluate[-1], "evaluation_sessions": len(window.evaluate),
        "regime_training_medians": thresholds, "models": evidence, "arms": {},
    }
    score_index = predictions[ARMS[0]].index
    cohort = frame.loc[score_index, ["date", "stock", "trade_date", "label_end", "target"]]
    returns = {cost: {} for cost in COSTS}
    for arm, pred in predictions.items():
        if not pred.index.equals(score_index):
            raise ValueError("Arms returned different signal cohorts")
        if not np.isfinite(pred.to_numpy()).all():
            raise ValueError(f"{arm} returned a non-finite prediction")
        if arm.startswith("catboost"):
            model = model_dir / evidence["arms"][arm]["model_file"]
            pd.testing.assert_frame_equal(replay_catboost(frame.loc[score_index], model), pred)

        joined = cohort.join(pred)
        joined["scored"] = joined["target"].notna() & joined["label_end"].le(window.evaluate[-1])
        recorded = joined.copy()
        recorded.loc[~recorded["scored"], "target"] = np.nan
        _write_table(backtest_dir / f"{arm}_predictions.parquet", recorded)
        scored = joined.loc[joined["scored"]]
        prediction = score_quantiles(scored)
        prediction["all_signal_rows"] = len(joined)
        prediction["regimes"] = {
            key: {
                level: score_quantiles(scored.loc[scored["date"].map(groups[key]).eq(level)])
                for level in ("low", "high")
            }
            for key in groups
        }
        result = {"prediction": prediction, "execution": {}}
        for cost, multiplier in COSTS.items():
            LOG.info("%s %s %s", window.period, arm, cost)
            simulated = simulate(
                frame, pred["score"], window.evaluate, data.benchmark_returns,
                run_id=context.run_id, cost_multiplier=multiplier, initial_cash=INITIAL_CASH,
                strategy=PORTFOLIO, market_status=data.market_status, roster=data.roster,
            )
            verified = _verify_ledger(simulated, window.evaluate)
            by_date = equity_returns(simulated["daily"], INITIAL_CASH)
            returns[cost][arm] = by_date
            attributed = groups.reindex(data.sessions).shift().reindex(window.evaluate)
            result["execution"][cost] = dict(
                simulated["metrics"], verification=verified, regimes=summarize_regime_returns(by_date, attributed),
            )
            for key in ("daily", "trades", "orders", "positions"):
                _write_table(backtest_dir / f"{arm}_{cost}_{key}.parquet", simulated[key])
        fold["arms"][arm] = result
    fold["contrasts"] = _comparisons([returns], seed)
    return fold, returns


def _comparisons(returns: list[dict], seed: int) -> dict:
    return {
        comparison: {
            cost: paired_block_interval(
                [(fold[cost][candidate], fold[cost][reference]) for fold in returns], seed=seed,
            )
            for cost in COSTS
        }
        for comparison, (candidate, reference) in COMPARISONS.items()
    }


def _decision(folds: list[dict], pooled: dict, point_in_time: bool) -> dict:
    dev = [fold for fold in folds if fold["period"].startswith("oos_")]
    holdout = next((fold for fold in folds if fold["period"] == "prospective_holdout"), None)
    candidates = {}
    for arm in ("catboost_base", "catboost_supertrend_atr"):
        comparison = "catboost_base_vs_historical" if arm == "catboost_base" else "catboost_extended_vs_historical"
        better = sum(
            fold["arms"][arm]["prediction"]["pinball"] < fold["arms"]["historical_quantiles"]["prediction"]["pinball"]
            for fold in dev
        )
        positive_stress = sum(fold["contrasts"][comparison]["double"]["net_return_difference"] > 0 for fold in dev)
        better_holdout = holdout is not None and (
            holdout["arms"][arm]["prediction"]["pinball"] < holdout["arms"]["historical_quantiles"]["prediction"]["pinball"]
            and all(holdout["contrasts"][comparison][cost]["net_return_difference"] > 0 for cost in COSTS)
        )
        candidates[arm] = {
            "oos_pinball_wins": better,
            "oos_positive_double_cost_folds": positive_stress,
            "oos_double_cost_ci_above_zero": pooled[comparison]["double"]["ci95_low"] > 0,
            "prospective_holdout_pass": better_holdout,
        }
    return {
        "advance_tcn": point_in_time and holdout is not None and any(
            item["oos_pinball_wins"] >= 2 and item["oos_positive_double_cost_folds"] >= 2
            and item["oos_double_cost_ci_above_zero"] and item["prospective_holdout_pass"]
            for item in candidates.values()
        ),
        "point_in_time_verified": point_in_time,
        "prospective_holdout_available": holdout is not None,
        "candidates": candidates,
    }


def _summary_table(folds: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "fold": fold["period"], "arm": arm, "cost": cost, **{
                column: result["prediction"][column] for column in ("rows", "pinball", "coverage_80", "width_80")
            },
            **{column: metrics[column] for column in (
                "total_return", "excess_return", "max_drawdown", "turnover",
                "trade_count", "order_count", "reject_rate", "partial_order_count",
            )},
        }
        for fold in folds
        for arm, result in fold["arms"].items()
        for cost, metrics in result["execution"].items()
    ])


def run(*, run_id: str, end: str, max_stocks: int, seed: int, iterations: int, input_paths: dict) -> Path:
    import catboost

    if run_id != slugify(run_id, limit=120):
        raise ValueError("Run ID must be a simple directory name of at most 120 characters")
    if pd.Timestamp(end) < pd.Timestamp(HISTORICAL_END):
        raise ValueError(f"End must cover the frozen historical span through {HISTORICAL_END}")
    if not 0 < iterations or not 3 <= max_stocks:
        raise ValueError("Need at least three stocks and one training iteration")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"status": "running", "phase": "loading_inputs"})
    try:
        LOG.info("Loading market data for %s", run_id)
        data = load_panel(START, end, max_stocks, seed, **input_paths)
        frame = data.frame.sort_values(["date", "stock"]).reset_index(drop=True)
        data = replace(data, frame=frame)
        windows = plan_windows(data.sessions)
        if data.benchmark_open.reindex(data.sessions).isna().any() or data.benchmark_close.reindex(data.sessions).isna().any():
            raise ValueError("The benchmark must have both open and close throughout the study")
        code_paths = (
            list(Path(__file__).parent.glob("*.py"))
            + [MY_STRATEGY_ROOT / "research_workbench" / name for name in ("data.py", "runner.py", "universe.py")]
            + list((MY_STRATEGY_ROOT / "execution").glob("*.py"))
        )
        config = {
            "start": START, "end": end, "max_stocks": max_stocks, "seed": seed, "iterations": iterations,
            "arms": list(ARMS), "base_features": list(BASE_FEATURES), "added_features": list(EXTRA_FEATURES),
            "portfolio": asdict(PORTFOLIO), "initial_cash": INITIAL_CASH,
            "expense_model": asdict(StudyCostModel()), "cost_multipliers": COSTS,
            "oos_starts": list(OOS_STARTS[:3]), "historical_replay_start": OOS_STARTS[3],
            "prospective": {
                "start_at_or_after": PROSPECTIVE_START, "market_sessions": PROSPECTIVE_SESSIONS,
                "method": "Fixed candidates and settings without using historical replay to change the prospective design",
            },
            "input_overrides": input_paths, "catboost_version": catboost.__version__,
            "code_sha256": {
                str(path.relative_to(MY_STRATEGY_ROOT)): sha256(path.read_bytes()).hexdigest()
                for path in sorted(code_paths)
            },
        }
        context = create_run_context(
            task="latent-quant", run_id=run_id, as_of_date=end, seed=seed, config=config,
            data_version=stable_hash(data.coverage),
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", data.coverage)
        _write_table(context.subdir("data_snapshots") / "panel.parquet", frame)
        _write_table(context.subdir("data_snapshots") / "market_status.parquet", data.market_status)
        _write_table(context.subdir("data_snapshots") / "roster.parquet", data.roster.reset_index())
        _write_table(context.subdir("data_snapshots") / "benchmark.parquet", pd.DataFrame({
            "date": data.sessions, "open": data.benchmark_open.reindex(data.sessions).to_numpy(),
            "close": data.benchmark_close.reindex(data.sessions).to_numpy(),
            "daily_return": data.benchmark_returns.reindex(data.sessions).to_numpy(),
        }))
        verified = _features_validation(frame, data.sessions)
        write_json(context.subdir("validation") / "feature_label_timing.json", verified)
        folds, development_returns = [], []
        for window in windows:
            LOG.info("Evaluating %s", window.period)
            write_json(root / "status.json", {"status": "running", "phase": window.period})
            fold, returns = _evaluate_fold(data, window, context, iterations=iterations, seed=seed)
            write_json(context.subdir("reports") / f"{window.period}.json", fold)
            folds.append(fold)
            if window.period.startswith("oos_"):
                development_returns.append(returns)
        pooled = _comparisons(development_returns, seed)
        decision = _decision(folds, pooled, data.coverage["point_in_time"]["verified"])
        report = {
            "run": context.to_dict(), "validation": verified, "input_coverage": data.coverage,
            "folds": folds, "oos_block_bootstrap": pooled, "decision": decision,
            "interpretation": {
                "historical_replay": "Previously evaluated date range, excluded from model selection and prospective acceptance",
                "oos": "Rolling research tests, accounts reset to cash in each fold",
                "execution": "Daily opening-price proxy and approximate limits when authoritative flags are absent",
                "positions": "Raw close or most recent usable mark. Dividends, splits and delisting recoveries are not simulated",
                "costs": "Assumed broker commission and slippage, with minimum commission, sell stamp tax and bilateral transfer fee",
                "benchmark_nav": "Close-to-close CSI 300; target uses T+1 to T+6 benchmark opens",
                "industry_limit": "Dated CAPCO classification when known; board cap remains active for unknown sectors",
                "liquidity_regime": "Median daily traded amount on the frozen cohort, grouped by training-period median",
                "pinball_and_coverage": "Each evaluation trading day has equal weight",
            },
        }
        report_path = context.subdir("reports") / "latent_quant.json"
        write_json(report_path, report)
        _write_table(context.subdir("reports") / "cost_and_prediction.csv", _summary_table(folds))
        write_json(root / "status.json", {"status": "completed", "phase": "complete", "advance_tcn": decision["advance_tcn"]})
        return report_path
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fixed-policy Latent-Quant CPU baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--end", default=HISTORICAL_END)
    parser.add_argument("--max-stocks", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=100)
    for key in INPUT_ARGS:
        if key == "status_paths":
            parser.add_argument("--status-path", dest=key, action="append")
        else:
            parser.add_argument("--" + key.replace("_", "-"))
    args = parser.parse_args()
    kwargs = vars(args)
    paths = {key: value for key in INPUT_ARGS if (value := kwargs.pop(key)) is not None}
    kwargs["run_id"] = args.run_id or make_run_id("latent-quant", args.end)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(**kwargs, input_paths=paths))


if __name__ == "__main__":
    main()
