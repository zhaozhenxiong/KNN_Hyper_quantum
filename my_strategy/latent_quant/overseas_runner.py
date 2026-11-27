"""Run independently tuned domestic and overseas historical comparisons."""

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

from .data import BASE_FEATURES, load_panel
from .evaluation import equity_returns, paired_block_interval, score_quantiles, summarize_regime_returns
from .execution import StudyCostModel, simulate
from .models import replay_catboost
from .overseas_data import CONTINUOUS_FEATURES, DEFAULT_OVERSEAS_ROOT, OVERSEAS_FEATURES, load_overseas
from .runner import (
    COSTS, HISTORICAL_END, INITIAL_CASH, PORTFOLIO, PROSPECTIVE_START, START,
    _features_validation, _regimes, _summary_table, _verify_ledger, plan_windows,
)
from .tuning import add_fold_pca, tune_fold

LOG = logging.getLogger(__name__)
CONTRASTS = {"raw_vs_domestic": ("overseas_raw", "domestic"), "pca_vs_raw": ("overseas_pca", "overseas_raw")}
MARKETS = ("sp500", "nikkei", "kospi")


def _paired(returns: list[dict], seed: int) -> dict:
    return {
        name: {
            cost: paired_block_interval(
                [(item[cost][candidate], item[cost][reference]) for item in returns], seed=seed,
            ) for cost in COSTS
        } for name, (candidate, reference) in CONTRASTS.items()
    }


def _run_fold(data, window: Window, context, seed: int) -> tuple[dict, dict]:
    frame, pca = add_fold_pca(data.frame, window, CONTINUOUS_FEATURES)
    raw = [*BASE_FEATURES, *OVERSEAS_FEATURES]
    features = {
        "domestic": list(BASE_FEATURES), "overseas_raw": raw,
        "overseas_pca": [*raw, "overseas_pc1", "overseas_pc2"],
    }
    if any(frame[column].notna().sum() == 0 for column in CONTINUOUS_FEATURES):
        raise ValueError("At least one required overseas feature has no observations")
    model_dir = context.subdir("models", window.period)
    fold_dir = context.subdir("backtests", window.period)
    write_json(model_dir / "pca.json", pca)
    state_rows = frame[["date", "overseas_pc1", "overseas_pc2"]].drop_duplicates()
    if state_rows["date"].duplicated().any():
        raise ValueError("The market state must be identical for all stocks on a session")
    _write_table(model_dir / "pca_by_date.parquet", state_rows)
    predictions, evidence = tune_fold(frame, window, features, seed=seed, model_dir=model_dir)
    first = predictions["domestic"].index
    cohort = frame.loc[first, ["date", "stock", "trade_date", "label_end", "target"]]
    scores = cohort["target"].notna() & cohort["label_end"].le(window.evaluate[-1])
    if not scores.any():
        raise ValueError("The evaluation fold has no matured common labels")
    states, thresholds = _regimes(frame, window)
    lagged = states.reindex(data.sessions).shift().reindex(window.evaluate)
    fold = {
        "period": window.period, "train_start": window.train_start, "valid_start": window.valid_start,
        "calibration_start": window.calibration_start, "evaluation_start": window.evaluate[0],
        "evaluation_end": window.evaluate[-1], "evaluation_sessions": len(window.evaluate),
        "models": evidence, "regime_training_medians": thresholds, "arms": {},
    }
    returns = {cost: {} for cost in COSTS}
    for arm, prediction in predictions.items():
        if not prediction.index.equals(first) or not np.isfinite(prediction.to_numpy()).all():
            raise ValueError(f"The {arm} prediction failed the common cohort check")
        meta = evidence["arms"][arm]
        model = model_dir / meta["model_file"]
        pd.testing.assert_frame_equal(replay_catboost(frame.loc[first], model), prediction)
        joined = cohort.join(prediction).assign(scored=scores)
        joined["target"] = joined["target"].where(scores)
        _write_table(fold_dir / f"{arm}_predictions.parquet", joined)
        scored = joined.loc[scores]
        measures = score_quantiles(scored)
        measures["all_signal_rows"] = len(joined)
        measures["regimes"] = {
            key: {level: score_quantiles(scored.loc[scored["date"].map(states[key]).eq(level)])
                  for level in ("low", "high")} for key in states
        }
        result = {"prediction": measures, "execution": {}}
        for cost, multiplier in COSTS.items():
            outcome = simulate(
                frame, prediction["score"], window.evaluate, data.benchmark_returns,
                run_id=context.run_id, cost_multiplier=multiplier, initial_cash=INITIAL_CASH,
                strategy=PORTFOLIO, market_status=data.market_status, roster=data.roster,
            )
            verified = _verify_ledger(outcome, window.evaluate)
            days = equity_returns(outcome["daily"], INITIAL_CASH)
            returns[cost][arm] = days
            result["execution"][cost] = dict(
                outcome["metrics"], verification=verified,
                mean_cash_weight=float((outcome["daily"]["cash"] / outcome["daily"]["equity"]).mean()),
                regimes=summarize_regime_returns(days, lagged),
            )
            for key in ("daily", "trades", "orders", "positions"):
                _write_table(fold_dir / f"{arm}_{cost}_{key}.parquet", outcome[key])
        fold["arms"][arm] = result
    return fold, returns


def _decision(folds: list[dict], pooled: dict) -> dict:
    development = [fold for fold in folds if fold["period"].startswith("oos_")]
    evidence = {}
    for name, (candidate, reference) in CONTRASTS.items():
        wins = sum(fold["arms"][candidate]["prediction"]["pinball"] <
                   fold["arms"][reference]["prediction"]["pinball"] for fold in development)
        stress_wins = sum(fold["contrasts"][name]["double"]["net_return_difference"] > 0
                          for fold in development)
        interval_above_zero = pooled[name]["double"]["ci95_low"] > 0
        evidence[name] = {
            "pinball_wins": wins, "double_cost_wins": stress_wins,
            "double_cost_ci_above_zero": interval_above_zero,
            "historical_numerical_gate": wins >= 2 and stress_wins >= 2 and interval_above_zero,
        }
    return {
        "historical": evidence, "qualified_forward_results": False, "advance_tcn": False,
        "qualification": "Historical comparison only. The v2 baseline and separately sealed overseas forward windows have no acceptance results.",
    }


def _coverage_by_fold(overseas: pd.DataFrame, window: Window) -> dict:
    observed = overseas.reindex(window.evaluate)
    return {
        market: int(observed[f"{market}_return_1d"].notna().sum())
        for market in MARKETS
    }


def run(*, run_id: str, source_root: str | Path, seed: int = 42) -> Path:
    import catboost

    if run_id != slugify(run_id, limit=120):
        raise ValueError("Run ID must be a simple directory name of at most 120 characters")
    root = artifact_run_dir(run_id, create=False)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "status.json", {"status": "running", "phase": "loading_inputs"})
    try:
        data = load_panel(START, HISTORICAL_END, 120, seed)
        if data.sessions[-1] >= pd.Timestamp(PROSPECTIVE_START):
            raise ValueError("Historical research cannot consume the v2 forward window")
        overseas, overseas_evidence = load_overseas(data.sessions, root=source_root)
        if not overseas.index.is_unique or not overseas.index.equals(data.sessions):
            raise ValueError("Overseas observations must share the A-share session index")
        frame = data.frame.sort_values(["date", "stock"]).reset_index(drop=True)
        frame = frame.join(overseas, on="date", validate="many_to_one")
        data = replace(data, frame=frame)
        windows = plan_windows(data.sessions)
        coverage = {window.period: _coverage_by_fold(overseas, window) for window in windows}
        if any(not all(counts.values()) for counts in coverage.values()):
            raise ValueError(f"An entire evaluation fold has no observations for an overseas market: {coverage}")
        code = [
            Path(__file__).with_name(name + ".py") for name in (
                "data", "evaluation", "execution", "models", "overseas_data", "overseas_runner", "runner", "tuning",
            )
        ]
        code.extend(MY_STRATEGY_ROOT / "research_workbench" / f"{name}.py" for name in ("data", "runner", "universe"))
        code.extend((MY_STRATEGY_ROOT / "execution").glob("*.py"))
        config = {
            "start": START, "end": HISTORICAL_END, "seed": seed, "stock_cap": 120,
            "market_data_path": str(Path(source_root).resolve()),
            "arms": ["domestic", "overseas_raw", "overseas_pca"],
            "market_cutoff": "Asia/Shanghai 18:00", "holdout_excluded_from": PROSPECTIVE_START,
            "domestic_features": list(BASE_FEATURES), "overseas_features": list(OVERSEAS_FEATURES),
            "pca_input": list(CONTINUOUS_FEATURES), "pca_components": 2,
            "portfolio": asdict(PORTFOLIO), "initial_cash": INITIAL_CASH,
            "cost_model": asdict(StudyCostModel()), "cost_multipliers": COSTS,
            "catboost_version": catboost.__version__,
            "code_sha256": {str(path.relative_to(MY_STRATEGY_ROOT)): sha256(path.read_bytes()).hexdigest()
                            for path in sorted(code)},
        }
        lineage = {"domestic": data.coverage, "overseas": overseas_evidence, "overseas_sessions_by_fold": coverage}
        context = create_run_context(
            task="latent-quant-overseas", run_id=run_id, as_of_date=HISTORICAL_END, seed=seed,
            config=config, data_version=stable_hash(lineage),
        )
        write_json(context.subdir("config") / "study.json", config)
        write_json(context.subdir("data_snapshots") / "coverage.json", lineage)
        _write_table(context.subdir("data_snapshots") / "overseas.parquet", overseas.rename_axis("date").reset_index())
        _write_table(context.subdir("data_snapshots") / "panel.parquet", frame)
        _write_table(context.subdir("data_snapshots") / "market_status.parquet", data.market_status)
        _write_table(context.subdir("data_snapshots") / "roster.parquet", data.roster.reset_index())
        _write_table(context.subdir("data_snapshots") / "benchmark.parquet", pd.DataFrame({
            "date": data.sessions, "open": data.benchmark_open.reindex(data.sessions).to_numpy(),
            "close": data.benchmark_close.reindex(data.sessions).to_numpy(),
            "close_return": data.benchmark_returns.reindex(data.sessions).to_numpy(),
        }))
        checks = _features_validation(frame, data.sessions)
        write_json(context.subdir("validation") / "feature_label_timing.json", checks)
        reports, development = [], []
        for window in windows:
            LOG.info("Tuning and evaluating %s", window.period)
            write_json(root / "status.json", {"status": "running", "phase": window.period})
            result, returns = _run_fold(data, window, context, seed)
            result["contrasts"] = _paired([returns], seed)
            result["market_observed_sessions"] = coverage[window.period]
            write_json(context.subdir("reports") / f"{window.period}.json", result)
            reports.append(result)
            if window.period.startswith("oos_"):
                development.append(returns)
        pooled = _paired(development, seed)
        decision = _decision(reports, pooled)
        summary = {
            "run": context.to_dict(), "input_coverage": lineage, "validation": checks,
            "folds": reports, "oos_block_bootstrap": pooled, "decision": decision,
            "training": "Equal tuning budgets; validation-only winner, reused unchanged for calibration and evaluation.",
            "historical_replay": "Prior evaluated date range, excluded from development aggregation and the forward decision.",
        }
        result = context.subdir("reports") / "overseas_research.json"
        write_json(result, summary)
        _write_table(context.subdir("reports") / "cost_and_prediction.csv", _summary_table(reports))
        write_json(context.subdir("reports") / "selected_hyperparameters.json", {
            fold["period"]: {arm: evidence["winner"] for arm, evidence in fold["models"]["arms"].items()}
            for fold in reports
        })
        write_json(root / "status.json", {"status": "completed", "phase": "complete", "advance_tcn": False})
        return result
    except Exception as error:
        write_json(root / "status.json", {"status": "failed", "error": str(error)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and evaluate domestic and overseas Latent-Quant feature sets")
    parser.add_argument("--source-root", default=str(DEFAULT_OVERSEAS_ROOT))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(run(run_id=args.run_id or make_run_id("latent-quant-overseas", HISTORICAL_END), source_root=args.source_root))


if __name__ == "__main__":
    main()
