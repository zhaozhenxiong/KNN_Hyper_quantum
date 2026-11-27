"""Replay prespecified portfolio and calibration comparisons on frozen experts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from itertools import product

import numpy as np
import pandas as pd

from my_strategy.core.run_context import create_run_context, make_run_id, stable_hash
from my_strategy.latent_quant import models
from my_strategy.latent_quant.evaluation import equity_returns, paired_block_interval
from my_strategy.latent_quant.execution import Strategy, StudyCostModel, execution_evidence_identity, scaled_study_cost, simulate
from my_strategy.latent_quant.runner import _verify_ledger

from . import expert_calibration as calibration, expert_research as research
from .runner import _write_table, write_json


def _validation_split(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.DatetimeIndex(rows["date"].unique()).sort_values()
    if len(dates) < 60:
        raise ValueError("组合调优需要至少 60 个有折外预测的内层交易日。")
    cutoff = dates[len(dates) // 2]
    first = rows.loc[rows["date"].lt(cutoff) & rows["label_end"].lt(cutoff)]
    second = rows.loc[rows["date"].ge(cutoff)]
    if first.empty or second.empty:
        raise ValueError("组合调优缺少成熟的拟合样本或独立验证期。")
    return first, second


def _validation_scores(archive: calibration.Archive) -> tuple[pd.DataFrame, dict]:
    first_fold = archive.report["folds"][0]
    rows = pd.read_parquet(archive.root / "backtests" / first_fold["period"] / "inner_oof.parquet")
    first, second = _validation_split(rows)
    first_outer = pd.Timestamp(first_fold["evaluation_start"])
    if (second["date"].max() >= first_outer or second["label_end"].isna().any()
            or second["label_end"].max() >= first_outer):
        raise ValueError("组合调优日期与标签结束日必须早于首个外折。")
    names = [name for name in ("supertrend_atr", "domestic_gpu", "overseas_gpu") if f"{name}_q50" in first]
    used = [{name: point / 4 for name, point in zip(names, points, strict=True) if point}
            for points in product(range(5), repeat=len(names)) if sum(points) == 4 and points[1] >= 2]

    def forecasts(partition):
        return {name: partition[[f"{name}_{q}" for q in models.QUANTILE_COLUMNS]].to_numpy() for name in names}

    earlier = forecasts(first)
    weights = min(used, key=lambda value: (research._pinball(first, research._fuse(value, earlier)), len(value)))
    evidence = research._fuse(weights, earlier)
    raw = research._fuse(weights, forecasts(second))
    adjusted, offsets = models._calibrate(np.vstack((evidence, raw)), first["target"].to_numpy())
    result = second[["date", "stock"]].assign(score=adjusted[:, 1])
    return result, {"weights": weights, "calibration": offsets, "fit_through": str(first["date"].max().date()),
                    "fit_labels_through": str(first["label_end"].max().date()),
                    "selection_start": str(second["date"].min().date()), "selection_end": str(second["date"].max().date()),
                    "selection_labels_through": str(second["label_end"].max().date()), "first_outer_start": str(first_outer.date())}


def _scores(frame: pd.DataFrame, scored: pd.DataFrame) -> pd.Series:
    return frame[["date", "stock"]].join(scored.set_index(["date", "stock"])[["score"]],
                                        on=["date", "stock"], validate="one_to_one")["score"]


def _simulate(archive, frame, scores, dates, policy, fee, multiplier, context, ledger, phase, method, cost):
    outcome = simulate(frame, scores, dates, archive.data.benchmark_returns, run_id=context.run_id,
                       initial_cash=archive.config["initial_cash"], strategy=policy, cost_multiplier=multiplier,
                       market_status=archive.data.market_status, roster=archive.data.roster, base_cost_model=fee)
    verification = _verify_ledger(outcome, dates, initial_cash=archive.config["initial_cash"])
    for key, tables in ledger.items():
        tables.append(outcome[key].assign(window=phase, method=method, cost=cost))
    cash_share = outcome["daily"]["cash"] / outcome["daily"]["equity"]
    return (outcome["metrics"] | {"verification": verification, "mean_cash_share": float(cash_share.mean())},
            equity_returns(outcome["daily"], archive.config["initial_cash"]))


def run(source_run_id: str, run_id: str) -> str:
    archive = calibration.open_archive(source_run_id)
    config = archive.config
    original = Strategy(**config["portfolio"])
    fee = StudyCostModel(**config["expense_model"])
    stress = scaled_study_cost(config["cost_multipliers"]["double"], fee)
    variable_round_trip = 2 * (stress.commission + stress.transfer_fee + stress.slippage) + stress.stamp_tax
    variants = {f"rebalance_{days}_top_{count}": replace(original, rebalance_days=days, top_n=count)
                for days, count in product((3, 5, 10), (5, 10, 20))}
    variants["original"] = original
    variants["round_trip_buffer"] = replace(original, min_score=variable_round_trip)
    variants = {name: value for name, value in variants.items() if name == "original" or value != original}
    parameters = {
        "source_run_id": source_run_id, "prediction_evidence_sha256": archive.report["prediction_evidence_sha256"],
        "execution_evidence": execution_evidence_identity(), "initial_cash": config["initial_cash"],
        "cost": config["expense_model"], "multipliers": config["cost_multipliers"],
        "policy_grid": {name: asdict(value) for name, value in variants.items()},
        "calibrators": ["current", "raw", "date_equal", "shrinkage"],
        "selection": "first_outer_fold_inner_oof_only", "variable_round_trip_buffer": variable_round_trip,
        "block_sizes": [5, 10, 20],
    }
    context = create_run_context(task="expert-policy-study", as_of_date=config["end"], seed=config["seed"],
                                 config=parameters, data_version=stable_hash(archive.report["input_coverage"]["input_version"]),
                                 run_id=run_id)
    context.write_metadata()
    config_dir = context.subdir("config")
    config_dir.mkdir(parents=True, exist_ok=True)
    write_json(config_dir / "study.json", parameters)
    ledger = {key: [] for key in ("daily", "orders", "positions", "trades")}
    tuning, boundaries = _validation_scores(archive)
    frame = archive.data.frame
    fit_dates = pd.DatetimeIndex(tuning["date"].unique()).sort_values()
    scoped = frame.loc[frame["date"].isin(fit_dates)]
    scores = _scores(scoped, tuning)
    trial = {}
    for name, value in variants.items():
        trial[name], _ = _simulate(archive, scoped, scores, fit_dates, value, fee, config["cost_multipliers"]["double"],
                                   context, ledger, "inner_selection", name, "double")
    qualifying = [name for name, m in trial.items()
                  if m["order_count"] and m["max_drawdown"] >= -0.15 and m["mean_daily_turnover"] <= 0.20]
    chosen = max(qualifying, key=lambda name: (trial[name]["total_return"], -trial[name]["mean_daily_turnover"],
                                              name == "original"), default="original")
    if trial[chosen]["total_return"] <= max(0, trial["original"]["total_return"]):
        chosen = "original"
    print(f"inner portfolio selection: {chosen}", flush=True)

    outer, series = {}, {}
    for fold in research._archived_folds(archive.report):
        period = fold["period"]
        print(f"calibration and policy replay: {period}", flush=True)
        start, end = (pd.Timestamp(fold[key]) for key in ("evaluation_start", "evaluation_end"))
        dates = archive.data.sessions[(archive.data.sessions >= start) & (archive.data.sessions <= end)]
        scoped = frame.loc[frame["date"].isin(dates)]
        rows, forecasts, _ = calibration.recover_fold(archive, fold, arm="fusion")
        forecasts = {name: rows[["date", "stock"]].assign(score=values[:, 1]) for name, values in forecasts.items()}
        options = {name: (scored, original) for name, scored in forecasts.items()}
        if chosen != "original":
            options["selected_policy"] = (forecasts["current"], variants[chosen])
        outer[period], series[period] = {}, {}
        for name, (forecast, policy) in options.items():
            scoped_score = _scores(scoped, forecast)
            counts = forecast.groupby("date")["score"].apply(lambda score: int(score.gt(policy.min_score).sum()))
            phase = [int(counts.reindex(dates[offset:-1:policy.rebalance_days]).fillna(0).gt(0).sum())
                     for offset in range(policy.rebalance_days)]
            outer[period][name] = {"positive_rebalances_by_phase": phase, "execution": {}}
            series[period][name] = {}
            for cost, multiplier in config["cost_multipliers"].items():
                result, returns = _simulate(archive, scoped, scoped_score, dates, policy, fee, multiplier,
                                            context, ledger, period, name, cost)
                outer[period][name]["execution"][cost] = result
                series[period][name][cost] = returns

    windows = [fold["period"] for fold in archive.report["folds"]]
    effects = {}
    for name in series[windows[0]]:
        if name == "current":
            continue
        effects[name] = {}
        for cost in config["cost_multipliers"]:
            pairs = [(series[period][name][cost], series[period]["current"][cost]) for period in windows]
            effects[name][cost] = {str(length): paired_block_interval(pairs, seed=context.seed, block_size=length)
                                   for length in parameters["block_sizes"]}
    if execution_evidence_identity() != parameters["execution_evidence"]:
        raise ValueError("组合对照期间历史撮合规则发生变化，请重新提交。")
    backtests = context.subdir("backtests", "comparison")
    backtests.mkdir(parents=True, exist_ok=True)
    for key, tables in ledger.items():
        _write_table(backtests / f"{key}.parquet", pd.concat(tables, ignore_index=True))
    reports = context.subdir("reports")
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / "expert_policy_study.json"
    write_json(path, {"run": context.to_dict(), "configuration": parameters, "selection_boundary": boundaries,
                      "chosen_policy": chosen, "inner_trials": trial, "outer": outer, "paired_effects": effects,
                      "historical_review_excluded_from_selection": True, "release_eligible": False})
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-id", default=calibration.SOURCE)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    print(run(args.source_run_id, args.run_id or make_run_id("expert-policy-study")), flush=True)


if __name__ == "__main__":
    main()
