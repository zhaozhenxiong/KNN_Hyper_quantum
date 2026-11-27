"""Evaluate the original Latent-Quant holdout from sealed same-day evidence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, datetime, time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import create_run_context, slugify, stable_hash
from my_strategy.research_workbench.runner import _write_table, write_json

from . import forward
from .evaluation import equity_returns, score_quantiles
from .execution import Strategy, StudyCostModel, simulate
from .forward_models import MANIFEST, replay_frozen_baseline
from .models import ARMS
from .runner import INITIAL_CASH, _comparisons, _verify_ledger


MODEL_ROOT = artifact_run_dir("latent-quant-forward-baseline-models-20261123-v1", create=False)
ACCEPTANCE_RUN = "latent-quant-forward-acceptance-v1"
_FORECAST = ["q10", "q50", "q90", "score"]
_ORDER = ["signal_date", "date", "stock", "side", "requested_shares"]


def _holdout_labels(panel: pd.DataFrame, status: pd.DataFrame, benchmark: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.DataFrame:
    dates = pd.Series(sessions, index=sessions)
    entry = panel["date"].map(dates.shift(-1))
    maturity = panel["date"].map(dates.shift(-forward.LABEL_OFFSET))
    keys = pd.MultiIndex.from_frame(panel[["stock", "date"]])
    statuses = status.set_index(["stock", "date"])["market_status"].reindex(keys).to_numpy()
    usable = panel["open"].gt(0) & panel["trade_open"].gt(0) & ~pd.Series(statuses, index=panel.index).isin({"suspended", "delisted"})
    if "has_trade_price" in panel:
        usable &= panel["has_trade_price"].eq(True)
    if "volume" in panel:
        usable &= panel["volume"].gt(0)
    stock_open = pd.Series(panel["open"].where(usable).to_numpy(), index=keys)
    lead = stock_open.reindex(pd.MultiIndex.from_arrays([panel["stock"], entry])).to_numpy()
    exit_ = stock_open.reindex(pd.MultiIndex.from_arrays([panel["stock"], maturity])).to_numpy()
    index_open = benchmark.set_index("date")["open"]
    target = (exit_ / lead) - (maturity.map(index_open) / entry.map(index_open))
    target = target.where(panel["signal_eligible"].eq(True) & np.isfinite(target))
    return pd.DataFrame({"trade_date": entry, "label_end": maturity, "target": target}, index=panel.index)


def _check_orders(simulated: pd.DataFrame, sealed: pd.DataFrame, sessions: pd.DatetimeIndex) -> None:
    def order_quantities(rows: pd.DataFrame) -> pd.DataFrame:
        result = rows[_ORDER].copy()
        result[["signal_date", "date"]] = result[["signal_date", "date"]].apply(pd.to_datetime)
        return result.sort_values(_ORDER).reset_index(drop=True)

    submitted = simulated.loc[simulated["requested_shares"].gt(0)]
    frozen = sealed.loc[pd.to_datetime(sealed["signal_date"]).ne(sessions[-1])]
    pd.testing.assert_frame_equal(order_quantities(submitted), order_quantities(frozen), check_dtype=False)
    if not simulated["filled_shares"].le(simulated["requested_shares"]).all():
        raise ValueError("A simulated fill exceeded the sealed order size")


def _check_daily_forecasts(panel: pd.DataFrame, sealed: pd.DataFrame, model_root: Path) -> None:
    replayed = replay_frozen_baseline(panel, model_root)
    for arm in ARMS:
        expected = replayed[arm].set_axis(panel.loc[replayed[arm].index, "stock"].to_numpy()).sort_index()
        actual = sealed.loc[sealed["arm"].eq(arm)].set_index("stock")[_FORECAST].sort_index()
        if expected.index.tolist() != actual.index.tolist():
            raise ValueError(f"Sealed forecast keys differ from the frozen {arm} model")
        np.testing.assert_allclose(expected.to_numpy(), actual.to_numpy(), rtol=0, atol=1e-12)


def _sealed_tables(root: Path, records: dict, calendar: dict, days: list[date], policy: forward.Policy, model_root: Path) -> dict:
    roles = ("panel", "benchmark", "market_status", "predictions", "orders")
    collected = {role: [] for role in roles}
    for day in days:
        record = records[day]
        folder = root / "captures" / str(day)
        paths = {role: folder / value["file"] for role, value in record["files"].items()}
        next_day = forward._ordered_sessions(calendar, date.fromordinal(day.toordinal() + 1))[0]
        if date.fromisoformat(record["next_trade_date"]) != next_day:
            raise ValueError(f"Sealed next-session date conflicts with the verified calendar for {day}")
        forward._verify_inputs(paths, policy, day, next_day, forward._timestamp(record["captured_at"]))
        for role in roles:
            rows = forward._table(paths[role], forward.FIELDS[role])
            rows["date"] = pd.to_datetime(rows["date"])
            if role == "panel":
                panel = rows
            elif role == "predictions":
                _check_daily_forecasts(panel, rows, model_root)
            collected[role].append(rows)
    return {role: pd.concat(tables, ignore_index=True) for role, tables in collected.items()}


def _evaluate(tables: dict, sessions: pd.DatetimeIndex, config: dict, roster: pd.DataFrame, run_id: str) -> tuple[dict, dict]:
    if config["initial_cash"] != INITIAL_CASH or config["expense_model"] != asdict(StudyCostModel()):
        raise ValueError("Current execution assumptions differ from the frozen v2 configuration")
    panel, benchmark, market, predictions, orders = (tables[k] for k in ("panel", "benchmark", "market_status", "predictions", "orders"))
    if not np.isfinite(benchmark[["open", "close"]].to_numpy(dtype=float)).all():
        raise ValueError("Sealed benchmark prices must be finite")
    frame = panel.sort_values(["date", "stock"]).reset_index(drop=True)
    labels = _holdout_labels(frame, market, benchmark, sessions)
    benchmark_returns = benchmark.set_index("date")["close"].reindex(sessions).pct_change(fill_method=None)
    strategy = Strategy(**config["portfolio"])
    returns = {cost: {} for cost in config["cost_multipliers"]}
    result, artifacts = {}, {}
    for arm in ARMS:
        forecast = predictions.loc[predictions["arm"].eq(arm), ["date", "stock", *_FORECAST]]
        scored = frame[["date", "stock"]].join(labels).join(forecast.set_index(["date", "stock"]), on=["date", "stock"], validate="one_to_one")
        eligible = scored["score"].notna()
        scored = scored.loc[eligible].copy()
        scored["scored"] = scored["target"].notna()
        prediction_metrics = score_quantiles(scored)
        if not prediction_metrics["rows"]:
            raise ValueError(f"{arm} has no mature in-window open-to-open labels")
        prediction_metrics["all_signal_rows"] = len(scored)
        result[arm] = {"prediction": prediction_metrics, "execution": {}}
        artifacts[f"{arm}_predictions.parquet"] = scored
        for cost, multiplier in config["cost_multipliers"].items():
            simulation = simulate(
                frame, scored["score"], sessions, benchmark_returns, run_id=run_id, cost_multiplier=multiplier,
                initial_cash=INITIAL_CASH, strategy=strategy, market_status=market, roster=roster,
            )
            frozen_orders = orders.loc[orders["arm"].eq(arm) & orders["cost"].eq(cost)]
            _check_orders(simulation["orders"], frozen_orders, sessions)
            verification = _verify_ledger(simulation, sessions)
            result[arm]["execution"][cost] = dict(simulation["metrics"], **verification, sealed_orders_verified=True)
            returns[cost][arm] = equity_returns(simulation["daily"], INITIAL_CASH)
            for key in ("daily", "orders", "positions", "trades"):
                artifacts[f"{arm}_{cost}_{key}.parquet"] = simulation[key]
    return {"arms": result, "contrasts": _comparisons([returns], seed=config["seed"])}, artifacts


def _decision(report: dict, historical: dict, point_in_time: bool) -> dict:
    decisions = {}
    pairs = {
        "catboost_base": "catboost_base_vs_historical",
        "catboost_supertrend_atr": "catboost_extended_vs_historical",
    }
    reference = report["arms"]["historical_quantiles"]["prediction"]["pinball"]
    for arm, contrast in pairs.items():
        previous = historical["candidates"][arm]
        pinball_wins = report["arms"][arm]["prediction"]["pinball"] < reference
        positive_after_costs = all(item["net_return_difference"] > 0 for item in report["contrasts"][contrast].values())
        decisions[arm] = {
            "holdout_lower_pinball": pinball_wins,
            "holdout_positive_net_difference_at_both_costs": positive_after_costs,
            "holdout_pass": pinball_wins and positive_after_costs,
            "development_pass": (previous["oos_pinball_wins"] >= 2 and previous["oos_positive_double_cost_folds"] >= 2
                                 and previous["oos_double_cost_ci_above_zero"]),
        }
    return {
        "archive_point_in_time_verified": point_in_time,
        "advance_tcn": point_in_time and any(item["holdout_pass"] and item["development_pass"] for item in decisions.values()),
        "candidates": decisions,
    }


def accept(
    *, run_id: str = ACCEPTANCE_RUN, root: Path = forward.ROOT, baseline: Path = forward.BASELINE,
    calendar_path: Path | None = forward.CALENDAR, model_root: Path = MODEL_ROOT,
) -> Path:
    """Verify the complete capture chain before evaluating or writing any output."""
    state = forward.status(root=root, baseline=baseline, calendar_path=calendar_path)
    if not state["ready_for_scoring"]:
        raise ValueError(f"留出期未达到正式验收条件（{state['valid_daily_captures']}/{state['target_sessions']} 已封存，{state['status']}）")
    if run_id != slugify(run_id, limit=120):
        raise ValueError("Run ID must be a simple directory name of at most 120 characters")
    output = artifact_run_dir(run_id, create=False)
    if output.exists():
        raise FileExistsError(f"An acceptance run already exists: {output}")
    policy = forward._policy(baseline)
    records, invalid = forward._capture_index(root, policy, date.fromisoformat(state["as_of"]))
    calendar = forward._study_calendar(root, records, calendar_path)
    days = forward._ordered_sessions(calendar, policy.start)[:policy.sessions]
    if invalid or len(days) != policy.sessions or not set(days).issubset(records):
        raise ValueError("The immutable capture chain failed revalidation")
    manifest_path = model_root / MANIFEST
    manifest = forward._json(manifest_path)
    starts_at = datetime.combine(policy.start, time(9, 30), tzinfo=forward.SHANGHAI)
    if (not manifest["verification"]["four_arms_exact_replay"] or manifest["first_signal"] != str(policy.start)
            or forward._timestamp(manifest["sealed_at"]) >= starts_at
            or manifest["source_files"]["config"]["sha256"] != forward._sha(policy.config)
            or manifest["source_files"]["roster"]["sha256"] != forward._sha(policy.roster)):
        raise ValueError("The replay models or their lineage differ from the pre-session lock")
    sessions = pd.DatetimeIndex(days)
    tables = _sealed_tables(root, records, calendar, days, policy, model_root)
    config = forward._json(policy.config)
    roster = pd.read_parquet(policy.roster).set_index("stock")
    report, artifacts = _evaluate(tables, sessions, config, roster, run_id)
    historical_path = baseline / "reports" / "latent_quant.json"
    report["decision"] = _decision(report, forward._json(historical_path)["decision"], manifest["point_in_time"]["verified"])
    checked, invalid = forward._capture_index(root, policy, date.fromisoformat(state["as_of"]))
    if invalid or any(checked.get(day) != records[day] for day in days):
        raise ValueError("A capture changed during the acceptance replay")
    lineage = {str(day): forward._sha(root / "captures" / str(day) / "manifest.json") for day in days}
    output.mkdir(parents=True, exist_ok=False)
    context = create_run_context(
        task="latent-quant-forward-acceptance", as_of_date=str(days[-1]), config=config,
        data_version=stable_hash(lineage), seed=config["seed"], run_id=run_id,
    )
    report.update(
        run=context.to_dict(), capture_manifests=lineage, frozen_model_sha256=forward._sha(manifest_path),
        development_report_sha256=forward._sha(historical_path),
        evaluation_start=str(days[0]), evaluation_end=str(days[-1]), evaluation_sessions=len(days),
        maximum_scoring_signal_sessions=len(days) - forward.LABEL_OFFSET,
    )
    for filename, rows in artifacts.items():
        _write_table(context.subdir("backtests") / filename, rows)
    report_path = context.subdir("reports") / "forward_acceptance.json"
    write_json(report_path, report)
    write_json(output / "status.json", {"status": "completed", "advance_tcn": report["decision"]["advance_tcn"]})
    return report_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=ACCEPTANCE_RUN)
    parser.add_argument("--calendar", type=Path, default=forward.CALENDAR)
    args = parser.parse_args(argv)
    try:
        print(accept(run_id=args.run_id, calendar_path=args.calendar))
    except (OSError, ValueError, KeyError, AssertionError) as error:
        print(f"Latent-Quant forward acceptance: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
