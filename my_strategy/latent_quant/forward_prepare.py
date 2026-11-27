"""Prepare and freeze Latent-Quant forecasts and next-open orders."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

from my_strategy.core.paths import artifact_run_dir

from . import forward
from .execution import Strategy, simulate

MARKET_INPUTS = ("calendar", "panel", "benchmark", "market_status", "announcements")
MODEL_ROOT = artifact_run_dir("latent-quant-forward-baseline-models-20261123-v1", create=False)
MODEL_LOCK_SOURCE = "sealed baseline_models.json SHA256 "


def _open_orders(
    panel: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    benchmark: pd.DataFrame,
    status: pd.DataFrame,
    roster: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    benchmark_returns = benchmark.set_index("date")["close"].pct_change(fill_method=None).reindex(dates)
    strategy = Strategy(**config["portfolio"])
    fields = [column for column in forward.FIELDS["orders"] if column not in {"arm", "cost"}]
    batches = []
    for arm, forecast in predictions.items():
        for cost, multiplier in config["cost_multipliers"].items():
            outcome = simulate(
                panel, forecast["score"], dates, benchmark_returns,
                run_id="latent-quant-forward-preparation", strategy=strategy,
                initial_cash=config["initial_cash"], cost_multiplier=multiplier, market_status=status, roster=roster,
            )
            orders = outcome["orders"]
            if outcome["trades"]["date"].eq(dates[-1].date().isoformat()).any():
                raise ValueError("Pre-order preparation must have no next-session market or fills")
            batches.append(
                orders.loc[orders["requested_shares"].gt(0), fields].assign(arm=arm, cost=cost),
            )
    return pd.concat(batches, ignore_index=True).reindex(columns=list(forward.FIELDS["orders"]))


def _daily_inputs(
    root: Path, role: str, captures: dict[date, dict], today: Path | None = None,
) -> pd.DataFrame:
    tables = []
    for day, capture in captures.items():
        evidence = capture["files"][role]
        file = root / "captures" / str(day) / evidence["file"]
        if forward._sha(file) != evidence["sha256"]:
            raise ValueError(f"Sealed {role} changed during preparation: {day}")
        tables.append(forward._table(file, forward.FIELDS[role]))
    if today is not None:
        tables.append(forward._table(today, forward.FIELDS[role]))
    if not tables:
        return pd.DataFrame(columns=forward.FIELDS[role])
    frames = [frame for frame in tables if not frame.empty]
    merged = pd.concat(frames, ignore_index=True) if frames else tables[0]
    if role == "orders":
        for column in ("signal_date", "date"):
            merged[column] = pd.to_datetime(merged[column]).dt.strftime("%Y-%m-%d")
    elif "date" in merged:
        merged["date"] = pd.to_datetime(merged["date"])
    return merged


def prepare(
    market_bundle: Path, *, models: Path = MODEL_ROOT, baseline: Path = forward.BASELINE,
    root: Path = forward.ROOT, calendar_path: Path | None = forward.CALENDAR,
    cutoff: time = forward.DEFAULT_CUTOFF,
) -> Path:
    """Replay previously sealed sessions before fixing the current next-open orders."""
    from .forward_models import replay_frozen_baseline

    cutoff = forward._cutoff(cutoff)
    now = forward._now()
    day = now.date()
    policy = forward._policy(baseline)
    if day < policy.start:
        raise ValueError("Preparation predates the frozen prospective start")
    if not forward.CLOSE <= now.time() <= cutoff:
        raise ValueError("Preparation requires the Shanghai after-close window")
    if (root / "captures" / str(day)).exists():
        raise FileExistsError(f"The signal session is already sealed: {day}")
    bundle_path = market_bundle.resolve(strict=True)
    bundle = forward._json(bundle_path)
    if bundle["trade_date"] != str(day) or set(bundle["inputs"]) != set(MARKET_INPUTS):
        raise ValueError("The market bundle must declare the current signal date and all five market inputs")
    inputs = {}
    for role in MARKET_INPUTS:
        spec = bundle["inputs"][role]
        source = (bundle_path.parent / spec["path"]).resolve(strict=True)
        digest = forward._sha(source)
        if spec.get("sha256", digest) != digest:
            raise ValueError(f"Market input hash changed: {role}")
        inputs[role] = {**spec, "path": str(source), "sha256": digest}
    captures, invalid = forward._capture_index(root, policy, day)
    if invalid:
        raise ValueError(f"Invalid prior captures: {invalid}")
    calendar = forward._study_calendar(root, captures, calendar_path)
    submitted = forward._calendar(Path(inputs["calendar"]["path"]))
    if not submitted.get(day):
        raise ValueError("The supplied calendar does not verify the signal session")
    forward._merge_calendar(calendar, submitted)
    sessions = forward._ordered_sessions(calendar, policy.start)
    if day not in sessions or (tomorrow := sessions.index(day) + 1) == len(sessions):
        raise ValueError("A continuous verified calendar is needed through the next session")
    previous = sessions[:tomorrow - 1]
    if missing := [str(session) for session in previous if session not in captures]:
        raise ValueError(f"Missing previously sealed sessions: {', '.join(missing)}")
    if set(captures) != set(previous):
        raise ValueError("A previous capture falls outside the verified session calendar")
    panel = _daily_inputs(root, "panel", captures, Path(inputs["panel"]["path"]))
    if not set(panel["date"].dt.date).issubset({*previous, day}) or panel.duplicated(["date", "stock"]).any():
        raise ValueError("The prepared panel must contain unique stocks within the sealed and current sessions")
    benchmark = _daily_inputs(root, "benchmark", captures, Path(inputs["benchmark"]["path"]))
    status = _daily_inputs(root, "market_status", captures, Path(inputs["market_status"]["path"]))
    roster = pd.read_parquet(policy.roster)
    if "stock" in roster:
        roster = roster.set_index("stock")
    if "delist_date" not in roster:
        roster = roster.assign(delist_date=pd.NaT)
    model_lock = models / "baseline_models.json"
    model_hash = forward._sha(model_lock)
    lock = forward._json(model_lock)
    first_open = datetime.combine(policy.start, time(9, 30), tzinfo=forward.SHANGHAI)
    if (lock["first_signal"] != str(policy.start) or date.fromisoformat(lock["input_cutoff"]) >= policy.start
            or forward._timestamp(lock["sealed_at"]) >= first_open
            or lock["source_files"]["config"]["sha256"] != forward._sha(policy.config)
            or lock["source_files"]["roster"]["sha256"] != forward._sha(policy.roster)
            or not lock["verification"]["four_arms_exact_replay"]):
        raise ValueError("The model manifest must match v2 and be sealed before the first session opens")
    model_source = f"{MODEL_LOCK_SOURCE}{model_hash}"
    for record in captures.values():
        source = record["files"]["predictions"]["source"]
        if source.startswith(MODEL_LOCK_SOURCE) and source != model_source:
            raise ValueError("The frozen model lock differs from a prior capture")
    forecasts = replay_frozen_baseline(panel, models)
    if set(forecasts) != policy.arms:
        raise ValueError("Frozen model predictions do not match the study arms")
    features = panel[["date", "stock"]]
    predictions = pd.concat(
        [features.loc[values.index].join(values).assign(arm=arm) for arm, values in forecasts.items()],
        ignore_index=True,
    )
    orders = _open_orders(
        panel, forecasts, pd.DatetimeIndex(sessions[:tomorrow + 1]), benchmark, status,
        roster, forward._json(policy.config),
    )
    has_signal_today = predictions["date"].dt.date.eq(day)
    order_today = orders["signal_date"].eq(str(day))
    previous_predictions = _daily_inputs(root, "predictions", captures)
    previous_orders = _daily_inputs(root, "orders", captures)
    if captures:
        comparisons = (
            ("predictions", predictions.loc[~has_signal_today], previous_predictions, ["date", "stock", "arm"]),
            ("orders", orders.loc[~order_today], previous_orders, ["signal_date", "stock", "arm", "cost"]),
        )
        for role, replayed, sealed, keys in comparisons:
            columns = list(forward.FIELDS[role])
            expected = sealed[columns].sort_values(keys).reset_index(drop=True)
            computed = replayed[columns].sort_values(keys).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(expected, computed, check_dtype=False, check_exact=True)
            except AssertionError as error:
                raise ValueError(f"Replayed {role} differ from previously sealed evidence") from error
    predictions = predictions.loc[has_signal_today]
    orders = orders.loc[order_today]
    if forward._sha(model_lock) != model_hash:
        raise ValueError("The baseline model manifest changed during prediction")

    with tempfile.TemporaryDirectory(prefix="khquant-forward-") as directory:
        stage = Path(directory)
        for role, frame in (("predictions", predictions), ("orders", orders)):
            artifact = stage / f"{role}.parquet"
            frame.to_parquet(artifact, index=False)
            inputs[role] = {
                "path": str(artifact), "source": model_source,
                "sha256": forward._sha(artifact), "observed_at": forward._now().isoformat(),
            }
        supplied = stage / "bundle.json"
        supplied.write_text(json.dumps({"trade_date": str(day), "inputs": inputs}), encoding="utf-8")
        return forward.capture(
            supplied, trade_date=day, baseline=baseline, root=root, calendar_path=calendar_path, cutoff=cutoff,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--market-bundle", type=Path, required=True, help="forward template without predictions/orders")
    parser.add_argument("--models", type=Path, default=MODEL_ROOT)
    parser.add_argument("--baseline", type=Path, default=forward.BASELINE)
    parser.add_argument("--root", type=Path, default=forward.ROOT)
    parser.add_argument("--calendar", type=Path, default=forward.CALENDAR, dest="calendar_path")
    parser.add_argument("--cutoff", type=time.fromisoformat, default=forward.DEFAULT_CUTOFF)
    args = vars(parser.parse_args(argv))
    try:
        manifest = prepare(**args)
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"Latent-Quant forward preparation: {error}", file=sys.stderr)
        return 2
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
