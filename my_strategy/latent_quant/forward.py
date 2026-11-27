"""Freeze contemporaneous inputs for the Latent-Quant prospective holdout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time
from hashlib import file_digest
import json
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from my_strategy.core.paths import MY_STRATEGY_ROOT, artifact_run_dir

SHANGHAI = ZoneInfo("Asia/Shanghai")
CLOSE = time(15)
DEFAULT_CUTOFF = time(18)
LABEL_OFFSET = 6
BASELINE = artifact_run_dir("latent-quant-20261122-cpu-v2", create=False)
ROOT = artifact_run_dir("latent-quant-forward-v1", create=False)
CALENDAR = MY_STRATEGY_ROOT / "data/raw/workbench_universe/20261120/calendar.csv"
FIELDS = {
    "calendar": ("date", "is_trading_day"),
    "panel": (
        "date", "stock", "open", "close", "trade_open", "trade_close",
        "amount", "board", "industry", "signal_eligible",
    ),
    "benchmark": ("date", "symbol", "open", "close"),
    "market_status": ("date", "stock", "market_status", "source"),
    "announcements": ("published_at", "source"),
    "predictions": ("date", "stock", "arm", "score", "q10", "q50", "q90"),
    "orders": ("signal_date", "date", "stock", "arm", "cost", "side", "requested_shares"),
}


def _cutoff(value: time) -> time:
    if value.tzinfo is not None or value < CLOSE or value.second or value.microsecond:
        raise ValueError("Capture cutoff must use HH:MM in Shanghai time at or after 15:00")
    return value


@dataclass(frozen=True)
class Policy:
    start: date
    sessions: int
    features: tuple[str, ...]
    arms: frozenset[str]
    costs: frozenset[str]
    config: Path
    roster: Path
    optional_features: frozenset[str] = frozenset()


def _now() -> datetime:
    return datetime.now(SHANGHAI)


def _timestamp(value: str) -> datetime:
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError("Evidence timestamps must include a UTC offset")
    return stamp.astimezone(SHANGHAI)


def _json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _sha(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _policy(baseline: Path) -> Policy:
    config = baseline / "config/study.json"
    roster = baseline / "data_snapshots/roster.parquet"
    payload = _json(config)
    count = int(payload["prospective"]["market_sessions"])
    if count <= LABEL_OFFSET:
        raise ValueError("The prospective window must exceed the label offset")
    return Policy(
        start=date.fromisoformat(payload["prospective"]["start_at_or_after"]),
        sessions=count, features=tuple(payload["base_features"] + payload["added_features"]),
        arms=frozenset(payload["arms"]), costs=frozenset(payload["cost_multipliers"]),
        config=config, roster=roster,
    )


def _table(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Capture inputs must be CSV or Parquet: {path}")
    if missing := set(required).difference(frame.columns):
        raise ValueError(f"{path.name} lacks fields {sorted(missing)}")
    return frame


def _calendar(path: Path) -> dict[date, bool]:
    frame = _table(path, FIELDS["calendar"])
    days = pd.to_datetime(frame["date"], errors="raise").dt.date
    if days.empty or days.isna().any() or not frame["is_trading_day"].isin([0, 1]).all():
        raise ValueError(f"Missing or invalid calendar records: {path}")
    ordered = sorted(days)
    if len(ordered) != (ordered[-1] - ordered[0]).days + 1 or days.duplicated().any():
        raise ValueError(f"Calendar must cover consecutive calendar dates: {path}")
    if any(day.weekday() >= 5 and active for day, active in zip(days, frame["is_trading_day"], strict=True)):
        raise ValueError(f"Calendar contains a weekend trading day: {path}")
    return {day: bool(active) for day, active in zip(days, frame["is_trading_day"], strict=True)}


def _merge_calendar(current: dict[date, bool], incoming: dict[date, bool]) -> None:
    if any(current[day] != incoming[day] for day in current.keys() & incoming.keys()):
        raise ValueError("Calendar conflicts with previously frozen dates")
    current.update(incoming)


def _ordered_sessions(calendar: dict[date, bool], start: date) -> list[date]:
    sessions: list[date] = []
    day = start
    while day in calendar:
        if calendar[day]:
            sessions.append(day)
        day = date.fromordinal(day.toordinal() + 1)
    return sessions


def _capture_index(
    root: Path, policy: Policy, as_of: date,
) -> tuple[dict[date, dict], dict[str, str]]:
    valid: dict[date, dict] = {}
    invalid: dict[str, str] = {}
    frozen_roles = {*FIELDS, "baseline_config", "baseline_roster"}
    for folder in sorted((root / "captures").glob("????-??-??")):
        try:
            day = date.fromisoformat(folder.name)
            if day > as_of:
                continue
            record = _json(folder / "manifest.json")
            stamp = _timestamp(record["captured_at"])
            cutoff = _cutoff(time.fromisoformat(record["cutoff"]))
            if (record["schema_version"] != 1 or record["trade_date"] != day.isoformat() or stamp.date() != day
                    or not CLOSE <= stamp.time() <= cutoff or day < policy.start or stamp > _now()):
                raise ValueError("Capture timestamp is outside the actual same-day acceptance window")
            entries = record["files"]
            if set(entries) != frozen_roles:
                raise ValueError("Capture evidence is incomplete")
            if entries["baseline_config"]["sha256"] != _sha(policy.config):
                raise ValueError("Baseline configuration differs from this study")
            if entries["baseline_roster"]["sha256"] != _sha(policy.roster):
                raise ValueError("Baseline roster differs from this study")
            for evidence in entries.values():
                name = evidence["file"]
                if Path(name).name != name or _sha(folder / name) != evidence["sha256"]:
                    raise ValueError(f"Capture evidence failed hash verification: {name}")
            valid[day] = record
        except (OSError, TypeError, ValueError, KeyError) as error:
            invalid[folder.name] = str(error)
    return valid, invalid


def _study_calendar(
    root: Path, valid: dict[date, dict], calendar_path: Path | None,
) -> dict[date, bool]:
    calendar = _calendar(calendar_path) if calendar_path is not None else {}
    for day, record in valid.items():
        relative = record["files"]["calendar"]["file"]
        _merge_calendar(calendar, _calendar(root / "captures" / str(day) / relative))
    return calendar


def status(
    *, root: Path = ROOT, baseline: Path = BASELINE, as_of: date | None = None,
    calendar_path: Path | None = CALENDAR, cutoff: time | None = None,
) -> dict[str, Any]:
    """Read and verify existing prospective evidence without creating files."""
    now = _now()
    as_of = as_of or now.date()
    if as_of > now.date():
        raise ValueError("Cannot inspect prospective status as a future date")
    policy = _policy(baseline)
    valid, invalid = _capture_index(root, policy, as_of)
    frozen_cutoff = time.fromisoformat(next(iter(valid.values()))["cutoff"]) if valid else None
    cutoff = _cutoff(cutoff or frozen_cutoff or DEFAULT_CUTOFF)
    if frozen_cutoff is not None and frozen_cutoff != cutoff:
        raise ValueError("The capture cutoff differs from the frozen study")
    calendar = _study_calendar(root, valid, calendar_path)
    sessions = _ordered_sessions(calendar, policy.start)
    evaluation = sessions[:policy.sessions]
    captured = [day for day in evaluation if day in valid]
    missing = [day for day in evaluation if day < as_of and day not in valid]
    if as_of in evaluation and as_of == now.date() and now.time() > cutoff and as_of not in valid:
        missing.append(as_of)
    ready = len(captured) == policy.sessions and not invalid
    today_wait = None
    if as_of == now.date() and as_of >= policy.start and as_of not in valid:
        if now.time() < CLOSE and calendar.get(as_of) is not False:
            today_wait = "awaiting_market_close"
        elif calendar.get(as_of) is None:
            today_wait = "verified_calendar_unavailable"
        elif calendar[as_of] and now.time() <= cutoff:
            today_wait = "awaiting_same_day_inputs"
    official_end = evaluation[-1] if len(evaluation) == policy.sessions else None
    all_labels_end = sessions[policy.sessions - 1 + LABEL_OFFSET] if len(sessions) >= policy.sessions + LABEL_OFFSET else None
    return {
        "as_of": str(as_of), "checked_at": now.isoformat(timespec="seconds"), "prospective_start": str(policy.start),
        "status": ("ready_for_scoring" if ready else "missed_or_invalid_capture" if missing or invalid
                   else today_wait or "collecting"),
        "acceptance_window_end": official_end, "all_signal_labels_end_if_known": all_labels_end,
        "target_sessions": policy.sessions, "confirmed_calendar_sessions": len(evaluation),
        "calendar_known_through": max(calendar, default=None), "valid_daily_captures": len(captured),
        "daily_captures_remaining": policy.sessions - len(captured), "missing_past_captures": missing,
        "invalid_captures": invalid, "calendar_needed": len(evaluation) < policy.sessions,
        "max_main_scoring_signal_sessions": policy.sessions - LABEL_OFFSET,
        "extra_sessions_to_mature_all_signals": LABEL_OFFSET, "ready_for_scoring": ready,
        "capture_cutoff": cutoff.isoformat(timespec="minutes"),
        "required_daily_inputs": list(FIELDS), "capture_root": str(root),
    }


def _verify_inputs(files: dict[str, Path], policy: Policy, day: date, next_day: date, now: datetime) -> None:
    roster = _table(files["baseline_roster"], ("board",))
    if "stock" not in roster:
        roster = roster.reset_index()
    boards = roster.set_index("stock")["board"]
    panel = _table(files["panel"], (*FIELDS["panel"], *policy.features))
    benchmark = _table(files["benchmark"], FIELDS["benchmark"])
    market = _table(files["market_status"], FIELDS["market_status"])
    prediction = _table(files["predictions"], FIELDS["predictions"])
    orders = _table(files["orders"], FIELDS["orders"])
    announcements = _table(files["announcements"], FIELDS["announcements"])
    for table in (panel, benchmark, market, prediction):
        if not pd.to_datetime(table["date"]).dt.date.eq(day).all():
            raise ValueError("Daily evidence must use the signal session only")
    for table in (panel, market, prediction, orders):
        if not table["stock"].isin(boards.index).all():
            raise ValueError("A stock is outside the frozen baseline roster")
    for table, keys in ((panel, ["stock"]), (market, ["stock"]), (prediction, ["stock", "arm"]),
                        (orders, ["stock", "arm", "cost"])):
        if table.duplicated(keys).any():
            raise ValueError(f"Duplicate input rows for {keys}")
    if not panel["board"].eq(panel["stock"].map(boards)).all() or not panel["signal_eligible"].isin([True, False]).all():
        raise ValueError("Panel board or signal eligibility is invalid")
    for outcome in ("target", "forward_return", "realized_return"):
        if outcome in panel.columns and panel[outcome].notna().any():
            raise ValueError("The signal snapshot contains a future outcome")
    valid_prices = panel[["trade_open", "trade_close"]].gt(0).all(axis=1)
    unavailable = set(market.loc[market["market_status"].isin(["suspended", "delisted"]), "stock"])
    if set(boards.index) - set(panel.loc[valid_prices, "stock"]) - unavailable:
        raise ValueError("Incomplete frozen roster without explicit non-trading evidence")
    if not market["market_status"].isin(["tradable", "suspended", "delisted", "unknown"]).all():
        raise ValueError("Invalid status in market evidence")
    if len(benchmark) != 1 or benchmark.iloc[0]["symbol"] != "000300.SH" or not benchmark[["open", "close"]].gt(0).all(axis=None):
        raise ValueError("The same-day CSI 300 benchmark open and close are required")
    eligible_rows = panel.loc[panel["signal_eligible"].eq(True)]
    columns = [*policy.features, "open", "close"]
    values = eligible_rows[columns].to_numpy(dtype=float)
    required = [index for index, column in enumerate(columns) if column not in policy.optional_features]
    optional = [index for index, column in enumerate(columns) if column in policy.optional_features]
    if (not np.isfinite(values[:, required]).all() or np.isinf(values[:, optional]).any()
            or not eligible_rows[["open", "close"]].gt(0).all(axis=None)):
        raise ValueError("An eligible stock lacks a finite frozen feature")
    eligible = set(eligible_rows["stock"])
    if eligible & unavailable or not valid_prices.loc[eligible_rows.index].all():
        raise ValueError("A stock without valid same-day trading evidence cannot be signal-eligible")
    expected_predictions = {(stock, arm) for stock in eligible for arm in policy.arms}
    if set(zip(prediction["stock"], prediction["arm"], strict=True)) != expected_predictions:
        raise ValueError("Predictions must cover every eligible stock and frozen study arm")
    quantiles = prediction[["q10", "q50", "q90"]].to_numpy(dtype=float)
    if not np.isfinite(quantiles).all() or (np.diff(quantiles, axis=1) < 0).any():
        raise ValueError("Prediction quantiles must be finite and ordered")
    if not np.isfinite(prediction["score"].to_numpy(dtype=float)).all():
        raise ValueError("Prediction scores must be finite")
    if not (
        pd.to_datetime(orders["signal_date"]).dt.date.eq(day).all()
        and pd.to_datetime(orders["date"]).dt.date.eq(next_day).all()
        and orders["arm"].isin(policy.arms).all() and orders["cost"].isin(policy.costs).all()
        and orders["side"].isin(["BUY", "SELL"]).all()
    ):
        raise ValueError("Orders must use frozen arms/costs and the next verified session")
    shares = orders["requested_shares"].to_numpy(dtype=float)
    if not (np.isfinite(shares) & (shares > 0) & (shares == np.floor(shares))).all():
        raise ValueError("All submitted orders must have a fixed positive integer share quantity")
    if {"filled_shares", "execution_price", "fee"}.intersection(orders.columns):
        raise ValueError("Pre-orders must omit next-session execution outcomes")
    if any(_timestamp(item) > now for item in announcements["published_at"]):
        raise ValueError("An announcement was published after the snapshot time")


def capture(
    bundle_path: Path, *, trade_date: date | None = None, root: Path = ROOT,
    baseline: Path = BASELINE, cutoff: time = DEFAULT_CUTOFF, calendar_path: Path | None = CALENDAR,
) -> Path:
    """Seal validated input files before the actual same-day cutoff."""
    now = _now()
    day = trade_date or now.date()
    cutoff = _cutoff(cutoff)
    if day != now.date() or not CLOSE <= now.time() <= cutoff:
        raise ValueError("Capture requires the current Shanghai date during the after-close window")
    policy = _policy(baseline)
    if day < policy.start:
        raise ValueError("Capture predates the frozen prospective holdout")
    bundle_path = bundle_path.resolve(strict=True)
    producer_hash = _sha(bundle_path)
    bundle = _json(bundle_path)
    entries = bundle["inputs"]
    if bundle["trade_date"] != day.isoformat() or set(entries) != set(FIELDS):
        raise ValueError("The bundle must declare the current day and all required roles")
    valid, invalid = _capture_index(root, policy, day)
    if invalid:
        raise ValueError(f"Previously captured evidence failed verification: {invalid}")
    if any(time.fromisoformat(record["cutoff"]) != cutoff for record in valid.values()):
        raise ValueError("The capture cutoff cannot change within one prospective study")
    destination = root / "captures" / day.isoformat()
    if destination.exists():
        raise FileExistsError(f"An immutable capture already exists for {day}: {destination}")
    known_calendar = _study_calendar(root, valid, calendar_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{day}-", dir=destination.parent) as staging:
        stage = Path(staging)
        files: dict[str, Path] = {}
        evidence: dict[str, dict] = {}
        baseline_sources = {"baseline_config": policy.config, "baseline_roster": policy.roster}
        for role in [*FIELDS, *baseline_sources]:
            if role in baseline_sources:
                path = baseline_sources[role]
                source = "frozen_baseline"
                observed = None
            else:
                spec = entries[role]
                path = (bundle_path.parent / spec["path"]).resolve(strict=True)
                source = spec["source"]
                observed = _timestamp(spec["observed_at"])
                if not isinstance(source, str) or not source.strip() or observed > now:
                    raise ValueError(f"Unverified evidence source or observation time: {role}")
                if role != "calendar" and (observed.date() != day or observed.time() < CLOSE):
                    raise ValueError(f"{role} must be observed after the current day's close")
            before = path.stat()
            target = stage / f"{role}{path.suffix.lower()}"
            shutil.copyfile(path, target)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError(f"Source changed during capture: {role}")
            digest = _sha(target)
            if role not in baseline_sources and (claimed := spec.get("sha256")) and claimed != digest:
                raise ValueError(f"Producer hash disagrees with capture: {role}")
            files[role] = target
            evidence[role] = {
                "file": target.name, "sha256": digest, "source": source, "source_path": str(path),
                "observed_at": observed.isoformat() if observed else None,
            }
        submitted_calendar = _calendar(files["calendar"])
        if not submitted_calendar.get(day):
            raise ValueError("The input calendar must verify the capture day as a trading session")
        _merge_calendar(known_calendar, submitted_calendar)
        following = _ordered_sessions(known_calendar, date.fromordinal(day.toordinal() + 1))
        if not following:
            raise ValueError("The verified calendar must cover the next trading session")
        _verify_inputs(files, policy, day, following[0], now)
        if _sha(bundle_path) != producer_hash:
            raise ValueError("Producer bundle changed during capture")
        finished = _now()
        if finished.date() != day or not CLOSE <= finished.time() <= cutoff:
            raise ValueError("Capture validation finished after the same-day cutoff")
        manifest = {
            "schema_version": 1, "trade_date": str(day), "captured_at": finished.isoformat(),
            "cutoff": cutoff.isoformat(timespec="minutes"), "next_trade_date": str(following[0]),
            "producer_bundle_sha256": producer_hash, "capture_code_sha256": _sha(Path(__file__)),
            "files": evidence,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if destination.exists():
            raise FileExistsError(f"An immutable capture already exists: {destination}")
        stage.rename(destination)
    return destination / "manifest.json"


def _template() -> dict[str, Any]:
    return {
        "trade_date": "YYYY-MM-DD",
        "inputs": {role: {
            "path": f"{role}.csv", "source": "actual provider or producing workflow",
            "observed_at": "YYYY-MM-DDTHH:MM:SS+08:00",
        } for role in FIELDS},
        "required_columns": FIELDS,
        "notes": [
            "Use a real provider-receipt timestamp with a UTC offset.",
            "Include the frozen base and added features in the panel. Its open/close fields are adjusted, and trade_open/trade_close are raw.",
            "Use empty orders, announcements or market-status tables with headers when they legitimately have no rows.",
            "All stocks with no valid bar must have explicit non-trading evidence.",
            "Use CSV or Parquet. The capture time is read from the local Shanghai clock. Producer source declarations are preserved for audit.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("template", help="print the input contract for the live data and signal producer")
    for command in ("status", "capture"):
        item = sub.add_parser(command)
        item.add_argument("--root", type=Path, default=ROOT)
        item.add_argument("--baseline", type=Path, default=BASELINE)
        item.add_argument("--calendar", type=Path, default=CALENDAR)
        item.add_argument("--cutoff", type=time.fromisoformat, default=DEFAULT_CUTOFF if command == "capture" else None)
        if command == "status":
            item.add_argument("--as-of", type=date.fromisoformat)
        else:
            item.add_argument("--trade-date", type=date.fromisoformat)
            item.add_argument("--bundle", type=Path, required=True)
    options = vars(parser.parse_args(argv))
    command = options.pop("command")
    try:
        if command == "template":
            result = _template()
        else:
            options["calendar_path"] = options.pop("calendar")
            if command == "status":
                result = status(**options)
            else:
                result = {"manifest": str(capture(options.pop("bundle"), **options))}
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    except (OSError, TypeError, ValueError, KeyError) as error:
        print(f"Latent-Quant forward: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
