"""Join frozen overseas closes and FOMC releases to A-share post-close sessions."""

from __future__ import annotations

from hashlib import file_digest
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_OVERSEAS_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw" / "latent_quant_overseas" / "20261123"
MARKETS = ("sp500", "nikkei", "kospi")
CONTINUOUS_FEATURES = tuple(
    f"{market}_{feature}" for market in MARKETS for feature in ("return_1d", "return_5d", "volatility_20d")
)
FRESHNESS_FEATURES = tuple(f"{market}_{feature}" for market in MARKETS for feature in ("age_days", "fresh"))
FED_FEATURES = (
    "fed_target_lower", "fed_target_upper", "fed_target_change_bps", "fomc_age_days", "fomc_new_statement",
)
MARKET_FEATURES = (*CONTINUOUS_FEATURES, *FRESHNESS_FEATURES)
OVERSEAS_FEATURES = (*MARKET_FEATURES, *FED_FEATURES)
_FILES = ("market_sessions.csv", "market_observations.csv", "fomc.csv")
_TIMESTAMP = r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})"


def _sha256(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()


def _read_frozen(root: Path) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    inputs = {"manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)}}
    tables = {}
    for name in _FILES:
        path = root / name
        digest = _sha256(path)
        record = manifest["files"][name]
        if digest != record["sha256"]:
            raise ValueError(f"Frozen snapshot SHA-256 mismatch at {path}")
        tables[name] = pd.read_csv(path)
        if tables[name].empty or len(tables[name]) != record["rows"]:
            raise ValueError(f"Frozen snapshot row-count mismatch at {path}")
        inputs[name] = {"path": str(path), "sha256": digest, "rows": len(tables[name])}
    return tables, manifest, inputs


def _timestamps(values: pd.Series, column: str) -> pd.Series:
    if not values.astype("string").str.fullmatch(_TIMESTAMP).fillna(False).all():
        raise ValueError(f"{column} requires an explicit UTC offset on every row")
    return pd.to_datetime(values, utc=True, format="mixed", errors="raise")


def _iso_date(day: pd.Timestamp) -> str | None:
    return None if pd.isna(day) else day.date().isoformat()


def _calendar_and_observations(
    schedule: pd.DataFrame, observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    schedule["date"] = pd.to_datetime(schedule["date"], format="%Y-%m-%d", errors="raise")
    observations["date"] = pd.to_datetime(observations["date"], format="%Y-%m-%d", errors="raise")
    schedule["close_at"] = _timestamps(schedule["close_at"], "close_at")
    observations["available_at"] = _timestamps(observations["available_at"], "available_at")
    observations["received_at"] = _timestamps(observations["received_at"], "received_at")
    observations["close"] = pd.to_numeric(observations["close"], errors="raise")
    if (
        schedule["date"].isna().any() or observations["date"].isna().any()
        or schedule.duplicated(["market", "date"]).any()
        or observations.duplicated(["market", "date", "available_at"]).any()
        or set(schedule["market"]) != set(MARKETS) or set(observations["market"]) != set(MARKETS)
        or not (observations["close"].gt(0) & np.isfinite(observations["close"])).all()
        or observations[["source", "availability_basis", "source_sha256"]].isna().any().any()
    ):
        raise ValueError("Invalid calendar, duplicate vintages, or missing market close evidence")
    observations = observations.merge(schedule, on=["market", "date"], how="left", validate="many_to_one")
    if observations["close_at"].isna().any() or observations["available_at"].lt(observations["close_at"]).any():
        raise ValueError("An overseas close was stamped before its market closed or outside its calendar")
    return schedule, observations.sort_values("available_at", kind="stable")


def _market_features(
    schedule: pd.DataFrame,
    observations: pd.DataFrame,
    cutoffs: pd.Series,
    market: str,
    max_age_days: int,
) -> pd.DataFrame:
    schedule = schedule.loc[schedule["market"].eq(market)].sort_values("date")
    market_days = pd.DatetimeIndex(schedule["date"])
    market_closes = pd.DatetimeIndex(schedule["close_at"])
    if not market_closes.is_monotonic_increasing:
        raise ValueError(f"Non-chronological session closes for {market}")
    events = iter(observations.loc[observations["market"].eq(market)].itertuples(index=False))
    pending = next(events, None)
    vintage = {}
    last_arrived = pd.NaT
    rows = []
    for day, cutoff in cutoffs.items():
        while pending is not None and pending.available_at <= cutoff:
            vintage[pending.date] = pending
            if pd.isna(last_arrived) or pending.date > last_arrived:
                last_arrived = pending.date
            pending = next(events, None)
        pos = market_closes.searchsorted(cutoff, side="right") - 1
        expected = market_days[pos] if pos >= 0 else pd.NaT
        observed = vintage.get(last_arrived)
        age = (day - observed.date).days if observed is not None else np.nan
        fresh = observed is not None and observed.date == expected and 0 <= age <= max_age_days
        row = {
            "expected_date": expected, "source_date": observed.date if observed is not None else pd.NaT,
            "close_at": observed.close_at if observed is not None else pd.NaT,
            "available_at": observed.available_at if observed is not None else pd.NaT,
            "received_at": observed.received_at if observed is not None else pd.NaT,
            "availability_basis": observed.availability_basis if observed is not None else None,
            "source": observed.source if observed is not None else None,
            "source_sha256": observed.source_sha256 if observed is not None else None,
            "age_days": age, "fresh": float(fresh), "return_1d": np.nan, "return_5d": np.nan, "volatility_20d": np.nan,
        }
        if fresh:
            history = [vintage.get(key) for key in market_days[max(0, pos - 20):pos + 1]]
            prices = np.array([item.close if item is not None else np.nan for item in history])
            if len(prices) >= 2 and np.isfinite(prices[-2:]).all():
                row["return_1d"] = prices[-1] / prices[-2] - 1
            if len(prices) >= 6 and np.isfinite(prices[-6:]).all():
                row["return_5d"] = prices[-1] / prices[-6] - 1
            if len(prices) == 21 and np.isfinite(prices).all():
                row["volatility_20d"] = float(np.std(prices[1:] / prices[:-1] - 1, ddof=0))
        rows.append(row)
    return pd.DataFrame(rows, index=cutoffs.index).add_prefix(f"{market}_")


def _fomc_features(events: pd.DataFrame, cutoffs: pd.Series) -> pd.DataFrame:
    events["event_date"] = pd.to_datetime(events["event_date"], format="%Y-%m-%d", errors="raise")
    events["available_at"] = _timestamps(events["available_at"], "FOMC available_at")
    events["received_at"] = _timestamps(events["received_at"], "FOMC received_at")
    events = events.sort_values("available_at")
    for name in ("target_lower", "target_upper"):
        events[name] = pd.to_numeric(events[name], errors="raise")
    if (
        events["event_date"].isna().any() or events["event_date"].duplicated().any()
        or not events["event_date"].is_monotonic_increasing
        or not events["target_lower"].ge(0).all() or not events["target_upper"].ge(events["target_lower"]).all()
        or events[["availability_basis", "source", "source_sha256"]].isna().any().any()
    ):
        raise ValueError("Invalid Federal Reserve statement data")
    midpoint = (events["target_lower"] + events["target_upper"]) / 2
    events["target_change_bps"] = midpoint.diff().mul(100)
    joined = pd.merge_asof(cutoffs.rename("cutoff_at").reset_index(), events,
                           left_on="cutoff_at", right_on="available_at", direction="backward").set_index("date")
    result = joined[["target_lower", "target_upper", "target_change_bps"]].add_prefix("fed_")
    for column in ("event_date", "availability_basis", "available_at", "received_at", "source", "source_sha256"):
        result[f"fomc_{column}"] = joined[column]
    result["fomc_age_days"] = (result.index - result["fomc_event_date"]).dt.days
    result["fomc_new_statement"] = result["fomc_available_at"].gt(cutoffs.shift()).astype(float)
    return result


def load_overseas(
    sessions: pd.DatetimeIndex,
    root: str | Path = DEFAULT_OVERSEAS_ROOT,
    *,
    cutoff_hour: int = 18,
    max_age_days: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Load causal close features and official target announcements from a sealed stage.

    The frame has a naive A-share session date index. Market returns use exact
    local sessions and FOMC targets refer to the latest public announcement.
    `available_at` describes the model clock and `received_at` records the actual
    source retrieval. For retrospective closes these two clocks are distinct.
    """
    sessions = pd.DatetimeIndex(sessions, name="date")
    if (sessions.empty or sessions.hasnans or sessions.has_duplicates or sessions.tz is not None
            or not sessions.is_monotonic_increasing or not sessions.equals(sessions.normalize())
            or not 0 <= cutoff_hour < 24 or max_age_days < 0):
        raise ValueError("Require sorted unique date-only A-share sessions and a valid cutoff and age")
    root = Path(root).resolve(strict=True)
    tables, manifest, inputs = _read_frozen(root)
    lower, upper = map(pd.Timestamp, manifest["date_scope"])
    if sessions[0] < lower or sessions[-1] > upper:
        raise ValueError("Requested A-share sessions fall outside the frozen archive range")
    cutoffs = pd.Series((sessions + pd.Timedelta(hours=cutoff_hour)).tz_localize("Asia/Shanghai").tz_convert("UTC"), index=sessions)
    schedule, observations = _calendar_and_observations(tables["market_sessions.csv"], tables["market_observations.csv"])
    pieces = [_market_features(schedule, observations, cutoffs, market, max_age_days) for market in MARKETS]
    fomc = _fomc_features(tables["fomc.csv"], cutoffs)
    frame = pd.concat([*pieces, fomc, cutoffs.rename("cutoff_at")], axis=1)
    nominal = observations["availability_basis"].str.startswith("nominal_")
    revisions_verified = bool(manifest.get("historical_vendor_revisions_verified", False))
    all_fresh = frame[[f"{market}_fresh" for market in MARKETS]].eq(1).all(axis=1)
    coverage = {
        "first_session": sessions[0].date().isoformat(), "last_session": sessions[-1].date().isoformat(),
        "sessions": len(sessions), "cutoff_asia_shanghai": f"{cutoff_hour:02}:00", "max_age_days": max_age_days,
        "all_markets_fresh_sessions": int(all_fresh.sum()), "inputs": inputs,
        "missing_feature_sessions": {name: int(frame[name].isna().sum()) for name in OVERSEAS_FEATURES},
        "markets": {market: {
            "observations": int(observations["market"].eq(market).sum()),
            "fresh_sessions": int(frame[f"{market}_fresh"].sum()),
            "first_observation": observations.loc[observations["market"].eq(market), "date"].min().date().isoformat(),
            "last_observation": observations.loc[observations["market"].eq(market), "date"].max().date().isoformat(),
            "latest_admitted_observation": _iso_date(frame[f"{market}_source_date"].max()),
        } for market in MARKETS},
        "fomc": {"official_statements": len(tables["fomc.csv"]), "new_release_sessions": int(frame["fomc_new_statement"].sum())},
        "point_in_time": {
            "verified": revisions_verified and not bool(nominal.any()), "nominal_market_observation_rows": int(nominal.sum()),
            "market_available_at": "Provider timestamp when evidenced, otherwise documented nominal market close plus buffer",
            "historical_vendor_revisions_verified": revisions_verified,
            "fomc_release_headers_verified": bool(tables["fomc.csv"]["availability_basis"].eq("official_release_header").all()),
        },
        "sources": manifest.get("sources", {}), "crosschecks": manifest.get("crosschecks", {}),
    }
    return frame, coverage
