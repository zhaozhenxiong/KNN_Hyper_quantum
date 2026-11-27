"""Overseas session cutoffs, missing bars, revisions and source integrity."""

from hashlib import sha256
import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.overseas_data import CONTINUOUS_FEATURES, MARKETS, load_overseas


def _write_snapshot(folder, market, observed, events, first, last):
    folder.mkdir(exist_ok=True)
    tables = {"market_sessions.csv": market, "market_observations.csv": observed, "fomc.csv": events}
    files = {}
    for name, values in tables.items():
        path = folder / name
        values.to_csv(path, index=False)
        files[name] = {"sha256": sha256(path.read_bytes()).hexdigest(), "rows": len(values)}
    (folder / "manifest.json").write_text(json.dumps({"date_scope": [first, last], "files": files}), encoding="utf-8")


@pytest.fixture
def inputs():
    dates = pd.bdate_range("2024-10-01", "2024-11-08")
    schedule, observations = [], []
    received = "2026-11-23T00:00:00Z"
    source = {"source": "synthetic fixture", "received_at": received, "source_sha256": "a" * 64}
    for market in MARKETS:
        for i, session in enumerate(dates):
            date = session.date().isoformat()
            if market == "sp500":
                close = pd.Timestamp(f"{date} 16:00", tz="America/New_York")
            else:
                zone = "Asia/Tokyo" if market == "nikkei" else "Asia/Seoul"
                clock = "15:00" if market == "nikkei" and date < "2024-11-05" else "15:30"
                close = pd.Timestamp(f"{date} {clock}", tz=zone)
            schedule.append({"market": market, "date": date, "close_at": close.isoformat()})
            observations.append({"market": market, "date": date, "close": 100 + i,
                                 "available_at": (close + pd.Timedelta(minutes=30)).isoformat(),
                                 "availability_basis": "nominal_test", **source})
    events = [{
        "event_date": date, "available_at": f"{date}T14:00:00{offset}", "target_lower": lower,
        "target_upper": lower + .25, "availability_basis": "official_release_header", **source,
    } for date, offset, lower in (("2024-07-31", "-04:00", 5.25), ("2024-09-18", "-04:00", 4.75), ("2024-11-07", "-05:00", 4.5))]
    return pd.DataFrame(schedule), pd.DataFrame(observations), pd.DataFrame(events)


def test_local_close_and_fed_release_use_beijing_cutoff(tmp_path, inputs):
    _write_snapshot(tmp_path, *inputs, "2024-10-01", "2024-11-08")
    dates = pd.DatetimeIndex(["2024-11-04", "2024-11-05", "2024-11-07", "2024-11-08"])
    panel, coverage = load_overseas(dates, tmp_path)
    assert panel.index.equals(dates)
    assert panel.at[dates[0], "sp500_source_date"] == pd.Timestamp("2024-11-01")
    assert panel.at[dates[1], "sp500_source_date"] == pd.Timestamp("2024-11-04")
    assert panel.at[dates[0], "sp500_close_at"].hour == 20
    assert panel.at[dates[1], "sp500_close_at"].hour == 21
    assert panel.at[dates[0], "nikkei_close_at"].hour == 6
    assert panel.at[dates[1], "nikkei_close_at"] == pd.Timestamp("2024-11-05T06:30Z")
    assert panel.at[dates[2], "fed_target_lower"] == 4.75
    assert panel.at[dates[3], "fed_target_lower"] == 4.5
    assert panel.at[dates[3], "fed_target_change_bps"] == -25
    assert panel.at[dates[3], "fomc_new_statement"] == 1
    assert panel[list(CONTINUOUS_FEATURES)].notna().all().all()
    assert coverage["point_in_time"]["fomc_release_headers_verified"]
    assert not coverage["point_in_time"]["verified"]


def test_late_new_vintage_only_changes_derived_features_after_it_arrives(tmp_path, inputs):
    calendar, market, fed = inputs
    baseline, revised = tmp_path / "baseline", tmp_path / "revised"
    old_quote = market.loc[market["market"].eq("nikkei") & market["date"].eq("2024-10-30")].iloc[0].copy()
    old_quote["available_at"] = "2024-11-07T02:00:00Z"
    old_quote["received_at"] = "2024-11-07T02:00:00Z"
    old_quote["availability_basis"] = "recorded_provider_update"
    old_quote["close"] *= 2
    new_market = pd.concat([market, old_quote.to_frame().T], ignore_index=True)
    _write_snapshot(baseline, calendar, market, fed, "2024-10-01", "2024-11-08")
    _write_snapshot(revised, calendar, new_market, fed, "2024-10-01", "2024-11-08")
    dates = pd.bdate_range("2024-10-25", "2024-11-08")
    before, _ = load_overseas(dates, baseline)
    after, _ = load_overseas(dates, revised)
    pd.testing.assert_frame_equal(before.loc[:"2024-11-06"], after.loc[:"2024-11-06"])
    assert before.at[dates[-1], "nikkei_volatility_20d"] != after.at[dates[-1], "nikkei_volatility_20d"]


def test_missing_expected_local_bar_does_not_become_a_multi_day_return(tmp_path, inputs):
    calendar, market, fed = inputs
    missing = market["market"].eq("kospi") & market["date"].eq("2024-11-06")
    _write_snapshot(tmp_path, calendar, market.loc[~missing], fed, "2024-10-01", "2024-11-08")
    date = pd.DatetimeIndex(["2024-11-06", "2024-11-07"])
    panel, _ = load_overseas(date, tmp_path)
    assert panel.at[date[0], "kospi_fresh"] == 0
    assert panel.at[date[0], "kospi_source_date"] == pd.Timestamp("2024-11-05")
    assert panel.at[date[1], "kospi_fresh"] == 1
    assert np.isnan(panel.at[date[1], "kospi_return_1d"])
    assert np.isnan(panel.at[date[1], "kospi_return_5d"])
    assert np.isnan(panel.at[date[1], "kospi_volatility_20d"])
    path = tmp_path / "market_observations.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_overseas(date, tmp_path)
