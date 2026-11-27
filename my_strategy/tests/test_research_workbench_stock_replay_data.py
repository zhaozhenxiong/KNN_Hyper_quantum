"""One-stock feature parity, warmup and end-bounded evidence checks."""

from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.data import BASE_FEATURES, EXTRA_FEATURES, _HFQ, load_panel
from my_strategy.research_workbench.stock_replay_data import StockReplayUnavailable, load_stock_replay_data
from my_strategy.tests.test_latent_quant_data import frozen_inputs


def _single_config(config: dict) -> dict:
    return {key: value for key, value in config.items() if key not in {"universe_anchor", "max_stocks"}}


def test_replay_reuses_frozen_features_with_warmup(frozen_inputs):
    configuration, dates = frozen_inputs
    original = load_panel(**configuration)
    replay = load_stock_replay_data("600000.SH", **_single_config(configuration))
    numeric = ["trade_open", "open", *BASE_FEATURES, *EXTRA_FEATURES, "target"]
    fields = ["stock", "date", "signal_eligible", *numeric]
    before = original.frame.loc[original.frame["stock"].eq("600000.SH"), fields].reset_index(drop=True)
    categorical = ["stock", "date", "signal_eligible"]
    pd.testing.assert_frame_equal(before[categorical], replay.frame[categorical], check_dtype=False)
    np.testing.assert_allclose(
        before[numeric].to_numpy(dtype=float, na_value=np.nan),
        replay.frame[numeric].to_numpy(dtype=float, na_value=np.nan),
    )
    assert all(pd.api.types.is_numeric_dtype(replay.frame[column]) for column in replay.feature_columns)
    assert replay.coverage["warmup_sessions"] == 32
    assert replay.frame.loc[0, "date"] == dates[32]
    assert replay.frame.loc[0, "signal_eligible"]
    assert replay.benchmark_returns.iloc[0] > 0
    assert replay.roster.index.tolist() == ["600000.SH"]


def test_future_prices_cannot_enter_replay_inputs_or_validation(frozen_inputs, tmp_path):
    configuration, dates = frozen_inputs
    replay_options = _single_config(configuration)
    cutoff = dates[45]
    replay_options["end"] = str(cutoff.date())
    adjusted_csv = pd.read_csv(replay_options["adjusted_path"])
    adjusted = tmp_path / "dated-adjusted.parquet"
    adjusted_csv.to_parquet(adjusted, index=False)
    replay_options["adjusted_path"] = adjusted
    original = load_stock_replay_data("600000.SH", **replay_options)

    future = adjusted_csv["date"].gt(str(cutoff.date()))
    adjusted_csv.loc[future, "factor_effective_date"] = "not-a-date"
    adjusted_csv.loc[future, "open"] *= 5
    adjusted_csv.to_parquet(adjusted, index=False)
    for key in ("benchmark_path", "execution_reference_path"):
        file = replay_options[key]
        observed = pd.read_csv(file)
        observed.loc[observed["date"].gt(str(cutoff.date())), "open"] *= 5
        observed.to_csv(file, index=False)
    with closing(sqlite3.connect(replay_options["raw_db"])) as connection, connection:
        connection.execute("UPDATE stock_daily_raw SET trade_open = trade_open * 3 WHERE date > ?", (str(cutoff.date()),))
        connection.execute("UPDATE stock_daily_normalized SET open = open * 3 WHERE date > ?", (str(cutoff.date()),))

    replay = load_stock_replay_data("600000.SH", **replay_options)
    pd.testing.assert_frame_equal(original.frame, replay.frame)
    pd.testing.assert_series_equal(original.benchmark_returns, replay.benchmark_returns)
    assert replay.coverage["input_version"] == original.coverage["input_version"]
    assert replay.calendar[-1] == cutoff
    assert replay.benchmark_open.index.max() == cutoff
    assert replay.frame["trade_date"].dropna().le(cutoff).all()
    assert replay.frame["label_end"].dropna().le(cutoff).all()


def test_missing_hfq_does_not_use_native_prices_for_the_model(frozen_inputs):
    configuration, _ = frozen_inputs
    file = Path(configuration["adjusted_path"])
    adjusted = pd.read_csv(file)
    adjusted.loc[adjusted["stock"].ne("600000.SH")].to_csv(file, index=False)
    with pytest.raises(StockReplayUnavailable) as error:
        load_stock_replay_data("600000.SH", **_single_config(configuration))
    assert error.value.code == "missing_adjusted_bars"
    with pytest.raises(StockReplayUnavailable) as out_of_scope:
        load_stock_replay_data("300750.SZ", **_single_config(configuration))
    assert out_of_scope.value.code == "unverified_roster"


@pytest.mark.skipif(not (_HFQ / "verified_market_hfq.parquet").is_file(), reason="requires the verified workspace archive")
def test_real_002436_warmup_and_partial_research_coverage():
    replay = load_stock_replay_data("002436.SZ", "2025-03-03", "2025-09-30")
    assert replay.coverage["raw_trade_rows"] == 147
    assert replay.coverage["adjusted_rows"] == 62
    assert replay.coverage["first_adjusted_bar_with_warmup"] == "2025-07-07"
    assert replay.coverage["eligible_first_session"] == "2025-08-04"
    assert replay.coverage["signal_eligible_rows"] == 42
    before = replay.frame["date"].lt("2025-07-07")
    assert replay.frame.loc[before, "open"].isna().all()
    assert np.isfinite(replay.frame.loc[before, "trade_open"]).all()
