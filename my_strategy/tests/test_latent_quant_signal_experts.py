import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.signal_experts import ALL_EXPERTS, BASE_EXPERTS, _signals, defensive_exposure, forecast_experts


def _panel(stocks: int = 8, days: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-02", periods=days)
    frame = pd.MultiIndex.from_product([dates, [f"60000{number}.SH" for number in range(stocks)]], names=["date", "stock"]).to_frame(index=False)
    stock_number = frame.groupby("stock", sort=False).ngroup().to_numpy()
    day_number = frame.groupby("stock", sort=False).cumcount().to_numpy()
    trend = (stock_number - (stocks - 1) / 2) * 0.0005 * day_number
    cycle = 0.06 * np.sin(day_number / 7 + stock_number)
    pulse = np.where((day_number + stock_number * 3) % 47 == 0, 0.08, 0.0)
    close = (10 + stock_number) * np.exp(trend + cycle + pulse)
    frame["close"] = close
    frame["open"] = frame.groupby("stock", sort=False)["close"].shift().fillna(frame["close"])
    frame["high"] = np.maximum(frame["open"], close) * 1.015
    frame["low"] = np.minimum(frame["open"], close) * 0.985
    frame["volume"] = 1_000_000 * (1 + 0.3 * np.sin(day_number / 5 + stock_number) ** 2)
    frame["target"] = frame.groupby("stock", sort=False)["close"].shift(-5) / close - 1
    frame["label_end"] = frame.groupby("stock", sort=False)["date"].shift(-5)
    return frame.reset_index(drop=True)


def _split(frame: pd.DataFrame, end: int = 150):
    dates = pd.DatetimeIndex(frame["date"].unique())
    train = frame.loc[frame["date"].isin(dates[:120])].copy()
    rows = frame.loc[frame["date"].isin(dates[126:end])].copy()
    return train, rows


def test_all_experts_share_ordered_quantiles_and_original_index():
    train, rows = _split(_panel())
    original = rows.copy()
    forecasts = forecast_experts(train, rows)
    assert tuple(forecasts) == ALL_EXPERTS
    pd.testing.assert_frame_equal(rows, original)
    for name, forecast in forecasts.items():
        assert forecast.quantiles.shape == (len(rows), 3), name
        assert np.isfinite(forecast.quantiles).all(), name
        assert (forecast.quantiles[:, 0] <= forecast.quantiles[:, 1]).all(), name
        assert (forecast.quantiles[:, 1] <= forecast.quantiles[:, 2]).all(), name
        assert forecast.raw_score.index.equals(rows.index), name
        assert forecast.state.index.equals(rows.index), name
        assert forecast.evidence["evaluation_rows"] == len(rows)
        assert forecast.evidence["train_state_rows"] > 0
        assert forecast.evidence["exact_group_rows"] + forecast.evidence["signed_fallback_rows"] + forecast.evidence["pooled_fallback_rows"] == len(rows)
    knn = forecasts["online_knn_volume_trend"].evidence
    assert knn["scaler"] == "initial_train_only"
    assert knn["update_rule"] == "label_end_strictly_before_signal_date"
    assert knn["knn_rows"] > 0
    assert knn["online_updates"] > 0


def test_causal_indicators_warm_up_and_sequential_entries_exit():
    panel = _panel(stocks=2)
    signals = _signals(panel)
    ordered = panel.sort_values(["stock", "date"], kind="stable")
    for _, positions in ordered.groupby("stock", sort=False).indices.items():
        rows = signals.iloc[positions]
        assert rows["donchian_breakout_state"].iloc[:55].isna().all()
        assert rows["ma_adx_trend_state"].iloc[:59].isna().all()
    assert signals["donchian_breakout_state"].eq(1).any()
    assert signals["donchian_breakout_state"].eq(-1).any()
    assert signals["rsi_bollinger_mean_reversion_state"].eq(1).any()
    assert signals["rsi_bollinger_mean_reversion_state"].eq(-1).any()
    assert signals["majority_vote_state"].abs().max() <= 2


def test_future_price_and_label_changes_do_not_change_already_emitted_quantiles():
    frame = _panel()
    train, rows = _split(frame)
    dates = pd.DatetimeIndex(frame["date"].unique())
    cutoff = dates[140]
    changed = rows.copy()
    future = changed["date"].ge(cutoff)
    changed.loc[future, ["open", "high", "low", "close"]] *= 3
    changed.loc[future, "volume"] *= 10
    changed.loc[future, "target"] = 100.0
    baseline = forecast_experts(train, rows)
    stressed = forecast_experts(train, changed)
    before = rows["date"].lt(cutoff).to_numpy()
    for name in ALL_EXPERTS:
        np.testing.assert_allclose(baseline[name].quantiles[before], stressed[name].quantiles[before], equal_nan=True)
        pd.testing.assert_series_equal(baseline[name].raw_score.loc[before], stressed[name].raw_score.loc[before], check_names=False)


def test_mature_label_boundary_and_empty_state_fallback_are_explicit():
    frame = _panel()
    train, rows = _split(frame, end=135)
    invalid = train.copy()
    invalid.loc[invalid.index[-1], "label_end"] = rows["date"].min()
    with pytest.raises(ValueError, match="mature before"):
        forecast_experts(invalid, rows, ("ma_adx_trend",))
    dates = pd.DatetimeIndex(frame["date"].unique())
    early = frame.loc[frame["date"].isin(dates[:20])].copy()
    later = frame.loc[frame["date"].isin(dates[26:30])].copy()
    forecast = forecast_experts(early, later, ("ma_adx_trend",))["ma_adx_trend"]
    assert forecast.evidence["train_state_rows"] == 0
    assert forecast.evidence["pooled_fallback_rows"] == len(later)
    np.testing.assert_allclose(forecast.quantiles, np.tile(early["target"].quantile([0.1, 0.5, 0.9]), (len(later), 1)))


def test_defensive_overlay_is_causal_and_has_explicit_warmup():
    dates = pd.bdate_range("2025-01-02", periods=120)
    returns = pd.Series([0.001] * 60 + [-0.02, 0.01] * 30, index=dates)
    evidence = defensive_exposure(returns)
    assert set(evidence["target_exposure"].unique()) <= {0.0, 0.5, 1.0}
    assert evidence["target_exposure"].iloc[:79].eq(0.5).all()
    assert evidence["defensive_state"].eq("defensive_cash").any()
    changed = returns.copy()
    changed.iloc[100:] = 0.5
    stressed = defensive_exposure(changed)
    pd.testing.assert_frame_equal(evidence.iloc[:100], stressed.iloc[:100])



    train, rows = _split(_panel())
    forecasts = forecast_experts(train, rows)
    vote = forecasts["majority_vote"]
    expected = sum(forecasts[name].state.clip(-1, 1).fillna(0) for name in BASE_EXPERTS).clip(-2, 2)
    pd.testing.assert_series_equal(vote.state.fillna(0), expected.fillna(0), check_names=False)
