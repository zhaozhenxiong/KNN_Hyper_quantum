import numpy as np
import pandas as pd

from my_strategy.latent_quant.dynamic_fusion import (
    EXPOSURES,
    dynamic_predict,
    fit_dynamic_fusion,
    fit_shadow_exposure,
    shadow_exposure_actions,
)


def _oof(days: int = 50, stocks: int = 20):
    dates = pd.bdate_range("2025-01-02", periods=days)
    rows = pd.MultiIndex.from_product([dates, [f"60000{number}.SH" for number in range(stocks)]], names=["date", "stock"]).to_frame(index=False)
    rng = np.random.default_rng(7)
    latent = rng.normal(0, 0.02, len(rows))
    rows["target"] = latent + rng.normal(0, 0.005, len(rows))
    rows["label_end"] = rows["date"] + pd.Timedelta(days=7)
    forecasts = {}
    for index, name in enumerate(("current_fusion", "trend", "mean_reversion")):
        noise = (0.02, 0.003, 0.015)[index]
        center = latent + rng.normal(0, noise, len(rows))
        forecasts[name] = np.column_stack([center - 0.02, center, center + 0.02])
    regimes = pd.Series(rows["date"].isin(dates[::2]).to_numpy(), index=rows.index)
    return rows, regimes, forecasts


def test_regularized_weights_and_binary_gate_stay_on_simplex():
    rows, regimes, forecasts = _oof()
    model = fit_dynamic_fusion(rows, regimes, forecasts)
    for field in ("global_weights", "risk_off_weights"):
        weights = np.array(list(model[field].values()))
        assert (weights >= 0).all()
        assert weights.sum() == 1.0
    assert model["global_weights"]["trend"] == max(model["global_weights"].values())
    assert model["evidence"]["risk_off"]["used"]
    assert 0 < model["evidence"]["risk_off"]["shrinkage_to_global"] <= 0.75
    assert model["evidence"]["best_single_expert"] == "trend"


def test_small_regime_falls_back_to_global_and_missing_views_use_equal():
    rows, regimes, forecasts = _oof(days=30)
    small = regimes & pd.Series(rows["date"].lt(rows["date"].unique()[5]).to_numpy(), index=rows.index)
    model = fit_dynamic_fusion(rows, small, forecasts)
    assert not model["evidence"]["risk_off"]["used"]
    assert model["risk_off_weights"] == model["global_weights"]
    changed = {name: values.copy() for name, values in forecasts.items()}
    changed["trend"][0] = np.nan
    changed["current_fusion"][20] = np.nan
    selected = rows.iloc[[0, 20]]
    values, daily, evidence = dynamic_predict(model, selected, pd.Series([True, False], index=selected.index), {name: item[[0, 20]] for name, item in changed.items()})
    np.testing.assert_allclose(values[0], np.mean([changed["current_fusion"][0], changed["mean_reversion"][0]], axis=0))
    np.testing.assert_allclose(values[1], np.mean([changed["trend"][20], changed["mean_reversion"][20]], axis=0))
    assert evidence["finite_expert_equal_fallback_rows"] == 2
    assert daily["risk_off"].tolist() == [True, False]
    assert np.isfinite(values).all()
    assert (np.diff(values, axis=1) >= 0).all()


def test_shadow_rl_is_train_only_and_emits_only_bounded_exposure():
    rows, regimes, forecasts = _oof()
    model = fit_dynamic_fusion(rows, regimes, forecasts)
    values, _, _ = dynamic_predict(model, rows, regimes, forecasts)
    shadow = fit_shadow_exposure(rows, regimes, values)
    assert shadow["enabled"] and shadow["execution_role"] == "shadow_only_no_stock_orders"
    actions = shadow_exposure_actions(shadow, rows.iloc[:40], regimes.iloc[:40], values[:40])
    assert len(actions) == 2
    assert set(actions["shadow_exposure"]) <= set(EXPOSURES)
    assert np.isfinite(actions["q_value"]).all()
    unchanged = shadow_exposure_actions(shadow, rows.iloc[:40], regimes.iloc[:40], values[:40])
    pd.testing.assert_frame_equal(actions, unchanged)
