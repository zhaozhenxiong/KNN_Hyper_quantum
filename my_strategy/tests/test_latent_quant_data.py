"""Checks for frozen input selection, official session labels and causal features."""

from contextlib import closing
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.latent_quant.data import BASE_FEATURES, EXTRA_FEATURES, load_panel
from my_strategy.storage.warehouse_schema import init_warehouse


@pytest.fixture
def frozen_inputs(tmp_path):
    sessions = pd.bdate_range("2023-09-01", periods=60).delete(41)
    stocks = {"000001.SZ": "sz_main", "600000.SH": "sh_main", "688001.SH": "star", "600001.SH": "sh_main"}
    adjusted, reference = [], []
    config = {name: tmp_path / file for name, file in (
        ("raw_db", "market.db"), ("roster_path", "roster.csv"), ("calendar_path", "calendar.csv"),
        ("benchmark_path", "index.csv"), ("adjusted_path", "adjusted.csv"),
        ("execution_reference_path", "reference.csv"),
    )}
    with closing(sqlite3.connect(config["raw_db"])) as connection, connection:
        init_warehouse(connection)
        for stock, board in stocks.items():
            for i, session in enumerate(sessions):
                if stock == "600000.SH" and i == 48:
                    continue
                if stock == "600001.SH" and i < 38:
                    continue
                if stock == "688001.SH" and i >= 55:
                    continue
                date = session.date().isoformat()
                open_ = 40 + i if stock != "688001.SH" else 120 - i
                prices = (open_, open_ + 2., open_ - 1., open_ + 1.)
                volume = 145 + i if stock == "600000.SH" else 14500 + i * 100
                connection.execute(
                    """INSERT INTO stock_daily_normalized
                       (stock,date,open,high,low,close,volume,has_trade_price,source_file,imported_at)
                       VALUES (?,?,?,?,?,?,?,1,'fixture','fixture')""", (stock, date, *prices, volume),
                )
                connection.execute(
                    """INSERT INTO stock_daily_raw
                       (stock,date,trade_open,trade_high,trade_low,trade_close,row_number,source_file,imported_at)
                       VALUES (?,?,?,?,?,?,0,'fixture','fixture')""", (stock, date, *prices),
                )
                price_fields = dict(zip(("open", "high", "low", "close"), prices, strict=True))
                adjusted.append({
                    "stock": stock, "date": date, "factor_effective_date": str(sessions[0].date()),
                    **{name: value * 2 for name, value in price_fields.items()},
                })
                reference.append({
                    "stock": stock, "date": date, **price_fields, "amount": (14500 + i * 100) * open_,
                    "volume": 14500 + i * 100, "preclose": open_,
                })
    pd.DataFrame(adjusted).to_csv(config["adjusted_path"], index=False)
    pd.DataFrame(reference).to_csv(config["execution_reference_path"], index=False)
    pd.DataFrame({"date": sessions, "is_trading_day": 1}).to_csv(config["calendar_path"], index=False)
    pd.DataFrame({
        "date": sessions, "open": np.arange(100, 100 + len(sessions)), "close": np.arange(101, 101 + len(sessions)),
    }).to_csv(config["benchmark_path"], index=False)
    pd.DataFrame([{
        "stock": stock, "exchange": stock[-2:], "board": board, "source": "fixture", "source_url": "https://example.com",
        "list_date": sessions[38] if stock == "600001.SH" else sessions[0],
        "delist_date": sessions[55] if stock == "688001.SH" else pd.NaT,
    } for stock, board in stocks.items()]).to_csv(config["roster_path"], index=False)
    status = tmp_path / "status.csv"
    pd.DataFrame({
        "stock": ["600000.SH"], "date": [sessions[48]], "market_status": ["suspended"], "source": ["fixture"],
    }).to_csv(status, index=False)
    config.update(
        start=str(sessions[32].date()), end=str(sessions[-1].date()), warmup_start=str(sessions[0].date()),
        universe_anchor=str(sessions[32].date()), max_stocks=3, status_paths=[status], classification_path=None,
    )
    return config, sessions


def test_shared_signal_eligibility_uses_t_bars_and_labels_use_exact_future_sessions(frozen_inputs):
    config, dates = frozen_inputs
    result = load_panel(**config)
    assert set(result.roster.index) == {"600000.SH", "000001.SZ", "688001.SH"}
    assert result.coverage["anchor_session"] == str(dates[31].date())
    assert result.coverage["historical_delistings"] == 1
    assert result.coverage["known_suspended_stock_days"] == 1
    assert result.coverage["execution_reference_volume_unit_repairs"] > 0
    assert not result.coverage["point_in_time"]["verified"]
    sh = result.frame.loc[result.frame["stock"].eq("600000.SH")].set_index("date")
    sz = result.frame.loc[result.frame["stock"].eq("000001.SZ")].set_index("date")
    assert sh.at[dates[32], "feature_volume"] == 17700
    assert sh.at[dates[32], "trade_open"] == 72
    assert sh.at[dates[32], "open"] == 144
    assert sh.at[dates[39], "trade_date"] == dates[40]
    assert sh.at[dates[39], "label_end"] == dates[45]
    assert sh.at[dates[39], "target"] == pytest.approx(85 / 80 - 145 / 140)
    assert result.benchmark_open.at[dates[39]] == 139
    assert result.benchmark_close.at[dates[39]] == 140
    assert result.benchmark_returns.at[dates[39]] == pytest.approx(140 / 139 - 1)
    assert sh.at[dates[42], "signal_eligible"]
    assert pd.isna(sh.at[dates[42], "target"])
    assert pd.isna(sh.at[dates[49], "return_1d"])
    assert sz.at[dates[-1], "signal_eligible"]
    assert pd.isna(sz.at[dates[-1], "target"])
    assert result.market_status.iloc[0]["market_status"] == "suspended"


def test_future_edits_change_labels_without_changing_past_features(frozen_inputs):
    config, dates = frozen_inputs
    baseline = load_panel(**config)
    cutoff = dates[42]
    adjusted = pd.read_csv(config["adjusted_path"], parse_dates=["date"])
    prices = ["open", "high", "low", "close"]
    adjusted.loc[adjusted["date"].gt(cutoff), prices] *= 4
    adjusted.to_csv(config["adjusted_path"], index=False)
    reference = pd.read_csv(config["execution_reference_path"], parse_dates=["date"])
    reference.loc[reference["date"].gt(cutoff), [*prices, "preclose", "amount"]] *= 4
    reference.to_csv(config["execution_reference_path"], index=False)
    with closing(sqlite3.connect(config["raw_db"])) as connection, connection:
        connection.execute(
            """UPDATE stock_daily_normalized SET open=open*4, high=high*4, low=low*4, close=close*4
               WHERE date > ?""", (str(cutoff.date()),),
        )
        connection.execute(
            """UPDATE stock_daily_raw SET trade_open=trade_open*4, trade_high=trade_high*4,
               trade_low=trade_low*4, trade_close=trade_close*4 WHERE date > ?""", (str(cutoff.date()),),
        )
    updated = load_panel(**config)
    columns = ["stock", "date", *BASE_FEATURES, *EXTRA_FEATURES, "signal_eligible"]
    past = baseline.frame["date"].le(cutoff)
    pd.testing.assert_frame_equal(baseline.frame.loc[past, columns], updated.frame.loc[past, columns])
    assert baseline.coverage["stocks"] == updated.coverage["stocks"]
    assert updated.coverage["inputs"]["adjusted"]["sha256"] != baseline.coverage["inputs"]["adjusted"]["sha256"]
    old = baseline.frame.loc[past, "target"].dropna()
    assert updated.frame.loc[old.index, "target"].ne(old).any()


def test_industry_is_available_only_after_publication_and_used_for_risk(frozen_inputs, tmp_path):
    config, dates = frozen_inputs
    classification = tmp_path / "industry.csv"
    classification.write_text(
        "stock,effective_from,available_at,source,version,industry\n"
        f"600000.SH,{dates[34].date()},{dates[34].date()},fixture,v1,J66\n",
        encoding="utf-8",
    )
    result = load_panel(**(config | {"classification_path": classification}))
    sh = result.frame.loc[result.frame["stock"].eq("600000.SH")].set_index("date")
    assert sh.at[dates[34], "industry"] == "unknown"
    assert sh.at[dates[35], "industry"] == "J66"
    assert sh.at[dates[35], "board"] == "sh_main"
    assert "industry" not in result.feature_columns
    assert result.coverage["industry_classification"]["first_known_session"] == str(dates[35].date())
    assert result.coverage["inputs"]["classification"]["path"] == str(classification)
