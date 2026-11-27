"""Regression tests for point-in-time workbench inputs and label alignment."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from my_strategy.research_workbench.data import (
    _make_features,
    attach_point_in_time_classifications,
    load_classification_history,
    load_research_data,
)
from my_strategy.storage.warehouse_schema import init_warehouse


class ResearchWorkbenchDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.raw = self.root / "raw.db"
        self.benchmark = self.root / "index.csv"
        self.dates = list(pd.bdate_range("2026-01-05", periods=10).strftime("%Y-%m-%d"))
        self.stock = "600000.SH"
        self.new_stock = "000001.SZ"
        with closing(sqlite3.connect(self.raw)) as connection, connection:
            init_warehouse(connection)
            for stock, skip in ((self.stock, {2}), (self.new_stock, {0, 1})):
                dates = [value for i, value in enumerate(self.dates) if i not in skip]
                connection.execute(
                    "INSERT INTO securities (stock, exchange, first_date, last_date, imported_at) VALUES (?, ?, ?, ?, 'test')",
                    (stock, stock[-2:], dates[0], dates[-1]),
                )
                for i, date in enumerate(self.dates):
                    if i in skip:
                        continue
                    price = 10 + i
                    connection.execute(
                        """INSERT INTO stock_daily_normalized
                        (stock, date, open, high, low, close, volume, amount, has_trade_price, source_file, imported_at)
                        VALUES (?, ?, ?, ?, ?, ?, 10000, 100000, 1, 'test', 'test')""",
                        (stock, date, price, price + 2, price - 1, price + 1),
                    )
                    connection.execute(
                        """INSERT INTO stock_daily_raw
                        (stock, date, row_number, trade_open, trade_high, trade_low, trade_close, source_file, imported_at)
                        VALUES (?, ?, 0, ?, ?, ?, ?, 'test', 'test')""",
                        (stock, date, price, price + 2, price - 1, price + 1),
                    )
        pd.DataFrame({"date": self.dates, "open": range(100, 110), "close": range(101, 111)}).to_csv(self.benchmark, index=False)

    def test_exact_open_labels_use_common_exchange_sessions(self):
        result = load_research_data(self.dates[0], self.dates[8], raw_db=self.raw, benchmark_path=self.benchmark, max_stocks=1)
        rows = result.frame.set_index("date")
        self.assertEqual(set(result.frame["stock"]), {self.stock})
        self.assertEqual(rows.iloc[0]["trade_date"], pd.Timestamp(self.dates[1]))
        self.assertEqual(rows.iloc[0]["label_end"], pd.Timestamp(self.dates[6]))
        self.assertAlmostEqual(rows.iloc[0]["target"], 16 / 11 - 106 / 101)
        self.assertTrue(pd.isna(rows.iloc[1]["target"]))
        self.assertEqual(rows.loc[self.dates[3], "prev_trade_close"], 12)
        self.assertEqual(result.coverage["next_session"], self.dates[9])
        self.assertTrue(pd.isna(rows.iloc[-1]["label_end"]))
        self.assertEqual(result.coverage["classification_counts"]["industry"], 0)
        self.assertTrue({"mean_reversion_zscore", "atr_pct", "amihud_illiquidity"}.issubset(result.feature_columns))

    def test_missing_benchmark_session_is_preserved(self):
        index = pd.read_csv(self.benchmark)
        index = index.loc[index["date"].ne(self.dates[6])]
        index.to_csv(self.benchmark, index=False)
        result = load_research_data(self.dates[0], self.dates[9], raw_db=self.raw, benchmark_path=self.benchmark, stocks=[self.stock])
        self.assertEqual(result.coverage["benchmark_missing_dates"], [self.dates[6]])
        self.assertEqual(result.frame.iloc[0]["label_end"], pd.Timestamp(self.dates[6]))
        self.assertTrue(pd.isna(result.frame.iloc[0]["target"]))
        self.assertTrue(pd.isna(result.benchmark_returns.loc[self.dates[7]]))

    def test_adjusted_prices_and_executable_ohlc_remain_distinct(self):
        path = self.root / "verified-adjusted.csv"
        opens = [20 + 2 * i + (5 if i >= 6 else 0) for i in range(10)]
        prices = pd.DataFrame({
            "stock": self.stock, "date": self.dates, "open": opens, "high": [v + 4 for v in opens],
            "low": [v - 2 for v in opens], "close": [v + 2 for v in opens],
        })
        prices.to_csv(path, index=False)
        result = load_research_data(
            self.dates[0], self.dates[9], raw_db=self.raw, benchmark_path=self.benchmark,
            stocks=[self.stock], price_basis="adjusted", adjusted_path=path,
        )
        self.assertAlmostEqual(result.frame.iloc[0]["target"], opens[6] / opens[1] - 106 / 101)
        self.assertEqual(result.frame.iloc[0]["trade_open"], 10)
        self.assertEqual(result.frame.iloc[0]["open"], 20)
        self.assertEqual(result.coverage["adjusted_rows"], len(result.frame))
        with self.assertRaisesRegex(ValueError, "verified adjusted price file"):
            load_research_data(self.dates[0], self.dates[9], price_basis="adjusted")

    def test_classification_respects_availability_effective_dates_and_revisions(self):
        path = self.root / "classifications.csv"
        path.write_text(
            "stock,effective_from,effective_to,available_at,source,version,industry,concepts\n"
            "600000.SH,2026-01-05,2026-01-09,2026-01-05T07:00:00Z,canonical,v1,bank,b|a|a\n"
            "600000.SH,2026-01-05,2026-01-09,2026-01-06T15:01:00+08:00,canonical,v2,finance,c\n"
            "600000.SH,2026-01-09,,2026-01-09,canonical,v3,utilities,\n",
            encoding="utf-8",
        )
        history = load_classification_history(path)
        frame = pd.DataFrame({"stock": self.stock, "exchange": "SH", "date": pd.to_datetime(self.dates[:6])})
        rows = attach_point_in_time_classifications(frame, history)
        self.assertEqual(rows["classification_version"].tolist(), ["v1", "v1", "v2", "v2", "", "v3"])
        self.assertEqual(rows["industry"].tolist(), ["bank", "bank", "finance", "finance", "unknown", "utilities"])
        self.assertEqual(rows["concepts"].tolist(), [("a", "b"), ("a", "b"), ("c",), ("c",), (), ()])
        self.assertEqual(rows["board"].unique().tolist(), ["sh_main"])

    def test_static_snapshot_is_rejected_as_historical_membership(self):
        path = self.root / "snapshot.csv"
        path.write_text("stock,board,industry\n600000.SH,sh_main,bank\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "effective_from"):
            load_classification_history(path)


def _factor_bars(count: int = 50) -> pd.DataFrame:
    close = np.arange(100.0, 100.0 + count)
    return pd.DataFrame({
        "stock": "600000.SH", "date": pd.bdate_range("2026-01-05", periods=count),
        "trade_open": close - 0.5, "trade_high": close + 2.0, "trade_low": close - 1.0,
        "trade_close": close, "amount": 100_000.0, "volume": 1000.0,
    })


def test_trailing_research_factors_and_missing_liquidity():
    bars = _factor_bars()
    bars.loc[25, "amount"] = 0.0
    bars.loc[27, "amount"] = np.nan
    rows = _make_features(bars, "raw_research", None)
    close = np.arange(100.0, 120.0)
    expected = (120.0 - np.mean(close)) / np.std(close)
    np.testing.assert_allclose(rows.loc[20, "mean_reversion_zscore"], expected)
    assert rows.loc[:19, "mean_reversion_zscore"].isna().all()
    np.testing.assert_allclose(rows.loc[14, "atr_pct"], 3.0 / 114.0)
    assert rows.loc[:13, "atr_pct"].isna().all()
    np.testing.assert_allclose(rows.loc[20, "amihud_illiquidity"], np.mean(1 / close) / 100_000)
    assert rows.loc[:19, "amihud_illiquidity"].isna().all()
    assert rows.loc[25:46, "amihud_illiquidity"].isna().all()
    assert rows.loc[47:, "amihud_illiquidity"].notna().all()


def test_research_factors_are_unchanged_by_future_bars():
    bars = _factor_bars()
    altered = bars.copy()
    cutoff = bars.loc[30, "date"]
    future = altered["date"].gt(cutoff)
    altered.loc[future, ["trade_open", "trade_high", "trade_low", "trade_close", "amount", "volume"]] *= 10.0
    altered = altered.sample(frac=1, random_state=13)
    before = _make_features(bars, "raw_research", None).set_index("date")
    after = _make_features(altered, "raw_research", None).set_index("date")
    pd.testing.assert_frame_equal(before.loc[:cutoff], after.loc[:cutoff])
    assert before.loc[cutoff:, "atr_pct"].ne(after.loc[cutoff:, "atr_pct"]).any()
    assert before.loc[cutoff:, "amihud_illiquidity"].ne(after.loc[cutoff:, "amihud_illiquidity"]).any()


if __name__ == "__main__":
    unittest.main()
