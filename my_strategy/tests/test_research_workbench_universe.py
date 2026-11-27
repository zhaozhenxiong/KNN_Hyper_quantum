from pathlib import Path
import sqlite3

import pandas as pd
import pytest

from my_strategy.research_workbench.cli import DEFAULTS, _config
from my_strategy.research_workbench.universe import load_universe, members, restrict_bars, target_coverage


def _fixture(path: Path) -> Path:
    path.write_text(
        "stock,exchange,board,list_date,delist_date,source,source_url\n"
        "600000.SH,SH,sh_main,1999-11-10,,SSE,https://www.sse.com.cn\n"
        "000001.SZ,SZ,sz_main,1991-04-03,2025-05-09,SZSE,https://www.szse.cn\n"
        "688003.SH,SH,star,2026-01-07,,SSE,https://www.sse.com.cn\n"
        "300001.SZ,SZ,chinext,2009-10-30,,SZSE,https://www.szse.cn\n"
        "920001.BJ,BJ,bse,2024-12-01,,BSE,https://www.bse.cn\n",
        encoding="utf-8",
    )
    return path


def test_scope_excludes_chinext_bse_and_filters_lifetimes(tmp_path):
    path = _fixture(tmp_path / "roster.csv")
    universe = load_universe(path, "main_star")
    assert list(universe.index) == ["000001.SZ", "600000.SH", "688003.SH"]
    assert "300001.SZ" in load_universe(path, "shsz").index
    assert "920001.BJ" not in load_universe(path, "shsz").index
    sample = pd.DataFrame(
        {
            "stock": ["000001.SZ", "000001.SZ", "688003.SH", "688003.SH", "300001.SZ"],
            "date": pd.to_datetime(["2025-05-08", "2025-05-09", "2026-01-06", "2026-01-07", "2026-01-07"]),
            "exchange": "unverified",
        }
    )
    actual = restrict_bars(sample, universe)
    assert actual.index.tolist() == [0, 3]
    assert actual.exchange.tolist() == ["SZ", "SH"]
    assert len(members(universe, pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-10"))) == 3


def test_latest_coverage_does_not_count_a_future_listing_or_retired_stock(tmp_path):
    roster = load_universe(_fixture(tmp_path / "roster.csv"), "main_star")
    with sqlite3.connect(":memory:") as connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock TEXT,date TEXT,has_trade_price INTEGER)")
        connection.executemany(
            "INSERT INTO stock_daily_normalized VALUES (?,?,1)",
            [(stock, "2025-05-09") for stock in ("000001.SZ", "600000.SH", "688003.SH", "300001.SZ")],
        )
        coverage = target_coverage(connection, roster, pd.Timestamp("2025-05-09"))
    assert coverage["active_stocks"] == coverage["active_trading_stocks"] == 1
    assert coverage["active_by_board"] == {"sh_main": 1}


def test_cli_uncapped_versioned_universe(tmp_path):
    path = _fixture(tmp_path / "roster.csv")
    values = dict(start="2024-01-02", end="2026-11-20", max_stocks=0, mode="seqlstm", holdout_start="2026-07-29")
    with pytest.raises(ValueError, match="股票上限"):
        _config(values, DEFAULTS)
    config = _config({**values, "universe_policy": "main_star", "universe_path": str(path)}, DEFAULTS)
    assert config["max_stocks"] == 0
    assert config["holdout_start"] == "2026-07-29"
    with pytest.raises(ValueError, match="名录"):
        _config({**values, "universe_policy": "main_star"}, DEFAULTS)
