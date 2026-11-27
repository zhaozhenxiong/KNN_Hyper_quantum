"""Traceability and causal replay checks for legacy B/S markers."""

from contextlib import closing
import csv
from io import StringIO
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench import stock_chart_legacy as legacy


def test_original_fill_sources_preserve_unknown_dates_and_selected_model(tmp_path, monkeypatch):
    symbol = "600000.SH"
    raw = tmp_path / "raw.sqlite"
    processed = tmp_path / "warehouse.sqlite"
    with closing(sqlite3.connect(raw)) as connection, connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock, date, close, has_trade_price)")
        connection.execute("INSERT INTO stock_daily_normalized VALUES (?, '2025-04-01', 11, 1)", (symbol,))
    with closing(sqlite3.connect(processed)) as connection, connection:
        connection.executescript(r"""
            CREATE TABLE runs(run_id, task);
            INSERT INTO runs VALUES ('real-sql-run', 'local-wfo');
            CREATE TABLE backtest_trades(run_id, stock, date, action, price, shares, fee, reason, source_file, row_number);
            CREATE INDEX idx_trades_run_stock_date ON backtest_trades(run_id, stock, date);
            INSERT INTO backtest_trades VALUES ('real-sql-run', '600000.SH', '2025-04-03 00:00:00', 'SELL', 12.5, 300, 5,
                                               'signal_date=2025-03-01', 'archived.csv', 9);
            INSERT INTO backtest_trades VALUES ('real-sql-run', '600000.SH', '2025-04-04 00:00:00', 'REDUCE', 12.6, 100, 5,
                                               '', 'D:\old-root\archived.csv', 10);
        """)
    root = tmp_path / "runs"
    selection = root / "real-seqlstm"
    (selection / "reports").mkdir(parents=True)
    (selection / "backtests").mkdir()
    metadata = {"status": "completed", "run_id": selection.name}
    (selection / "workbench-status.json").write_text(json.dumps(metadata), encoding="utf-8")
    (selection / "reports/workbench.json").write_text(json.dumps({
        **metadata, "config": {"mode": "seqlstm"}, "summary": {"selected_model": "LightGBM"},
    }), encoding="utf-8")
    (selection / "backtests/trades.csv").write_text(
        "symbol,date,signal_date,action,run_id,price,shares,fee\n"
        "600000.SH,2025-04-04,2025-04-02,BUY,real-seqlstm,10.45,500,5.8\n"
        "600001.SH,2025-04-04,2025-04-02,BUY,real-seqlstm,9.35,600,5.8\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(legacy, "raw_db_path", lambda: raw)
    monkeypatch.setattr(legacy, "processed_db_path", lambda: processed)
    monkeypatch.setattr(legacy, "ARTIFACT_RUNS_ROOT", root)
    assert {item["id"] for item in legacy.catalog(symbol)} == {"rule:ma_cross", "sqlite:real-sql-run", "workbench:real-seqlstm"}
    payload = legacy.view(symbol, "workbench:real-seqlstm", "2025-04-01", "2025-04-04")
    assert payload["source"]["selected_model"] == "LightGBM"
    event, = payload["events"]
    assert event["time"] == event["execution_date"] == "2025-04-04"
    assert event["signal_date"] == "2025-04-02" and event["shares"] == 500
    assert event["stage"] == "fill" and event["side"] == "BUY" and event["requested_shares"] is None
    event, = legacy.view(symbol, "sqlite:real-sql-run", "2025-04-03", "2025-04-03")["events"]
    assert event["signal_date"] is None and event["execution_date"] == "2025-04-03"
    assert event["side"] == "SELL" and event["source_row"] == 9 and event["price"] == 12.5
    assert event["action"] == "SELL" and "legacy-evidence" in event["source_url"]
    reduced, = legacy.view(symbol, "sqlite:real-sql-run", "2025-04-04", "2025-04-04")["events"]
    assert reduced["action"] == "REDUCE" and reduced["side"] == "SELL"
    document = legacy.evidence(symbol, "sqlite:real-sql-run", reduced["id"])
    row, = csv.DictReader(StringIO(document.decode("utf-8-sig")))
    assert row["stock"] == symbol and row["action"] == "REDUCE" and row["price"] == "12.6"
    assert row["source_row"] == "10"
    with pytest.raises(FileNotFoundError):
        legacy.evidence("600001.SH", "sqlite:real-sql-run", reduced["id"])
    with pytest.raises(ValueError):
        legacy.evidence(symbol, "workbench:../outside", "untrusted")


def test_ma_replay_uses_warmup_and_future_changes_leave_past_signals_unchanged(tmp_path, monkeypatch):
    dates = [str(day.date()) for day in pd.bdate_range("2025-01-02", periods=90)]
    price = np.r_[np.arange(28) + 10, 38 - np.arange(30) * 0.6, np.arange(32) + 20]
    frame = pd.DataFrame({"date": dates, "close": price})
    raw = tmp_path / "bars.db"
    with closing(sqlite3.connect(raw)) as connection, connection:
        connection.execute("CREATE TABLE stock_daily_normalized(stock, date, close, has_trade_price)")
        connection.executemany("INSERT INTO stock_daily_normalized VALUES ('600000.SH', ?, ?, 1)", zip(dates, price, strict=True))
    monkeypatch.setattr(legacy, "raw_db_path", lambda: raw)
    full = legacy.view("600000.SH", "rule:ma_cross", dates[0], dates[-1])["events"]
    assert [event["side"] for event in full] == ["SELL", "BUY"]
    assert all(event["stage"] == "signal" and event["execution_date"] is None and event["price"] is None for event in full)
    visible = legacy.view("600000.SH", "rule:ma_cross", dates[30], dates[40])["events"]
    assert visible == [event for event in full if dates[30] <= event["time"] <= dates[40]]
    mutated = frame.copy()
    mutated.loc[mutated["date"].gt(dates[45]), "close"] *= 70
    past = [event for event in full if event["time"] <= dates[45]]
    changed = legacy._rule_signals(mutated, dates[0], dates[-1])
    assert [event for event in changed if event["time"] <= dates[45]] == past
