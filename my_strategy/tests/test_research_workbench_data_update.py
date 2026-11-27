from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.data_manager.downloader import DataDownloader
from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.research_workbench import data_update


def _calendar(path: Path):
    path.write_text(
        "date,is_trading_day\n"
        "2026-11-20,1\n2026-11-21,0\n2026-11-22,0\n2026-11-23,1\n2026-11-24,1\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("moment,expected", [
    ("2026-11-24T09:30:00+08:00", "2026-11-23"),
    ("2026-11-24T15:59:59+08:00", "2026-11-23"),
    ("2026-11-24T16:00:00+08:00", "2026-11-24"),
    ("2026-11-22T18:00:00+08:00", "2026-11-20"),
])
def test_latest_complete_session_uses_verified_close(tmp_path, moment, expected):
    calendar = _calendar(tmp_path / "calendar.csv")
    assert data_update.latest_complete_session(
        datetime.fromisoformat(moment), calendar_path=calendar, update_after_hour=16,
    ) == expected


def test_latest_complete_session_rejects_bad_calendar(tmp_path):
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("date,is_trading_day\n2026-11-23,1\n2026-11-23,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="重复"):
        data_update.latest_complete_session(
            datetime.fromisoformat("2026-11-24T10:00:00+08:00"),
            calendar_path=calendar,
            update_after_hour=16,
        )


def test_validate_target_checks_ohlc_and_duplicate_keys():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        """CREATE TABLE stock_daily_normalized (
               stock TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL,
               volume REAL, has_trade_price INTEGER
           )"""
    )
    valid = ("000001.SZ", "2026-11-23", 10, 11, 9, 10.5, 1000, 1)
    connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?)", valid)
    assert data_update._validate_target(connection, "2026-11-23") == {
        "target_rows": 1, "target_trade_rows": 1, "invalid_trade_rows": 0,
        "duplicate_stock_dates": 0, "passed": True,
    }
    connection.execute(
        "INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?)",
        ("000002.SZ", "2026-11-23", 12, 11, 9, 10.5, 1000, 1),
    )
    connection.execute("INSERT INTO stock_daily_normalized VALUES (?,?,?,?,?,?,?,?)", valid)
    validation = data_update._validate_target(connection, "2026-11-23")
    assert validation["invalid_trade_rows"] == 1
    assert validation["duplicate_stock_dates"] == 1
    assert validation["passed"] is False
    connection.close()


def _run_worker(tmp_path, monkeypatch, report, after, validation=None):
    markets = iter(({"latest_date": "2026-11-20", "data_start": "2020-01-01", "total_stocks": 2, "latest_stocks": 2}, after))
    monkeypatch.setattr(data_update, "_connection", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(data_update, "_market", lambda _connection: next(markets))
    monkeypatch.setattr(data_update, "_validate_target", lambda _connection, _target: validation or {
        "target_rows": 2, "target_trade_rows": 1, "invalid_trade_rows": 0,
        "duplicate_stock_dates": 0, "passed": True,
    })
    calls = []

    class Updater:
        def __init__(self, config):
            calls.append(("config", config))

        def update_pool(self, *, mode, end_date, progress):
            calls.append(("update", mode, end_date))
            for index, row in enumerate(report.to_dict("records"), 1):
                progress(index, len(report), row)
            return report

    monkeypatch.setattr(data_update, "DailyUpdater", Updater)
    monkeypatch.setattr(data_update, "load_data_config", lambda: {"update": {}})

    class Access:
        def record_data_update(self, **kwargs):
            calls.append(("record", kwargs["update_id"], kwargs["end_date"]))
            statuses = report["status"]
            return {
                "update_id": kwargs["update_id"], "total": len(report),
                "success": int(statuses.eq("success").sum()),
                "failed": int(statuses.eq("failed").sum()), "skipped": 0,
                "storage_mode": "db", "raw_db_path": "fixture.db",
            }

    monkeypatch.setattr(data_update, "get_data_access", lambda: Access())
    result_path = tmp_path / "update.result.json"
    result = data_update.run_update("update-fixture", "2026-11-23", result_path)
    assert json.loads(result_path.read_text(encoding="utf-8"))["job_id"] == "update-fixture"
    assert ("update", "db", "2026-11-23") in calls
    assert ("record", "update-fixture", "2026-11-23") in calls
    return result


def test_worker_distinguishes_complete_partial_and_failed(tmp_path, monkeypatch):
    complete_report = pd.DataFrame([
        {"code": "000001.SZ", "status": "success", "last_date": "2026-11-23", "error_msg": ""},
        {"code": "000002.SZ", "status": "success", "last_date": "2026-11-23", "error_msg": ""},
    ])
    after = {"latest_date": "2026-11-23", "data_start": "2020-01-01", "total_stocks": 2, "latest_stocks": 2}
    complete = _run_worker(tmp_path, monkeypatch, complete_report, after)
    assert complete["status"] == "completed"
    assert complete["summary"]["target_success"] == 2

    partial_report = complete_report.copy()
    partial_report.loc[1, ["status", "last_date", "error_msg"]] = ["failed", "", "provider unavailable"]
    partial = _run_worker(tmp_path, monkeypatch, partial_report, after)
    assert partial["status"] == "partial"
    assert partial["summary"]["target_success"] == 1
    assert "provider unavailable" in partial["failure_examples"][0]

    stale_report = complete_report.copy()
    stale_report.loc[1, "last_date"] = "2026-11-20"
    stale = _run_worker(tmp_path, monkeypatch, stale_report, after)
    assert stale["status"] == "partial"
    assert stale["summary"]["failed"] == 0
    assert stale["summary"]["target_success"] == 1

    failed = _run_worker(tmp_path, monkeypatch, partial_report, {**after, "latest_date": "2026-11-20"})
    assert failed["status"] == "failed"


def test_worker_refuses_preexisting_future_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(data_update, "_connection", lambda: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(data_update, "_market", lambda _connection: {"latest_date": "2026-11-24"})
    with pytest.raises(ValueError, match="拒绝回写"):
        data_update.run_update("update-fixture", "2026-11-23", tmp_path / "result.json")


def test_update_pool_reports_progress_without_changing_order(monkeypatch):
    pool = pd.DataFrame({"stock": ["000001.SZ", "000002.SZ", "000003.SZ"]})
    monkeypatch.setattr(
        "my_strategy.data_manager.updater.StockPoolManager",
        lambda _config: type("Pool", (), {"build_pool": lambda _self, mode, stocks: pool})(),
    )
    updater = DailyUpdater({"update": {"sleep_seconds": 0}})
    monkeypatch.setattr(updater, "update_one", lambda stock, end_date: {"code": stock, "status": "success", "last_date": end_date})
    progress = []
    report = updater.update_pool(mode="custom", end_date="2026-11-23", progress=lambda *args: progress.append(args))
    assert report["code"].tolist() == pool["stock"].tolist()
    assert [(item[0], item[1]) for item in progress] == [(1, 3), (2, 3), (3, 3)]


def test_akshare_daily_uses_current_em_interface(monkeypatch):
    calls = []
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7897")

    def history(**kwargs):
        assert "HTTP_PROXY" not in os.environ
        assert os.environ["NO_PROXY"] == "*"
        calls.append(kwargs)
        if kwargs["adjust"]:
            return pd.DataFrame({
                "日期": ["2026-11-23"], "开盘": [10], "最高": [11], "最低": [9],
                "收盘": [10.5], "成交量": [1000], "成交额": [10500],
            })
        return pd.DataFrame({
            "日期": ["2026-11-23"], "开盘": [20], "最高": [22], "最低": [18], "收盘": [21],
        })

    monkeypatch.setitem(sys.modules, "akshare", SimpleNamespace(
        stock_zh_a_hist_em=history,
        stock_zh_a_hist=lambda **_kwargs: pytest.fail("retired AkShare interface was called"),
    ))
    result = DataDownloader._download_stock_daily_akshare("000001.SZ", "2026-11-21", "2026-11-23")
    assert [call["adjust"] for call in calls] == ["qfq", ""]
    assert result.loc[0, "open"] == 10
    assert result.loc[0, "trade_close"] == 21
    assert os.environ["HTTP_PROXY"] == "http://127.0.0.1:7897"
    assert "NO_PROXY" not in os.environ


@pytest.mark.parametrize("code,source_volume,expected_volume", [
    ("002307.SZ", 604805, 60480500),
    ("300001.SZ", 125765, 12576500),
    ("688001.SH", 10276186, 10276186),
])
def test_tencent_daily_normalizes_prices_and_market_volume_units(
    monkeypatch, code, source_volume, expected_volume,
):
    sessions = []

    class Session:
        def __init__(self):
            self.trust_env = True
            self.closed = False
            sessions.append(self)

        def get(self, _url, *, params, timeout):
            assert self.trust_env is False
            assert timeout == 20
            symbol = params["param"].split(",")[0]
            adjusted = params["param"].endswith("qfq")
            bars = [
                [day, "2.55", "2.62", "2.63", "2.51", str(source_volume), {},
                 "2.82", "11827.35", "0.00", "0.00"]
                for day in ("2026-11-20", "2026-11-23")
            ]
            key = "day" if symbol.startswith("sh688") else "qfqday" if adjusted else "day"
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {"code": 0, "data": {symbol: {key: bars}}},
            )

        def close(self):
            self.closed = True

    monkeypatch.setitem(sys.modules, "requests", SimpleNamespace(Session=Session))
    result = DataDownloader().download_stock_daily(
        code, "2026-11-20", "2026-11-23", sources=["tencent"],
    )
    assert result.ok, result.error
    assert result.source == "tencent"
    assert result.data["date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-11-20", "2026-11-23"]
    assert result.data["volume"].tolist() == [expected_volume, expected_volume]
    assert result.data["amount"].tolist() == [118273500.0, 118273500.0]
    assert result.data["trade_close"].tolist() == [2.62, 2.62]
    assert len(sessions) == 1 and sessions[0].closed
