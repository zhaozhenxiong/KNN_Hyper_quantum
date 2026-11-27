from email.message import Message
from io import BytesIO
import json
import sqlite3
from types import SimpleNamespace

import pandas as pd
import pytest

from my_strategy.core import paths
from my_strategy.latent_quant.execution import Strategy
from my_strategy.research_workbench import latent_api, stock_chart_api as chart, stock_chart_legacy


class Request:
    def __init__(self, body=None):
        payload = json.dumps(body).encode()
        self.headers = Message()
        self.headers["Content-Type"] = "application/json"
        self.headers["Content-Length"] = str(len(payload))
        self.rfile = BytesIO(payload)

    def _json(self, status, payload):
        self.status = status
        self.payload = json.loads(json.dumps(payload, allow_nan=False))

    def _send(self, status, content, content_type):
        self.status, self.payload = status, content


def _get(route, query=""):
    request = Request()
    assert chart.handle_get(request, f"/api/stock-chart/{route}", query)
    return request


def _post(body):
    request = Request(body)
    assert chart.handle_post(request, "/api/stock-chart/replay")
    return request


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.fixture
def study(tmp_path, monkeypatch):
    root = tmp_path / "runs"
    monkeypatch.setattr(latent_api, "RUNS", root)
    monkeypatch.setattr(stock_chart_legacy, "catalog", lambda symbol: [{"id": "rule:ma_cross", "label": "均线"}])
    identities = [
        {"symbol": "002650.SZ", "name": "样例公司", "exchange": "SZ", "first_date": "2025-03-03", "last_date": "2025-03-05"},
        {"symbol": "000001.SZ", "name": "另一只股票", "exchange": "SZ", "first_date": "2025-03-03", "last_date": "2025-03-04"},
    ]
    monkeypatch.setattr(chart, "_securities", lambda tag=None: [row for row in identities if tag is None or row["symbol"] == tag])
    first, second, third = "2025-03-03", "2025-03-04", "2025-03-05"

    def market_connection():
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE securities (first_date, last_date, raw_rows)")
        connection.execute("INSERT INTO securities VALUES (?, ?, ?)", (first, third, 5))
        return connection

    def warehouse_prices(connection, stocks, start, end):
        return pd.DataFrame([
            {"stock": stocks[0], "date": day, "trade_open": 20.0, "trade_high": 21.0, "trade_low": 19.0,
             "trade_close": 20.0, "volume": 1000, "source": "raw", "has_trade_price": 1}
            for day in (first, second) if start <= day <= end
        ])

    evidence = tmp_path / "status.csv"
    pd.DataFrame([{"stock": "000001.SZ", "date": third, "market_status": "suspended",
                   "source": "notice", "source_url": "official"}]).to_csv(evidence, index=False)
    monkeypatch.setattr(chart, "_connection", market_connection)
    monkeypatch.setattr(chart, "_read_bars", warehouse_prices)
    monkeypatch.setattr(chart, "STATUS_FILE", evidence)
    baseline = root / chart.DEFAULT_RUN
    domestic = root / "domestic-test"

    def report(folder, profile, arms):
        _json(folder / "status.json", {"status": "completed"})
        _json(folder / "reports" / latent_api.REPORTS[profile][0], {
            "run": {"run_id": folder.name},
            "folds": [{"period": "oos_1", "evaluation_start": first, "evaluation_end": third,
                       "arms": {arm: {"execution": {"normal": {"metrics": {}}}} for arm in arms}}],
        })

    report(baseline, "baseline", ["catboost_base"])
    source = baseline / "data_snapshots"
    _table(source / "roster.parquet", [{"stock": "002650.SZ"}])
    _table(source / "panel.parquet", [
        {"stock": "002650.SZ", "date": day, "trade_open": p, "trade_high": p + 1, "trade_low": p - 1,
         "trade_close": p + 0.1, "open": p + 100, "volume": 5000, "source": "raw", "has_trade_price": 1}
        for day, p in zip((first, second, third), (10, 11, 12), strict=True)
    ])
    _table(source / "market_status.parquet", [
        {"stock": "002650.SZ", "date": third, "market_status": "suspended", "source": "notice"},
    ])
    report(domestic, "domestic", ["domestic_fixed", "domestic_tuned"])
    _json(domestic / "config/study.json", {"source_run_id": baseline.name})
    _json(domestic / "data_snapshots/coverage.json", {
        "source_run_id": baseline.name,
        "frozen_input_sha256": {path.name: chart._fingerprint(path) for path in source.glob("*.parquet")},
    })

    reference = {
        "stock": "002650.SZ", "signal_date": first, "date": second, "side": "BUY", "requested_shares": 1000,
        "open": 11.0, "reference_close": 10.1, "reference_date": first, "reason": "", "target_weight": 0.095,
        "status": "partial",
    }
    complete = {
        "symbol": "002650.SZ", "signal_date": first, "date": second, "action": "BUY", "shares": 800,
        "price": 11.0055, "fee": 5.08, "transfer_fee_paid": 0.08, "target_weight": 0.095, "reason": "",
    }
    for folder, arms in ((baseline, ["catboost_base"]), (domestic, ["domestic_fixed", "domestic_tuned"])):
        where = folder / "backtests/oos_1"
        predictions = "domestic_predictions.parquet" if folder == domestic else "catboost_base_predictions.parquet"
        _table(where / predictions, [{"date": first, "stock": "002650.SZ", "q10": -0.2, "q50": 0.1, "q90": 0.3,
                                     "score": 0.1, "target": 0.12, "label_end": third}])
        for arm in arms:
            _table(where / f"{arm}_normal_orders.parquet", [
                {**reference, "stock": "000001.SZ"}, reference,
                {**reference, "signal_date": second, "date": third, "side": "SELL", "requested_shares": 800,
                 "status": "rejected", "reason": "suspended", "reference_date": second, "open": None},
            ])
            _table(where / f"{arm}_normal_trades.parquet", [{**complete, "symbol": "000001.SZ"}, complete])
            _table(where / f"{arm}_normal_positions.parquet", [{"stock": "002650.SZ", "date": second, "shares": 800,
                                                               "mark_price": 11.1, "mark_date": second, "market_value": 8880}])
    return baseline, domestic


def test_signal_fill_rejection_and_stable_source_rows(study):
    baseline, _ = study
    response = _get("view")
    assert response.status == 200
    assert response.payload["context"]["price_basis"] == "raw"
    assert response.payload["context"]["run_id"] == chart.DEFAULT_RUN
    assert response.payload["bars"][0]["open"] == 10
    assert response.payload["bars"][-1]["open"] is None
    assert response.payload["nontrading"] == [{"time": "2025-03-05", "status": "suspended", "source": "notice"}]
    arm = response.payload["strategies"][0]
    signal, fill, second_signal, rejection = arm["events"]
    assert (signal["time"], signal["stage"], signal["price"], signal["fee"]) == ("2025-03-03", "signal", 10.1, None)
    assert (fill["time"], fill["stage"], fill["price"], fill["fee"]) == ("2025-03-04", "fill", 11.0055, 5.08)
    assert fill["status"] == "partial" and fill["filled_shares"] == 800 and fill["requested_shares"] == 1000
    assert fill["source_row"] == 1 and signal["source_row"] == 1
    assert fill["source_url"].endswith("/oos_1--catboost_base_normal_trades.parquet")
    assert second_signal["side"] == "SELL" and rejection["reason"] == "suspended" and rejection["price"] is None
    assert arm["positions"][0]["shares"] == 800
    truncated = _get("view", f"symbol=002650.SZ&run_id={baseline.name}&end=2025-03-03").payload
    assert truncated["strategies"][0]["events"] == [signal]
    assert truncated["strategies"][0]["predictions"][0]["target"] is None
    assert arm["predictions"][0]["target"] == 0.12


def test_catalog_search_allowlist_and_domestic_lineage(study):
    baseline, domestic = study
    catalog = _get("catalog", "symbol=002650.SZ").payload
    assert catalog["defaults"]["run_id"] == chart.DEFAULT_RUN
    assert [item["profile"] for item in catalog["runs"]] == ["baseline", "domestic"]
    assert catalog["legacy_sources"][0]["id"] == "rule:ma_cross"
    assert _get("search", "q=002650").payload["symbols"][0]["name"] == "样例公司"
    assert _get("view", "symbol=../../etc").status == 400
    assert _get("view", f"run_id={baseline.name}&arm=unknown").status == 400
    assert _get("view", f"run_id={baseline.name}&fold=../x").status == 400
    assert _get("view", f"run_id={baseline.name}&end=2025-03-02").status == 400
    assert _get("view", f"run_id={baseline.name}&fee_path=/secret").status == 400
    other = _get("view", f"symbol=000001.SZ&run_id={baseline.name}").payload
    assert other["strategies"][0]["status"] == "not_covered" and len(other["bars"]) == 3
    assert other["context"]["price_source"] == "raw_warehouse_fallback"
    only_halt = _get("view", "symbol=000001.SZ&start=2025-03-05&end=2025-03-05").payload
    assert only_halt["bars"][0]["status"] == "suspended"
    result = _get("view", f"symbol=002650.SZ&run_id={domestic.name}").payload
    assert result["strategies"][0]["predictions"] == result["strategies"][1]["predictions"]
    assert result["sources"][0]["run_id"] == baseline.name
    path = baseline / "data_snapshots/panel.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    rejected = _get("view", f"symbol=002650.SZ&run_id={domestic.name}")
    assert rejected.status == 409 and rejected.payload["code"] == "source_hash_mismatch"


def test_replay_uses_entire_fold_keeps_source_and_reloads_from_cache(study, monkeypatch):
    from my_strategy.research_workbench import stock_chart_replay, stock_replay_data, stock_replay_models

    baseline, _ = study
    monkeypatch.setattr(paths, "ARTIFACT_RUNS_ROOT", baseline.parent)
    query = {"run_id": baseline.name, "fold": "oos_1", "cost": "normal"}
    assert _post({**query, "symbol": "002650.SZ"}).status == 409
    assert _post({**query, "symbol": "000001.SZ", "raw_path": "outside"}).status == 400
    _json(baseline / "config/study.json", {"initial_cash": 1_000_000, "seed": 42, "cost_multipliers": {"normal": 1}})
    delivered = baseline / "models/oos_1"
    delivered.mkdir(parents=True)
    for name in stock_chart_replay._MODELS["baseline"]:
        (delivered / name).write_bytes(b"fixed model")
    original_hash = chart._fingerprint(baseline / "data_snapshots/panel.parquet")
    requested = []
    days = pd.date_range("2025-03-03", "2025-03-05")
    symbol = "000001.SZ"
    frame = pd.DataFrame({
        "date": days, "stock": symbol, "signal_eligible": True, "amount": 10_000_000,
        "volume": 1000, "board": "sz_main", "source": "verified_raw", "has_trade_price": True,
        "trade_open": 10.0, "trade_high": 10.8, "trade_low": 9.6, "trade_close": 10.5,
    })
    statuses = pd.DataFrame({"stock": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]"),
                             "market_status": pd.Series(dtype=str), "source": pd.Series(dtype=str)})

    def load(code, start, end):
        requested.append((code, start, end))
        return SimpleNamespace(
            frame=frame, sessions=days, coverage={"eligible_rows": 3},
            benchmark_open=pd.Series(4000.0, index=days), benchmark_close=pd.Series(4000.0, index=days),
            benchmark_returns=pd.Series(0.0, index=days), market_status=statuses,
            roster=pd.DataFrame({"delist_date": [pd.NaT]}, index=[symbol]),
        )

    forecasts = pd.DataFrame({"q10": [-0.1] * 3, "q50": [0.1] * 3, "q90": [0.3] * 3, "score": [1.0] * 3})
    monkeypatch.setattr(stock_replay_data, "load_stock_replay_data", load)
    monkeypatch.setattr(stock_replay_models, "replay", lambda *args: (
        {"catboost_base": forecasts}, {"catboost_base": Strategy()},
        {"requested_device": "GPU", "effective_devices": {"catboost_base": "CPU"}},
    ))

    first = _post({**query, "symbol": symbol, "start": "2025-03-04"})
    assert first.status == 200, first.payload
    assert requested == [(symbol, "2025-03-03", "2025-03-05")]
    info = first.payload["context"]["replay"]
    assert info["mode"] == "single_stock" and info["source_run_id"] == baseline.name and not info["cached"]
    assert info["devices"]["effective_devices"] == {"catboost_base": "CPU"}
    signals = first.payload["strategies"][0]
    assert signals["status"] == "replayed" and len(signals["events"]) == 1
    fill = signals["events"][0]
    assert fill["stage"] == "fill" and fill["time"] == "2025-03-04" and fill["signal_date"] == "2025-03-03"
    assert fill["source_url"].startswith(f"/api/stock-chart/replays/{info['run_id']}/downloads/")
    file = _get(fill["source_url"].removeprefix("/api/stock-chart/"))
    assert file.status == 200 and file.payload.startswith(b"PAR1")
    assert _get(f"replays/{info['run_id']}/downloads/metadata.json").status == 404
    assert _get(f"replays/{baseline.name}/downloads/panel.parquet").status == 404
    assert chart._fingerprint(baseline / "data_snapshots/panel.parquet") == original_hash
    second = _post({**query, "symbol": symbol, "end": "2025-03-03"})
    assert second.status == 200
    assert second.payload["context"]["replay"]["run_id"] == info["run_id"]
    assert second.payload["context"]["replay"]["cached"] and len(requested) == 1
    assert second.payload["strategies"][0]["events"][0]["stage"] == "signal"
    loaded = _get("view", f"symbol={symbol}&run_id={baseline.name}").payload
    assert loaded["context"]["replay"]["cached"] and len(loaded["strategies"][0]["events"]) == 2
