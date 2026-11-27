import json

import pandas as pd
import pytest

from my_strategy.research_workbench import fuyao_sector


def _stamp(day):
    return int(pd.Timestamp(day, tz="Asia/Shanghai").timestamp() * 1000)


class Client:
    def __init__(self):
        self.calls = []

    def get(self, endpoint, params):
        self.calls.append((endpoint, params))
        if endpoint.endswith("ths-index-list"):
            return {"code": 0, "request_id": "catalog-request", "data": {"timestamp": _stamp("2026-11-26"), "item": [
                {"thscode": "881270.TI", "name": "元件"}, {"thscode": "881273.TI", "name": "白酒"},
            ]}}
        if endpoint.endswith("historical-batch"):
            groups = []
            for code in params["thscodes"].split(","):
                groups.append({"thscode": code, "item": [{
                    "date_ms": _stamp("2026-11-20"), "open_price": 100, "high_price": 110,
                    "low_price": 99, "close_price": 108, "volume": 1000, "turnover": 108000,
                }]})
            return {"code": 0, "request_id": "history-request", "data": {"item": groups}}
        groups = [{"thscode": stock, "index_list": ([{"thscode": "881270.TI", "name": "元件", "type": "industry"}]
                  if stock == "002436.SZ" else [])} for stock in params["thscode"].split(",")]
        return {"code": 0, "request_id": "membership-request", "data": {"item": groups}}


def test_download_seals_hashes_request_ids_and_never_persists_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("FUYAO_API_KEY", "not-written-to-disk")
    client = Client()
    result = fuyao_sector.download_snapshot(
        ["002436.SZ", "600519.SH"], "2026-11-01", "2026-11-20", root=tmp_path, client=client,
    )
    loaded = fuyao_sector.load_latest_snapshot(root=tmp_path)
    assert len(loaded["catalog"]) == 2
    assert loaded["daily"]["date"].dt.strftime("%Y-%m-%d").unique().tolist() == ["2026-11-20"]
    assert loaded["manifest"]["classified_stocks"] == 1
    assert {entry["request_id"] for entry in loaded["manifest"]["requests"]} == {
        "catalog-request", "history-request", "membership-request",
    }
    assert all("X-api-key" not in entry["params"] for entry in loaded["manifest"]["requests"])
    assert "not-written-to-disk" not in "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    history = next(entry for entry in client.calls if entry[0].endswith("historical-batch"))[1]
    assert history["start"] == _stamp("2026-11-01") and history["end"] == _stamp("2026-11-20")

    daily = result["snapshot"] + "/industry_daily.parquet"
    with open(daily, "ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="哈希不一致"):
        fuyao_sector.load_latest_snapshot(root=tmp_path)


def test_current_membership_never_backfills_and_later_snapshot_replaces_mapping():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2026-11-20", "2026-11-26", "2026-11-27"]),
        "stock": "002436.SZ", "original_industry": "old-capco",
    })
    memberships = pd.DataFrame([
        {"stock": "002436.SZ", "industry": "881270.TI", "industry_name": "元件", "effective_from": "2026-11-26"},
        {"stock": "002436.SZ", "industry": "881999.TI", "industry_name": "新行业", "effective_from": "2026-11-27"},
    ])
    result = fuyao_sector.apply_membership_snapshots(frame, memberships)
    assert result["industry"].tolist() == ["unknown", "881270.TI", "881999.TI"]
    assert result["original_industry"].eq("old-capco").all()
    assert result["industry_effective_from"].iloc[0] is pd.NaT or pd.isna(result["industry_effective_from"].iloc[0])


def test_multiple_industries_for_one_stock_are_rejected(tmp_path):
    class Ambiguous(Client):
        def get(self, endpoint, params):
            if endpoint.endswith("ths-index-list-by-stock"):
                return {"code": 0, "data": {"item": [{"thscode": "002436.SZ", "index_list": [
                    {"thscode": "881270.TI"}, {"thscode": "881273.TI"},
                ]}]}}
            return super().get(endpoint, params)

    with pytest.raises(ValueError, match="多个行业"):
        fuyao_sector.download_snapshot(["002436.SZ"], "2026-11-01", "2026-11-20", root=tmp_path, client=Ambiguous())


def test_missing_key_is_an_explicit_download_error(monkeypatch):
    monkeypatch.delenv("FUYAO_API_KEY", raising=False)
    with pytest.raises(ValueError, match="FUYAO_API_KEY"):
        fuyao_sector.FuyaoClient()
