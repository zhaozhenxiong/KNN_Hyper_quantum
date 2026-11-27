"""Frozen Tonghuashun industry data from the user-authorized Fuyao API.

The API key is read only from ``FUYAO_API_KEY`` and is never persisted.  The
service exposes current stock/industry relationships, not historical membership;
consequently a snapshot becomes effective on its actual observation date and is
never backfilled.  Native index history is independent of that membership limit.
"""

from __future__ import annotations

from datetime import datetime
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Iterable
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from my_strategy.core.paths import DATA_ROOT


PROVIDER = "ths_fuyao"
BASE_URL = "https://fuyao.aicubes.cn"
SHANGHAI = ZoneInfo("Asia/Shanghai")
SNAPSHOT = re.compile(r"20\d{6}T\d{6}(?:\d{6})?\+0800")


def snapshot_root() -> Path:
    return Path(os.environ.get("KHQUANT_FUYAO_SECTOR_ROOT") or DATA_ROOT / "raw" / "fuyao" / "sector")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _millis(day: str) -> int:
    value = pd.Timestamp(day)
    if value.tzinfo is not None or value != value.normalize():
        raise ValueError("同花顺历史查询日期须为 YYYY-MM-DD")
    return int(value.tz_localize(SHANGHAI).timestamp() * 1000)


def _batches(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start:start + size]


class FuyaoClient:
    """Small synchronous client; response logging deliberately excludes headers."""

    def __init__(self, *, timeout: float = 45.0, pause: float = 0.12):
        key = os.environ.get("FUYAO_API_KEY", "").strip()
        if not key:
            raise ValueError("同花顺数据下载需要 FUYAO_API_KEY 环境变量。")
        self.session = requests.Session()
        self.session.headers.update({"X-api-key": key, "Accept": "application/json"})
        self.timeout = timeout
        self.pause = pause

    def get(self, endpoint: str, params: dict) -> dict:
        last_error = None
        for attempt in range(4):
            try:
                response = self.session.get(BASE_URL + endpoint, params=params, timeout=self.timeout)
                if response.status_code == 429 or response.status_code >= 500:
                    last_error = RuntimeError(f"Fuyao HTTP {response.status_code}")
                else:
                    response.raise_for_status()
                    payload = response.json()
                    if payload.get("code") == 0:
                        time.sleep(self.pause)
                        return payload
                    last_error = RuntimeError(
                        f"Fuyao code={payload.get('code')}: {payload.get('message')}; "
                        f"request_id={payload.get('request_id')}"
                    )
                    if payload.get("code") not in {3002, 4001}:
                        break
            except (requests.RequestException, ValueError) as error:
                last_error = error
            if attempt < 3:
                time.sleep((attempt + 1) * 2.0)
        raise RuntimeError(f"同花顺请求失败：{endpoint}: {last_error}") from last_error


def _request(client, staging: Path, relative: str, endpoint: str, params: dict, log: list[dict]) -> dict:
    observed = datetime.now(SHANGHAI).isoformat(timespec="seconds")
    payload = client.get(endpoint, params)
    _write_json(staging / relative, payload)
    log.append({
        "endpoint": endpoint, "params": params, "observed_at": observed,
        "code": payload.get("code"), "request_id": payload.get("request_id"),
        "server_timestamp": (payload.get("data") or {}).get("timestamp"),
        "raw_file": relative,
    })
    return payload


def download_snapshot(stocks: Iterable[str], start: str, end: str, *, root: Path | None = None,
                      client=None) -> dict:
    """Download and seal one immutable catalog/history/current-membership snapshot."""
    if _millis(start) > _millis(end):
        raise ValueError("同花顺历史开始日期晚于结束日期")
    symbols = sorted({str(stock).strip().upper() for stock in stocks})
    if not symbols or any(re.fullmatch(r"\d{6}\.(?:SH|SZ|BJ)", stock) is None for stock in symbols):
        raise ValueError("成分反查需要完整的沪深北股票代码")
    client = client or FuyaoClient()
    destination_root = (root or snapshot_root()).resolve()
    destination_root.mkdir(parents=True, exist_ok=True)
    observed = datetime.now(SHANGHAI)
    name = observed.strftime("%Y%m%dT%H%M%S%f+0800")
    target = destination_root / name
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=destination_root))
    log: list[dict] = []
    try:
        catalog_payload = _request(
            client, staging, "raw/catalog.json", "/api/a-share-index/catalog/ths-index-list",
            {"tag": "industry"}, log,
        )
        catalog = pd.DataFrame((catalog_payload.get("data") or {}).get("item") or [])
        if catalog.empty or not {"thscode", "name"}.issubset(catalog):
            raise ValueError("同花顺行业目录为空或字段不完整")
        catalog = catalog.loc[:, ["thscode", "name"]].rename(columns={"thscode": "industry", "name": "industry_name"})
        catalog["industry"] = catalog["industry"].astype(str).str.upper()
        if catalog["industry"].duplicated().any() or catalog.isna().any(axis=None):
            raise ValueError("同花顺行业目录含重复或缺失代码")
        catalog = catalog.sort_values("industry").reset_index(drop=True)
        codes = catalog["industry"].tolist()
        history_rows = []
        for index, batch in enumerate(_batches(codes, 10)):
            params = {"thscodes": ",".join(batch), "interval": "1d", "start": _millis(start), "end": _millis(end)}
            payload = _request(
                client, staging, f"raw/history/{index:03d}.json",
                "/api/a-share-index/prices/historical-batch", params, log,
            )
            for group in (payload.get("data") or {}).get("item") or []:
                for bar in group.get("item") or []:
                    history_rows.append({"industry": group["thscode"].upper(), **bar})
        history = pd.DataFrame(history_rows)
        required = {"industry", "date_ms", "open_price", "high_price", "low_price", "close_price", "volume", "turnover"}
        if history.empty or not required.issubset(history):
            raise ValueError("同花顺行业历史行情为空或字段不完整")
        history["date"] = pd.to_datetime(history["date_ms"], unit="ms", utc=True).dt.tz_convert(SHANGHAI).dt.tz_localize(None).dt.normalize()
        history = history.rename(columns={
            "open_price": "open", "high_price": "high", "low_price": "low", "close_price": "close",
            "turnover": "amount",
        })
        history = history.loc[:, ["date", "industry", "open", "high", "low", "close", "volume", "amount", "date_ms"]]
        if history.duplicated(["date", "industry"]).any():
            raise ValueError("同花顺行业历史含重复日期/行业键")
        names = catalog.set_index("industry")["industry_name"]
        history["industry_name"] = history["industry"].map(names)
        history = history.sort_values(["industry", "date"]).reset_index(drop=True)

        membership_rows = []
        for index, batch in enumerate(_batches(symbols, 100)):
            payload = _request(
                client, staging, f"raw/memberships/{index:03d}.json",
                "/api/a-share-index/constituents/ths-index-list-by-stock",
                {"thscode": ",".join(batch), "type": "industry"}, log,
            )
            for group in (payload.get("data") or {}).get("item") or []:
                industries = group.get("index_list") or []
                if len({item.get("thscode") for item in industries}) > 1:
                    raise ValueError(f"个股反查返回多个行业，需人工核对：{group.get('thscode')}")
                for item in industries:
                    membership_rows.append({"stock": group["thscode"].upper(), "industry": item["thscode"].upper()})
        memberships = pd.DataFrame(membership_rows, columns=["stock", "industry"])
        if memberships.duplicated("stock").any():
            raise ValueError("同花顺当前行业快照含重复股票")
        unknown = sorted(set(symbols) - set(memberships["stock"]))
        memberships = pd.concat([memberships, pd.DataFrame({"stock": unknown, "industry": "unknown"})], ignore_index=True)
        memberships["effective_from"] = observed.strftime("%Y-%m-%d")
        memberships["observed_at"] = observed.isoformat(timespec="seconds")
        memberships["industry_name"] = memberships["industry"].map(names).fillna("未分类（快照前或无当前行业）")
        memberships = memberships.sort_values("stock").reset_index(drop=True)

        catalog.to_parquet(staging / "catalog.parquet", index=False)
        history.to_parquet(staging / "industry_daily.parquet", index=False)
        memberships.to_parquet(staging / "memberships.parquet", index=False)
        files = {
            str(path.relative_to(staging)): {"sha256": _sha(path), "size_bytes": path.stat().st_size}
            for path in sorted(staging.rglob("*")) if path.is_file()
        }
        manifest = {
            "schema_version": 1, "provider": PROVIDER, "observed_at": observed.isoformat(timespec="seconds"),
            "effective_from": observed.strftime("%Y-%m-%d"), "history_start": start, "history_end": end,
            "catalog_rows": len(catalog), "history_rows": len(history), "membership_rows": len(memberships),
            "classified_stocks": int(memberships["industry"].ne("unknown").sum()),
            "requests": log, "files": files,
            "membership_basis": "current_snapshot_effective_from_observation_date; no historical backfill",
            "native_index_notice": "同花顺原生行业指数；历史成分宽度不可由当前成分推导。成交额变化不是主力净流入。",
        }
        _write_json(staging / "manifest.json", manifest)
        staging.rename(target)
    except Exception:
        # Preserve no partial snapshot pointer.  A hidden staging directory is
        # never accepted by load_latest_snapshot().
        raise
    pointer = {"snapshot": name, "manifest_sha256": _sha(target / "manifest.json")}
    temporary = destination_root / f".latest-{os.getpid()}.json"
    _write_json(temporary, pointer)
    temporary.replace(destination_root / "latest.json")
    return {"snapshot": str(target), "manifest": manifest, "pointer": pointer}


def load_latest_snapshot(*, root: Path | None = None, verify: bool = True) -> dict:
    destination = (root or snapshot_root()).resolve()
    pointer_path = destination / "latest.json"
    if not pointer_path.is_file():
        raise ValueError("同花顺板块冻结快照尚未准备，请先运行受控下载。")
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if not isinstance(pointer.get("snapshot"), str) or SNAPSHOT.fullmatch(pointer["snapshot"]) is None:
        raise ValueError("同花顺最新快照指针无效")
    folder = destination / pointer["snapshot"]
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("provider") != PROVIDER or manifest.get("schema_version") != 1:
        raise ValueError("同花顺快照提供方或版本不受支持")
    if pointer.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("同花顺快照清单哈希不一致")
    if verify:
        for relative, reference in manifest["files"].items():
            path = (folder / relative).resolve()
            if folder not in path.parents or not path.is_file() or _sha(path) != reference["sha256"]:
                raise ValueError(f"同花顺冻结文件哈希不一致：{relative}")
    return {
        "root": folder, "manifest": manifest,
        "catalog": pd.read_parquet(folder / "catalog.parquet"),
        "daily": pd.read_parquet(folder / "industry_daily.parquet"),
        "memberships": pd.read_parquet(folder / "memberships.parquet"),
    }


def apply_membership_snapshots(frame: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    """Apply observed relationships only on/after each snapshot effective date."""
    required = {"stock", "industry", "effective_from"}
    if not required.issubset(memberships):
        raise ValueError("同花顺行业快照缺少生效日或行业字段")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["stock"] = result["stock"].astype(str).str.upper()
    result["industry"] = "unknown"
    result["industry_name"] = "未分类（快照前或无当前行业）"
    result["industry_effective_from"] = pd.NaT
    relations = memberships.copy()
    relations["stock"] = relations["stock"].astype(str).str.upper()
    relations["effective_from"] = pd.to_datetime(relations["effective_from"], errors="raise").dt.normalize()
    if relations.duplicated(["stock", "effective_from"]).any():
        raise ValueError("同花顺行业快照在同一生效日含重复股票")
    for effective, rows in relations.groupby("effective_from", sort=True):
        active = result["date"].ge(effective)
        if not active.any():
            continue
        mapping = rows.drop_duplicates("stock").set_index("stock")
        industries = result.loc[active, "stock"].map(mapping["industry"]).fillna("unknown")
        result.loc[active, "industry"] = industries.to_numpy()
        if "industry_name" in mapping:
            names = result.loc[active, "stock"].map(mapping["industry_name"]).fillna("未分类（快照前或无当前行业）")
            result.loc[active, "industry_name"] = names.to_numpy()
        result.loc[active, "industry_effective_from"] = effective
    return result
