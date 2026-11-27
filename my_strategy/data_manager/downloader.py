"""Online data downloaders. This module never writes cache."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from .validator import DataValidator


@dataclass
class DownloadResult:
    ok: bool
    data: pd.DataFrame | None = None
    source: str = ""
    error: str = ""


@contextmanager
def _akshare_direct_connection():
    proxy_keys = (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    )
    saved = {key: os.environ.get(key) for key in proxy_keys}
    for key in proxy_keys:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for key in proxy_keys:
            os.environ.pop(key, None)
            if saved[key] is not None:
                os.environ[key] = saved[key]


class DataDownloader:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def download_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: str,
        sources: Iterable[str] | None = None,
    ) -> DownloadResult:
        if sources is None:
            ds_cfg = self.config.get("data_sources", {})
            sources = [ds_cfg.get("primary", "akshare"), *ds_cfg.get("fallback", ["baostock"])]

        errors = []
        for source in sources:
            source = str(source).lower()
            try:
                if source == "akshare":
                    df = self._download_stock_daily_akshare(code, start_date, end_date)
                elif source == "tencent":
                    df = self._download_stock_daily_tencent(code, start_date, end_date)
                elif source == "baostock":
                    df = self._download_stock_daily_baostock(code, start_date, end_date)
                elif source == "local_csv":
                    df = self._download_stock_daily_local_csv(code, start_date, end_date)
                else:
                    errors.append(f"{source}: unsupported source")
                    continue

                if df is None or df.empty:
                    errors.append(f"{source}: empty data")
                    continue
                clean = DataValidator.clean_stock_daily(df, code)
                valid = DataValidator.validate_stock_daily(clean, min_rows=1, require_min_rows=False)
                if not valid.ok:
                    errors.append(f"{source}: {valid.error}")
                    continue
                return DownloadResult(True, clean, source, "")
            except Exception as exc:
                errors.append(f"{source}: {exc}")
        return DownloadResult(False, None, "", " | ".join(errors))

    def download_stock_basic(self, min_count: int = 3000) -> DownloadResult:
        try:
            import akshare as ak

            spot = ak.stock_zh_a_spot_em()
            rename = {
                "代码": "code",
                "名称": "name",
                "最新价": "latest_price",
                "总市值": "total_market_cap",
                "成交额": "turnover",
            }
            out = spot.rename(columns={k: v for k, v in rename.items() if k in spot.columns}).copy()
            if "code" not in out.columns or "name" not in out.columns:
                return DownloadResult(False, None, "akshare", f"missing code/name columns: {spot.columns.tolist()}")
            out["code"] = out["code"].map(_normalize_stock_code)
            if "total_market_cap" in out.columns:
                out["total_market_cap"] = pd.to_numeric(out["total_market_cap"], errors="coerce")
            if "turnover" in out.columns:
                out["turnover"] = pd.to_numeric(out["turnover"], errors="coerce")
            out["source"] = "akshare"
            out["updated_at"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            if len(out) < min_count:
                return DownloadResult(False, None, "akshare", f"too few all-A rows: {len(out)} < {min_count}")
            return DownloadResult(True, out.reset_index(drop=True), "akshare", "")
        except Exception as exc:
            return DownloadResult(False, None, "akshare", str(exc))

    def download_index_daily(self, index_code: str, start_date: str, end_date: str) -> DownloadResult:
        try:
            import akshare as ak

            symbol = index_code.split(".")[0]
            df = ak.index_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
            )
            df = df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"})
            clean = DataValidator.clean_stock_daily(df, index_code)
            return DownloadResult(not clean.empty, clean, "akshare", "" if not clean.empty else "empty data")
        except Exception as exc:
            return DownloadResult(False, None, "akshare", str(exc))

    @staticmethod
    def _download_stock_daily_akshare(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import akshare as ak

        symbol = code.split(".")[0]
        with _akshare_direct_connection():
            adjusted = ak.stock_zh_a_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq",
            )
            raw = ak.stock_zh_a_hist_em(
                symbol=symbol,
                period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="",
            )
        adjusted = adjusted.rename(
            columns={
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "成交额": "amount",
                "涨跌幅": "pct_chg",
                "换手率": "turnover",
            }
        )
        raw = raw.rename(
            columns={
                "日期": "date",
                "开盘": "trade_open",
                "最高": "trade_high",
                "最低": "trade_low",
                "收盘": "trade_close",
            }
        )
        return _merge_adjusted_with_trade_price(adjusted, raw)

    @staticmethod
    def _download_stock_daily_tencent(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import requests

        symbol = ("sh" if code.endswith(".SH") else "sz") + code.split(".")[0]
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
        if pd.isna(start) or pd.isna(end) or end < start:
            raise ValueError("invalid Tencent date range")
        session = requests.Session()
        session.trust_env = False

        def fetch(adjust: str) -> pd.DataFrame:
            key = "qfqday" if adjust else "day"
            selected = []
            for year in range(start.year, end.year + 1):
                year_start = pd.Timestamp(year=year, month=1, day=1)
                year_end = pd.Timestamp(year=year, month=12, day=31)
                params = {
                    "param": f"{symbol},day,{year_start.date()},{year_end.date()},640,{adjust}",
                }
                response = session.get(
                    "https://proxy.finance.qq.com/ifzqgtimg/appstock/app/newfqkline/get",
                    params=params,
                    timeout=20,
                )
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or payload.get("code") != 0:
                    raise ValueError("invalid Tencent historical response")
                data = payload.get("data")
                if data == []:
                    continue
                if not isinstance(data, dict) or not isinstance(data.get(symbol), dict):
                    raise ValueError("Tencent response does not match the requested stock")
                bars = data[symbol].get(key)
                if adjust and bars is None:
                    bars = data[symbol].get("day")
                if not isinstance(bars, list):
                    raise ValueError(f"Tencent response is missing {key} bars")
                for bar in bars:
                    if not isinstance(bar, list) or len(bar) < 9:
                        raise ValueError("malformed Tencent daily bar")
                    day = pd.Timestamp(bar[0]).normalize()
                    if day < max(start, year_start) or day > min(end, year_end):
                        continue
                    op, close, high, low = (float(bar[index]) for index in (1, 2, 3, 4))
                    volume = float(bar[5])
                    turnover = float(bar[7])
                    amount = float(bar[8]) * 10000.0
                    values = (op, close, high, low, volume, turnover, amount)
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError("non-finite Tencent daily value")
                    if min(op, close, high, low) <= 0 or high < max(op, close, low) or low > min(op, close, high):
                        raise ValueError("invalid Tencent OHLC range")
                    if volume < 0 or turnover < 0 or amount < 0:
                        raise ValueError("negative Tencent volume, turnover or amount")
                    row = {
                        "date": day,
                        "volume": volume if symbol.startswith("sh688") else volume * 100.0,
                        "amount": amount,
                        "turnover": turnover,
                    }
                    if adjust:
                        row.update(open=op, high=high, low=low, close=close)
                    else:
                        row.update(trade_open=op, trade_high=high, trade_low=low, trade_close=close)
                    selected.append(row)
            frame = pd.DataFrame(selected)
            if not frame.empty and frame["date"].duplicated().any():
                raise ValueError("duplicate Tencent stock/date rows")
            return frame

        try:
            adjusted = fetch("qfq")
            raw = fetch("")
        finally:
            session.close()
        return _merge_adjusted_with_trade_price(adjusted, raw)

    @staticmethod
    def _download_stock_daily_baostock(code: str, start_date: str, end_date: str) -> pd.DataFrame:
        import baostock as bs

        lg = bs.login()
        if getattr(lg, "error_code", "0") != "0":
            raise RuntimeError(getattr(lg, "error_msg", "baostock login failed"))
        try:
            market = "sh" if code.endswith(".SH") else "sz"
            bs_code = f"{market}.{code.split('.')[0]}"
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if getattr(rs, "error_code", "0") != "0":
                raise RuntimeError(getattr(rs, "error_msg", "baostock query failed"))
            adjusted = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])

            raw_rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3",
            )
            raw_rows = []
            while raw_rs.next():
                raw_rows.append(raw_rs.get_row_data())
            if getattr(raw_rs, "error_code", "0") != "0":
                raise RuntimeError(getattr(raw_rs, "error_msg", "baostock raw query failed"))
            raw = pd.DataFrame(raw_rows, columns=["date", "trade_open", "trade_high", "trade_low", "trade_close"])
            return _merge_adjusted_with_trade_price(adjusted, raw)
        finally:
            bs.logout()

    def _download_stock_daily_local_csv(self, code: str, start_date: str, end_date: str) -> pd.DataFrame:
        data_root = Path(self.config.get("data_root", "my_strategy/data"))
        candidates = [
            data_root / "raw" / "stock_daily" / f"{code}.csv",
            data_root / "raw" / "custom_pool" / f"{code}.csv",
        ]
        for path in candidates:
            if path.exists():
                df = pd.read_csv(path)
                df["date"] = pd.to_datetime(df["date"])
                return df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
        return pd.DataFrame()


def _normalize_stock_code(code: str) -> str:
    code = str(code).strip().split(".")[0].zfill(6)
    suffix = "SH" if code.startswith("6") else "SZ"
    return f"{code}.{suffix}"


def _merge_adjusted_with_trade_price(adjusted: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if adjusted is None or adjusted.empty:
        return adjusted
    adjusted = adjusted.copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"])
    if raw is None or raw.empty:
        return adjusted
    raw = raw[["date", "trade_open", "trade_high", "trade_low", "trade_close"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    return adjusted.merge(raw, on="date", how="left")
