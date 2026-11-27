"""Daily incremental update workflow."""

from __future__ import annotations

from datetime import datetime
import time

import pandas as pd

from .config import load_data_config
from .downloader import DataDownloader
from .local_cache import LocalCache
from .stock_pool import StockPoolManager, normalize_stock_code
from .validator import DataValidator


class DailyUpdater:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.cache = LocalCache(self.config)
        self.downloader = DataDownloader(self.config)

    def update_one(self, code: str, start_date: str | None = None, end_date: str | None = None) -> dict:
        code = normalize_stock_code(code)
        update_cfg = self.config.get("update", {})
        start_date = start_date or update_cfg.get("start_date", "2015-01-01")
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        old_df = self.cache.read_stock_daily(code)
        old_last_date = None if old_df.empty else pd.to_datetime(old_df["date"]).max()
        old_has_trade_prices = False
        if "trade_close" in old_df.columns:
            old_has_trade_prices = pd.to_numeric(old_df["trade_close"], errors="coerce").notna().any()

        download_start = pd.to_datetime(start_date)
        if update_cfg.get("auto_incremental", True) and old_last_date is not None and old_has_trade_prices:
            download_start = old_last_date + pd.Timedelta(days=1)

        if old_last_date is not None and old_last_date >= pd.to_datetime(end_date) and old_has_trade_prices:
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "success", "")
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date})
            return row

        if download_start > pd.to_datetime(end_date):
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "success", "")
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "skip", "start_date": "", "end_date": end_date})
            return row

        result = self.downloader.download_stock_daily(code, download_start.strftime("%Y-%m-%d"), end_date)
        if not result.ok or result.data is None or result.data.empty:
            row = self._status_row(code, old_last_date, len(old_df), "local_cache", "failed", result.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        new_df = DataValidator.clean_stock_daily(result.data, code)
        new_valid = DataValidator.validate_stock_daily(new_df, min_rows=1, require_min_rows=False)
        if not new_valid.ok:
            row = self._status_row(code, old_last_date, len(old_df), result.source, "failed", new_valid.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        merged = pd.concat([old_df, new_df], ignore_index=True) if not old_df.empty else new_df
        merged = DataValidator.clean_stock_daily(merged, code)
        merged_valid = DataValidator.validate_stock_daily(
            merged,
            min_rows=int(self.config.get("cache", {}).get("min_rows_for_valid_stock", 100)),
            require_min_rows=True,
        )
        if not merged_valid.ok:
            row = self._status_row(code, old_last_date, len(old_df), result.source, "failed", merged_valid.error)
            self.cache.update_status(row)
            self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
            return row

        self.cache.save_stock_daily(code, merged, source=result.source)
        last_date = pd.to_datetime(merged["date"]).max()
        row = self._status_row(code, last_date, len(merged), result.source, "success", "")
        self.cache.update_status(row)
        self.cache.append_log({**row, "time": now, "action": "update", "start_date": download_start.strftime("%Y-%m-%d"), "end_date": end_date})
        return row

    def update_pool(
        self,
        mode: str | None = None,
        stocks: list[str] | None = None,
        end_date: str | None = None,
        max_stocks: int | None = None,
        progress=None,
    ) -> pd.DataFrame:
        pool = StockPoolManager(self.config).build_pool(mode=mode, stocks=stocks)
        if max_stocks and max_stocks > 0:
            pool = pool.head(max_stocks).copy()
        rows = []
        total = len(pool)
        sleep_seconds = float(self.config.get("update", {}).get("sleep_seconds", 0.8))
        for index, (_, item) in enumerate(pool.iterrows(), 1):
            row = self.update_one(item["stock"], end_date=end_date)
            rows.append(row)
            if progress is not None:
                progress(index, total, row)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
        return pd.DataFrame(rows)

    @staticmethod
    def _status_row(code: str, last_date, rows: int, source: str, status: str, error_msg: str) -> dict:
        if pd.isna(last_date):
            last_date_text = ""
        elif last_date is None:
            last_date_text = ""
        else:
            last_date_text = pd.to_datetime(last_date).strftime("%Y-%m-%d")
        return {
            "code": code,
            "last_date": last_date_text,
            "last_update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "rows": rows,
            "source": source,
            "status": status,
            "error_msg": error_msg,
        }
