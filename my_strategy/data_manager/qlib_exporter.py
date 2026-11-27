"""Placeholder exporter for future Qlib integration."""

from __future__ import annotations

from pathlib import Path

from .config import load_data_config, project_path
from .local_cache import LocalCache
from my_strategy.storage.access_layer import get_data_access


class QlibExporter:
    def __init__(self, config: dict | None = None):
        self.config = config or load_data_config()
        self.cache = LocalCache(self.config)

    def export_stock_daily_csv(self, output_dir: str | Path | None = None) -> Path:
        output = project_path(output_dir or (self.cache.data_root / "qlib_data" / "stock_daily_csv"))
        output.mkdir(parents=True, exist_ok=True)
        storage = get_data_access()
        stocks = storage.list_stocks()
        if not stocks and storage.settings.write_files:
            stocks = [path.stem for path in self.cache.stock_daily_dir.glob("*.parquet")]
        for stock in stocks:
            df = self.cache.read_stock_daily(stock)
            if not df.empty:
                df.to_csv(output / f"{stock}.csv", index=False, encoding="utf-8-sig")
        return output
