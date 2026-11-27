#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DB-first repository for raw daily market data."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any

import pandas as pd

from my_strategy.storage.db_paths import raw_db_path
from my_strategy.storage.warehouse_schema import create_views, init_warehouse


class RawMarketRepository:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path else raw_db_path()
        if not self.db_path.is_absolute():
            self.db_path = Path.cwd() / self.db_path

    def exists(self) -> bool:
        return self.db_path.exists()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        init_warehouse(conn)
        return conn

    def list_stocks(self) -> list[str]:
        if not self.db_path.exists():
            return []
        with self.connect() as conn:
            rows = conn.execute("SELECT stock FROM securities ORDER BY stock").fetchall()
        return [str(row["stock"]) for row in rows]

    def has_stock(self, code: str) -> bool:
        if not self.db_path.exists():
            return False
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM stock_daily_normalized WHERE stock = ? LIMIT 1", (code,)).fetchone()
        return row is not None

    def read_stock_daily(self, code: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        clauses = ["stock = ?"]
        params: list[Any] = [code]
        if start_date:
            clauses.append("date >= ?")
            params.append(str(start_date))
        if end_date:
            clauses.append("date <= ?")
            params.append(str(end_date))
        sql = f"""
            SELECT stock AS code, date, open, high, low, close,
                   factor_open, factor_high, factor_low, factor_close,
                   volume, amount, turnover, pct_chg,
                   open AS trade_open, high AS trade_high, low AS trade_low, close AS trade_close,
                   source, source_file, imported_at AS updated_at
            FROM stock_daily_normalized
            WHERE {' AND '.join(clauses)}
            ORDER BY date
        """
        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def read_stock_daily_batch(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
        *,
        chunk_size: int = 500,
    ) -> pd.DataFrame:
        """Read daily bars for many stocks with chunked SQL IN queries."""
        if not self.db_path.exists() or not stocks:
            return pd.DataFrame()
        frames: list[pd.DataFrame] = []
        ordered_stocks = [str(stock) for stock in stocks if str(stock).strip()]
        with self.connect() as conn:
            conn.execute("PRAGMA temp_store=MEMORY")
            conn.execute("PRAGMA cache_size=-200000")
            conn.execute("PRAGMA mmap_size=268435456")
            for start in range(0, len(ordered_stocks), chunk_size):
                chunk = ordered_stocks[start : start + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                clauses = [f"stock IN ({placeholders})"]
                params: list[Any] = list(chunk)
                if start_date:
                    clauses.append("date >= ?")
                    params.append(str(start_date))
                if end_date:
                    clauses.append("date <= ?")
                    params.append(str(end_date))
                sql = f"""
                    SELECT stock, stock AS code, date, open, high, low, close,
                           factor_open, factor_high, factor_low, factor_close,
                           volume, amount, turnover, pct_chg,
                           open AS trade_open, high AS trade_high, low AS trade_low, close AS trade_close,
                           source, source_file, imported_at AS updated_at
                    FROM stock_daily_normalized
                    WHERE {' AND '.join(clauses)}
                    ORDER BY stock, date
                """
                frame = pd.read_sql_query(sql, conn, params=params)
                if not frame.empty:
                    frames.append(frame)
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values(["stock", "date"]).reset_index(drop=True)

    def stock_daily_coverage(
        self,
        stocks: list[str],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, dict[str, Any]]:
        if not self.db_path.exists() or not stocks:
            return {}
        out: dict[str, dict[str, Any]] = {}
        ordered_stocks = [str(stock) for stock in stocks if str(stock).strip()]
        with self.connect() as conn:
            for start in range(0, len(ordered_stocks), 500):
                chunk = ordered_stocks[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                clauses = [f"stock IN ({placeholders})"]
                params: list[Any] = list(chunk)
                if start_date:
                    clauses.append("date >= ?")
                    params.append(str(start_date))
                if end_date:
                    clauses.append("date <= ?")
                    params.append(str(end_date))
                rows = conn.execute(
                    f"""
                    SELECT stock, MIN(date) AS first_date, MAX(date) AS last_date, COUNT(*) AS rows
                    FROM stock_daily_normalized
                    WHERE {' AND '.join(clauses)}
                    GROUP BY stock
                    """,
                    params,
                ).fetchall()
                for row in rows:
                    out[str(row["stock"])] = {
                        "first_date": row["first_date"],
                        "last_date": row["last_date"],
                        "rows": int(row["rows"] or 0),
                    }
        return out

    def upsert_stock_daily(self, code: str, df: pd.DataFrame, *, source: str = "unknown") -> int:
        if df is None or df.empty:
            return 0
        imported_at = datetime.now().astimezone().isoformat(timespec="seconds")
        out = df.copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out = out.dropna(subset=["date"])
        out["stock"] = code
        out["source"] = source
        out["updated_at"] = imported_at
        out["source_file"] = f"db://stock_daily/{code}"
        out["imported_at"] = imported_at
        out["row_number"] = range(len(out))

        raw_cols = [
            "stock",
            "date",
            "row_number",
            "code",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "turnover",
            "pct_chg",
            "trade_open",
            "trade_high",
            "trade_low",
            "trade_close",
            "source",
            "updated_at",
            "source_file",
            "imported_at",
        ]
        raw = _select_columns(out, raw_cols)
        normalized = _normalize_daily(out, code=code, imported_at=imported_at)
        first_date = normalized["date"].min() if not normalized.empty else None
        last_date = normalized["date"].max() if not normalized.empty else None
        with self.connect() as conn:
            conn.execute("DELETE FROM stock_daily_raw WHERE stock = ?", (code,))
            conn.execute("DELETE FROM stock_daily_normalized WHERE stock = ?", (code,))
            _append_dataframe(conn, "stock_daily_raw", raw)
            _append_dataframe(conn, "stock_daily_normalized", normalized)
            conn.execute(
                """
                INSERT INTO securities (stock, exchange, name, first_date, last_date, raw_rows, source_file, updated_at, imported_at)
                VALUES (?, ?, '', ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stock) DO UPDATE SET
                    first_date=excluded.first_date,
                    last_date=excluded.last_date,
                    raw_rows=excluded.raw_rows,
                    source_file=excluded.source_file,
                    updated_at=excluded.updated_at,
                    imported_at=excluded.imported_at
                """,
                (
                    code,
                    code.split(".")[-1] if "." in code else "",
                    first_date,
                    last_date,
                    int(len(raw)),
                    f"db://stock_daily/{code}",
                    imported_at,
                    imported_at,
                ),
            )
            create_views(conn)
        return int(len(normalized))

    def update_status(self, row: dict[str, Any]) -> None:
        payload = {
            "code": row.get("code", ""),
            "last_date": row.get("last_date", ""),
            "last_update_time": row.get("last_update_time", ""),
            "rows": int(row.get("rows") or 0),
            "source": row.get("source", ""),
            "status": row.get("status", ""),
            "error_msg": row.get("error_msg", ""),
            "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO data_source_status (code, last_date, last_update_time, rows, source, status, error_msg, updated_at)
                VALUES (:code, :last_date, :last_update_time, :rows, :source, :status, :error_msg, :updated_at)
                ON CONFLICT(code) DO UPDATE SET
                    last_date=excluded.last_date,
                    last_update_time=excluded.last_update_time,
                    rows=excluded.rows,
                    source=excluded.source,
                    status=excluded.status,
                    error_msg=excluded.error_msg,
                    updated_at=excluded.updated_at
                """,
                payload,
            )

    def append_log(self, row: dict[str, Any]) -> None:
        payload = {key: row.get(key, "") for key in ["time", "code", "action", "start_date", "end_date", "rows", "source", "status", "error_msg"]}
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO update_log (time, code, action, start_date, end_date, rows, source, status, error_msg)
                VALUES (:time, :code, :action, :start_date, :end_date, :rows, :source, :status, :error_msg)
                """,
                payload,
            )

    def read_stock_basic(self) -> pd.DataFrame:
        if not self.db_path.exists():
            return pd.DataFrame()
        with self.connect() as conn:
            df = pd.read_sql_query("SELECT * FROM stock_basic ORDER BY code", conn)
        if df.empty:
            return df
        payloads = []
        for _, row in df.iterrows():
            payload = json.loads(row.get("payload_json") or "{}")
            payload.update({col: row[col] for col in df.columns if col not in {"payload_json", "updated_at"} and pd.notna(row[col])})
            payloads.append(payload)
        return pd.DataFrame(payloads)

    def save_stock_basic(self, df: pd.DataFrame) -> int:
        if df is None or df.empty:
            return 0
        now = datetime.now().astimezone().isoformat(timespec="seconds")
        rows = []
        for _, item in df.iterrows():
            payload = item.to_dict()
            code = str(payload.get("code") or payload.get("stock") or "")
            if not code:
                continue
            rows.append(
                {
                    "code": code,
                    "name": payload.get("name"),
                    "stock_name": payload.get("stock_name"),
                    "market": payload.get("market"),
                    "latest_price": payload.get("latest_price"),
                    "total_market_cap": payload.get("total_market_cap"),
                    "turnover": payload.get("turnover"),
                    "payload_json": json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
                    "updated_at": now,
                }
            )
        with self.connect() as conn:
            conn.execute("DELETE FROM stock_basic")
            if rows:
                pd.DataFrame(rows).to_sql("stock_basic", conn, if_exists="append", index=False)
        return len(rows)


def _select_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame()
    for col in columns:
        out[col] = df[col] if col in df.columns else None
    return out


def _normalize_daily(df: pd.DataFrame, *, code: str, imported_at: str) -> pd.DataFrame:
    out = df.copy()
    for col in ["open", "high", "low", "close", "trade_open", "trade_high", "trade_low", "trade_close", "volume", "amount", "turnover", "pct_chg"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for price_col, trade_col in [("open", "trade_open"), ("high", "trade_high"), ("low", "trade_low"), ("close", "trade_close")]:
        if trade_col in out.columns:
            trade = pd.to_numeric(out[trade_col], errors="coerce")
            base = pd.to_numeric(out.get(price_col), errors="coerce")
            out[price_col] = trade.where(trade > 0, base)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out = out.dropna(subset=["date"])
    out["_has_trade_price"] = (pd.to_numeric(out.get("trade_close"), errors="coerce") > 0).fillna(False).astype(int) if "trade_close" in out.columns else 0
    out["_order"] = range(len(out))
    out = out.sort_values(["date", "_has_trade_price", "_order"], ascending=[True, False, True])
    duplicate_counts = out.groupby("date")["date"].transform("size")
    out = out.drop_duplicates(subset=["date"], keep="first")
    normalized = pd.DataFrame(
        {
            "stock": code,
            "date": out["date"],
            "open": out.get("open"),
            "high": out.get("high"),
            "low": out.get("low"),
            "close": out.get("close"),
            "factor_open": out.get("open"),
            "factor_high": out.get("high"),
            "factor_low": out.get("low"),
            "factor_close": out.get("close"),
            "volume": out.get("volume"),
            "amount": out.get("amount"),
            "turnover": out.get("turnover"),
            "pct_chg": out.get("pct_chg"),
            "source": out.get("source"),
            "source_file": out.get("source_file"),
            "duplicate_source_rows": duplicate_counts.loc[out.index].astype(int).values,
            "has_trade_price": out["_has_trade_price"].astype(int).values,
            "imported_at": imported_at,
        }
    )
    return normalized


def _append_dataframe(conn: sqlite3.Connection, table: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    clean = df.where(pd.notna(df), None)
    clean.to_sql(table, conn, if_exists="append", index=False, chunksize=2000)
    return int(len(clean))
