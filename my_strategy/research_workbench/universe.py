"""Versioned stock lifetimes and shared workbench coverage rules."""

from __future__ import annotations

from hashlib import file_digest
from pathlib import Path
import re
import sqlite3

import pandas as pd


BOARD_POLICIES = {
    "main_star": frozenset(("sh_main", "sz_main", "star")),
    "shsz": frozenset(("sh_main", "sz_main", "star", "chinext")),
}
POLICIES = frozenset(("legacy", *BOARD_POLICIES))
_CODE = re.compile(r"\d{6}\.(SH|SZ|BJ)")
_BOARDS_BY_EXCHANGE = {
    "SH": frozenset(("sh_main", "star")),
    "SZ": frozenset(("sz_main", "chinext")),
    "BJ": frozenset(("bse",)),
}


def load_universe(path: str | Path, policy: str) -> pd.DataFrame:
    """Load source-backed lifetimes. Delisting date is the exclusive endpoint."""
    resolved = Path(path).resolve(strict=True)
    if policy not in BOARD_POLICIES:
        raise ValueError("A roster requires an explicit main_star or shsz policy.")
    data = pd.read_csv(resolved, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    fields = ("stock", "exchange", "board", "list_date", "delist_date", "source", "source_url")
    missing = set(fields) - set(data)
    if missing or data.empty:
        raise ValueError(f"Roster is empty or missing fields: {sorted(missing)}")
    data = data[list(fields)].apply(lambda values: values.str.strip())
    for row in data.itertuples(index=False):
        if (
            _CODE.fullmatch(row.stock) is None or row.stock[-2:] != row.exchange
            or row.board not in _BOARDS_BY_EXCHANGE.get(row.exchange, ())
            or not row.source or not row.source_url
        ):
            raise ValueError(f"Unverified or invalid exchange/board/source for {row.stock}")
    if data["stock"].duplicated().any():
        raise ValueError("A roster needs one non-ambiguous lifetime per canonical stock.")
    for field in ("list_date", "delist_date"):
        values = data[field].replace("", None)
        if not values.dropna().str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            raise ValueError(f"Roster {field} needs an ISO date.")
        data[field] = pd.to_datetime(values, format="%Y-%m-%d", errors="raise")
    if data["list_date"].isna().any() or data["delist_date"].le(data["list_date"]).any():
        raise ValueError("Roster contains a missing listing date or invalid lifetime.")
    data = data.loc[data["board"].isin(BOARD_POLICIES[policy])].set_index("stock").sort_index()
    if data.empty:
        raise ValueError(f"No stocks satisfy roster policy {policy}.")
    with resolved.open("rb") as file:
        checksum = file_digest(file, "sha256").hexdigest()
    data.attrs.update(path=str(resolved), sha256=checksum, policy=policy)
    return data


def members(roster: pd.DataFrame, first: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
    """Return instruments whose lifetimes intersect the requested closed interval."""
    return roster.loc[roster["list_date"].le(last) & (roster["delist_date"].isna() | roster["delist_date"].gt(first))]


def restrict_bars(bars: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Keep only listed stock-date rows, using exchange-backed identity."""
    first = bars["stock"].map(roster["list_date"])
    last = bars["stock"].map(roster["delist_date"])
    valid = bars["date"].ge(first) & (last.isna() | bars["date"].lt(last))
    result = bars.loc[valid].copy()
    result["exchange"] = result["stock"].map(roster["exchange"])
    return result


def target_coverage(
    connection: sqlite3.Connection, roster: pd.DataFrame, date: pd.Timestamp,
) -> dict:
    """Count target-session raw bars on the date-specific listed universe."""
    active = members(roster, date, date)
    traded = {
        stock for (stock,) in connection.execute(
            "SELECT stock FROM stock_daily_normalized WHERE date=? AND has_trade_price=1",
            (date.date().isoformat(),),
        )
    }
    trading = active.index.isin(traded)
    return {
        "date": date.date().isoformat(),
        "policy": roster.attrs["policy"],
        "roster_sha256": roster.attrs["sha256"],
        "active_stocks": len(active),
        "active_trading_stocks": int(trading.sum()),
        "active_by_board": active["board"].value_counts().to_dict(),
        "active_trading_by_board": active.loc[trading, "board"].value_counts().to_dict(),
        "active_missing_stocks": active.index[~trading].tolist(),
    }
