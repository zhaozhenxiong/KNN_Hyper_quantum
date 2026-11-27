#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Refresh AI-Agent chart data and redraw current figures/top charts.

This script fixes old daily CSVs that only had close prices by merging local
OHLC prices from the local data center, then redraws existing figures and the
current top chart set with the shared candlestick chart renderer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.charting import plot_backtest_chart

DEFAULT_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed" / "ai_agent_screen"
RAW_DAILY_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "raw" / "stock_daily"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh AI-Agent daily OHLC data and redraw charts.")
    parser.add_argument("--output-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--top-count", type=int, default=10)
    parser.add_argument("--skip-daily-fix", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument("--skip-top-charts", action="store_true")
    parser.add_argument("--max-files", type=int, default=0, help="Debug cap for daily files.")
    parser.add_argument("--mark-full-cycles-only", action="store_true", default=True)
    parser.add_argument("--all-trade-marks", dest="mark_full_cycles_only", action="store_false")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

    args = parse_args()
    root = resolve_path(args.output_root)
    started = time.time()
    report: dict[str, Any] = {
        "output_root": str(root),
        "started_at": pd.Timestamp.now().isoformat(),
        "daily_fix": {},
        "figures": {},
        "top_charts": {},
        "issues": [],
    }

    print(f"刷新目录: {root}")
    if not args.skip_daily_fix:
        report["daily_fix"] = refresh_daily_files(root, max_files=args.max_files, issues=report["issues"])
    if not args.skip_figures:
        report["figures"] = redraw_existing_figures(root, args.mark_full_cycles_only, issues=report["issues"])
    if not args.skip_top_charts:
        report["top_charts"] = redraw_top_charts(root, args.top_count, args.mark_full_cycles_only, issues=report["issues"])

    report["elapsed_seconds"] = round(time.time() - started, 2)
    report["finished_at"] = pd.Timestamp.now().isoformat()
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "chart_refresh_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(_compact_report(report, report_path), ensure_ascii=False, indent=2))


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def refresh_daily_files(root: Path, max_files: int = 0, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = issues if issues is not None else []
    files = list(_daily_files(root))
    if max_files:
        files = files[:max_files]
    rows = []
    ok = missing = errors = updated = suspicious = 0
    cache: dict[str, pd.DataFrame | None] = {}
    start_ts = time.time()

    print(f"核对/补齐 daily: {len(files)} 个文件")
    for idx, path in enumerate(files, start=1):
        stock = stock_from_daily_path(path)
        try:
            daily = pd.read_csv(path)
            fixed, info = enrich_daily_with_local_ohlc(daily, stock, cache)
            info.update({"file": str(path), "stock": stock})
            rows.append(info)
            if info["status"] == "missing_local":
                missing += 1
            elif info["status"] == "ok":
                ok += 1
                if info["changed"]:
                    fixed.to_csv(path, index=False, encoding="utf-8-sig")
                    updated += 1
                if info["max_close_gap"] >= 0.03:
                    suspicious += 1
                    issues.append(info)
            else:
                errors += 1
                issues.append(info)
        except Exception as exc:
            errors += 1
            issue = {"file": str(path), "stock": stock, "status": "error", "error": str(exc)[:240]}
            rows.append(issue)
            issues.append(issue)

        if idx == 1 or idx % 200 == 0 or idx == len(files):
            elapsed = max(time.time() - start_ts, 1)
            print(f"  daily [{idx}/{len(files)}] updated={updated} missing={missing} suspicious={suspicious} rate={idx/elapsed:.1f}/s")

    report_path = root / "reports" / "daily_price_check.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report_path, index=False, encoding="utf-8-sig")
    return {
        "files": len(files),
        "ok": ok,
        "updated": updated,
        "missing_local": missing,
        "errors": errors,
        "suspicious_close_gap": suspicious,
        "report_file": str(report_path),
    }


def enrich_daily_with_local_ohlc(
    daily: pd.DataFrame,
    stock: str,
    cache: dict[str, pd.DataFrame | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if daily.empty or "date" not in daily.columns:
        return daily, {"status": "bad_daily", "changed": False, "max_close_gap": 0.0, "missing_rows": 0}

    out = daily.copy()
    out["_date_key"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    duplicate_dates_before = int(out["_date_key"].duplicated(keep=False).sum())
    local = load_local_ohlc(stock, cache)
    if local is None or local.empty:
        out.drop(columns=["_date_key"], inplace=True, errors="ignore")
        return out, {"status": "missing_local", "changed": False, "max_close_gap": 0.0, "missing_rows": len(daily)}

    local = local.copy()
    local["_date_key"] = pd.to_datetime(local["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged = out.merge(
        local[["_date_key", "open", "high", "low", "close", "volume"]].rename(
            columns={col: f"local_{col}" for col in ["open", "high", "low", "close", "volume"]}
        ),
        on="_date_key",
        how="left",
    )

    old_close = pd.to_numeric(merged.get("close"), errors="coerce")
    local_close = pd.to_numeric(merged["local_close"], errors="coerce")
    close_gap = ((old_close - local_close).abs() / local_close.abs()).where(local_close > 0)
    max_gap = float(close_gap.max(skipna=True) if close_gap.notna().any() else 0.0)
    missing_rows = int(local_close.isna().sum())

    changed = False
    if "factor_close" not in merged.columns and "close" in merged.columns:
        merged["factor_close"] = merged["close"]
        changed = True
    for col in ["open", "high", "low", "close", "volume"]:
        local_col = f"local_{col}"
        if local_col not in merged.columns:
            continue
        source = pd.to_numeric(merged[local_col], errors="coerce")
        current = pd.to_numeric(merged[col], errors="coerce") if col in merged.columns else pd.Series(index=merged.index, dtype=float)
        new_values = source.where(source.notna(), current)
        if col not in merged.columns or not new_values.equals(current):
            merged[col] = new_values
            changed = True

    merged.drop(columns=[c for c in merged.columns if c.startswith("local_")], inplace=True, errors="ignore")
    if duplicate_dates_before:
        merged = merged.drop_duplicates(subset=["_date_key"], keep="last").copy()
        changed = True
    merged.sort_values("_date_key", inplace=True)
    merged.drop(columns=["_date_key"], inplace=True, errors="ignore")
    front = [c for c in ["date", "open", "high", "low", "close", "factor_close", "volume"] if c in merged.columns]
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]
    return merged, {
        "status": "ok",
        "changed": bool(changed),
        "max_close_gap": round(max_gap, 6),
        "missing_rows": missing_rows,
        "duplicate_date_rows": duplicate_dates_before,
        "rows": int(len(merged)),
    }


def load_local_ohlc(stock: str, cache: dict[str, pd.DataFrame | None]) -> pd.DataFrame | None:
    if stock in cache:
        return cache[stock]
    path = RAW_DAILY_ROOT / f"{stock}.parquet"
    if not path.exists():
        cache[stock] = None
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        cache[stock] = None
        return None
    if raw.empty:
        cache[stock] = None
        return None
    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw["date"], errors="coerce")
    has_trade_price = pd.Series(False, index=raw.index)
    for base_col in ["open", "high", "low", "close"]:
        trade_col = f"trade_{base_col}"
        if trade_col in raw.columns:
            trade = pd.to_numeric(raw[trade_col], errors="coerce")
            base = pd.to_numeric(raw[base_col], errors="coerce")
            out[base_col] = trade.where(trade > 0, base)
            if base_col == "close":
                has_trade_price = trade > 0
        else:
            out[base_col] = pd.to_numeric(raw[base_col], errors="coerce")
    out["high"] = out[["open", "high", "close"]].max(axis=1)
    out["low"] = out[["open", "low", "close"]].min(axis=1)
    out["volume"] = pd.to_numeric(raw.get("volume", 0), errors="coerce")
    out["_has_trade_price"] = has_trade_price.fillna(False).astype(int)
    out = out.dropna(subset=["date", "open", "high", "low", "close"])
    out["_date_key"] = out["date"].dt.strftime("%Y-%m-%d")
    out.sort_values(["_date_key", "_has_trade_price"], ascending=[True, False], inplace=True)
    out = out.drop_duplicates(subset=["_date_key"], keep="first")
    out.drop(columns=["_has_trade_price", "_date_key"], inplace=True, errors="ignore")
    out = out.sort_values("date").reset_index(drop=True)
    cache[stock] = out
    return out


def redraw_existing_figures(root: Path, mark_full_cycles_only: bool, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = issues if issues is not None else []
    files = sorted((root / "figures").glob("*.png"))
    ok = skipped = errors = 0
    print(f"重画现有 figures: {len(files)} 张")
    for path in files:
        stock = stock_from_chart_name(path.name)
        if not stock:
            skipped += 1
            continue
        daily_path, trades_path = paths_for_stock(root, stock)
        if not daily_path.exists():
            skipped += 1
            continue
        try:
            daily = pd.read_csv(daily_path)
            trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
            title = title_for_stock(root, stock, path.name)
            plot_backtest_chart(title, daily, trades, path, score_col="final_score", mark_full_cycles_only=mark_full_cycles_only)
            ok += 1
        except Exception as exc:
            errors += 1
            issues.append({"stock": stock, "file": str(path), "status": "figure_error", "error": str(exc)[:240]})
    return {"figures": len(files), "redrawn": ok, "skipped": skipped, "errors": errors}


def redraw_top_charts(root: Path, top_count: int, mark_full_cycles_only: bool, issues: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    issues = issues if issues is not None else []
    top_dir = root / "top_charts"
    top_dir.mkdir(parents=True, exist_ok=True)
    for old in top_dir.glob("*.png"):
        old.unlink()

    candidates_path = root / "reports" / "buy_candidates.csv"
    ranking_path = root / "reports" / "ranking.csv"
    source = candidates_path if candidates_path.exists() else ranking_path
    if not source.exists():
        return {"source_file": "", "redrawn": 0, "errors": 0}
    candidates = pd.read_csv(source)
    if "rank" not in candidates.columns:
        candidates["rank"] = range(1, len(candidates) + 1)

    ok = errors = skipped = 0
    for _, row in candidates.head(top_count).iterrows():
        stock = str(row.get("stock", "")).strip()
        if not stock:
            skipped += 1
            continue
        daily_path = Path(str(row.get("daily_file", ""))) if str(row.get("daily_file", "")) else paths_for_stock(root, stock)[0]
        trades_path = Path(str(row.get("trades_file", ""))) if str(row.get("trades_file", "")) else paths_for_stock(root, stock)[1]
        if not daily_path.exists():
            skipped += 1
            continue
        try:
            daily = pd.read_csv(daily_path)
            trades = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
            rank = int(row.get("rank", ok + 1))
            stock_name = str(row.get("stock_name", "") or "")
            score = float(pd.to_numeric(pd.Series([row.get("screen_score", 0)]), errors="coerce").fillna(0).iloc[0])
            safe_name = safe_filename(stock_name or stock)
            out_path = top_dir / f"rank_{rank:02d}_{stock.replace('.', '_')}_{safe_name}_score_{score:.3f}.png"
            plot_backtest_chart(f"{stock_name} {stock}".strip(), daily, trades, out_path, score_col="final_score", mark_full_cycles_only=mark_full_cycles_only)
            ok += 1
        except Exception as exc:
            errors += 1
            issues.append({"stock": stock, "status": "top_chart_error", "error": str(exc)[:240]})
    return {"source_file": str(source), "redrawn": ok, "skipped": skipped, "errors": errors}


def _daily_files(root: Path):
    daily_dir = root / "daily"
    if daily_dir.exists():
        yield from sorted(daily_dir.glob("*_daily.csv"))
    signals_dir = root / "signals"
    if signals_dir.exists():
        yield from sorted(signals_dir.glob("*_multi_agent_daily.csv"))


def paths_for_stock(root: Path, stock: str) -> tuple[Path, Path]:
    tag = stock.replace(".", "_")
    daily = root / "daily" / f"{tag}_daily.csv"
    trades = root / "trades" / f"{tag}_trades.csv"
    if daily.exists() or trades.exists():
        return daily, trades
    return root / "signals" / f"{tag}_multi_agent_daily.csv", root / "signals" / f"{tag}_multi_agent_trades.csv"


def stock_from_daily_path(path: Path) -> str:
    name = path.name
    match = re.match(r"(\d{6})_(SZ|SH|BJ)(?:_multi_agent)?_daily\.csv$", name, flags=re.IGNORECASE)
    return f"{match.group(1)}.{match.group(2).upper()}" if match else ""


def stock_from_chart_name(name: str) -> str:
    match = re.search(r"(\d{6})_(SZ|SH|BJ)", name, flags=re.IGNORECASE)
    return f"{match.group(1)}.{match.group(2).upper()}" if match else ""


def title_for_stock(root: Path, stock: str, chart_name: str = "") -> str:
    name = ""
    for csv_name in ["ranking.csv", "buy_candidates.csv", "focus_watchlist_backtest.csv"]:
        path = root / "reports" / csv_name
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=lambda col: col in {"stock", "stock_name"})
            hit = df[df["stock"].astype(str).eq(stock)]
            if not hit.empty:
                name = str(hit.iloc[0].get("stock_name", "") or "")
                break
        except Exception:
            continue
    if not name and chart_name:
        parts = chart_name.replace(".png", "").split("_")
        if len(parts) >= 3:
            maybe_name = parts[2]
            if not maybe_name.startswith("multi") and not maybe_name.isdigit():
                name = maybe_name
    return f"{name} {stock}".strip()


def safe_filename(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r'[\\/:*?"<>|\s]+', "_", text)
    return text.strip("_") or "unknown"


def _compact_report(report: dict[str, Any], report_path: Path) -> dict[str, Any]:
    return {
        "report_file": str(report_path),
        "daily_fix": report.get("daily_fix", {}),
        "figures": report.get("figures", {}),
        "top_charts": report.get("top_charts", {}),
        "issues_count": len(report.get("issues", [])),
        "elapsed_seconds": report.get("elapsed_seconds"),
    }


if __name__ == "__main__":
    main()
