"""Run one bounded, post-close update of every stock in the raw warehouse."""

from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, time
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd

from my_strategy.data_manager.config import load_data_config
from my_strategy.data_manager.updater import DailyUpdater
from my_strategy.research_workbench import latent_api
from my_strategy.storage.access_layer import get_data_access
from my_strategy.storage.db_paths import raw_db_path

PROGRESS_PREFIX = "KHQUANT_DATA_UPDATE_PROGRESS="
SHANGHAI = ZoneInfo("Asia/Shanghai")


def latest_complete_session(
    now: datetime | None = None,
    *,
    calendar_path: Path | None = None,
    update_after_hour: int | None = None,
) -> str:
    """Return the latest verified session whose regular close has passed."""
    current = now or datetime.now(SHANGHAI)
    if current.tzinfo is None:
        current = current.replace(tzinfo=SHANGHAI)
    else:
        current = current.astimezone(SHANGHAI)
    if update_after_hour is None:
        update_after_hour = int(load_data_config().get("update", {}).get("update_after_hour", 16))
    if not 0 <= update_after_hour <= 23:
        raise ValueError("数据更新收盘时刻必须为 0 至 23。")
    path = calendar_path or latent_api._calendar_path()
    frame = pd.read_csv(path)
    if not {"date", "is_trading_day"}.issubset(frame.columns):
        raise ValueError("已验证交易日历缺少 date 或 is_trading_day。")
    days = pd.to_datetime(frame["date"], errors="coerce")
    if days.isna().any() or days.duplicated().any():
        raise ValueError("已验证交易日历包含无效或重复日期。")
    sessions = pd.DatetimeIndex(days.loc[frame["is_trading_day"].eq(1)]).sort_values()
    today = pd.Timestamp(current.date())
    close_passed = current.time() >= time(update_after_hour, 0)
    eligible = sessions[sessions <= today] if close_passed else sessions[sessions < today]
    if eligible.empty:
        raise ValueError("已验证交易日历中没有最近完整交易日。")
    return eligible[-1].date().isoformat()


def _connection() -> sqlite3.Connection:
    path = raw_db_path().resolve(strict=True)
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def _market(connection: sqlite3.Connection) -> dict:
    first, last, total = connection.execute(
        """SELECT MIN(first_date), (SELECT MAX(date) FROM stock_daily_normalized),
                  COUNT(*) FROM securities WHERE raw_rows > 0"""
    ).fetchone()
    latest_stocks = connection.execute(
        "SELECT COUNT(*) FROM stock_daily_normalized WHERE date=? AND has_trade_price=1", (last,),
    ).fetchone()[0] if last else 0
    return {
        "data_start": first,
        "latest_date": last,
        "total_stocks": int(total),
        "latest_stocks": int(latest_stocks),
    }


def _validate_target(connection: sqlite3.Connection, target: str) -> dict:
    rows, traded = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(has_trade_price=1), 0) FROM stock_daily_normalized WHERE date=?",
        (target,),
    ).fetchone()
    invalid = connection.execute(
        """SELECT COUNT(*) FROM stock_daily_normalized
           WHERE date=? AND has_trade_price=1 AND (
               open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL OR volume IS NULL
               OR open<=0 OR high<=0 OR low<=0 OR close<=0 OR volume<=0
               OR high<low OR high<open OR high<close OR low>open OR low>close
           )""",
        (target,),
    ).fetchone()[0]
    duplicates = connection.execute(
        """SELECT COUNT(*) FROM (
               SELECT stock, date FROM stock_daily_normalized WHERE date=? GROUP BY stock, date HAVING COUNT(*)>1
           )""",
        (target,),
    ).fetchone()[0]
    return {
        "target_rows": int(rows),
        "target_trade_rows": int(traded),
        "invalid_trade_rows": int(invalid),
        "duplicate_stock_dates": int(duplicates),
        "passed": bool(traded and not invalid and not duplicates),
    }


def _write_result(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_update(job_id: str, end_date: str, result_path: Path) -> dict:
    if not latent_api.ID.fullmatch(job_id):
        raise ValueError("更新任务编号无效。")
    target = datetime.strptime(end_date, "%Y-%m-%d").date().isoformat()
    started_at = latent_api._timestamp()
    with closing(_connection()) as connection:
        before = _market(connection)
    if before["latest_date"] and before["latest_date"] > target:
        raise ValueError(f"原始仓已包含晚于目标日 {target} 的行情 {before['latest_date']}，拒绝回写。")

    config = load_data_config()
    updater = DailyUpdater(config)

    def progress(processed: int, total: int, row: dict) -> None:
        if processed == total or processed % 25 == 0 or processed == 1:
            print(PROGRESS_PREFIX + json.dumps({
                "processed": processed,
                "total": total,
                "last_code": row.get("code", ""),
                "last_status": row.get("status", ""),
            }, ensure_ascii=False), flush=True)

    report = updater.update_pool(mode="db", end_date=target, progress=progress)
    if report.empty:
        raise ValueError("原始仓股票池为空，没有执行行情更新。")
    summary = get_data_access().record_data_update(
        update_id=job_id,
        started_at=started_at,
        finished_at=latent_api._timestamp(),
        mode="db",
        stocks_requested="",
        end_date=target,
        report=report,
    )
    statuses = report.get("status", pd.Series(dtype=str)).astype(str)
    last_dates = report.get("last_date", pd.Series(dtype=str)).astype(str)
    target_success = int(((statuses == "success") & (last_dates == target)).sum())
    all_reached = bool(len(report)) and int(last_dates.eq(target).sum()) == len(report)
    with closing(_connection()) as connection:
        after = _market(connection)
        validation = _validate_target(connection, target)
    if after["latest_date"] and after["latest_date"] > target:
        raise ValueError(f"更新后出现晚于目标日的行情 {after['latest_date']}。")
    reached = after["latest_date"] == target and validation["passed"]
    if reached and all_reached and not summary["failed"]:
        status = "completed"
        message = f"原始仓全部股票的日线已更新至 {target}。"
    elif reached and target_success:
        status = "partial"
        message = f"原始行情已部分更新至 {target}，{summary['failed']} 只股票失败。"
    else:
        status = "failed"
        message = f"更新未形成通过验证的 {target} 原始日线覆盖。"
    failures = report.loc[statuses == "failed"].head(5)
    failure_examples = [
        f"{row.get('code', '')}: {str(row.get('error_msg', ''))[:180]}"
        for _, row in failures.iterrows()
    ]
    result = {
        "schema": "khquant_data_update_v1",
        "job_id": job_id,
        "status": status,
        "message": message,
        "mode": "db",
        "scope_label": "原始仓全部股票",
        "target_date": target,
        "started_at": started_at,
        "finished_at": latent_api._timestamp(),
        "summary": {**summary, "target_success": target_success},
        "before": before,
        "after": after,
        "validation": validation,
        "failure_examples": failure_examples,
    }
    _write_result(result_path, result)
    print(json.dumps({key: result[key] for key in ("job_id", "status", "message", "summary")}, ensure_ascii=False), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Update all KHQuant raw-warehouse stocks after market close.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--result", required=True, type=Path)
    args = parser.parse_args()
    run_update(args.job_id, args.end_date, args.result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
