#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Maintain AI-Agent screen SQL, generated files and model artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from my_strategy.model_housekeeper import cleanup_model_artifacts
from my_strategy.storage.cleanup_files import cleanup_generated_files
from my_strategy.storage.cleanup_sqlite import cleanup_ai_agent_screen_db

DEFAULT_PROCESSED_ROOT = PROJECT_ROOT / "my_strategy" / "data" / "processed"
DEFAULT_SCREEN_ROOT = DEFAULT_PROCESSED_ROOT / "ai_agent_screen"
DEFAULT_DB_PATH = DEFAULT_SCREEN_ROOT / "ai_agent_screen.db"


def summarize_maintenance_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return compact counters for an operator-facing maintenance preview."""
    sql = report.get("sql") or {}
    sql_actions = sql.get("actions") or []
    sql_rows = sum(int(item.get("rows", 0) or 0) for item in sql_actions)

    files = report.get("files") or {}
    models = report.get("models") or {}
    before = int(sql.get("database_size_before_bytes", 0) or 0)
    after = int(sql.get("database_size_after_bytes", before) or before)
    return {
        "mode": report.get("mode", ""),
        "db_path": sql.get("db_path", ""),
        "db_size_before_bytes": before,
        "db_size_after_bytes": after,
        "sql_candidate_rows": sql_rows,
        "file_candidate_count": int(files.get("candidate_count", 0) or 0),
        "file_candidate_size_bytes": int(files.get("candidate_size_bytes", 0) or 0),
        "model_candidate_count": int(models.get("candidate_count", 0) or 0),
        "model_candidate_size_bytes": int(models.get("candidate_size_bytes", 0) or 0),
        "kept_detail_runs": sql.get("kept_detail_runs", []),
        "kept_summary_runs": sql.get("kept_summary_runs", []),
    }


def format_maintenance_preview(report: dict[str, Any]) -> str:
    """Format a concise preview instead of dumping a long JSON object."""
    summary = summarize_maintenance_report(report)
    executed = bool(report.get("execute"))
    title = "维护结果" if executed else "维护预览"
    row_label = "已删除 SQL 行数" if executed else "预计删除 SQL 行数"
    file_label = "已删除文件" if executed else "预计删除文件"
    model_label = "已删除旧模型/报告" if executed else "预计删除旧模型/报告"
    size_label = "SQLite 清理后大小" if executed else "SQLite 当前大小"
    size_value = summary["db_size_after_bytes"] if executed else summary["db_size_before_bytes"]
    lines = [
        "",
        title,
        f"SQLite: {summary['db_path'] or '-'}",
        f"{size_label}: {_format_bytes(size_value)}",
        f"{row_label}: {summary['sql_candidate_rows']:,}",
        f"{file_label}: {summary['file_candidate_count']} 个，约 {_format_bytes(summary['file_candidate_size_bytes'])}",
        f"{model_label}: {summary['model_candidate_count']} 个，约 {_format_bytes(summary['model_candidate_size_bytes'])}",
    ]
    if summary["kept_detail_runs"]:
        lines.append(f"保留明细 run: {', '.join(summary['kept_detail_runs'])}")
    if summary["kept_summary_runs"]:
        lines.append(f"保留汇总 run: {', '.join(summary['kept_summary_runs'][:3])}")
    return "\n".join(lines)


def _format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if amount < 1024 or unit == "TB":
            return f"{amount:.2f} {unit}" if unit != "B" else f"{int(amount)} B"
        amount /= 1024


def run_maintenance(args: argparse.Namespace) -> dict[str, Any]:
    report: dict[str, Any] = {
        "execute": bool(args.execute),
        "mode": "execute" if args.execute else "dry_run",
        "sql": cleanup_ai_agent_screen_db(
            db_path=args.db_path,
            keep_latest_runs=args.keep_latest_runs,
            keep_detail_runs=args.keep_detail_runs,
            keep_summary_runs=args.keep_summary_runs,
            keep_rl_experience_runs=args.keep_rl_experience_runs,
            vacuum=args.vacuum,
            execute=args.execute,
        ),
    }
    if args.delete_files:
        report["files"] = cleanup_generated_files(
            processed_root=args.processed_root,
            screen_root=args.output_root,
            keep_run_logs=args.keep_run_logs,
            keep_figures=args.keep_figures,
            include_legacy_backtests=args.include_legacy_backtests,
            execute=args.execute,
        )
    if args.delete_models:
        report["models"] = cleanup_model_artifacts(
            keep_recent=args.keep_models_per_kind,
            keep_top=args.keep_top_models_per_kind,
            execute=args.execute,
        )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run or execute AI-Agent data/model retention.")
    parser.add_argument("--execute", action="store_true", help="Actually delete planned rows/files/models.")
    parser.add_argument("--vacuum", action="store_true", help="Run VACUUM after SQL deletion. Requires --execute.")
    parser.add_argument("--delete-files", action="store_true", help="Include generated file cleanup.")
    parser.add_argument("--delete-models", action="store_true", help="Include model artifact cleanup.")
    parser.add_argument("--include-legacy-backtests", action="store_true", help="Allow pruning old non-screen backtest roots.")
    parser.add_argument("--output-root", default=str(DEFAULT_SCREEN_ROOT))
    parser.add_argument("--processed-root", default=str(DEFAULT_PROCESSED_ROOT))
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--keep-latest-runs", type=int, default=3)
    parser.add_argument("--keep-detail-runs", type=int, default=1)
    parser.add_argument("--keep-summary-runs", type=int, default=10)
    parser.add_argument("--keep-rl-experience-runs", type=int, default=1)
    parser.add_argument("--keep-run-logs", type=int, default=10)
    parser.add_argument("--keep-figures", type=int, default=80)
    parser.add_argument("--keep-models-per-kind", type=int, default=2)
    parser.add_argument("--keep-top-models-per-kind", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    result = run_maintenance(args)
    if not args.execute:
        print(format_maintenance_preview(result))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
