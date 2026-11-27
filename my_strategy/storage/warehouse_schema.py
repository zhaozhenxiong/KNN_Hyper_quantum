#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SQLite schema for the KHQuant unified local warehouse."""

from __future__ import annotations

import sqlite3

from my_strategy.storage.warehouse_integrity import ensure_integrity_schema

SCHEMA_VERSION = 2


def init_warehouse(conn: sqlite3.Connection) -> None:
    """Create the core warehouse schema if it does not already exist."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS warehouse_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS import_runs (
            import_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            mode TEXT NOT NULL,
            source_root TEXT NOT NULL,
            db_path TEXT NOT NULL,
            dry_run INTEGER NOT NULL DEFAULT 0,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS securities (
            stock TEXT PRIMARY KEY,
            exchange TEXT,
            name TEXT,
            first_date TEXT,
            last_date TEXT,
            raw_rows INTEGER,
            source_file TEXT,
            updated_at TEXT,
            imported_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily_raw (
            stock TEXT NOT NULL,
            date TEXT NOT NULL,
            row_number INTEGER NOT NULL,
            code TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            trade_open REAL,
            trade_high REAL,
            trade_low REAL,
            trade_close REAL,
            source TEXT,
            updated_at TEXT,
            source_file TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (stock, date, source_file, row_number)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_daily_normalized (
            stock TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            factor_open REAL,
            factor_high REAL,
            factor_low REAL,
            factor_close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            source TEXT,
            source_file TEXT NOT NULL,
            duplicate_source_rows INTEGER NOT NULL DEFAULT 1,
            has_trade_price INTEGER NOT NULL DEFAULT 0,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (stock, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_quality_issues (
            issue_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            category TEXT NOT NULL,
            severity TEXT NOT NULL,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            source_file TEXT,
            message TEXT NOT NULL,
            details_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_source_status (
            code TEXT PRIMARY KEY,
            last_date TEXT,
            last_update_time TEXT,
            rows INTEGER,
            source TEXT,
            status TEXT,
            error_msg TEXT,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS update_log (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            code TEXT,
            action TEXT,
            start_date TEXT,
            end_date TEXT,
            rows INTEGER,
            source TEXT,
            status TEXT,
            error_msg TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS data_update_runs (
            update_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            mode TEXT,
            stocks_requested TEXT,
            end_date TEXT,
            storage_mode TEXT NOT NULL,
            raw_db_path TEXT NOT NULL,
            total INTEGER NOT NULL,
            success INTEGER NOT NULL,
            failed INTEGER NOT NULL,
            skipped INTEGER NOT NULL,
            report_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_basic (
            code TEXT PRIMARY KEY,
            name TEXT,
            stock_name TEXT,
            market TEXT,
            latest_price REAL,
            total_market_cap REAL,
            turnover REAL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_snapshots (
            factor_version TEXT NOT NULL,
            stock TEXT NOT NULL,
            code TEXT,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            factor_open REAL,
            factor_high REAL,
            factor_low REAL,
            factor_close REAL,
            volume REAL,
            amount REAL,
            turnover REAL,
            pct_chg REAL,
            trade_open REAL,
            trade_high REAL,
            trade_low REAL,
            trade_close REAL,
            ma5 REAL,
            ma10 REAL,
            ma20 REAL,
            ma60 REAL,
            rsi14 REAL,
            return_1d REAL,
            return_5d REAL,
            return_20d REAL,
            vol_ma20 REAL,
            volume_ratio REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            volatility_20d REAL,
            volume_window INTEGER NOT NULL,
            computed_at TEXT NOT NULL,
            source_start TEXT,
            source_end TEXT,
            PRIMARY KEY (factor_version, stock, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS factor_snapshot_runs (
            snapshot_id TEXT PRIMARY KEY,
            factor_version TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            source_start TEXT,
            source_end TEXT,
            stocks INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            volume_window INTEGER NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_feature_snapshots (
            feature_version TEXT NOT NULL,
            stock TEXT NOT NULL,
            code TEXT,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            amount REAL,
            return_1d REAL,
            return_3d REAL,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            ma5_ratio REAL,
            ma10_ratio REAL,
            ma20_ratio REAL,
            ma60_ratio REAL,
            ma20_ma60_ratio REAL,
            rsi14 REAL,
            macd REAL,
            macd_signal REAL,
            macd_hist REAL,
            volatility_20d REAL,
            volume_ratio REAL,
            amount_change_5d REAL,
            high_low_range REAL,
            close_to_20d_high REAL,
            close_to_20d_low REAL,
            boll_pct_b REAL,
            boll_bandwidth REAL,
            macd_5_10_6 REAL,
            macd_5_10_6_signal REAL,
            macd_5_10_6_hist REAL,
            kdj_k REAL,
            kdj_d REAL,
            kdj_j REAL,
            macd_kdj_resonance REAL,
            ar_26 REAL,
            br_26 REAL,
            cci_14 REAL,
            obv_change_20d REAL,
            sar_gap REAL,
            expma_5_10_ratio REAL,
            future_return_5d REAL,
            future_up_5d REAL,
            future_return_10d REAL,
            future_up_10d REAL,
            future_return_20d REAL,
            future_up_20d REAL,
            factor_version TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            source_start TEXT,
            source_end TEXT,
            PRIMARY KEY (feature_version, stock, date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_feature_snapshot_runs (
            snapshot_id TEXT PRIMARY KEY,
            feature_version TEXT NOT NULL,
            factor_version TEXT NOT NULL,
            computed_at TEXT NOT NULL,
            source_start TEXT,
            source_end TEXT,
            stocks INTEGER NOT NULL,
            rows INTEGER NOT NULL,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            task TEXT,
            source_root TEXT NOT NULL,
            artifact_root TEXT,
            start TEXT,
            end TEXT,
            stocks INTEGER,
            success INTEGER,
            failed INTEGER,
            ml_model TEXT,
            dl_model TEXT,
            rl_model TEXT,
            elapsed_seconds REAL,
            summary_json TEXT NOT NULL DEFAULT '{}'
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_artifacts (
            run_id TEXT NOT NULL,
            artifact_type TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            size_bytes INTEGER,
            modified_at TEXT,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (run_id, artifact_type, file_path)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_feature_labels (
            feature_version TEXT NOT NULL,
            stock TEXT NOT NULL,
            date TEXT NOT NULL,
            future_return_5d REAL,
            future_up_5d REAL,
            future_return_10d REAL,
            future_up_10d REAL,
            future_return_20d REAL,
            future_up_20d REAL,
            computed_at TEXT NOT NULL,
            PRIMARY KEY (feature_version, stock, date),
            FOREIGN KEY (feature_version, stock, date)
              REFERENCES model_feature_snapshots(feature_version, stock, date)
              ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_output_references (
            run_id TEXT NOT NULL,
            scope_key TEXT NOT NULL DEFAULT '',
            output_type TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            authority TEXT NOT NULL,
            metadata_json TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL,
            PRIMARY KEY (run_id, scope_key, output_type)
        )
        """
    )
    conn.execute('CREATE TABLE IF NOT EXISTS backtest_stock_results (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS backtest_reports (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS backtest_daily_ledger (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS backtest_trades (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS agent_signal_records (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute('CREATE TABLE IF NOT EXISTS backtest_failures (run_id TEXT NOT NULL, imported_at TEXT NOT NULL)')
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS model_training_runs (
            model_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            run_id TEXT,
            model_type TEXT,
            device TEXT,
            report_path TEXT NOT NULL,
            model_path TEXT,
            created_at TEXT,
            metrics_json TEXT NOT NULL DEFAULT '{}',
            samples_json TEXT NOT NULL DEFAULT '{}',
            stocks_json TEXT NOT NULL DEFAULT '{}',
            feature_columns_json TEXT NOT NULL DEFAULT '[]',
            params_json TEXT NOT NULL DEFAULT '{}',
            imported_at TEXT NOT NULL,
            PRIMARY KEY (kind, model_name)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS comparison_reports (
            comparison_id TEXT PRIMARY KEY,
            imported_at TEXT NOT NULL,
            source_root TEXT NOT NULL,
            baseline_run_id TEXT,
            candidate_run_id TEXT,
            summary_json TEXT NOT NULL DEFAULT '{}',
            diff_file TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_snapshots (
            snapshot_id TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            run_id TEXT,
            stock TEXT NOT NULL,
            as_of_date TEXT NOT NULL,
            published_at TEXT NOT NULL,
            signal_time TEXT,
            execution_time TEXT,
            execution_assumption TEXT,
            final_action TEXT NOT NULL,
            final_score REAL,
            target_weight REAL,
            model_version TEXT NOT NULL DEFAULT '{}',
            data_version TEXT NOT NULL,
            config_hash TEXT NOT NULL,
            git_commit TEXT NOT NULL,
            seed INTEGER,
            data_manifest_hash TEXT NOT NULL,
            replay_fingerprint TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            UNIQUE(channel, stock, as_of_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS signal_snapshot_comparisons (
            comparison_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            comparison_type TEXT NOT NULL,
            baseline_snapshot_id TEXT NOT NULL,
            candidate_snapshot_id TEXT NOT NULL,
            status TEXT NOT NULL,
            summary_json TEXT NOT NULL
        )
        """
    )
    create_indexes(conn)
    ensure_integrity_schema(conn)
    create_views(conn)
    upsert_meta(conn, "schema_version", str(SCHEMA_VERSION))


def create_indexes(conn: sqlite3.Connection) -> None:
    index_specs = [
        ("idx_raw_stock_date", "stock_daily_raw", "stock", "date"),
        ("idx_norm_date_stock", "stock_daily_normalized", "date", "stock"),
        ("idx_quality_entity", "data_quality_issues", "entity_type", "entity_id"),
        ("idx_source_status_status", "data_source_status", "status", "last_update_time"),
        ("idx_data_update_runs_finished", "data_update_runs", "finished_at", "mode"),
        ("idx_factor_snapshots_stock_date", "factor_snapshots", "stock", "date", "factor_version"),
        ("idx_factor_snapshots_date_stock", "factor_snapshots", "date", "stock", "factor_version"),
        ("idx_factor_snapshot_runs_version", "factor_snapshot_runs", "factor_version", "computed_at"),
        ("idx_model_features_stock_date", "model_feature_snapshots", "stock", "date", "feature_version"),
        ("idx_model_features_date_stock", "model_feature_snapshots", "date", "stock", "feature_version"),
        ("idx_model_feature_runs_version", "model_feature_snapshot_runs", "feature_version", "computed_at"),
        ("idx_results_run_score", "backtest_stock_results", "run_id", "screen_score"),
        ("idx_results_stock_run", "backtest_stock_results", "stock", "run_id"),
        ("idx_daily_run_stock_date", "backtest_daily_ledger", "run_id", "stock", "date"),
        ("idx_trades_run_stock_date", "backtest_trades", "run_id", "stock", "date"),
        ("idx_signals_run_stock_date", "agent_signal_records", "run_id", "stock", "date"),
        ("idx_reports_run_stock", "backtest_reports", "run_id", "stock"),
        ("idx_models_kind_run", "model_training_runs", "kind", "run_id"),
        ("idx_artifacts_run_type", "run_artifacts", "run_id", "artifact_type"),
        ("idx_output_references_artifact", "run_output_references", "artifact_id"),
        ("idx_signal_snapshots_stock_asof", "signal_snapshots", "stock", "as_of_date"),
        ("idx_signal_snapshots_channel_asof", "signal_snapshots", "channel", "as_of_date"),
        ("idx_signal_snapshot_comparisons_type", "signal_snapshot_comparisons", "comparison_type", "created_at"),
    ]
    for spec in index_specs:
        name, table, *cols = spec
        existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
        if all(col in existing for col in cols):
            quoted = ", ".join(f'"{col}"' for col in cols)
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{name}" ON "{table}" ({quoted})')


def create_views(conn: sqlite3.Connection) -> None:
    """Create convenience views when their source columns are available."""
    _create_view_if_columns(
        conn,
        "v_latest_stock_daily",
        "stock_daily_normalized",
        ["stock", "date", "open", "high", "low", "close", "volume", "amount", "source"],
        """
        CREATE VIEW v_latest_stock_daily AS
        SELECT stock, date, open, high, low, close, volume, amount, source, source_file, has_trade_price
        FROM stock_daily_normalized
        """,
    )
    _create_view_if_columns(
        conn,
        "v_model_feature_inference",
        "model_feature_snapshots",
        ["feature_version", "stock", "date", "close", "return_1d", "ma5_ratio"],
        """
        CREATE VIEW v_model_feature_inference AS
        SELECT feature_version, stock, code, date, open, high, low, close, volume, amount,
               return_1d, return_3d, return_5d, return_10d, return_20d,
               ma5_ratio, ma10_ratio, ma20_ratio, ma60_ratio, ma20_ma60_ratio,
               rsi14, macd, macd_signal, macd_hist, volatility_20d, volume_ratio,
               amount_change_5d, high_low_range, close_to_20d_high, close_to_20d_low,
               factor_version, computed_at, source_start, source_end
        FROM model_feature_snapshots
        """,
    )
    _create_view_if_columns(
        conn,
        "v_latest_screen_ranking",
        "backtest_stock_results",
        ["run_id", "stock", "stock_name", "rank", "latest_action", "screen_score"],
        """
        CREATE VIEW v_latest_screen_ranking AS
        WITH latest AS (
            SELECT run_id
            FROM runs
            WHERE task IN ('multi-agent-backtest', 'model-compare-screen', 'screen-run')
            ORDER BY imported_at DESC
            LIMIT 1
        )
        SELECT r.*
        FROM backtest_stock_results r
        JOIN latest l ON r.run_id = l.run_id
        ORDER BY CAST(r.rank AS INTEGER)
        """,
    )
    _create_view_if_columns(
        conn,
        "v_buy_candidates",
        "backtest_stock_results",
        ["run_id", "stock", "stock_name", "rank", "latest_action", "screen_score"],
        """
        CREATE VIEW v_buy_candidates AS
        SELECT *
        FROM backtest_stock_results
        WHERE latest_action = 'buy'
        ORDER BY run_id, CAST(rank AS INTEGER)
        """,
    )
    _create_view_if_columns(
        conn,
        "v_trade_ledger",
        "backtest_trades",
        ["run_id", "stock", "date", "action", "price", "shares"],
        """
        CREATE VIEW v_trade_ledger AS
        SELECT run_id, stock, date, action, price, shares, position_before, position_after,
               cash_flow, fee, cash_after, equity_after, target_weight, holding_cost_after,
               ledger_version, source_file, row_number
        FROM backtest_trades
        ORDER BY run_id, stock, date, row_number
        """,
    )
    if _has_columns(conn, "backtest_daily_ledger", ["run_id", "stock", "date", "equity", "position_weight"]) and _has_columns(
        conn, "agent_signal_records", ["run_id", "stock", "date", "final_score", "final_action"]
    ):
        conn.execute(
            """
            CREATE VIEW IF NOT EXISTS v_backtest_daily_with_signals AS
            SELECT
                d.run_id,
                d.stock,
                d.date,
                d.open,
                d.high,
                d.low,
                d.close,
                d.equity,
                d.position_weight,
                d.target_weight,
                d.effective_target_weight,
                s.final_score,
                s.final_action,
                s.entry_probability,
                s.exit_probability,
                s.ml_score_signal,
                s.dl_score_signal,
                s.prob_score_signal,
                s.rl_target_weight
            FROM backtest_daily_ledger d
            LEFT JOIN agent_signal_records s
              ON d.run_id = s.run_id AND d.stock = s.stock AND d.date = s.date
            """
        )
    _create_view_if_columns(
        conn,
        "v_model_runs",
        "model_training_runs",
        ["model_name", "kind", "run_id", "model_type", "device", "report_path"],
        """
        CREATE VIEW v_model_runs AS
        SELECT model_name, kind, run_id, model_type, device, report_path, model_path,
               created_at, metrics_json, samples_json, stocks_json, imported_at
        FROM model_training_runs
        ORDER BY kind, created_at DESC, model_name DESC
        """,
    )
    _create_view_if_columns(
        conn,
        "v_run_lineage",
        "runs",
        ["run_id", "task", "source_root", "ml_model", "dl_model", "rl_model"],
        """
        CREATE VIEW v_run_lineage AS
        SELECT run_id, task, source_root, artifact_root, start, end, stocks, success, failed,
               ml_model, dl_model, rl_model, elapsed_seconds, imported_at
        FROM runs
        ORDER BY imported_at DESC, run_id DESC
        """,
    )


def _create_view_if_columns(conn: sqlite3.Connection, view_name: str, table: str, cols: list[str], sql: str) -> None:
    if not _has_columns(conn, table, cols):
        return
    safe_sql = sql.replace("CREATE VIEW ", "CREATE VIEW IF NOT EXISTS ", 1)
    try:
        conn.execute(safe_sql)
    except sqlite3.OperationalError as exc:
        if "already exists" not in str(exc).lower():
            raise


def _has_columns(conn: sqlite3.Connection, table: str, cols: list[str]) -> bool:
    existing = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
    return all(col in existing for col in cols)


def upsert_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        """
        INSERT INTO warehouse_meta (key, value, updated_at)
        VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
        """,
        (key, value),
    )
