# Data Policy

## Local First

Use local data center paths and project scripts first. Avoid temporary online fetches for normal backtests.

Standard single-stock command includes `--no-update` when the user says current local data is already updated:

```powershell
python .\my_strategy\run_multi_agent_backtest.py --stocks 002436.SZ --start 2026-01-01 --end 2026-07-09 --no-update
```

Drop `--no-update` only when the user asks to update local data first.

For same-day realtime quote refresh, use the canonical updater instead of temporary scripts:

```powershell
python .\my_strategy\scripts\update_daily_data.py --today-realtime --end-date 2026-07-10
```

This updater writes into `data/raw/stock_daily` only after a valid same-day quote is returned, so failed online reads should not pollute the local cache.

## Price Policy

The long-term rule is:

- Use real/unadjusted trade prices for trading simulation and chart price axes.
- Use adjusted/factor prices only for indicators when needed.
- Chart `open/high/low/close` must be real price OHLC.

Local parquet files may contain duplicate dates: one row can carry adjusted prices and another can carry real `trade_*` prices.

When duplicate dates exist:

1. Prefer rows with positive `trade_close`.
2. Use `trade_open/trade_high/trade_low/trade_close` for chart/trade OHLC when present.
3. Fall back to `open/high/low/close`.
4. Ensure one row per date.

`DataAgent` is expected to deduplicate this way. If results look wrong, inspect:

```text
my_strategy\agents\data_agent.py
```

## Daily CSV Health Checks

Every generated daily file should have:

```text
date, open, high, low, close, factor_close, equity, position_weight
```

Run this when chart data looks wrong:

```powershell
python .\my_strategy\scripts\refresh_ai_agent_charts.py
```

Expected healthy final report:

- `missing_local = 0`
- `errors = 0`
- `suspicious_close_gap = 0`
- duplicate dates = 0

Reports:

```text
my_strategy\data\processed\ai_agent_screen\reports\chart_refresh_report.json
my_strategy\data\processed\ai_agent_screen\reports\daily_price_check.csv
```
