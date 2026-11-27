---
name: khquant-ai-agent
description: "Local KHQuant AI-Agent project guide for D:\\quant_strategy. Use when Codex works on this quant strategy workspace: single-stock or batch backtests, AI Agent strategy decisions, ML/DL/RL model training or selection, local A-share data updates, SQLite screening results, candlestick chart refreshes, recommendation top lists, or maintenance cleanup. This skill prevents workflow drift by enforcing the project's fixed paths, local-data-first policy, real-price trading/charting rules, output conventions, and validation checks."
---

# KHQuant AI Agent

Use this skill before touching the local quant project at `D:\quant_strategy`.

## First Moves

1. Work from `D:\quant_strategy`.
2. Prefer existing scripts and project conventions over ad hoc code.
3. Use local data first. Do not switch to temporary online pulls unless the user explicitly asks.
4. Before reporting a backtest or chart result, check the generated `report`, `daily`, `trades`, and figure path.
5. If prices, duplicate dates, or chart axes look wrong, refresh charts/data before drawing conclusions.
6. Keep daily AI Agent outputs in `my_strategy\data\processed\ai_agent_screen`; do not create parallel roots such as `ai_agent_backtest` or `ai_agent_screen_auto`.
7. For buy/sell point optimization, prefer `scripts\optimize_fusion_from_existing_results.py` on existing daily outputs before retraining models.

## Reference Map

Read only the references needed for the task:

- Project paths and ownership: `references/project-structure.md`
- Data source and price policy: `references/data-policy.md`
- Single-stock, batch, model, and recommendation workflows: `references/workflows.md`
- Chart requirements and refresh commands: `references/charting.md`
- SQLite/model/file cleanup rules: `references/maintenance.md`

## Standard Validation

For backtests:

- Confirm the stock code and Chinese name when possible.
- Confirm the date range.
- Verify daily output has no duplicated dates.
- Verify a figure exists and show it when useful.
- Summarize latest action, latest position, return, drawdown, Sharpe, latest trade, latest trade price, and active ML/DL/Prob/RL signals.

For code edits:

- Keep changes inside `my_strategy` unless the user asks otherwise.
- Compile changed Python files with `python -m compileall`.
- Do not delete generated data without a dry-run preview or an explicit cleanup command.
