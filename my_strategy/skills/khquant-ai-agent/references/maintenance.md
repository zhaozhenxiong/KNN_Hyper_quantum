# Maintenance

Main command:

```powershell
python .\my_strategy\maintenance.py --delete-files --delete-models --vacuum
```

This is a dry-run by default. It should print a preview.

Execute only after confirmation or explicit user request:

```powershell
python .\my_strategy\maintenance.py --execute --delete-files --delete-models --vacuum
```

Backtest integration:

```text
run_multi_agent_backtest.py
```

Post-run maintenance is available, but use `--no-auto-maintenance` for quick single-stock checks unless cleanup is requested.

Rules:

- Keep recent summary runs for ranking comparisons.
- Keep only latest detail runs for heavy daily/trade/signal tables.
- Protect current best ML/DL/RL models.
- Protect RL experience for the current best RL source run.
- Delete only after dry-run preview unless the command includes `--execute`.
- Use `--maintenance-no-confirm` only for unattended jobs.

Important database:

```text
my_strategy\data\processed\ai_agent_screen\ai_agent_screen.db
```

Legacy duplicate output roots should be cleanup candidates, not active targets:

```text
my_strategy\data\processed\ai_agent_backtest
my_strategy\data\processed\ai_agent_screen_auto
my_strategy\data\processed\ai_agent_screen_100b
my_strategy\data\processed\strategy_screen
```

Maintenance logs:

```text
my_strategy\data\processed\maintenance_logs
```

If SQLite or model outputs grow unexpectedly:

1. Run dry-run maintenance.
2. Check candidate row/file/model counts.
3. Execute cleanup only if the preview matches intent.
4. Re-run dry-run to confirm no remaining candidates.
