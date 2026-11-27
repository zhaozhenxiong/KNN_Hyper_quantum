# Project Structure

Workspace root:

```text
D:\quant_strategy
```

Important project paths:

```text
D:\quant_strategy\my_strategy
D:\quant_strategy\my_strategy\configs\multi_agent_config.yaml
D:\quant_strategy\my_strategy\data\raw\stock_daily
D:\quant_strategy\my_strategy\data\processed\ai_agent_screen
D:\quant_strategy\my_strategy\data\processed\model_registry
D:\quant_strategy\my_strategy\data\processed\ml
D:\quant_strategy\my_strategy\data\processed\deep_learning
D:\quant_strategy\my_strategy\data\processed\rl
```

Output meaning:

- `ai_agent_screen`: the only daily AI Agent output area. It owns single-stock backtests, batch rankings, buy candidates, top charts, focus figures, and the main SQLite database.
- `data/raw/stock_daily`: local daily parquet files. Treat as the primary data source.
- `model_registry/best_models.json`: current best ML and DL model registry.
- `rl/best_rl_model.json`: current best RL policy registry.

Core scripts:

```text
my_strategy\run_multi_agent_backtest.py
my_strategy\scripts\batch_multi_agent_backtest.py
my_strategy\scripts\refresh_ai_agent_charts.py
my_strategy\maintenance.py
my_strategy\train_and_select_models.py
my_strategy\train_ml_model.py
my_strategy\train_deep_model.py
my_strategy\train_rl_agent.py
```

Main data modules:

```text
my_strategy\agents\data_agent.py
my_strategy\data_manager\
my_strategy\storage\screen_sqlite.py
my_strategy\storage\cleanup_sqlite.py
my_strategy\storage\cleanup_files.py
```

Main strategy/model modules:

```text
my_strategy\agents\multi_agent_engine.py
my_strategy\agents\fusion_decision.py
my_strategy\agents\probabilistic_agent.py
my_strategy\agents\rl_policy_agent.py
my_strategy\ml\
my_strategy\ml\probabilistic.py
my_strategy\deep_learning\
my_strategy\rl\
my_strategy\model_selector.py
my_strategy\fusion_registry.py
```

Main chart module:

```text
my_strategy\charting.py
```
