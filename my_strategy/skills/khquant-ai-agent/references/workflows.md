# Workflows

## Single-Stock Backtest

Use this for requests like "回测兴森科技", "看看洁美科技", or "跑一下 002436.SZ".

1. Confirm code from local data when needed:

```powershell
Get-ChildItem .\my_strategy\data\raw\stock_daily -Filter "002859*"
```

2. Run the standard command:

```powershell
python .\my_strategy\run_multi_agent_backtest.py --stocks 002859.SZ --start 2026-01-01 --end 2026-07-09 --no-update --no-auto-maintenance
```

Use `--no-auto-maintenance` for quick ad hoc single-stock checks so the response is not interrupted by cleanup prompts.

3. Read outputs:

```text
my_strategy\data\processed\ai_agent_screen\reports\<TAG>_multi_agent_report.json
my_strategy\data\processed\ai_agent_screen\signals\<TAG>_multi_agent_daily.csv
my_strategy\data\processed\ai_agent_screen\signals\<TAG>_multi_agent_trades.csv
my_strategy\data\processed\ai_agent_screen\figures\<TAG>_<NAME>_<START>_<END>_multi_agent.png
```

4. Validate:

- Daily duplicate dates are zero.
- Latest row date matches requested end or last available trading day.
- Figure exists.
- Recent trade annotation appears on chart when there is a trade.

5. Report:

- latest action
- latest position
- total return
- annual return
- max drawdown
- Sharpe
- trade count
- win rate
- final equity
- latest trade date/action/price
- image path rendered with Markdown

## Batch Screening

Use this for large stock pools and top recommendations:

```powershell
python .\my_strategy\run_multi_agent_backtest.py --stocks local --start 2026-01-01 --end 2026-07-09 --no-update
```

For large pools, `run_multi_agent_backtest.py` may train/select models first unless disabled. Use the project's automatic best-model logic rather than manually choosing old models unless the user requests it.

Primary reports:

```text
my_strategy\data\processed\ai_agent_screen\reports\ranking.csv
my_strategy\data\processed\ai_agent_screen\reports\buy_candidates.csv
my_strategy\data\processed\ai_agent_screen\reports\recommend_buy_top.csv
my_strategy\data\processed\ai_agent_screen\reports\focus_watchlist_backtest.csv
```

Top chart directory:

```text
my_strategy\data\processed\ai_agent_screen\top_charts
```

## Model Use

Default behavior:

- Automatically refresh/load best ML/DL models from `model_registry/best_models.json`.
- Automatically load best RL model from `rl/best_rl_model.json`.
- Run `ProbabilisticAgent` from `my_strategy\ml\probabilistic.py` when `agents.probabilistic.enabled=true`.
- Treat Bayesian probability and Markov state as ML/probability-layer risk context, not as deep learning.
- Keep RL `allow_entry=false` behavior unless the user asks RL to open positions independently.

Best model registries:

```text
my_strategy\data\processed\model_registry\best_models.json
my_strategy\data\processed\rl\best_rl_model.json
my_strategy\data\processed\model_registry\best_fusion_policy.json
```

When training many stocks, compare with the historical best model and keep the best. Do not pile up models indefinitely; use maintenance cleanup.

Fusion policy:

- `FusionDecisionPolicy` combines Agent scores with activation functions such as `sigmoid`, `tanh`, or `softsign`.
- `run_multi_agent_backtest.py` automatically loads `best_fusion_policy.json` when it exists.
- When the user asks to optimize buy/sell points without retraining, use existing daily outputs first:

```powershell
python .\my_strategy\scripts\optimize_fusion_from_existing_results.py --trials 24 --max-stocks 300
```

This script reads existing `ai_agent_screen\daily\*_daily.csv` files and saves the best fusion policy without retraining ML/DL models.

Probabilistic daily fields:

```text
prob_score, prob_score_signal, prob_confidence, bayes_prob_up, bayes_score,
markov_state, markov_next_state, markov_state_prob, markov_score
```

## Recommendation Top List

When asked for recommended buys, inspect:

```text
my_strategy\data\processed\ai_agent_screen\reports\recommend_buy_top.csv
my_strategy\data\processed\ai_agent_screen\reports\buy_candidates.csv
```

Prefer candidates with:

- `latest_action` in `buy`, `hold`, or actionable watchlist context
- positive or high `screen_score`
- acceptable drawdown
- recent full-cycle trade context
- nonzero ML/DL/RL support when available

Always state that this is a backtest/screening result, not investment advice.
