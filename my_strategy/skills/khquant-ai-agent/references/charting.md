# Charting Rules

Shared chart function:

```text
my_strategy\charting.py
```

Chart requirements:

- Use candlesticks when daily data has `open/high/low/close`.
- A-share color convention: red up, green down.
- Show real-price y-axis, not adjusted price.
- Show holding cost line when present.
- Show strategy NAV on the same price panel, using the right axis.
- Show all buy/sell trade points.
- Mark complete opens with `B`.
- Mark complete clears with `S`.
- Mark add-on buys with an up triangle.
- Mark partial sells/reduces with a down triangle.
- Annotate the latest marked buy/sell price, e.g. `最近买入: 42.25`.
- Include drawdown subplot.
- Include strategy/ML/DL/Prob score plus `entry_probability`/`exit_probability` when those fields exist.
- Include position and position-change subplot.
- Title should contain Chinese stock name and code.

Refresh current screening charts:

```powershell
python .\my_strategy\scripts\refresh_ai_agent_charts.py
```

To redraw charts without touching daily CSVs:

```powershell
python .\my_strategy\scripts\refresh_ai_agent_charts.py --skip-daily-fix
```

Current screening figure directories:

```text
my_strategy\data\processed\ai_agent_screen\figures
my_strategy\data\processed\ai_agent_screen\top_charts
```

Single-stock figure directory:

```text
my_strategy\data\processed\ai_agent_screen\figures
```

If a chart price axis looks wrong:

1. Check duplicate dates in the daily CSV.
2. Check that OHLC exists.
3. Run `refresh_ai_agent_charts.py`.
4. If a single-stock run caused the issue, inspect `DataAgent` deduplication before rerunning.
