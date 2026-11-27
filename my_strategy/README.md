# Personal Quant Strategy Workspace

这个目录是当前项目的个人策略层，不直接修改 `Qbot-main/` 和 `qlib-main/` 源码。

## KHQuant 研究工作台

四页本地工作台、8126 启动参数、个股连续研究图、冻结预测工件契约、空仓平线诊断和验证命令见 [research_workbench/README.md](research_workbench/README.md)。页面加载不自动推理；真实行情、派生结构、冻结预测与回放账本分别保留证据边界。

## 常用命令

运行 Qbot/Qlib 风格策略筛选实验台：

```powershell
python .\my_strategy\strategy_lab\batch_runner.py
```

打开可交互策略筛选 Notebook：

```powershell
jupyter notebook .\my_strategy\strategy_lab\strategy_lab_notebook.ipynb
```

运行 KHQuant-MultiAgent-V1：

```powershell
python .\my_strategy\run_multi_agent_backtest.py
```

训练并启用 AI Agent 机器学习模型：

```powershell
python .\my_strategy\scripts\build_stock_metadata.py
python .\my_strategy\train_ml_model.py --stocks local --start 2021-01-01 --end 2026-07-07 --train-end 2024-12-31 --valid-end 2025-12-31 --model-type lightgbm --model-name lightgbm_cs_rank_5d_local_all --label-mode cross_sectional_rank --base-label future_return_5d --neutral-group auto --no-update
python .\my_strategy\run_multi_agent_backtest.py --stocks 002436.SZ --start 2026-01-01 --end 2026-07-07 --ml-model lightgbm_cs_rank_5d_local_all --no-update
python .\my_strategy\train_deep_model.py --stocks local --start 2021-01-01 --end 2026-07-07 --train-end 2024-12-31 --valid-end 2025-12-31 --model-type lstm --lookback 20 --epochs 12 --batch-size 4096 --hidden-size 64 --model-name dl_lstm_cs_rank_5d_local_all --max-samples 160000 --no-update
python .\my_strategy\run_multi_agent_backtest.py --stocks 002436.SZ --start 2023-01-01 --end 2026-07-07 --ml-model lightgbm_cs_rank_5d_local_all --dl-model dl_lstm_cs_rank_5d_local_all --no-update
```

运行 5 分钟分时回测：

```powershell
python .\my_strategy\run_intraday_backtest.py --stocks 002436.SZ --period 5 --start 2026-07-01 --end 2026-07-07
```

单次运行配置里的综合策略：

```powershell
python .\my_strategy\backtest_runner.py
```

运行参数扫描：

```powershell
python .\my_strategy\backtest_runner.py --scan
```

直接运行主回测脚本：

```powershell
python .\khquant_backtester.py --stock 002436.SZ --start 2022-01-01 --end 2026-07-06 --strategy combined
```

把最新 daily CSV 转成 Qbot 可读信号：

```powershell
python .\my_strategy\qbot_bridge.py --daily .\results\daily\YOUR_DAILY_FILE.csv --stock 002436.SZ
```

检查 Qlib 环境：

```powershell
python .\my_strategy\qlib_runner.py
```

## 目录约定

- `my_strategy/strategy_lab/`: Qbot/Qlib 风格策略筛选实验台
- `my_strategy/agents/`: 多 Agent 决策模块
- `my_strategy/configs/`: 多 Agent 配置
- `my_strategy/outputs/signals/`: 多 Agent 信号、成交、Qbot 桥接文件
- `my_strategy/outputs/reports/`: 多 Agent 绩效报告
- `results/charts/`: 回测图表
- `results/daily/`: 每日净值和持仓
- `results/trades/`: 成交记录
- `results/reports/`: JSON 绩效报告
- `results/scans/`: 参数扫描结果
- `results/qbot_signals/`: 给 Qbot/外部程序读取的信号

## KHQuant-MultiAgent-V1

第一版多 Agent 框架包含：

- `DataAgent`: 拉取并清洗行情数据
- `FactorAgent`: 计算 MA、RSI、MACD、成交量、波动率等因子
- `TrendAgent`: 判断趋势强弱
- `ReversalAgent`: 判断 RSI 超买超卖
- `VolumeAgent`: 判断量能变化
- `MLAgent`: 预留 Qlib 预测分数入口，默认关闭
- `RiskAgent`: 止损、止盈、回撤、波动率风控
- `PortfolioAgent`: 根据综合分数和风控动作决定目标仓位
- `QbotBridgeAgent`: 导出 Qbot 可读信号表

## 机器学习 Agent

第一版机器学习模块位于 `my_strategy/ml/`，先支持传统监督学习模型，不直接上深度学习或强化学习。

- `feature_builder.py`: 日线技术特征、动量、波动、成交量、价格位置
- `label_builder.py`: 未来 5/10/20 日收益和涨跌标签
- `trainers.py`: Ridge、ElasticNet、Logistic、RandomForest、HistGradientBoosting，可选 LightGBM/XGBoost
- `predictor.py`: 加载模型并生成 `ml_score`、`ml_prediction`、`ml_confidence`
- `agents/ml_prediction_agent.py`: 把 ML 分数接入 MultiAgentEngine

模型输出位置：

- `my_strategy/data/processed/ml/models/`
- `my_strategy/data/processed/ml/reports/`

建议先用 `random_forest` 或 `hist_gb` 验证流程；安装 LightGBM/XGBoost 后，再训练更强的横截面选股模型。

## 分时数据和分时回测

分时数据统一走 KHQuant Local Data Center，缓存位置：

- `my_strategy/data/raw/stock_minute/1m/*.parquet`
- `my_strategy/data/raw/stock_minute/5m/*.parquet`
- `my_strategy/data/raw/stock_minute/15m/*.parquet`
- `my_strategy/data/raw/stock_minute/30m/*.parquet`
- `my_strategy/data/raw/stock_minute/60m/*.parquet`

分时回测入口是 `my_strategy/run_intraday_backtest.py`，配置在 `my_strategy/configs/intraday_config.yaml`。第一版 `KHQuant-Intraday-V1` 使用 MA、VWAP、RSI、量比打分；当前 K 线收盘后生成信号，下一根 K 线开盘执行，避免未来函数。

输出位置：

- `my_strategy/data/processed/intraday_backtest/signals/`
- `my_strategy/data/processed/intraday_backtest/reports/`
- `my_strategy/data/processed/intraday_backtest/figures/`

## 调仓和图表标记

多 Agent 回测配置在 `my_strategy/configs/multi_agent_config.yaml`：

- `rebalance_interval_days`: 普通调仓最小间隔，默认 5 个自然日；风控清仓不受限制
- `min_weight_change`: 仓位变化小于该阈值时不交易，默认 0.08
- `mark_full_cycles_only`: 图表只标记完整开仓/完整清仓，默认开启

如果图上的买卖点仍然太密，可以把 `rebalance_interval_days` 调到 `10` 或 `20`，或者把 `min_weight_change` 调到 `0.12` 以上。

## 策略筛选实验台

`strategy_lab` 当前会测试这些候选：

- `ma_cross`
- `macd`
- `rsi_reversal`
- `boll_reversion`
- `kdj_macd`
- `stoch_rsi`
- `adx_macd`
- `aroon`
- `volume_breakout`
- `trend_combo`
- `reversal_combo`
- `balanced_combo`
- `multi_agent_v1`

输出位置：

- `my_strategy/strategy_lab/outputs/reports/strategy_ranking.csv`
- `my_strategy/strategy_lab/outputs/reports/best_strategy_report.json`
- `my_strategy/strategy_lab/outputs/charts/best_strategy_equity.png`
- `my_strategy/strategy_lab/strategy_lab_notebook.ipynb`

Qlib 候选模型会记录在 `best_strategy_report.json` 中；当前环境未安装 `qlib` 时只做占位检测。
