#!/usr/bin/env python
"""用开盘价跑多Agent回测"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from my_strategy.run_multi_agent_backtest import (
    load_config, DataAgent, FactorAgent, MultiAgentEngine,
    simulate_trades, save_outputs, ensure_output_dirs, run_stock as original_run_stock
)

# 修补run_stock：用开盘价代替收盘价
_original_simulate = simulate_trades

def patched_run_stock(stock, config):
    """用开盘价运行的版本"""
    data_cfg = config["data"]
    auto_update = bool(data_cfg.get("auto_update", True))
    start = pd.to_datetime(data_cfg["start"])
    end = pd.to_datetime(data_cfg["end"])
    warmup_days = int(data_cfg.get("warmup_days", 180))
    load_start = (start - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    
    data_agent = DataAgent(data_cfg.get("source", "local_data_center"), auto_update=auto_update)
    raw = data_agent.load(stock, load_start, data_cfg["end"])
    volume_window = config.get("agents", {}).get("volume", {}).get("volume_window", 20)
    factors = FactorAgent.add_factors(raw, volume_window=volume_window)
    
    # 🟢 用开盘价替换收盘价
    factors["close_orig"] = factors["close"]
    factors["close"] = factors["open"]
    if "trade_close" in factors.columns:
        factors["trade_close"] = factors["trade_open"] if "trade_open" in factors.columns else factors["open"]
    
    engine = MultiAgentEngine(config)
    decisions = engine.generate_decisions(factors)
    
    period_mask = (pd.to_datetime(factors["date"]) >= start) & (pd.to_datetime(factors["date"]) <= end)
    factors = factors.loc[period_mask].reset_index(drop=True)
    decisions = decisions.loc[(pd.to_datetime(decisions["date"]) >= start) & (pd.to_datetime(decisions["date"]) <= end)].reset_index(drop=True)
    
    if factors.empty or decisions.empty:
        raise RuntimeError(f"{stock} 在 {data_cfg['start']} ~ {data_cfg['end']} 没有足够的因子数据")
    
    daily, trades = simulate_trades(factors, decisions, config)
    paths = save_outputs(stock, decisions, daily, trades, config)
    
    report = __import__('json').loads(paths["report"].read_text(encoding="utf-8"))
    metrics = report["metrics"]
    latest = report["latest_decision"]
    print(f"\n{'=' * 72}")
    print(f"{config.get('strategy_name')} / {stock} (开盘价)")
    print(f"总收益: {metrics['total_return']:.2%} | 年化: {metrics['annual_return']:.2%} | 最大回撤: {metrics['max_drawdown']:.2%} | 夏普: {metrics['sharpe_ratio']:.2f}")
    print(f"交易次数: {metrics['n_trades']} | 胜率: {metrics['win_rate']:.1%}")
    print(f"最新决策: {latest['date']} {latest['final_action']} score={latest['final_score']} weight={latest['target_weight']}")
    return paths

if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="")
    parser.add_argument("--start", default="")
    parser.add_argument("--end", default="")
    parser.add_argument("--config", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "my_strategy", "configs", "multi_agent_config.yaml"))
    args = parser.parse_args()
    
    config = load_config(__import__('pathlib').Path(args.config))
    if args.stocks:
        config.setdefault("data", {})["stocks"] = [s.strip() for s in args.stocks.split(",") if s.strip()]
    if args.start:
        config.setdefault("data", {})["start"] = args.start
    if args.end:
        config.setdefault("data", {})["end"] = args.end
    
    for stock in config["data"]["stocks"]:
        try:
            patched_run_stock(stock, config)
        except Exception as e:
            print(f"❌ {stock} 出错: {e}")
