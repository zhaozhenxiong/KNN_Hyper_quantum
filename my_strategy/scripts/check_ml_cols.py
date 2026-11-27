"""检查ML列是否在decision中"""
import sys; sys.path.insert(0, "D:/quant_strategy")
import pandas as pd, yaml
from my_strategy.agents.data_agent import DataAgent
from my_strategy.agents.factor_agent import FactorAgent
from my_strategy.agents.multi_agent_engine import MultiAgentEngine
from my_strategy.ml.predictor import enrich_with_ml_predictions

config = yaml.safe_load(open("my_strategy/configs/multi_agent_config.yaml", encoding="utf-8"))
config["agents"]["ml"]["enabled"] = True
config["agents"]["ml"]["model_name"] = "lightgbm_cs_rank_5d_local_all"

da = DataAgent("local_data_center", auto_update=False)
raw = da.load("002635.SZ", "2022-01-01", "2026-07-08")
factors = FactorAgent.add_factors(raw, volume_window=20)
print("Before ML:", [c for c in factors.columns if "ml" in c.lower()], flush=True)

factors = enrich_with_ml_predictions(factors, "lightgbm_cs_rank_5d_local_all", None)
print("After ML:", [c for c in factors.columns if "ml" in c.lower()], flush=True)

engine = MultiAgentEngine(config)
decisions = engine.generate_decisions(factors)
print("Decisions cols:", list(decisions.columns), flush=True)
ml_in_dec = [c for c in decisions.columns if "ml" in c.lower()]
print("ML in decisions:", ml_in_dec, flush=True)
if ml_in_dec:
    print("Latest ML:", decisions[ml_in_dec].iloc[-3:].to_dict("records"), flush=True)
