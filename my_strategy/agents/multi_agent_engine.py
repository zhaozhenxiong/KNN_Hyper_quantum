#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Coordinator for all quant decision agents."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from .deep_learning_agent import DeepLearningAgent
from .fusion_decision import FusionDecisionPolicy
from .portfolio_agent import PortfolioAgent
from .probabilistic_agent import ProbabilisticAgent
from .risk_agent import RiskAgent
from .rl_policy_agent import RLPolicyAgent
from .ml_prediction_agent import MLPredictionAgent
from .ml_score_agent import MLScoreAgent
from .strategy_agents import MLAgent, ReversalAgent, TrendAgent, VolumeAgent


class MultiAgentEngine:
    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        agents_cfg = self.config.get("agents", {})
        decision_cfg = self.config.get("decision", {})
        risk_cfg = self.config.get("risk", {})
        portfolio_cfg = self.config.get("portfolio", {})

        trend_cfg = agents_cfg.get("trend", {})
        reversal_cfg = agents_cfg.get("reversal", {})
        volume_cfg = agents_cfg.get("volume", {})
        ml_cfg = agents_cfg.get("ml", {})
        ml_score_cfg = agents_cfg.get("ml_score", {})
        dl_cfg = agents_cfg.get("deep_learning", {})
        prob_cfg = agents_cfg.get("probabilistic", {})
        rl_cfg = agents_cfg.get("reinforcement_learning", {})

        self.strategy_agents = [
            TrendAgent(
                weight=trend_cfg.get("weight", 0.45),
                short_ma=trend_cfg.get("short_ma", "ma20"),
                long_ma=trend_cfg.get("long_ma", "ma60"),
                enabled=trend_cfg.get("enabled", True),
            ),
            ReversalAgent(
                weight=reversal_cfg.get("weight", 0.30),
                rsi_low=reversal_cfg.get("rsi_low", 35),
                rsi_high=reversal_cfg.get("rsi_high", 75),
                enabled=reversal_cfg.get("enabled", True),
            ),
            VolumeAgent(
                weight=volume_cfg.get("weight", 0.25),
                volume_ratio_buy=volume_cfg.get("volume_ratio_buy", 1.3),
                enabled=volume_cfg.get("enabled", True),
            ),
            MLScoreAgent(
                weight=ml_score_cfg.get("weight", ml_cfg.get("weight", 0.25)),
                score_column=ml_score_cfg.get("score_column", ml_cfg.get("score_column", "ml_score")),
                prediction_column=ml_score_cfg.get("prediction_column", ml_cfg.get("prediction_column", "pred_rank_future_return_5d")),
                confidence_column=ml_score_cfg.get("confidence_column", ml_cfg.get("confidence_column", "ml_confidence")),
                buy_threshold=ml_score_cfg.get("buy_threshold", 0.35),
                sell_threshold=ml_score_cfg.get("sell_threshold", -0.25),
                neutral_zone=ml_score_cfg.get("neutral_zone", 0.10),
                max_score=ml_score_cfg.get("max_score", 1.0),
                enabled=ml_score_cfg.get("enabled", ml_cfg.get("enabled", False)),
            ),
            DeepLearningAgent(
                weight=dl_cfg.get("weight", 0.20),
                score_column=dl_cfg.get("score_column", "dl_score"),
                prediction_column=dl_cfg.get("prediction_column", "dl_prediction"),
                confidence_column=dl_cfg.get("confidence_column", "dl_confidence"),
                buy_threshold=dl_cfg.get("buy_threshold", 0.35),
                sell_threshold=dl_cfg.get("sell_threshold", -0.25),
                neutral_zone=dl_cfg.get("neutral_zone", 0.10),
                max_score=dl_cfg.get("max_score", 1.0),
                enabled=dl_cfg.get("enabled", False),
            ),
            ProbabilisticAgent(
                weight=prob_cfg.get("weight", 0.15),
                score_column=prob_cfg.get("score_column", "prob_score"),
                confidence_column=prob_cfg.get("confidence_column", "prob_confidence"),
                buy_threshold=prob_cfg.get("buy_threshold", 0.30),
                sell_threshold=prob_cfg.get("sell_threshold", -0.25),
                neutral_zone=prob_cfg.get("neutral_zone", 0.08),
                max_score=prob_cfg.get("max_score", 1.0),
                enabled=prob_cfg.get("enabled", False),
            ),
            MLAgent(
                weight=ml_cfg.get("weight", 0.20),
                prediction_column=ml_cfg.get("prediction_column", "pred_return_5d"),
                enabled=False,
            ),
        ]
        self.diagnostic_agents = [
            MLPredictionAgent(
                weight=0.0,
                score_column=ml_cfg.get("score_column", "ml_score"),
                prediction_column=ml_cfg.get("prediction_column", "pred_rank_future_return_5d"),
                confidence_column=ml_cfg.get("confidence_column", "ml_confidence"),
                enabled=ml_cfg.get("enabled", False),
            )
        ]
        self.rl_agent = RLPolicyAgent(
            model_name=rl_cfg.get("model_name", ""),
            model_root=rl_cfg.get("model_root"),
            weight=rl_cfg.get("weight", 0.10),
            blend=rl_cfg.get("blend", 0.15),
            min_confidence=rl_cfg.get("min_confidence", 0.05),
            allow_entry=rl_cfg.get("allow_entry", False),
            enabled=rl_cfg.get("enabled", False),
        )

        self.risk_agent = RiskAgent(
            stop_loss=risk_cfg.get("stop_loss", 0.08),
            take_profit=risk_cfg.get("take_profit", 0.20),
            max_drawdown=risk_cfg.get("max_drawdown", 0.15),
            max_volatility=risk_cfg.get("max_volatility", 0.08),
        )
        self.buy_threshold = decision_cfg.get("buy_threshold", 0.55)
        self.hold_threshold = decision_cfg.get("hold_threshold", 0.15)
        self.sell_threshold = decision_cfg.get("sell_threshold", -0.25)
        self.portfolio_agent = PortfolioAgent(
            min_entry_weight=portfolio_cfg.get("min_entry_weight", 0.30),
            mid_weight=portfolio_cfg.get("mid_weight", 0.50),
            high_weight=portfolio_cfg.get("high_weight", 0.70),
            max_weight=portfolio_cfg.get("max_weight", risk_cfg.get("max_single_position", 0.90)),
            reduce_weight=portfolio_cfg.get("reduce_weight", 0.30),
            hold_threshold=self.hold_threshold,
            buy_threshold=self.buy_threshold,
            high_threshold=portfolio_cfg.get("high_threshold", 0.75),
        )
        fusion_cfg = dict(decision_cfg.get("fusion", {}))
        fusion_cfg.setdefault(
            "weights",
            {
                "trend": trend_cfg.get("weight", 0.45),
                "reversal": reversal_cfg.get("weight", 0.30),
                "volume": volume_cfg.get("weight", 0.25),
                "ml": ml_score_cfg.get("weight", ml_cfg.get("weight", 0.25)),
                "dl": dl_cfg.get("weight", 0.20),
                "prob": prob_cfg.get("weight", 0.15),
            },
        )
        self.fusion_policy = FusionDecisionPolicy(fusion_cfg, portfolio_cfg={**risk_cfg, **portfolio_cfg})

    def decide_one_day(self, row, context: dict[str, Any] | None = None) -> dict[str, Any]:
        context = context or {}
        agent_results = []
        weighted_score = 0.0
        total_weight = 0.0

        for agent in self.strategy_agents:
            if not agent.enabled:
                continue
            signal = agent.run(row, context)
            agent_results.append(signal)
            weighted_score += signal.score * agent.weight
            total_weight += agent.weight
        for agent in self.diagnostic_agents:
            if not agent.enabled:
                continue
            agent_results.append(agent.run(row, context))

        fallback_score = weighted_score / total_weight if total_weight else 0.0
        risk_signal = self.risk_agent.run(row, context)

        if risk_signal.action == "force_sell":
            fallback_action = "sell"
        elif risk_signal.action == "reduce":
            fallback_action = "reduce"
        elif risk_signal.action == "take_profit":
            fallback_action = "sell"
        elif fallback_score >= self.buy_threshold:
            fallback_action = "buy"
        elif fallback_score <= self.sell_threshold:
            fallback_action = "sell"
        elif fallback_score >= self.hold_threshold:
            fallback_action = "hold"
        else:
            fallback_action = "watch"

        fallback_target_weight = self.portfolio_agent.calc_target_weight(fallback_score, risk_signal.action)
        fusion = self.fusion_policy.decide(
            agent_results=agent_results,
            row=row,
            risk_signal=risk_signal,
            fallback_score=fallback_score,
            fallback_action=fallback_action,
            fallback_weight=fallback_target_weight,
        )
        final_score = fusion.final_score
        final_action = fusion.final_action
        base_target_weight = fusion.base_target_weight
        target_weight = base_target_weight
        rl_signal = None
        if self.rl_agent.enabled:
            rl_context = dict(context)
            rl_context["base_final_score"] = final_score
            rl_context["base_target_weight"] = base_target_weight
            rl_signal = self.rl_agent.run(row, rl_context)
            agent_results.append(rl_signal)
            rl_weight = float(rl_signal.extra.get("rl_target_weight", base_target_weight))
            rl_confidence = float(rl_signal.extra.get("rl_confidence", 0.0))
            can_apply_rl = rl_confidence >= self.rl_agent.min_confidence
            if not self.rl_agent.allow_entry and base_target_weight <= 0:
                can_apply_rl = False
            if final_action == "sell" or risk_signal.action in {"force_sell", "take_profit"}:
                can_apply_rl = False
            if risk_signal.action == "reduce":
                rl_weight = min(rl_weight, self.portfolio_agent.reduce_weight)
            if can_apply_rl:
                blend = self.rl_agent.blend
                target_weight = (1.0 - blend) * base_target_weight + blend * rl_weight
        reason = {
            signal.agent_name: {
                "score": round(signal.score, 4),
                "confidence": round(signal.confidence, 4),
                "action": signal.action,
                "reason": signal.reason,
            }
            for signal in agent_results
        }
        reason["RiskAgent"] = {
            "score": round(risk_signal.score, 4),
            "confidence": round(risk_signal.confidence, 4),
            "action": risk_signal.action,
            "reason": risk_signal.reason,
        }
        reason["FusionDecision"] = {
            "score": round(fusion.final_score, 4),
            "confidence": round(fusion.confidence_score, 4),
            "action": fusion.final_action,
            "reason": fusion.reason,
        }
        ml_score_signal = next((signal for signal in agent_results if signal.agent_name == "MLScoreAgent"), None)
        ml_prediction_signal = next((signal for signal in agent_results if signal.agent_name == "MLPredictionAgent"), None)
        dl_signal = next((signal for signal in agent_results if signal.agent_name == "DeepLearningAgent"), None)
        prob_signal = next((signal for signal in agent_results if signal.agent_name == "ProbabilisticAgent"), None)

        result = {
            "date": row["date"],
            "close": float(row["close"]),
            "final_score": round(final_score, 4),
            "final_action": final_action,
            "target_weight": round(float(target_weight), 4),
            "base_target_weight": round(float(base_target_weight), 4),
            "fallback_score": round(float(fallback_score), 6),
            "fallback_action": fallback_action,
            "agent_weighted_score": round(float(fusion.raw_score), 6),
            "agent_confidence_score": round(float(fusion.confidence_score), 6),
            "fusion_score": round(float(fusion.final_score), 6),
            "fusion_action": fusion.final_action,
            "fusion_target_weight": round(float(fusion.target_weight), 6),
            "entry_probability": round(float(fusion.entry_probability), 6),
            "exit_probability": round(float(fusion.exit_probability), 6),
            "fusion_activation": fusion.activation,
            "agent_score_breakdown": json.dumps(fusion.breakdown, ensure_ascii=False),
            "risk_action": risk_signal.action,
            "reason": reason,
            "reason_json": json.dumps(reason, ensure_ascii=False),
        }
        ml_signal = ml_score_signal or ml_prediction_signal
        if ml_signal is not None:
            result["ml_score"] = round(float(ml_signal.extra.get("ml_score", ml_signal.score)), 6)
            result["ml_score_signal"] = round(float(ml_signal.extra.get("ml_score_signal", ml_signal.score)), 6)
            result["ml_confidence"] = round(float(ml_signal.extra.get("ml_confidence", ml_signal.confidence)), 6)
            result["ml_prediction"] = round(float(ml_signal.extra.get("ml_prediction", 0)), 8)
            result["ml_model_name"] = ml_signal.extra.get("ml_model_name", "")
            result["ml_prediction_kind"] = ml_signal.extra.get("ml_prediction_kind", "")
            result["ml_score_action"] = ml_signal.extra.get("ml_score_action", ml_signal.action)
            result["ml_score_reason"] = ml_signal.extra.get("ml_score_reason", ml_signal.reason)
        if dl_signal is not None:
            result["dl_score"] = round(float(dl_signal.extra.get("dl_score", dl_signal.score)), 6)
            result["dl_score_signal"] = round(float(dl_signal.extra.get("dl_score_signal", dl_signal.score)), 6)
            result["dl_confidence"] = round(float(dl_signal.extra.get("dl_confidence", dl_signal.confidence)), 6)
            result["dl_prediction"] = round(float(dl_signal.extra.get("dl_prediction", 0)), 8)
            result["dl_model_name"] = dl_signal.extra.get("dl_model_name", "")
            result["dl_model_type"] = dl_signal.extra.get("dl_model_type", "")
            result["dl_prediction_kind"] = dl_signal.extra.get("dl_prediction_kind", "")
            result["dl_action"] = dl_signal.extra.get("dl_action", dl_signal.action)
            result["dl_reason"] = dl_signal.extra.get("dl_reason", dl_signal.reason)
        if prob_signal is not None:
            result["prob_score"] = round(float(prob_signal.extra.get("prob_score", prob_signal.score)), 6)
            result["prob_score_signal"] = round(float(prob_signal.extra.get("prob_score_signal", prob_signal.score)), 6)
            result["prob_confidence"] = round(float(prob_signal.extra.get("prob_confidence", prob_signal.confidence)), 6)
            result["prob_action"] = prob_signal.extra.get("prob_action", prob_signal.action)
            result["prob_model_name"] = prob_signal.extra.get("prob_model_name", "")
            result["prob_reason"] = prob_signal.extra.get("prob_reason", prob_signal.reason)
            result["bayes_prob_up"] = round(float(prob_signal.extra.get("bayes_prob_up", 0.5)), 6)
            result["bayes_score"] = round(float(prob_signal.extra.get("bayes_score", 0)), 6)
            result["markov_state"] = prob_signal.extra.get("markov_state", "")
            result["markov_next_state"] = prob_signal.extra.get("markov_next_state", "")
            result["markov_state_prob"] = round(float(prob_signal.extra.get("markov_state_prob", 0)), 6)
            result["markov_score"] = round(float(prob_signal.extra.get("markov_score", 0)), 6)
        if rl_signal is not None:
            result["rl_model_name"] = rl_signal.extra.get("rl_model_name", "")
            result["rl_target_weight"] = round(float(rl_signal.extra.get("rl_target_weight", base_target_weight)), 6)
            result["rl_confidence"] = round(float(rl_signal.extra.get("rl_confidence", 0.0)), 6)
            result["rl_action"] = rl_signal.extra.get("rl_action", rl_signal.action)
            result["rl_q_values"] = rl_signal.extra.get("rl_q_values", "{}")
            result["rl_reason"] = rl_signal.extra.get("rl_reason", rl_signal.reason)
        return result

    def generate_decisions(self, df: pd.DataFrame) -> pd.DataFrame:
        decisions = []
        context = {
            "current_position": 0,
            "current_position_weight": 0.0,
            "entry_price": None,
            "current_drawdown": 0,
        }
        portfolio_value = 1.0
        peak_value = 1.0

        for _, row in df.iterrows():
            if pd.isna(row.get("ma60")):
                continue
            result = self.decide_one_day(row, context)

            if context["current_position"] > 0:
                portfolio_value *= 1 + float(row.get("return_1d", 0) or 0)
            peak_value = max(peak_value, portfolio_value)
            current_drawdown = portfolio_value / peak_value - 1
            context["current_drawdown"] = current_drawdown

            if result["final_action"] == "buy" and context["current_position"] == 0:
                context["current_position"] = 1
                context["entry_price"] = float(row["close"])
            elif result["final_action"] == "sell":
                context["current_position"] = 0
                context["current_position_weight"] = 0.0
                context["entry_price"] = None
            else:
                context["current_position_weight"] = float(result.get("target_weight", context.get("current_position_weight", 0.0)) or 0.0)
            if result["final_action"] != "sell":
                context["current_position_weight"] = float(result.get("target_weight", context.get("current_position_weight", 0.0)) or 0.0)

            result["paper_portfolio_value"] = round(portfolio_value, 6)
            result["paper_drawdown"] = round(current_drawdown, 6)
            decisions.append(result)

        return pd.DataFrame(decisions)
