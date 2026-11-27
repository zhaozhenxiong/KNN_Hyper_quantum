#!/usr/bin/env python
"""Tests for strategy-level acceptance checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from my_strategy.data_manager.config import load_data_config
from my_strategy.validation.strategy_acceptance import audit_backtest_report, run_leakage_acceptance


class StrategyAcceptanceTests(unittest.TestCase):
    def test_docker_runtime_uses_gpu_capable_torch_package(self) -> None:
        requirements = Path("my_strategy/requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("https://download.pytorch.org/whl/cpu", requirements)
        self.assertIn("https://download.pytorch.org/whl/cu130", requirements)
        self.assertIn('torch==2.13.0+cu130; sys_platform == "linux"', requirements)
        self.assertIn('torch==2.13.0; sys_platform == "darwin"', requirements)
        self.assertNotIn("+cpu", requirements)

    def test_lstm_auto_device_prefers_cuda_then_mps(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this Python environment")

        from my_strategy.deep_learning.models import select_torch_device

        with patch.object(torch.cuda, "is_available", return_value=True):
            self.assertEqual(select_torch_device(torch), "cuda")

        with patch.object(torch.cuda, "is_available", return_value=False):
            with patch.object(torch.backends.mps, "is_available", return_value=True):
                self.assertEqual(select_torch_device(torch), "mps")
                self.assertEqual(select_torch_device(torch, "cuda"), "mps")
                self.assertEqual(select_torch_device(torch, "mps"), "mps")

    def test_lstm_predict_falls_back_to_cpu_when_saved_device_is_cuda(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed in this Python environment")

        from my_strategy.deep_learning.models import TorchLSTMRegressor

        x = pd.DataFrame(
            {
                "f1_lag1": [0.1, 0.2, 0.3, 0.4],
                "f2_lag1": [0.2, 0.3, 0.4, 0.5],
                "f1_lag2": [0.3, 0.4, 0.5, 0.6],
                "f2_lag2": [0.4, 0.5, 0.6, 0.7],
            }
        ).to_numpy(dtype="float32", copy=True)
        y = pd.Series([0.01, 0.02, -0.01, 0.03]).to_numpy(dtype="float32", copy=True)
        model = TorchLSTMRegressor(lookback=2, hidden_size=4, epochs=1, batch_size=2, device="cpu").fit(x, y)
        model.device = "cuda"
        with patch.object(torch.cuda, "is_available", return_value=False):
            pred = model.predict(x[:1])
        self.assertEqual(len(pred), 1)

    def test_data_root_can_be_overridden_for_docker(self) -> None:
        with patch.dict("os.environ", {"KHQUANT_DATA_ROOT": "/app/my_strategy/data"}):
            config = load_data_config()
        self.assertEqual(config["data_root"], "/app/my_strategy/data")
        self.assertEqual(config["stock_pool"]["custom_pool_path"], "/app/my_strategy/data/raw/custom_pool/my_pool.csv")

    def test_leakage_acceptance_passes_project_features(self) -> None:
        report = run_leakage_acceptance(
            run_id="unit-leakage",
            feature_columns=["ma_5", "rsi_14", "volume_ratio"],
            train_end="2025-12-31",
            valid_end="2026-06-30",
            data_end="2026-07-15",
        )
        self.assertEqual(report.status, "success")
        self.assertEqual(report.checks["feature_lookahead"], "pass")

    def test_leakage_acceptance_blocks_future_named_features(self) -> None:
        report = run_leakage_acceptance(
            run_id="unit-leakage-bad",
            feature_columns=["ma_5", "next_close"],
            train_end="2025-12-31",
            valid_end="2026-06-30",
            data_end="2026-07-15",
        )
        self.assertEqual(report.status, "failed")
        self.assertEqual(report.checks["feature_lookahead"], "fail")

    def test_backtest_artifact_audit_passes_valid_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            daily_path = root / "daily.csv"
            trades_path = root / "trades.csv"
            figure_path = root / "figure.png"
            report_path = root / "report.json"

            pd.DataFrame(
                {
                    "date": ["2026-01-01", "2026-01-02"],
                    "open": [10.0, 10.5],
                    "high": [11.0, 11.0],
                    "low": [9.5, 10.0],
                    "close": [10.5, 10.8],
                    "factor_close": [10.5, 10.8],
                    "equity": [100000.0, 100500.0],
                    "position_weight": [0.0, 0.2],
                }
            ).to_csv(daily_path, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                {
                    "date": ["2026-01-02"],
                    "symbol": ["002436.SZ"],
                    "side": ["BUY"],
                    "shares": [100],
                    "price": [10.8],
                }
            ).to_csv(trades_path, index=False, encoding="utf-8-sig")
            figure_path.write_bytes(b"png")
            report_path.write_text(
                json.dumps(
                    {
                        "strategy_name": "unit",
                        "stock": "002436.SZ",
                        "stock_name": "unit",
                        "metrics": {
                            "total_return": 0.005,
                            "annual_return": 0.1,
                            "max_drawdown": 0.0,
                            "sharpe_ratio": 1.0,
                            "n_trades": 1,
                            "win_rate": 0.0,
                        },
                        "latest_decision": {"date": "2026-01-02", "final_action": "buy"},
                        "files": {"daily": str(daily_path), "trades": str(trades_path), "figure": str(figure_path)},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            report = audit_backtest_report(report_path, run_id="unit-full", require_figures=True, require_ledger=True)
            self.assertEqual(report.status, "success", report.to_dict())
            self.assertEqual(report.checks["daily_contract"], "pass")
            self.assertEqual(report.checks["broker_ledger"], "pass")

    def test_backtest_artifact_audit_fails_missing_daily(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report_path = root / "report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "strategy_name": "unit",
                        "stock": "002436.SZ",
                        "stock_name": "unit",
                        "metrics": {
                            "total_return": 0.0,
                            "annual_return": 0.0,
                            "max_drawdown": 0.0,
                            "sharpe_ratio": 0.0,
                            "n_trades": 0,
                            "win_rate": 0.0,
                        },
                        "latest_decision": {"date": "2026-01-02"},
                        "files": {"daily": str(root / "missing.csv")},
                    }
                ),
                encoding="utf-8",
            )
            report = audit_backtest_report(report_path, run_id="unit-full", require_figures=False, require_ledger=False)
            self.assertEqual(report.status, "failed")
            self.assertEqual(report.checks["daily_contract"], "fail")


if __name__ == "__main__":
    unittest.main()
