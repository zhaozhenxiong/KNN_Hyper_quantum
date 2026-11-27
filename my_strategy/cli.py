#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Unified KHQuant command line entrypoint.

The CLI keeps local Python and Docker workflows aligned. It delegates to the
existing project scripts so behavior stays consistent with the established
entrypoints.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


SCRIPT_COMMANDS = {
    "update-data": ["my_strategy/scripts/update_daily_data.py"],
    "train-models": ["my_strategy/train_and_select_models.py"],
    "train-backtest-fast": ["my_strategy/scripts/train_backtest_fast.py"],
    "train-ml": ["my_strategy/train_ml_model.py"],
    "train-dl": ["my_strategy/train_deep_model.py"],
    "train-rl": ["my_strategy/train_rl_agent.py"],
    "screen-fast": ["my_strategy/scripts/fast_market_screen.py"],
    "factor-snapshots": ["my_strategy/scripts/build_factor_snapshots.py"],
    "backtest": ["my_strategy/run_multi_agent_backtest.py"],
    "backtest-batch": ["my_strategy/scripts/batch_multi_agent_backtest.py"],
    "backtest-fast": ["my_strategy/scripts/batch_multi_agent_backtest.py", "--fast", "--workers", "auto"],
    "verify-db": ["my_strategy/scripts/migrate_db_first_storage.py", "--verify-db-first", "--cleanup-dry-run"],
    "model-db-sync": ["my_strategy/scripts/migrate_db_first_storage.py", "--migrate-model-artifacts"],
    "maintenance": ["my_strategy/maintenance.py"],
    "package": ["my_strategy/scripts/package_project.py"],
    "benchmark": ["my_strategy/scripts/benchmark_runtime.py"],
    "benchmark-train-data": ["my_strategy/scripts/benchmark_training_dataset.py"],
    "implementation-run": ["my_strategy/scripts/implementation_run.py"],
    "parity-diagnostics": ["my_strategy/scripts/parity_diagnostics.py"],
    "research-workbench": ["-m", "my_strategy.research_workbench.cli", "serve"],
    "research-run": ["-m", "my_strategy.research_workbench.cli", "run"],
    "latent-quant": ["-m", "my_strategy.latent_quant.runner"],
    "latent-quant-overseas": ["-m", "my_strategy.latent_quant.overseas_runner"],
    "latent-quant-forward": ["-m", "my_strategy.latent_quant.forward"],
    "latent-quant-forward-prepare": ["-m", "my_strategy.latent_quant.forward_prepare"],
    "latent-quant-forward-accept": ["-m", "my_strategy.latent_quant.forward_accept"],
    "doctor": ["my_strategy/scripts/docker_runtime_doctor.py"],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KHQuant unified CLI")
    parser.add_argument("command", choices=sorted(SCRIPT_COMMANDS))
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to the selected command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    command = [sys.executable, *SCRIPT_COMMANDS[ns.command], *ns.args]
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
