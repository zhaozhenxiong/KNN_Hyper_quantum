"""Research figures reproduce verified daily and prediction values."""

import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench import expert_multiview_figures as figures


def _archive(tmp_path):
    root = tmp_path / "run"
    folder = root / "backtests/oos_1"
    folder.mkdir(parents=True)
    (root / "reports").mkdir()
    (root / "status.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    verification = {"ledger_tables": 1, "matrix_rows": 1, "all_daily_dates_unique": True, "all_trade_counts_match": True}
    (root / "reports/artifact_verification.json").write_text(json.dumps(verification), encoding="utf-8")
    fold = {"period": "oos_1", "evaluation_start": "2026-01-01", "evaluation_end": "2026-01-03", "evaluation_sessions": 3}
    report = {"run": {"run_id": "run"}, "folds": [fold], "interpretation": {"score": "excess return"}}
    (root / "reports/expert_multiview_study.json").write_text(json.dumps(report), encoding="utf-8")
    dates = pd.date_range("2026-01-01", periods=3)
    for index, arm in enumerate(figures.ARMS):
        equity = np.array([100.0, 101.0 + index, 102.0 + index])
        pd.DataFrame({"date": dates, "equity": equity, "market_value": equity * 0.5,
                      "benchmark": [100.0, 99.0, 98.0]}).to_parquet(folder / f"{arm}_double_daily.parquet", index=False)
    for index, view in enumerate(figures.SCORE_VIEWS):
        frame = pd.DataFrame({"date": np.repeat(dates, 2), "stock": [f"{row:06}.SZ" for row in range(6)],
                              "q50": np.linspace(-0.02, 0.02, 6) + index * 0.001})
        frame.to_parquet(folder / f"{view}_predictions.parquet", index=False)
    return root


def test_chart_tables_recompute_nav_drawdown_exposure_and_score_shares(tmp_path):
    root = _archive(tmp_path)
    daily = figures._daily_frame(root, "oos_1")
    assert daily["date"].is_unique
    assert daily["fusion_rank_nav"].tolist() == pytest.approx([1.0, 1.01, 1.02])
    assert daily["fusion_rank_drawdown"].tolist() == pytest.approx([0.0, 0.0, 0.0])
    assert daily["fusion_rank_exposure"].tolist() == pytest.approx([0.5, 0.5, 0.5])
    assert daily["benchmark_nav"].tolist() == pytest.approx([1.0, 0.99, 0.98])
    scores = figures._score_frame(root, "oos_1")
    assert len(scores) == 36
    for view in figures.SCORE_VIEWS:
        assert scores[f"{view}_count"].sum() == 6
        assert scores[f"{view}_density"].sum() == pytest.approx(1.0)


def test_generate_writes_light_dark_figures_and_accessible_tables(tmp_path):
    root = _archive(tmp_path)
    destination = figures.generate(root)
    for mode in figures.PALETTES:
        for suffix in ("png", "svg"):
            assert (destination / f"oos_1_research_overview_{mode}.{suffix}").stat().st_size > 0
    assert (destination / "oos_1_daily_chart_data.csv").is_file()
    assert (destination / "oos_1_score_distribution.csv").is_file()
    assert (destination / "index.html").is_file()
    with pytest.raises(FileExistsError):
        figures.generate(root)
