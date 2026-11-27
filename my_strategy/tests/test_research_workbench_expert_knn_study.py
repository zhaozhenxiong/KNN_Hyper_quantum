from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd

from my_strategy.latent_quant.execution import StudyCostModel
from my_strategy.latent_quant.runner import COSTS, PORTFOLIO
from my_strategy.latent_quant.signal_experts import ExpertForecast
from my_strategy.research_workbench import expert_knn_study as knn
from my_strategy.research_workbench.runner import Window


def test_fold_uses_calibration_output_as_evaluation_rows(tmp_path, monkeypatch):
    dates = pd.to_datetime([
        "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04",
        "2025-01-06", "2025-01-07", "2025-01-09", "2025-01-10",
    ])
    frame = pd.DataFrame({
        "date": dates, "stock": "000001.SZ", "signal_eligible": True,
        "label_end": dates + pd.Timedelta(days=1), "target": np.linspace(-0.02, 0.02, len(dates)),
        "trade_date": dates + pd.Timedelta(days=1),
    })
    window = Window("oos_1", dates[0], dates[3], dates[5], pd.DatetimeIndex(dates[-2:]))
    quantiles = np.array([
        [-0.02, 0.00, 0.02], [-0.01, 0.01, 0.03], [0.00, 0.02, 0.04],
    ])

    def forecast(train, rows, *, names):
        assert names == (knn.ARM,)
        assert len(rows) == 3
        return {knn.ARM: ExpertForecast(
            quantiles, pd.Series(np.zeros(len(rows)), index=rows.index),
            pd.Series(np.ones(len(rows)), index=rows.index), {"causal": True},
        )}

    def execute(data, prediction, sessions, context, folder, arm, policy, cost_model):
        assert len(prediction) == len(sessions) == 2
        assert prediction.index.equals(frame.index[-2:])
        execution = {
            cost: {"total_return": 0.0, "verification": {"cash": True, "equity": True, "next_session": True}}
            for cost in COSTS
        }
        return execution, {cost: pd.Series(dtype=float) for cost in COSTS}

    class Context:
        run_id = "shape-test"

        def subdir(self, *parts):
            path = tmp_path.joinpath(*parts)
            path.mkdir(parents=True, exist_ok=True)
            return path

    monkeypatch.setattr(knn, "forecast_experts", forecast)
    monkeypatch.setattr(knn.research, "_execution", execute)
    policy = replace(PORTFOLIO, selection_mode="cross_sectional_rank", min_score=None)
    fold, _ = knn._fold(
        SimpleNamespace(frame=frame), window, Context(), policy=policy, cost_model=StudyCostModel(),
    )
    assert fold["evaluation_sessions"] == 2
    saved = pd.read_parquet(tmp_path / "backtests/oos_1" / f"{knn.ARM}_predictions.parquet")
    assert len(saved) == 2
    assert saved[["date", "stock"]].reset_index(drop=True).equals(
        frame.loc[frame.index[-2:], ["date", "stock"]].reset_index(drop=True),
    )


def test_artifact_verification_counts_pretrade_rejections_in_order_ledger(tmp_path):
    day = pd.Timestamp("2025-01-02")
    fold = {
        "period": "oos_1", "evaluation_sessions": 1,
        "arms": {knn.ARM: {"execution": {}}},
    }
    for cost in COSTS:
        fold["arms"][knn.ARM]["execution"][cost] = {
            "order_count": 1, "pretrade_rejected_count": 1, "trade_count": 1,
            "verification": {"cash": True, "equity": True, "next_session": True},
        }
        folder = tmp_path / "backtests/oos_1"
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"date": [day]}).to_parquet(folder / f"{knn.ARM}_{cost}_daily.parquet", index=False)
        pd.DataFrame({"date": [day, day]}).to_parquet(folder / f"{knn.ARM}_{cost}_orders.parquet", index=False)
        pd.DataFrame({"date": [day]}).to_parquet(folder / f"{knn.ARM}_{cost}_trades.parquet", index=False)
        pd.DataFrame({"date": [day]}).to_parquet(folder / f"{knn.ARM}_{cost}_positions.parquet", index=False)
        pd.DataFrame({"date": [day]}).to_parquet(folder / f"{knn.ARM}_{cost}_selection_daily.parquet", index=False)
    result = knn._verify_outputs(tmp_path, [fold], pd.DataFrame([{"row": 1}, {"row": 2}]))
    assert result["all_order_counts_match"] is True
    assert result["all_ledger_checks_pass"] is True
