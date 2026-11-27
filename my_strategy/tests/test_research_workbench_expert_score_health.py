import json

import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench import expert_score_health as health


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_target_metadata_is_recomputed_instead_of_ignored():
    rows = pd.DataFrame({"target": [0.01, -0.02]})
    observed = health._verified_target(rows, {})
    assert observed["rows"] == 2
    assert health._verified_target(rows, {"target": observed}) == observed
    with pytest.raises(ValueError, match="目标分布"):
        health._verified_target(rows, {"target": observed | {"positive_rate": 1.0}})


def test_distribution_is_finite_and_uses_strict_positive_boundary():
    result = health._distribution(pd.Series([0.0, 0.01, -0.01, np.nan, np.inf]))
    assert result["rows"] == 3
    assert result["missing"] == 2
    assert result["unique"] == 3
    assert result["positive_rate"] == 1 / 3
    assert result["median"] == 0
    assert health._distribution([])["rows"] == 0


def test_single_tree_median_is_flagged_only_with_real_score_collapse():
    collapsed = health._learned_median_health(1, pd.Series([0.01] * 100 + [-0.01] * 100))
    assert collapsed["early_stopped_median"]
    assert collapsed["collapsed"]
    informative = health._learned_median_health(8, pd.Series([0.01] * 100 + [-0.01] * 100))
    assert not informative["early_stopped_median"]
    assert not informative["collapsed"]


def test_archive_health_recovers_raw_scores_and_flags_weighted_collapsed_median(tmp_path):
    fold = "oos_3"
    run_id = "synthetic-expert"
    report = {
        "run": {"run_id": run_id},
        "portfolio": {"min_score": 0.0},
        "folds": [{
            "period": fold,
            "models": {"arms": {"domestic_gpu": {"winner": {"tree_counts": [40, 1, 60]}}}},
        }],
    }
    offsets = [0.01, -0.02, 0.03]
    _write(tmp_path / "models" / fold / "fusion.json", {
        "winner": {"weights": {"domestic_gpu": 0.5, "supertrend_atr": 0.5}},
        "calibration": {"fusion": {
            "offsets": offsets, "raw_crossing_rate": 0, "crossing_rate_after_offsets": 0,
        }},
    })
    dates = pd.date_range("2026-04-01", periods=4).repeat(3)
    raw = np.array([-0.01, 0.0, 0.01] * 4)
    (tmp_path / "backtests" / fold).mkdir(parents=True)
    pd.DataFrame({"date": dates, "q50": raw + offsets[1]}).to_parquet(
        tmp_path / "backtests" / fold / "fusion_predictions.parquet", index=False,
    )
    pd.DataFrame({"date": dates, "q50": raw}).to_parquet(
        tmp_path / "backtests" / fold / "domestic_gpu_predictions.parquet", index=False,
    )
    result = health.inspect_archive(tmp_path, report)
    assert result["source_run_id"] == run_id
    assert result["status"] == "attention"
    evidence = result["folds"][0]
    assert evidence["raw_q50"]["rows"] == 12
    assert evidence["raw_q50"]["positive_rate"] == 1 / 3
    assert evidence["calibrated_q50"]["positive_rate"] == 0
    assert evidence["raw_qualified_rows"] == 4
    assert evidence["qualified_rows"] == 0
    assert evidence["contributing_models"][0]["q50_tree_count"] == 1
    assert evidence["contributing_models"][0]["collapsed"]
    assert evidence["entry"]["comparison"] == "strictly_greater"
    assert result["registry_mutated"] is False


def test_archive_health_does_not_invert_quantiles_after_crossing(tmp_path):
    fold = "oos_1"
    _write(tmp_path / "models" / fold / "fusion.json", {
        "winner": {"weights": {"supertrend_atr": 1.0}},
        "calibration": {"fusion": {
            "offsets": [0.0, -0.02, 0.0], "raw_crossing_rate": 0.1, "crossing_rate_after_offsets": 0.2,
        }},
    })
    (tmp_path / "backtests" / fold).mkdir(parents=True)
    pd.DataFrame({"date": pd.to_datetime(["2025-01-02"]), "q50": [-0.01]}).to_parquet(
        tmp_path / "backtests" / fold / "fusion_predictions.parquet", index=False,
    )
    report = {"run": {"run_id": "crossed"}, "portfolio": {"min_score": 0.0}, "folds": [{"period": fold}]}
    evidence = health.inspect_archive(tmp_path, report)["folds"][0]
    assert evidence["raw_q50"] is None
    assert "原生推理" in evidence["raw_recovery"]


def test_pure_knn_health_uses_rank_semantics_and_frozen_scores(tmp_path):
    fold = "oos_1"
    values = pd.DataFrame({
        "date": pd.to_datetime(["2025-10-09", "2025-10-10", "2025-10-13"]), "q50": [-0.01, 0.01, 0.02],
    })
    (tmp_path / "backtests" / fold).mkdir(parents=True)
    values.to_parquet(tmp_path / "backtests" / fold / "online_knn_volume_trend_predictions.parquet", index=False)
    report = {
        "run": {"run_id": "pure-knn"}, "training_engine": "online_knn_volume_trend",
        "folds": [{
            "period": fold,
            "arms": {"online_knn_volume_trend": {
                "calibration": {"offsets": [0.0, 0.005, 0.0]},
                "score_evidence": {
                    "raw_evaluation_q50": health._distribution([-0.015, 0.005, 0.015]),
                    "calibrated_evaluation_q50": health._distribution(values["q50"]),
                    "qualified_rows": 3,
                    "entry": {"selection_mode": "cross_sectional_rank", "min_score": None,
                              "comparison": "finite_cross_sectional_rank"},
                },
            }},
        }],
    }
    result = health.inspect_archive(tmp_path, report)
    evidence = result["folds"][0]
    assert result["status"] == "ok"
    assert evidence["entry"]["min_score"] is None
    assert evidence["qualified_rows"] == 3
    assert evidence["raw_qualified_rows"] is None
    assert evidence["calibration_offset_q50"] == 0.005
    assert "横截面排名" in result["summary"]


def test_archive_health_reads_verified_knn_distributions_without_fixed_offsets(tmp_path):
    fold = "oos_2"
    raw_distribution = {"rows": 3, "median": -0.01, "positive_rate": 1 / 3}
    report = {
        "run": {"run_id": "knn-expert"}, "portfolio": {"min_score": 0.0},
        "folds": [{
            "period": fold,
            "fusion": {"method": "online_knn_conditional_residual_quantiles", "source_fusion_weights": {"domestic_gpu": 1.0}},
            "arms": {"fusion": {"score_evidence": {
                "raw_evaluation_q50": raw_distribution, "raw_qualified_rows": 1,
            }}},
        }],
    }
    (tmp_path / "backtests" / fold).mkdir(parents=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2025-10-09", "2025-10-10", "2025-10-13"]), "q50": [-0.01, 0.01, 0.02],
    }).to_parquet(tmp_path / "backtests" / fold / "fusion_predictions.parquet", index=False)
    evidence = health.inspect_archive(tmp_path, report)["folds"][0]
    assert evidence["raw_q50"] == raw_distribution
    assert evidence["raw_qualified_rows"] == 1
    assert evidence["qualified_rows"] == 2
    assert evidence["calibration_offset_q50"] is None
    assert "条件 KNN" in evidence["raw_recovery"]


def test_hybrid_health_reports_native_scores_weights_fallback_and_cuda(tmp_path):
    fold = "oos_1"
    table = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
        "knn_q50": [0.01, 0.02, 0.015, np.nan],
        "knn_raw_q50": [0.005, 0.015, 0.01, np.nan],
        "multimodal_q50": [100.0, 200.0, np.nan, 300.0],
        "knn_rank": [0.5, 1.0, 1.0, np.nan],
        "multimodal_rank": [0.5, 1.0, np.nan, 1.0],
        "hybrid_score": [0.5, 1.0, 0.8, np.nan],
        "effective_knn_weight": [0.6, 0.6, 1.0, 0.0],
        "effective_multimodal_weight": [0.4, 0.4, 0.0, 0.0],
        "fallback_reason": ["", "", "multimodal_missing_or_invalid", "knn_missing_or_invalid"],
    })
    (tmp_path / "backtests" / fold).mkdir(parents=True)
    table.to_parquet(tmp_path / "backtests" / fold / "knn60_multimodal40_predictions.parquet", index=False)
    coverage = {
        "evaluation_rows": 4, "full_60_40_rows": 2, "knn_fallback_rows": 1, "invalid_knn_rows": 1,
        "fallback_reasons": {"multimodal_missing_or_invalid": 1, "knn_missing_or_invalid": 1},
        "hybrid_score": health._distribution(table["hybrid_score"]),
        "knn_rank": health._distribution(table["knn_rank"]),
        "multimodal_rank": health._distribution(table["multimodal_rank"]),
        "entry": {"selection_mode": "cross_sectional_rank", "min_score": None},
    }
    report = {
        "run": {"run_id": "hybrid-health"}, "training_engine": "knn60_multimodal40",
        "folds": [{
            "period": fold,
            "arms": {
                "online_knn_volume_trend": {
                    "calibration": {"offsets": [0.0, 0.005, 0.0]},
                    "score_evidence": {
                        "raw_evaluation_q50": health._distribution(table["knn_raw_q50"]),
                        "calibrated_evaluation_q50": health._distribution(table["knn_q50"]),
                    },
                },
                "knn60_multimodal40": {"score_evidence": coverage},
            },
            "hybrid": {
                "knn_backend": {"requested_device": "cuda", "actual_device": "cuda:1", "backend": "torch_cuda_float64"},
                "fusion_weights": {"domestic_gpu": 0.7, "sector_gpu": 0.3},
                "multimodal_validation_health": {"passed": True},
            },
        }],
    }
    result = health.inspect_archive(tmp_path, report)
    evidence = result["folds"][0]
    assert result["status"] == "attention" and result["mode"] == "knn60_multimodal40"
    assert evidence["coverage"]["full_60_40_rows"] == 2
    assert evidence["coverage"]["knn_fallback_rows"] == 1
    assert evidence["knn_backend"]["actual_device"] == "cuda:1"
    assert evidence["multimodal_q50"]["median"] == 200.0
    assert evidence["qualified_rows"] == 3
    table.loc[0, "effective_multimodal_weight"] = 0.0
    table.to_parquet(tmp_path / "backtests" / fold / "knn60_multimodal40_predictions.parquet", index=False)
    with pytest.raises(ValueError, match="覆盖率"):
        health.inspect_archive(tmp_path, report)
