import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench.models import fit_predict, load_predict, sample_training_rows, sequence_ready


def _panel() -> tuple[pd.DataFrame, dict]:
    dates = pd.bdate_range("2024-01-02", periods=32)
    generator = np.random.default_rng(42)
    rows = []
    for stock in range(4):
        for step, date in enumerate(dates):
            momentum = generator.normal()
            rows.append({
                "stock": str(stock), "date": date,
                "target": momentum / 100 if step < 30 else np.nan,
                "label_end": dates[step + 2] if step < 30 else pd.NaT,
                "board": "main" if stock < 2 else "growth",
                "industry": "hardware" if step < 12 else "new-sector",
                "concepts": ("exports",) if step < 12 else ("unseen-concept",),
                "momentum": momentum, "volatility": generator.uniform(0.1, 1.0),
            })
    frame = pd.DataFrame(rows).sample(frac=1, random_state=13)
    masks = {
        "train_mask": frame["date"].le(dates[11]),
        "valid_mask": frame["date"].between(dates[14], dates[20]),
        "predict_mask": frame["date"].ge(dates[24]),
    }
    return frame, masks


def test_lightgbm_candidate_uses_train_vocabulary_and_round_trips(tmp_path):
    pytest.importorskip("lightgbm")
    frame, masks = _panel()
    result = fit_predict(
        frame, **masks, feature_columns=["momentum", "volatility"], model="lightgbm",
        static_columns=["board", "industry", "concepts"],
        params={"n_estimators": 8, "min_child_samples": 2, "n_jobs": 1, "task": "classification"},
        bundle_path=tmp_path / "tree",
    )
    assert result.metadata["status"] == "candidate"
    assert result.metadata["prediction_kind"] == "probability_positive"
    assert result.metadata["vocabularies"] == {"board": ["growth", "main"], "industry": ["hardware"], "concepts": ["exports"]}
    assert result.predictions.index.equals(frame.index[masks["predict_mask"]])
    assert result.predictions.between(0.0, 1.0).all()
    assert result.predictions.loc[frame["target"].isna()].notna().all()
    replay = load_predict(frame, bundle_path=tmp_path / "tree", predict_mask=masks["predict_mask"])
    pd.testing.assert_series_equal(result.predictions, replay.predictions)


def test_lstm_unseen_categories_and_future_rows_do_not_change_earlier_predictions(tmp_path):
    torch = pytest.importorskip("torch")
    frame, masks = _panel()
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = fit_predict(
            frame, **masks, feature_columns=["momentum", "volatility"], model="lstm", device="cpu",
            static_columns=["board", "industry", "concepts"], bundle_path=tmp_path / "lstm",
            params={"lookback": 4, "hidden_size": 6, "epochs": 2, "batch_size": 16},
        )
        future = frame.copy()
        final_day = future["date"].eq(future["date"].max())
        future.loc[final_day, "momentum"] += 99.0
        future.loc[masks["predict_mask"], "industry"] = "another-unseen"
        future.loc[masks["predict_mask"], "concepts"] = pd.Series(
            [("another-unseen-concept",)] * int(masks["predict_mask"].sum()), index=frame.index[masks["predict_mask"]],
        )
        replay = load_predict(future, bundle_path=tmp_path / "lstm", predict_mask=masks["predict_mask"])
        early = ~final_day.reindex(result.predictions.index)
        pd.testing.assert_series_equal(result.predictions[early], replay.predictions[early])
        assert result.predictions.notna().all()
        assert not np.allclose(result.predictions[~early], replay.predictions[~early])
        expected_center = frame.loc[masks["train_mask"], ["momentum", "volatility"]].median().to_numpy()
        np.testing.assert_allclose(result.metadata["feature_center"], expected_center)
    finally:
        torch.set_num_threads(prior_threads)


def test_model_rejects_label_lookahead_before_training():
    frame, masks = _panel()
    frame.loc[masks["train_mask"], "label_end"] = frame.loc[masks["valid_mask"], "date"].min()
    with pytest.raises(ValueError, match="training labels must end"):
        fit_predict(frame, **masks, feature_columns=["momentum"], model="lightgbm")


def test_sequence_sampling_balances_stocks_and_ignores_input_row_order():
    frame, masks = _panel()
    ready = sequence_ready(frame, 4)
    for _, group in frame.assign(ready=ready).groupby("stock"):
        assert group.sort_values("date")["ready"].tolist() == [False] * 3 + [True] * 29
    rows = np.flatnonzero(ready & masks["train_mask"])
    sampled = sample_training_rows(frame, rows, 15, seed=42)
    assert len(sampled) == 15
    assert sorted(frame.iloc[sampled]["stock"].value_counts()) == [3, 4, 4, 4]
    changed = frame.iloc[::-1]
    reversed_candidates = (ready & masks["train_mask"]).reindex(changed.index)
    reordered = sample_training_rows(changed, np.flatnonzero(reversed_candidates), 15, seed=42)
    assert set(frame.index[sampled]) == set(changed.index[reordered])


def test_seqlstm_bundle_caps_training_and_reuses_batch_preprocessing(tmp_path):
    torch = pytest.importorskip("torch")
    frame, masks = _panel()
    budget = 13
    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = fit_predict(
            frame, **masks, feature_columns=["momentum", "volatility"], model="seqlstm", device="cpu",
            static_columns=["board", "industry"], bundle_path=tmp_path / "seqlstm",
            params={"lookback": 4, "hidden_size": 6, "epochs": 1, "max_train_samples": budget},
        )
        assert model.metadata["model"] == "seqlstm"
        assert model.metadata["train_sequences"] == budget
        assert model.metadata["vocabularies"]["industry"] == ["hardware"]
        restored = load_predict(frame, bundle_path=tmp_path / "seqlstm", predict_mask=masks["predict_mask"])
        pd.testing.assert_series_equal(model.predictions, restored.predictions)
    finally:
        torch.set_num_threads(prior_threads)
