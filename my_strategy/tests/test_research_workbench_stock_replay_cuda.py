import numpy as np
import pandas as pd
import pytest

from my_strategy.research_workbench.stock_replay_cuda import predict_frozen_on_cuda


@pytest.mark.parametrize("nan_mode", ["Min", "Max"])
def test_cuda_frozen_quantiles_match_native_cpu_at_split_borders_and_missing_values(nan_mode):
    torch = pytest.importorskip("torch")
    catboost = pytest.importorskip("catboost")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for frozen tree parity")

    rng = np.random.default_rng(413)
    train = pd.DataFrame(rng.standard_normal((256, 3)), columns=["return", "volatility", "breadth"])
    target = train["return"] * 0.5 + train["volatility"] * 0.3
    train.iloc[::11] = np.nan
    model = catboost.CatBoostRegressor(
        loss_function="MultiQuantile:alpha=0.1,0.5,0.9", iterations=12, depth=3,
        random_seed=42, nan_mode=nan_mode, allow_writing_files=False, verbose=False, thread_count=2,
    ).fit(train, target)
    test = train.copy()
    for index, borders in model.get_borders().items():
        for row, border in enumerate(borders):
            value = np.float32(border)
            test.iloc[3 * row:3 * row + 3, index] = [
                np.nextafter(value, np.float32(-np.inf)), value, np.nextafter(value, np.float32(np.inf)),
            ]
        test.iloc[-1, index] = np.nan

    gpu = predict_frozen_on_cuda(model, test)
    native = model.predict(test, task_type="CPU")
    np.testing.assert_allclose(gpu, native, atol=1e-10, rtol=0)
