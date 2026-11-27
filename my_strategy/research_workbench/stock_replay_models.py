"""Apply archived Latent-Quant models and policies to one stock without refitting."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from my_strategy.latent_quant import models
from my_strategy.latent_quant.execution import Strategy, StudyCostModel
from my_strategy.latent_quant.runner import COSTS, INITIAL_CASH
from my_strategy.research_workbench.runner import Window


GPU_TOLERANCE = 1e-7
_RULES = {"historical_quantiles", "supertrend_atr"}
_DOMESTIC_ARMS = {"domestic_fixed", "domestic_tuned"}
_OVERSEAS_ARMS = {"domestic", "overseas_raw", "overseas_pca"}


def _artifact(root: Path, *parts: str) -> Path:
    path = root.joinpath(*parts).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"缺少可信冻结文件 {'/'.join(parts)}")
    return path


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"冻结文件 {path.name} 读取失败") from error


def _policy(run_folder: Path, profile: str, fold: dict) -> tuple[dict[str, Strategy], dict]:
    expected = {"baseline": set(models.ARMS), "domestic": _DOMESTIC_ARMS, "overseas": _OVERSEAS_ARMS}
    if set(fold["arms"]) != expected.get(profile):
        raise ValueError("该运行缺少预期的冻结策略记录")
    config = _json(_artifact(run_folder, "config", "study.json"))
    expense = config.get("expense_model", config.get("cost_model"))
    if expense != asdict(StudyCostModel()) or config.get("initial_cash") != INITIAL_CASH or config.get("cost_multipliers") != COSTS:
        raise ValueError("此归档的资金或成交费用与当前回放引擎不同；可继续查看已有账本。")
    try:
        policies = {
            arm: Strategy(**(settings["strategy"] if profile == "domestic" else config["portfolio"]))
            for arm, settings in fold["arms"].items()
        }
    except (KeyError, TypeError) as error:
        raise ValueError("该折的冻结仓位参数不完整") from error
    return policies, config


def _rule_predictions(run_folder: Path, window: Window, frame: pd.DataFrame, evidence: dict) -> dict[str, pd.DataFrame]:
    columns = ["date", "stock", "signal_eligible", "target", "label_end", "st_direction", "atr_pct"]
    reference = pd.read_parquet(
        _artifact(run_folder, "data_snapshots", "panel.parquet"), columns=columns,
        filters=[("date", ">=", window.train_start), ("date", "<", window.evaluate[0])],
    )
    train_mask, _, calibration_mask = window.masks(reference)
    accepted = reference["signal_eligible"].eq(True) & np.isfinite(reference["target"]) & reference["label_end"].gt(reference["date"])
    train, calibration = reference.loc[train_mask & accepted], reference.loc[calibration_mask & accepted]
    if train.empty or calibration.empty:
        raise ValueError("冻结面板缺少可复算的规则训练或校准数据")
    observations = pd.concat([calibration[["stock", "st_direction", "atr_pct"]], frame[["stock", "st_direction", "atr_pct"]]])
    pooled = train["target"].quantile(list(models.QUANTILE_LEVELS)).to_numpy()
    historical, _ = models._historical_quantiles(train, observations, pooled)
    trend, _ = models._trend_quantiles(train, observations, pooled)
    output = {}
    for arm, raw in (("historical_quantiles", historical), ("supertrend_atr", trend)):
        values, rebuilt = models._calibrate(raw, calibration["target"].to_numpy(dtype=float))
        offsets = evidence[arm]["calibration"]["offsets"]
        if not np.allclose(rebuilt["offsets"], offsets, atol=1e-12, rtol=0):
            raise ValueError(f"{arm} 复算校准参数与冻结记录不同")
        output[arm] = models._prediction_table(values, frame.index)
    output["supertrend_atr"]["score"] = models._trend_score(frame)
    return output


def _join_daily(frame: pd.DataFrame, path: Path, features: list[str]) -> pd.DataFrame:
    daily = pd.read_parquet(path, columns=["date", *features]).set_index("date")
    if not daily.index.is_unique or not frame["date"].isin(daily.index).all():
        raise ValueError(f"{path.name} 缺少该折应有的同日冻结输入")
    return frame.drop(columns=features, errors="ignore").join(daily, on="date", validate="many_to_one")


def _overseas_inputs(run_folder: Path, fold: str, frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    state = _json(_artifact(run_folder, "models", fold, "pca.json"))
    overseas = _join_daily(frame, _artifact(run_folder, "data_snapshots", "overseas.parquet"), config["overseas_features"])
    return _join_daily(overseas, _artifact(run_folder, "models", fold, "pca_by_date.parquet"), state["output_features"])


def _catboost_prediction(
    run_folder: Path, fold: str, arm: str, frame: pd.DataFrame, evidence: dict, devices: dict, cutoffs: set[float],
) -> tuple[pd.DataFrame, str]:
    from catboost import CatBoostError, CatBoostRegressor

    model_file = f"{arm}.cbm"
    if evidence.get("model_file") != model_file:
        raise ValueError(f"{arm} 缺少原生冻结模型记录")
    model_path = _artifact(run_folder, "models", fold, model_file)
    metadata = _json(_artifact(run_folder, "models", fold, f"{arm}.json"))
    features = metadata["features"]
    if features != evidence["features"] or metadata["calibration"] != evidence["calibration"]:
        raise ValueError(f"{arm} 冻结模型侧车与折内报告不一致")
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"{arm} 缺少冻结模型输入 {', '.join(sorted(missing))}")
    inputs = models._matrix(frame, features)
    offsets = np.asarray(metadata["calibration"]["offsets"])

    def calibrated(raw: np.ndarray) -> np.ndarray:
        return np.sort(np.sort(raw, axis=1) + offsets, axis=1)

    def cpu_predictions() -> np.ndarray:
        try:
            return calibrated(model.predict(inputs, thread_count=4, task_type="CPU"))
        except CatBoostError as error:
            raise ValueError(f"{arm} 无法读取冻结模型的输出：{error}") from error

    model = CatBoostRegressor()
    try:
        model.load_model(str(model_path))
    except CatBoostError as error:
        raise ValueError(f"{arm} 无法读取原生冻结模型：{error}") from error
    chosen = None
    device = "CPU"
    if devices["fallback_reason"] is None:
        accelerated = None
        candidate_device = "GPU"
        if devices["native_gpu_limit"] is None:
            try:
                accelerated = model.predict(inputs, thread_count=4, task_type="GPU")
            except CatBoostError as error:
                reason = str(error).strip()
                if "Model is not one-dimensional" in reason:
                    devices["native_gpu_limit"] = reason
                else:
                    devices["fallback_reason"] = reason
        if devices["native_gpu_limit"] is not None:
            try:
                from .stock_replay_cuda import FrozenCudaUnavailable, predict_frozen_on_cuda
            except ImportError as error:
                devices["fallback_reason"] = str(error).strip()
            else:
                try:
                    accelerated = predict_frozen_on_cuda(model, inputs)
                    candidate_device = "PyTorch CUDA"
                except FrozenCudaUnavailable as error:
                    devices["fallback_reason"] = str(error).strip()
        if accelerated is not None and np.shape(accelerated) != (len(frame), 3):
            devices["fallback_reason"] = f"{arm} GPU 输出维度与冻结三分位数不一致"
        elif accelerated is not None:
            gpu = calibrated(accelerated)
            cpu = cpu_predictions()
            difference = float(np.max(np.abs(gpu - cpu)))
            devices["max_absolute_differences"][arm] = difference if np.isfinite(difference) else None
            matches = np.isfinite(difference) and difference <= GPU_TOLERANCE and all(
                np.array_equal(gpu[:, 1] > cutoff, cpu[:, 1] > cutoff) for cutoff in cutoffs
            )
            if matches:
                chosen, device = gpu, candidate_device
            else:
                chosen = cpu
                devices["fallback_reason"] = f"{arm} 未通过 GPU/CPU 精度或下单方向一致性校验"
    if chosen is None:
        chosen = cpu_predictions()
    if not np.isfinite(chosen).all():
        raise ValueError(f"{arm} 冻结模型返回非有限分位数")
    return models._prediction_table(chosen, frame.index), device


def replay(
    run_folder: Path, profile: str, report: dict, fold: str, frame: pd.DataFrame, sessions: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, Strategy], dict]:
    """Predict only requested eligible sessions, with frozen per-fold models and policies."""
    item = next((entry for entry in report["folds"] if entry["period"] == fold), None)
    if item is None or Path(fold).name != fold:
        raise ValueError("没有可用的冻结评估窗口")
    policies, config = _policy(run_folder, profile, item)
    begin = pd.Timestamp(item["evaluation_start"])
    finish = pd.Timestamp(item["evaluation_end"])
    if not frame.index.is_unique or frame["stock"].nunique() > 1 or frame.duplicated(["stock", "date"]).any():
        raise ValueError("新股输入需使用唯一股票与日期记录")
    rows = frame.loc[frame["date"].isin(sessions) & frame["date"].between(begin, finish) & frame["signal_eligible"].eq(True)]
    evidence = item["models"]["arms"]
    devices = {"requested_device": "GPU", "effective_devices": {}, "native_gpu_limit": None, "fallback_reason": None,
               "max_absolute_differences": {}, "absolute_tolerance": GPU_TOLERANCE}
    if rows.empty:
        devices["effective_devices"] = {arm: "not_run" for arm in policies}
        return {arm: models._prediction_table(np.empty((0, 3)), rows.index) for arm in policies}, policies, devices
    predictions = {}
    if profile == "baseline":
        window = Window(fold, pd.Timestamp(item["train_start"]), pd.Timestamp(item["valid_start"]),
                        pd.Timestamp(item["calibration_start"]), pd.DatetimeIndex([begin]))
        predictions.update(_rule_predictions(run_folder, window, rows, evidence))
        devices["effective_devices"].update({arm: "rule" for arm in _RULES})
    elif profile == "overseas":
        rows = _overseas_inputs(run_folder, fold, rows, config)
    computed = {}
    for arm in policies:
        if arm not in predictions:
            model_arm = "domestic" if profile == "domestic" else arm
            if model_arm not in computed:
                computed[model_arm] = _catboost_prediction(
                    run_folder, fold, model_arm, rows, evidence[model_arm], devices,
                    {policy.min_score for policy in policies.values()},
                )
            predictions[arm], device = computed[model_arm]
            devices["effective_devices"][arm] = device
    return predictions, policies, devices
