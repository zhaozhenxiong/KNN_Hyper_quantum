"""Causal walk-forward KNN calibration for archived multimodal quantiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import expert_calibration as calibration


LOOKBACK_SESSIONS = 60


def online_knn_residual_quantiles(
    cal_rows: pd.DataFrame, cal_raw: np.ndarray, rows: pd.DataFrame, raw: np.ndarray,
    *, lookback_sessions: int = LOOKBACK_SESSIONS,
) -> tuple[np.ndarray, dict]:
    """Refit on each signal date using only labels matured strictly before it."""
    if lookback_sessions < 1 or cal_rows.empty or rows.empty:
        raise ValueError("在线 KNN 需要正回看窗口及非空校准、评估样本")
    if not cal_rows["label_end"].max() < rows["date"].min():
        raise ValueError("初始校准标签越过评估起点")
    pool = pd.concat([cal_rows, rows], ignore_index=True)
    pool_raw = np.vstack([cal_raw, raw])
    dates = pd.to_datetime(pool["date"]).to_numpy(dtype="datetime64[ns]")
    labels = pd.to_datetime(pool["label_end"]).to_numpy(dtype="datetime64[ns]")
    known = pool["target"].notna().to_numpy() & np.isfinite(pool["target"].to_numpy(dtype=float))
    result = np.empty_like(raw)
    daily = []
    evaluation_dates = pd.DatetimeIndex(rows["date"].unique()).sort_values()
    for day in evaluation_dates:
        current = np.flatnonzero(rows["date"].eq(day).to_numpy()) + len(cal_rows)
        eligible = known & (dates < day.to_datetime64()) & (labels < day.to_datetime64())
        available_dates = np.unique(dates[eligible])
        if not len(available_dates):
            raise ValueError(f"{day.date()} 缺少已成熟的在线校准标签")
        cutoff = available_dates[-min(lookback_sessions, len(available_dates))]
        history = np.flatnonzero(eligible & (dates >= cutoff))
        values, evidence = calibration._knn_residual_quantiles(
            pool.iloc[history], pool_raw[history], pool.iloc[current], pool_raw[current],
        )
        result[current - len(cal_rows)] = values
        daily.append({
            "date": day.date().isoformat(), "history_rows": len(history),
            "sample_rows": evidence["sample_rows"], "history_start": str(pd.Timestamp(cutoff).date()),
            "last_label_end": str(pd.Timestamp(labels[history].max()).date()),
            "history_sessions": len(available_dates[-lookback_sessions:]),
        })
    if not np.isfinite(result).all() or (np.diff(result, axis=1) < 0).any():
        raise ValueError("在线 KNN 校准产生非有限或无序分位数")
    return result, {
        "method": "online_knn_conditional_residual_quantiles",
        "lookback_sessions": lookback_sessions,
        "label_boundary": "label_end_strictly_before_each_signal_date",
        "daily": daily,
    }


def recover_online(archive: calibration.Archive, fold: dict, rows: pd.DataFrame,
                   variants: dict, evidence: dict) -> tuple[np.ndarray, dict]:
    """Recover only the calibration raw forecasts; reuse the verified evaluation recovery."""
    train, cal = (calibration._settled(archive.data.frame, fold[stage]) for stage in ("train", "calibration"))
    cal_raw = calibration._raw_predict(archive, fold, evidence["weights"], train, cal, {})
    return online_knn_residual_quantiles(cal, cal_raw, rows, variants["raw"])
