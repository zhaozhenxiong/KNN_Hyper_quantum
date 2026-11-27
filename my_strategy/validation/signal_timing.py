"""Signal and execution timing checks."""

from __future__ import annotations

from typing import Any

import pandas as pd


def validate_signal_execution_timing(
    frame: pd.DataFrame,
    *,
    signal_time_col: str = "signal_time",
    execution_time_col: str = "execution_time",
    allow_same_timestamp: bool = False,
) -> dict[str, Any]:
    issues: list[str] = []
    if signal_time_col not in frame.columns or execution_time_col not in frame.columns:
        return {
            "ok": True,
            "issues": ["timing_columns_missing"],
            "checked": False,
        }
    signal_time = pd.to_datetime(frame[signal_time_col])
    execution_time = pd.to_datetime(frame[execution_time_col])
    invalid = execution_time < signal_time
    if not allow_same_timestamp:
        invalid = invalid | (execution_time == signal_time)
    invalid_count = int(invalid.sum())
    if invalid_count:
        issues.append(f"execution_not_after_signal:{invalid_count}")
    return {"ok": not issues, "issues": issues, "checked": True, "rows": int(len(frame))}
