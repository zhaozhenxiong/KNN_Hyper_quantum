"""Future-function and validation leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class LeakageIssue:
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": self.message, "severity": self.severity}


def validate_time_split(
    *,
    train_end: str,
    valid_end: str,
    data_end: str | None = None,
) -> dict[str, Any]:
    issues: list[LeakageIssue] = []
    train_ts = pd.to_datetime(train_end)
    valid_ts = pd.to_datetime(valid_end)
    if train_ts >= valid_ts:
        issues.append(LeakageIssue("TRAIN_VALID_OVERLAP", "train_end must be earlier than valid_end"))
    if data_end:
        data_ts = pd.to_datetime(data_end)
        if valid_ts > data_ts:
            issues.append(LeakageIssue("VALID_AFTER_DATA_END", "valid_end is after data_end"))
    return {"ok": not issues, "issues": [issue.to_dict() for issue in issues]}


def validate_feature_label_timing(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
    label_horizon_days: int | None = None,
    feature_available_col: str | None = None,
    label_end_col: str | None = None,
) -> dict[str, Any]:
    issues: list[LeakageIssue] = []
    if date_col not in frame.columns:
        issues.append(LeakageIssue("MISSING_DATE", f"missing {date_col}"))
        return {"ok": False, "issues": [issue.to_dict() for issue in issues]}
    dates = pd.to_datetime(frame[date_col])
    if feature_available_col and feature_available_col in frame.columns:
        available = pd.to_datetime(frame[feature_available_col])
        if (available > dates).any():
            issues.append(LeakageIssue("FEATURE_AVAILABLE_AFTER_SIGNAL", "feature availability is after signal date"))
    if label_end_col and label_end_col in frame.columns:
        label_end = pd.to_datetime(frame[label_end_col])
        if (label_end <= dates).any():
            issues.append(LeakageIssue("LABEL_NOT_FORWARD", "label end date must be after feature date"))
    if label_horizon_days is not None and label_horizon_days <= 0:
        issues.append(LeakageIssue("INVALID_LABEL_HORIZON", "label horizon must be positive"))
    if dates.is_monotonic_increasing is False:
        issues.append(LeakageIssue("DATES_NOT_SORTED", "dates are not sorted", severity="warning"))
    return {"ok": not any(issue.severity == "error" for issue in issues), "issues": [issue.to_dict() for issue in issues]}


def validate_feature_columns_no_lookahead(
    feature_columns: list[str],
    *,
    label_columns: list[str] | None = None,
    allowed_future_columns: list[str] | None = None,
) -> dict[str, Any]:
    labels = {item.lower() for item in (label_columns or [])}
    allowed = {item.lower() for item in (allowed_future_columns or [])}
    future_tokens = ("future_", "next_", "lead_", "forward_", "tomorrow")
    issues: list[LeakageIssue] = []
    for column in feature_columns:
        name = str(column).strip()
        lower = name.lower()
        if lower in labels or lower in allowed:
            continue
        if any(token in lower for token in future_tokens):
            issues.append(
                LeakageIssue(
                    "FUTURE_NAMED_FEATURE",
                    f"feature column appears to contain future information: {name}",
                )
            )
    return {"ok": not issues, "issues": [issue.to_dict() for issue in issues], "checked": len(feature_columns)}
