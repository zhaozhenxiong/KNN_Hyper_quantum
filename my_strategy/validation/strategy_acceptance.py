"""Strategy-level leakage and backtest acceptance checks."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from my_strategy.validation.data_quality import validate_ohlcv
from my_strategy.validation.leakage import (
    validate_feature_columns_no_lookahead,
    validate_feature_label_timing,
    validate_time_split,
)
from my_strategy.validation.signal_timing import validate_signal_execution_timing


REQUIRED_DAILY_COLUMNS = ["date", "open", "high", "low", "close", "factor_close", "equity", "position_weight"]
REQUIRED_REPORT_KEYS = ["strategy_name", "stock", "stock_name", "metrics", "latest_decision", "files"]
REQUIRED_METRIC_KEYS = ["total_return", "annual_return", "max_drawdown", "sharpe_ratio", "n_trades", "win_rate"]
LEDGER_REQUIRED_COLUMNS = ["date", "symbol", "side", "shares", "price"]


@dataclass
class AcceptanceIssue:
    code: str
    message: str
    severity: str = "error"
    path: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {"code": self.code, "message": self.message, "severity": self.severity}
        if self.path:
            data["path"] = self.path
        return data


@dataclass
class AcceptanceReport:
    run_id: str
    mode: str
    status: str
    checks: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    issues: list[AcceptanceIssue] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        fatal = [issue.to_dict() for issue in self.issues if issue.severity == "error"]
        warnings = [issue.to_dict() for issue in self.issues if issue.severity != "error"]
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "fatal_issues": fatal,
            "warnings": warnings,
            "checks": self.checks,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }


def _status_from_issues(issues: list[AcceptanceIssue], *, fail_on_warning: bool) -> str:
    if any(issue.severity == "error" for issue in issues):
        return "failed"
    if fail_on_warning and any(issue.severity != "error" for issue in issues):
        return "failed"
    return "success"


def _merge_validation_issues(
    issues: list[AcceptanceIssue],
    *,
    check_name: str,
    result: dict[str, Any],
) -> None:
    for item in result.get("issues", []):
        if isinstance(item, dict):
            issues.append(
                AcceptanceIssue(
                    code=str(item.get("code", check_name)),
                    message=str(item.get("message", item)),
                    severity=str(item.get("severity", "error")),
                )
            )
        else:
            severity = "warning" if "missing" in str(item) else "error"
            issues.append(AcceptanceIssue(code=check_name.upper(), message=str(item), severity=severity))


def run_leakage_acceptance(
    *,
    run_id: str,
    feature_columns: list[str],
    train_end: str,
    valid_end: str,
    data_end: str,
    label_horizon_days: int = 5,
    fail_on_warning: bool = False,
) -> AcceptanceReport:
    """Run static strategy-level future-function checks."""

    checks: dict[str, str] = {}
    issues: list[AcceptanceIssue] = []

    feature_result = validate_feature_columns_no_lookahead(feature_columns, label_columns=["future_return_5d"])
    checks["feature_lookahead"] = "pass" if feature_result["ok"] else "fail"
    _merge_validation_issues(issues, check_name="feature_lookahead", result=feature_result)

    split_result = validate_time_split(train_end=train_end, valid_end=valid_end, data_end=data_end)
    checks["time_split"] = "pass" if split_result["ok"] else "fail"
    _merge_validation_issues(issues, check_name="time_split", result=split_result)

    sample_dates = pd.date_range(end=pd.to_datetime(data_end), periods=4, freq="D")
    timing_sample = pd.DataFrame(
        {
            "date": sample_dates,
            "feature_available_at": sample_dates,
            "label_end_at": sample_dates + pd.Timedelta(days=label_horizon_days),
        }
    )
    feature_timing = validate_feature_label_timing(
        timing_sample,
        feature_available_col="feature_available_at",
        label_end_col="label_end_at",
        label_horizon_days=label_horizon_days,
    )
    checks["feature_label_timing"] = "pass" if feature_timing["ok"] else "fail"
    _merge_validation_issues(issues, check_name="feature_label_timing", result=feature_timing)

    signal_timing_sample = pd.DataFrame(
        {
            "signal_time": ["2026-01-01 15:00:00", "2026-01-02 15:00:00"],
            "execution_time": ["2026-01-02 09:30:00", "2026-01-05 09:30:00"],
        }
    )
    execution_timing = validate_signal_execution_timing(signal_timing_sample)
    checks["signal_execution_timing"] = "pass" if execution_timing["ok"] else "fail"
    _merge_validation_issues(issues, check_name="signal_execution_timing", result=execution_timing)

    status = _status_from_issues(issues, fail_on_warning=fail_on_warning)
    return AcceptanceReport(
        run_id=run_id,
        mode="leakage-only",
        status=status,
        checks=checks,
        metrics={"feature_columns_checked": int(feature_result.get("checked", len(feature_columns)))},
        issues=issues,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _resolve_report_file_path(report_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (report_path.parent / path).resolve()


def _check_required_columns(
    issues: list[AcceptanceIssue],
    *,
    frame: pd.DataFrame,
    required: list[str],
    code: str,
    path: Path,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        issues.append(AcceptanceIssue(code=code, message="missing columns: " + ", ".join(missing), path=str(path)))


def audit_backtest_report(
    report_path: Path,
    *,
    run_id: str,
    require_figures: bool = True,
    require_ledger: bool = True,
    fail_on_warning: bool = False,
) -> AcceptanceReport:
    """Audit one KHQuant backtest report and its linked artifacts."""

    report_path = report_path.resolve()
    issues: list[AcceptanceIssue] = []
    checks: dict[str, str] = {}
    metrics: dict[str, Any] = {}
    artifacts: dict[str, str] = {"report": str(report_path)}

    if not report_path.exists():
        issue = AcceptanceIssue("MISSING_REPORT", "report file does not exist", path=str(report_path))
        return AcceptanceReport(run_id=run_id, mode="full-backtest", status="failed", checks={"artifacts": "fail"}, issues=[issue])

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        issue = AcceptanceIssue("INVALID_REPORT_JSON", f"cannot read report json: {exc}", path=str(report_path))
        return AcceptanceReport(run_id=run_id, mode="full-backtest", status="failed", checks={"artifacts": "fail"}, issues=[issue])

    missing_report_keys = [key for key in REQUIRED_REPORT_KEYS if key not in report]
    if missing_report_keys:
        issues.append(AcceptanceIssue("REPORT_CONTRACT", "missing report keys: " + ", ".join(missing_report_keys), path=str(report_path)))
    checks["report_contract"] = "pass" if not missing_report_keys else "fail"

    report_metrics = report.get("metrics", {}) if isinstance(report.get("metrics"), dict) else {}
    missing_metrics = [key for key in REQUIRED_METRIC_KEYS if key not in report_metrics]
    if missing_metrics:
        issues.append(AcceptanceIssue("METRICS_CONTRACT", "missing metric keys: " + ", ".join(missing_metrics), path=str(report_path)))
    checks["metrics_contract"] = "pass" if not missing_metrics else "fail"
    metrics.update({key: report_metrics.get(key) for key in REQUIRED_METRIC_KEYS if key in report_metrics})

    files = report.get("files", {}) if isinstance(report.get("files"), dict) else {}
    daily_path = _resolve_report_file_path(report_path, files.get("daily"))
    trades_path = _resolve_report_file_path(report_path, files.get("trades"))
    figure_path = _resolve_report_file_path(report_path, files.get("figure"))

    for label, path in [("daily", daily_path), ("trades", trades_path), ("figure", figure_path)]:
        if path:
            artifacts[label] = str(path)

    daily: pd.DataFrame | None = None
    if not daily_path or not daily_path.exists():
        issues.append(AcceptanceIssue("MISSING_DAILY", "daily csv is missing", path=str(daily_path or "")))
        checks["daily_contract"] = "fail"
        checks["ohlcv_quality"] = "fail"
    else:
        daily = _read_csv(daily_path)
        _check_required_columns(issues, frame=daily, required=REQUIRED_DAILY_COLUMNS, code="DAILY_CONTRACT", path=daily_path)
        checks["daily_contract"] = "pass" if all(column in daily.columns for column in REQUIRED_DAILY_COLUMNS) else "fail"
        ohlcv = validate_ohlcv(daily)
        checks["ohlcv_quality"] = "pass" if ohlcv["ok"] else "fail"
        _merge_validation_issues(issues, check_name="ohlcv_quality", result=ohlcv)
        metrics["daily_rows"] = int(len(daily))
        metrics["duplicate_daily_dates"] = int(pd.to_datetime(daily["date"]).duplicated().sum()) if "date" in daily.columns else None
        if "equity" in daily.columns:
            equity = pd.to_numeric(daily["equity"], errors="coerce")
            if equity.isna().any() or (equity <= 0).any():
                issues.append(AcceptanceIssue("INVALID_EQUITY", "equity contains NaN or non-positive values", path=str(daily_path)))
        if "position_weight" in daily.columns:
            weights = pd.to_numeric(daily["position_weight"], errors="coerce")
            if weights.isna().any() or (weights < -0.001).any() or (weights > 1.5).any():
                issues.append(AcceptanceIssue("INVALID_POSITION_WEIGHT", "position_weight is outside accepted bounds", path=str(daily_path)))

    if not trades_path or not trades_path.exists():
        if require_ledger:
            issues.append(AcceptanceIssue("MISSING_TRADES", "trades csv is missing", path=str(trades_path or "")))
            checks["broker_ledger"] = "fail"
        else:
            checks["broker_ledger"] = "warn"
    else:
        trades = _read_csv(trades_path)
        metrics["trade_rows"] = int(len(trades))
        if require_ledger and not trades.empty:
            _check_required_columns(issues, frame=trades, required=LEDGER_REQUIRED_COLUMNS, code="LEDGER_CONTRACT", path=trades_path)
        checks["broker_ledger"] = "pass" if not any(issue.code == "LEDGER_CONTRACT" for issue in issues) else "fail"

    if require_figures:
        if not figure_path or not figure_path.exists() or figure_path.stat().st_size <= 0:
            issues.append(AcceptanceIssue("MISSING_FIGURE", "figure is missing or empty", path=str(figure_path or "")))
            checks["artifacts"] = "fail"
        else:
            checks["artifacts"] = "pass"
    else:
        checks.setdefault("artifacts", "pass")

    if daily is not None and "date" in daily.columns:
        latest_report_date = str(report.get("latest_decision", {}).get("date", ""))
        latest_daily_date = str(pd.to_datetime(daily["date"]).max().date())
        if latest_report_date and latest_daily_date not in latest_report_date:
            issues.append(
                AcceptanceIssue(
                    "LATEST_DATE_MISMATCH",
                    f"latest report date {latest_report_date} does not match daily max {latest_daily_date}",
                    severity="warning",
                    path=str(report_path),
                )
            )

    status = _status_from_issues(issues, fail_on_warning=fail_on_warning)
    return AcceptanceReport(
        run_id=run_id,
        mode="full-backtest",
        status=status,
        checks=checks,
        metrics=metrics,
        issues=issues,
        artifacts=artifacts,
    )


def write_acceptance_report(report: AcceptanceReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path
