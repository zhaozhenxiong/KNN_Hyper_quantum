"""Prepare expert research inputs with fixed or point-in-time stock membership."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from my_strategy.core.paths import artifact_run_dir
from my_strategy.core.run_context import stable_hash
from my_strategy.latent_quant.data import (
    BASE_FEATURES, EXTRA_FEATURES, _BOARD_POLICY, _HFQ, _PRICES, _RAW_DIR, _STAGE, _TRADE_PRICES, _UPDATE,
    _market_adjusted, _match_execution_reference, _read_market, _sample_universe, _status_evidence, _stock_features,
)
from my_strategy.latent_quant.domestic_runner import BASELINE_ID, FrozenInputs, _load_frozen, _read, _sha
from my_strategy.latent_quant.runner import HISTORICAL_END, OOS_STARTS, START
from my_strategy.research_workbench.data import (
    _STOCK, _attach_labels, _date, _load_benchmark, _read_bars, attach_point_in_time_classifications,
    load_classification_history,
)
from my_strategy.research_workbench.stock_replay_data import _bounded_prices, _calendar, _day
from my_strategy.research_workbench.universe import BOARD_POLICIES, load_universe, members, restrict_bars


ORIGINAL_PRESET = "frozen_v2"
DEFAULT_START, DEFAULT_END = OOS_STARTS[0], "2026-07-28"
WARMUP = "2023-09-01"
POOLS = ("frozen_120", "manual", "board", "full")
DEFAULT_UNIVERSE_POLICY = _BOARD_POLICY
SUPPORTED_BOARDS = tuple(sorted(BOARD_POLICIES[DEFAULT_UNIVERSE_POLICY]))
UNIVERSE_SCOPE_LABELS = {
    "main_star": "沪深主板+科创板",
    "shsz": "沪深主板+创业板+科创板",
}
FULL_MEMBERSHIP = "daily_exchange_list_delist"
_OPTIONS = {
    "preset", "pool", "symbols", "boards", "stock_cap", "seed", "start", "end", "commission_bps",
    "universe_policy", "sector_view", "sector_provider",
}
_ARCHIVE = {
    "roster": _STAGE / "roster.csv", "calendar": _STAGE / "calendar.csv", "benchmark": _STAGE / "csi300.csv",
    "adjusted": _HFQ / "verified_market_hfq.parquet", "reference": _HFQ / "sina_klc_market_reference.parquet",
    "classification": _STAGE / "industries" / "capco-industries.csv",
    "status_roster": _STAGE / "verified_nontrading_evidence.csv",
    "status_updates": _UPDATE / "verified_nontrading.csv", "status_star": _UPDATE / "star_suspensions.csv",
}
_RAW = _RAW_DIR / "khquant_raw.db"
_PANEL = ("panel.parquet", "benchmark.parquet", "roster.parquet", "market_status.parquet", "coverage.json")
_KEYS = ["stock", "date"]


class ExpertSpecError(ValueError):
    """Invalid or insufficient evidence for the requested study cohort."""

    def __init__(self, message: str, *, blockers: list[dict] | None = None):
        super().__init__(message)
        self.blockers = blockers or []


def universe_policy(spec: Mapping | None) -> str:
    """Legacy omission is main_star, preserving old spec and identity hashes."""
    return (spec or {}).get("universe_policy", DEFAULT_UNIVERSE_POLICY)


def universe_scope_label(spec: Mapping | None) -> str:
    return UNIVERSE_SCOPE_LABELS[universe_policy(spec)]


def _strings(value, field: str) -> list[str]:
    if not isinstance(value, (list, tuple)) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ExpertSpecError(f"{field} 必须为非空字符串条目的数组。")
    return [item.strip() for item in value]


def normalize_expert_spec(payload: Mapping | None = None) -> dict:
    """Canonicalize historical dates and an explicit pool. No arbitrary paths are accepted."""
    if payload is not None and (not isinstance(payload, Mapping) or set(payload) - _OPTIONS):
        raise ExpertSpecError("专家研究请求含未支持字段。")
    options = payload or {}
    preset = options.get("preset", ORIGINAL_PRESET if payload is None else "custom")
    pool = options.get("pool", "frozen_120")
    if preset not in (ORIGINAL_PRESET, "custom") or pool not in POOLS:
        raise ExpertSpecError("未知的研究预设或股票池。")
    policy = options.get("universe_policy", DEFAULT_UNIVERSE_POLICY)
    if policy not in BOARD_POLICIES or (pool != "full" and policy != DEFAULT_UNIVERSE_POLICY):
        raise ExpertSpecError("股票池策略仅支持全量池的 main_star 或 shsz。")
    symbols = sorted({stock.upper() for stock in _strings(options.get("symbols", []), "symbols")})
    boards = sorted({board.lower() for board in _strings(options.get("boards", []), "boards")})
    if any(_STOCK.fullmatch(stock) is None for stock in symbols):
        raise ExpertSpecError("股票代码须包含交易所后缀，如 002436.SZ。")
    if set(boards) - set(SUPPORTED_BOARDS):
        raise ExpertSpecError("当前历史名单与复权档案只支持沪深主板和科创板。")
    if pool == "manual" and not symbols or pool == "board" and not boards:
        raise ExpertSpecError("请选择要研究的股票或板块。")
    if (pool != "manual" and symbols) or (pool != "board" and boards):
        raise ExpertSpecError("股票代码仅适用于自选池，板块仅适用于板块池。")
    cap = options.get("stock_cap", {"manual": len(symbols), "full": None}.get(pool, 120))
    seed = options.get("seed", 42)
    if pool == "full" and cap is not None:
        raise ExpertSpecError("全量核验股票池按每日在市名单纳入成员，不接受抽样上限。")
    if (pool != "full" and (type(cap) is not int or cap < 1)) or type(seed) is not int or not 0 <= seed <= 2**31 - 1:
        raise ExpertSpecError("样本上限须为正整数，种子须为 0 至 2147483647 的整数。")
    if pool == "frozen_120" and cap != 120 or pool == "manual" and cap < len(symbols):
        raise ExpertSpecError("冻结样本固定 120 只，自选样本不会按上限静默截断。")
    if pool == "manual":
        cap = len(symbols)
    start = options.get("start", DEFAULT_START)
    end = options.get("end", HISTORICAL_END if pool == "full" else DEFAULT_END)
    try:
        valid = isinstance(start, str) and isinstance(end, str) and _date(START) <= _date(start) <= _date(end) <= _date(HISTORICAL_END)
    except ValueError as error:
        raise ExpertSpecError("研究日期须为 YYYY-MM-DD。") from error
    if not valid:
        raise ExpertSpecError(f"评估范围须落在 {START} 至 {HISTORICAL_END}。")
    if preset == ORIGINAL_PRESET and (pool != "frozen_120" or (start, end, seed) != (DEFAULT_START, DEFAULT_END, 42)):
        raise ExpertSpecError("原冻结预设的股票池、评估日期及种子为已记录的固定口径。")
    spec = {"preset": preset, "pool": pool, "symbols": symbols, "boards": boards, "stock_cap": cap,
            "seed": seed, "start": start, "end": end}
    # Do not add a default-valued key: legacy specs, archives and prediction keys
    # already mean main_star. A non-default policy must always be explicit.
    if pool == "full" and policy != DEFAULT_UNIVERSE_POLICY:
        spec["universe_policy"] = policy
    if "commission_bps" in options:
        spec["commission_bps"] = options["commission_bps"]
    if "sector_view" in options:
        if type(options["sector_view"]) is not bool:
            raise ExpertSpecError("板块视图开关必须为布尔值。")
        if options["sector_view"]:
            spec["sector_view"] = True
    if "sector_provider" in options:
        provider = options["sector_provider"]
        if provider not in {"capco", "ths_fuyao"}:
            raise ExpertSpecError("板块数据提供方仅支持 capco 或 ths_fuyao。")
        spec["sector_provider"] = provider
    return spec


@dataclass(frozen=True)
class _Scope:
    spec: dict
    calendar: pd.DatetimeIndex
    sessions: pd.DatetimeIndex
    evaluation: pd.DatetimeIndex
    historical_roster: pd.DataFrame
    roster: pd.DataFrame
    anchor: pd.Timestamp | None
    use_frozen: bool
    rejections: dict[str, str]


def _frozen_paths() -> dict[str, Path]:
    run = artifact_run_dir(BASELINE_ID, create=False)
    return {"study.json": run / "config" / "study.json", **{name: run / "data_snapshots" / name for name in _PANEL}}


def _scope(payload: Mapping | None) -> _Scope:
    spec = normalize_expert_spec(payload)
    cutoff = _date(HISTORICAL_END if spec["preset"] == ORIGINAL_PRESET else spec["end"])
    calendar = _calendar(_ARCHIVE["calendar"], _date(WARMUP), cutoff)
    sessions = calendar[calendar >= _date(START)]
    evaluation = sessions[(sessions >= _date(spec["start"])) & (sessions <= _date(spec["end"]))]
    if spec["pool"] == "full":
        roster = load_universe(_ARCHIVE["roster"], universe_policy(spec))
        selected = members(roster, sessions[0], sessions[-1]) if len(sessions) else roster.iloc[:0]
        return _Scope(spec, calendar, sessions, evaluation, roster, selected, None, False, {})
    anchor = calendar[calendar < _date(START)][-1]
    frozen = pd.read_parquet(_frozen_paths()["roster.parquet"]).set_index("stock")
    if len(frozen) != 120 or not frozen.index.is_unique:
        raise ExpertSpecError("冻结 v2 股票池缺失或成员异常。")
    if spec["pool"] == "frozen_120":
        return _Scope(spec, calendar, sessions, evaluation, frozen, frozen, anchor, True, {})

    roster = load_universe(_ARCHIVE["roster"], _BOARD_POLICY)
    active = members(roster, anchor, anchor)
    with closing(sqlite3.connect(_RAW.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        if spec["pool"] == "board":
            boards = roster.loc[roster["board"].isin(spec["boards"])]
            stocks, _ = _sample_universe(connection, boards, calendar, _date(START), spec["stock_cap"], spec["seed"])
            rejected = {}
        else:
            quotes = {stock for stock, in connection.execute(
                """SELECT stock FROM stock_daily_normalized
                   WHERE date = ? AND has_trade_price = 1 AND open > 0 AND volume > 0""", (_day(anchor),),
            )}
            rejected = {
                stock: ("缺少同口径交易所名单" if stock not in roster.index else
                        "锚定交易日未上市或已退市" if stock not in active.index else "锚定交易日缺少有效原始行情")
                for stock in spec["symbols"] if stock not in active.index or stock not in quotes
            }
            stocks = [stock for stock in spec["symbols"] if stock not in rejected]
    use_frozen = set(stocks).issubset(frozen.index)
    selected = (frozen if use_frozen else roster).loc[stocks]
    return _Scope(spec, calendar, sessions, evaluation, roster, selected, anchor, use_frozen, rejected)


def _read_raw(scope: _Scope) -> pd.DataFrame:
    with closing(sqlite3.connect(_RAW.as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("BEGIN")
        return _read_bars(connection, scope.roster.index.tolist(), WARMUP, _day(scope.calendar[-1]))


def _raw_sha(raw: pd.DataFrame) -> str:
    if raw.empty:
        return sha256(raw.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()
    digest = sha256()
    for start in range(0, len(raw), 50_000):
        chunk = raw.iloc[start : start + 50_000]
        digest.update(chunk.to_csv(index=False, header=start == 0, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def _identity(scope: _Scope, raw: pd.DataFrame | None = None) -> dict:
    sources = _frozen_paths() if scope.use_frozen else _ARCHIVE
    digests = {name: {"path": str(path), "sha256": _sha(path)} for name, path in sources.items()}
    if not scope.use_frozen:
        digests["raw_db"] = {"path": str(_RAW), "sha256": _raw_sha(_read_raw(scope) if raw is None else raw),
                             "scope": "ordered selected stock/date bars through cutoff including warmup"}
    roster = scope.roster.sort_index()[["exchange", "board", "list_date", "delist_date", "source", "source_url"]]
    return {
        "schema_version": 1, "pool": scope.spec["pool"], "symbols": scope.roster.index.tolist(),
        "selected_roster_sha256": sha256(roster.to_csv(lineterminator="\n").encode("utf-8")).hexdigest(),
        "source_kind": "frozen_v2" if scope.use_frozen else "verified_local_archive",
        "anchor_session": _day(scope.anchor) if scope.anchor is not None else None,
        "input_start": START, "input_cutoff": _day(scope.calendar[-1]),
        **({"membership": FULL_MEMBERSHIP} if scope.anchor is None else {}),
        **({"universe_policy": universe_policy(scope.spec)}
           if universe_policy(scope.spec) != DEFAULT_UNIVERSE_POLICY else {}),
        "feature_dependency_sha256": {
            path.name: _sha(path) for path in (
                Path(__file__), Path(__file__).with_name("stock_replay_data.py"),
                Path(__file__).with_name("universe.py"), Path(__file__).with_name("data.py"),
            )
        } | {"latent_quant/data.py": _sha(Path(__file__).parents[1] / "latent_quant" / "data.py")},
        "inputs": digests,
    }


def _require_selection(scope: _Scope) -> None:
    if scope.rejections or scope.roster.empty:
        failures = "、".join(f"{stock} {reason}" for stock, reason in scope.rejections.items())
        raise ExpertSpecError(f"不能建立完整的时点股票池。{failures}")
    if scope.evaluation.empty:
        raise ExpertSpecError("所选评估区间没有可核验的交易日。")


def expert_input_identity(spec: Mapping | None = None) -> dict:
    """Fingerprint sources and selected roster without constructing market-wide features."""
    scope = _scope(spec)
    _require_selection(scope)
    return _identity(scope)


input_identity = expert_input_identity


def _adjusted_maturity(adjusted: pd.DataFrame, scope: _Scope) -> pd.DataFrame:
    # A valid feature needs twenty returns and thus twenty-one consecutive
    # closes. A five-day label needs opening prices at T+1 and T+6.
    dates = pd.Series(scope.calendar, index=scope.calendar)
    potential = np.zeros(len(dates), dtype=bool)
    if not adjusted.empty and len(dates) >= 27:
        listed = adjusted.loc[adjusted["date"].isin(scope.calendar), _KEYS]
        if scope.anchor is None:
            listed = restrict_bars(listed, scope.roster)[_KEYS]
        codes = scope.roster.index.get_indexer(listed["stock"])
        sessions = scope.calendar.get_indexer(listed["date"])
        observed = np.zeros((len(scope.roster), len(dates)), dtype=bool)
        observed[codes, sessions] = True
        rolling = np.lib.stride_tricks.sliding_window_view(observed, 21, axis=1).all(axis=-1)
        potential[20:-6] = (rolling[:, :-6] & observed[:, 21:-5] & observed[:, 26:]).any(axis=0)
    return pd.DataFrame({"date": dates[potential], "label_end": dates.shift(-6)[potential], "target": 0.0})


def _expected_listed_days(roster: pd.DataFrame, sessions: pd.DatetimeIndex) -> pd.Series:
    """Count sessions where each exchange lifetime is open, with exclusive delisting."""
    starts = sessions.searchsorted(roster["list_date"].to_numpy())
    ends = sessions.searchsorted(roster["delist_date"].to_numpy())
    return pd.Series(np.maximum(0, ends - starts), index=roster.index)


def _full_membership(scope: _Scope) -> dict:
    return {
        "basis": FULL_MEMBERSHIP, "boards": sorted(BOARD_POLICIES[universe_policy(scope.spec)]),
        "historical_members": len(scope.roster),
        "listed_after_input_start": int(scope.roster["list_date"].gt(scope.sessions[0]).sum()),
        "delisted_by_cutoff": int(scope.roster["delist_date"].le(scope.sessions[-1]).sum()),
        "active_on_cutoff": len(members(scope.roster, scope.sessions[-1], scope.sessions[-1])),
    }


def _maturity(scope: _Scope, panel: pd.DataFrame) -> tuple[list[dict], list[str]]:
    from .expert_research import plan_expert_calendar

    try:
        windows = plan_expert_calendar(scope.sessions, scope.spec)
    except ValueError as error:
        return [], [str(error)]
    if scope.use_frozen:
        settled = panel.loc[panel["signal_eligible"].eq(True) & np.isfinite(panel["target"])
                            & panel["label_end"].gt(panel["date"])]
        basis = "eligible_mature_labels"
    else:
        settled = _adjusted_maturity(panel, scope)
        basis = "adjusted_coverage_upper_bound"
    records, blockers = [], []
    for window in windows:
        train, validation, calibration = [int(settled.loc[mask, "date"].nunique()) for mask in window.masks(settled)]
        mature = settled["date"].isin(window.evaluate) & settled["label_end"].le(window.evaluate[-1])
        evaluated = int(settled.loc[mature, "date"].nunique())
        blocker = None
        if train < 100 or validation == 0 or calibration == 0 or evaluated == 0:
            blocker = f"{window.period} 缺少 100 个成熟训练日、独立验证／校准日或窗口内成熟标签。"
            blockers.append(blocker)
        records.append({
            "id": window.period, "train_start": _day(window.train_start), "validation_start": _day(window.valid_start),
            "calibration_start": _day(window.calibration_start), "evaluation_start": _day(window.evaluate[0]),
            "evaluation_end": _day(window.evaluate[-1]), "evaluation_sessions": len(window.evaluate),
            "maturity": {"basis": basis, "train_sessions": train, "validation_sessions": validation,
                         "calibration_sessions": calibration, "evaluation_sessions": evaluated},
            "blocked_reason": blocker,
        })
    return records, blockers


def _shsz_policy_evidence(scope: _Scope, panel: pd.DataFrame, observed: dict) -> tuple[list[dict], dict]:
    """Audit ChiNext explicitly; an empty roster subset is never a valid SHSZ fallback."""
    blockers = []
    chinext = scope.roster.index[scope.roster["board"].eq("chinext")].tolist()
    historical_boards = set(scope.historical_roster["board"])
    evidence = {
        "roster_stocks": len(scope.historical_roster),
        "roster_board_counts": scope.historical_roster["board"].value_counts().to_dict(),
        "roster_chinext_stocks": len(chinext),
        "adjusted_stocks": int(panel["stock"].nunique()),
        "adjusted_chinext_stocks": 0,
        "evaluation_adjusted_chinext_stocks": sum(bool(observed.get(stock)) for stock in chinext),
        "raw_chinext_stocks": 0,
        "reference_chinext_stocks": 0,
        "aligned_chinext_stocks": 0,
        "aligned_chinext_stock_days": 0,
        "evaluation_aligned_chinext_stocks": 0,
        "sources": {
            "roster": {"path": str(_ARCHIVE["roster"]), "readable": True},
            "adjusted": {"path": str(_ARCHIVE["adjusted"]), "readable": True},
        },
    }
    source_paths = {
        "reference": str(_ARCHIVE["reference"]), "raw_db": str(_RAW), "classification": str(_ARCHIVE["classification"]),
        "market_status": [str(_ARCHIVE[key]) for key in ("status_roster", "status_updates", "status_star")],
    }
    source_errors = []

    def read_source(name, operation):
        try:
            result = operation()
        except Exception as error:  # Evidence failures belong in the structured preflight report.
            source_errors.append(name)
            evidence["sources"][name] = {"path": source_paths[name], "readable": False,
                                         "error": f"{type(error).__name__}: {error}"}
            return None
        evidence["sources"][name] = {"path": source_paths[name], "readable": True}
        return result

    adjusted_keys = {
        (stock, pd.Timestamp(day)) for stock, day in panel[_KEYS].itertuples(index=False, name=None)
        if stock in chinext
    }
    adjusted_stocks = {stock for stock, _ in adjusted_keys}
    evidence["adjusted_chinext_stocks"] = len(adjusted_stocks)

    def reference_evidence():
        path = _ARCHIVE["reference"]
        required = {*_KEYS, *_PRICES, "volume", "amount"}
        if chinext:
            return _read_market(path, (*_KEYS, *_PRICES, "volume", "amount"), chinext)
        columns = pq.read_schema(path).names if path.suffix.lower() == ".parquet" else pd.read_csv(path, nrows=0).columns
        if missing := required - set(columns):
            raise ValueError(f"Execution reference is missing fields: {sorted(missing)}")
        return pd.DataFrame({"stock": pd.Series(dtype=str), "date": pd.Series(dtype="datetime64[ns]")})

    reference = read_source("reference", reference_evidence)
    reference_keys = set() if reference is None else {
        (stock, pd.Timestamp(day)) for stock, day in reference.loc[
            reference["date"].between(scope.calendar[0], scope.calendar[-1]), _KEYS,
        ].itertuples(index=False, name=None)
    }
    reference_stocks = {stock for stock, _ in reference_keys}
    evidence["reference_chinext_stocks"] = len(reference_stocks)

    def raw_market_keys():
        with closing(sqlite3.connect(_RAW.as_uri() + "?mode=ro", uri=True)) as connection:
            connection.execute("BEGIN")
            rows = connection.execute(
                """SELECT DISTINCT stock, date FROM stock_daily_normalized
                   WHERE date BETWEEN ? AND ? AND has_trade_price = 1 AND open > 0 AND volume > 0""",
                (_day(scope.calendar[0]), _day(scope.calendar[-1])),
            )
            return {(stock, pd.Timestamp(day)) for stock, day in rows if stock in chinext}

    raw_keys = read_source("raw_db", raw_market_keys) or set()
    raw_stocks = {stock for stock, _ in raw_keys}
    evidence["raw_chinext_stocks"] = len(raw_stocks)
    aligned_keys = adjusted_keys & reference_keys & raw_keys
    evidence["aligned_chinext_stocks"] = len({stock for stock, _ in aligned_keys})
    evidence["aligned_chinext_stock_days"] = len(aligned_keys)
    evaluation = set(scope.evaluation)
    evidence["evaluation_aligned_chinext_stocks"] = len({
        stock for stock, day in aligned_keys if day in evaluation
    })

    history = read_source("classification", lambda: load_classification_history(_ARCHIVE["classification"]))
    evidence["classification_chinext_rows"] = 0 if history is None else int(history["stock"].isin(chinext).sum())
    status = read_source("market_status", lambda: _status_evidence(
        [_ARCHIVE[key] for key in ("status_roster", "status_updates", "status_star")], chinext, scope.evaluation,
    ))
    evidence["market_status_chinext_rows"] = 0 if status is None else len(status)

    if "chinext" not in historical_boards:
        blockers.append({
            "code": "missing_chinext_roster", "board": "chinext",
            "message": "SHSZ 预检受阻：本地交易所名录没有创业板（chinext）成员，禁止回退为沪深主板+科创板。",
            "evidence": {"roster": str(_ARCHIVE["roster"]), "actual_board_counts": evidence["roster_board_counts"]},
        })
    elif not chinext:
        blockers.append({
            "code": "missing_chinext_membership", "board": "chinext",
            "message": "SHSZ 预检受阻：创业板名录与请求的上市／退市期间没有交集。",
            "evidence": {"roster": str(_ARCHIVE["roster"])},
        })
    else:
        required = {key: evidence[key] for key in (
            "evaluation_adjusted_chinext_stocks", "raw_chinext_stocks", "reference_chinext_stocks",
            "aligned_chinext_stock_days", "evaluation_aligned_chinext_stocks",
        )}
        if any(count == 0 for count in required.values()):
            blockers.append({
                "code": "insufficient_chinext_market_evidence", "board": "chinext",
                "message": "SHSZ 预检受阻：创业板缺少对齐的复权、原始行情或执行参考证据，不能建立全量冻结。",
                "evidence": required,
            })
    for name in dict.fromkeys(source_errors):
        blockers.append({
            "code": "invalid_shsz_evidence", "source": name,
            "message": f"SHSZ 预检证据不可读：{name}。",
            "evidence": evidence["sources"][name],
        })
    return blockers, evidence


def require_full_universe_evidence(spec: Mapping | None) -> dict:
    """Reject a new SHSZ input build unless its explicit policy evidence passes."""
    report = expert_preflight(spec)
    if report["status"] == "blocked" or report.get("blocker_details"):
        details = report.get("blocker_details") or [
            {"code": "full_preflight_blocked", "message": message} for message in report["blockers"]
        ]
        raise ExpertSpecError("；".join(report["blockers"]) or "SHSZ 全量股票池预检受阻。", blockers=details)
    return report


def expert_preflight(spec: Mapping | None = None) -> dict:
    """Preview actual sessions and the requested cohort without training models."""
    scope = _scope(spec)
    first, last = scope.evaluation[[0, -1]] if len(scope.evaluation) else (None, None)
    selected = scope.roster.index.tolist()
    observed = {}
    panel = pd.DataFrame(columns=_KEYS)
    windows, blockers = [], []
    if selected and first is not None:
        if scope.use_frozen:
            file = _frozen_paths()["panel.parquet"]
            panel = pd.read_parquet(file, columns=["stock", "date", "signal_eligible"],
                                    filters=[("stock", "in", selected), ("date", ">=", first), ("date", "<=", last)])
            observed = panel.groupby("stock")["signal_eligible"].sum().to_dict()
            panel = pd.read_parquet(file, columns=["date", "signal_eligible", "label_end", "target"],
                                    filters=[("stock", "in", selected), ("label_end", "<=", scope.sessions[-1])])
            evidence = "frozen_signal_eligible_days"
        else:
            panel = pd.read_parquet(_ARCHIVE["adjusted"], columns=["stock", "date"],
                                    filters=[("stock", "in", selected), ("date", ">=", scope.calendar[0].date()),
                                             ("date", "<=", last.date())])
            panel["date"] = pd.to_datetime(panel["date"])
            if scope.anchor is None:
                panel = restrict_bars(panel, scope.roster)[_KEYS]
            observed = panel.loc[panel["date"].between(first, last)].groupby("stock")["date"].nunique().to_dict()
            evidence = "adjusted_bars_before_feature_warmup"
        windows, blockers = _maturity(scope, panel)
    else:
        evidence = "no_sources"
        if first is None:
            blockers.append("所选评估区间没有可核验的交易日。")
    expected_days = (_expected_listed_days(scope.roster, scope.evaluation) if scope.anchor is None
                     else pd.Series(len(scope.evaluation), index=scope.roster.index))
    in_evaluation = expected_days.loc[expected_days.gt(0)]
    missing = [stock for stock in in_evaluation.index if not observed.get(stock)]
    incomplete = [stock for stock, days in in_evaluation.items() if 0 < observed.get(stock, 0) < days]
    warnings = []
    if missing:
        warnings.append("部分股票在请求区间没有可确认的特征或复权价格。")
    if not scope.use_frozen:
        warnings.append("扩展股票的预检按复权覆盖计数，实际可用信号还需通过连续特征、成交额和时间切分检查。")
    if scope.spec["pool"] == "board" and len(selected) < scope.spec["stock_cap"]:
        warnings.append("锚定日有效股票数少于请求上限。")
    blocker_details, policy_evidence = [], None
    if scope.spec["pool"] == "full" and universe_policy(scope.spec) == "shsz":
        blocker_details, policy_evidence = _shsz_policy_evidence(scope, panel, observed)
        blockers.extend(blocker["message"] for blocker in blocker_details)
    return {
        "status": "blocked" if blockers or scope.rejections or len(missing) == len(in_evaluation) else "partial" if missing else "ready",
        "spec": scope.spec, "universe_policy": universe_policy(scope.spec), "scope_label": universe_scope_label(scope.spec),
        "blocker_details": blocker_details, **({"policy_evidence": policy_evidence} if policy_evidence is not None else {}),
        "first_trade_session": _day(first) if first is not None else None,
        "last_trade_session": _day(last) if last is not None else None,
        "first_training_input": _day(scope.sessions[0]) if len(scope.sessions) else None,
        "evaluation_sessions": len(scope.evaluation),
        "anchor_session": _day(scope.anchor) if scope.anchor is not None else None,
        "selected_symbols": selected, "selected_stocks": len(selected),
        "board_counts": scope.roster["board"].value_counts().to_dict(), "rejections": scope.rejections, "warnings": warnings,
        **({"roster_board_counts": scope.historical_roster["board"].value_counts().to_dict()}
           if scope.anchor is None else {}),
        "windows": windows, "blockers": blockers,
        "coverage": {"basis": evidence, "by_symbol": {stock: int(observed.get(stock, 0)) for stock in selected},
                     "without_evidence": missing, "partially_covered": incomplete},
        **({"membership": _full_membership(scope)} if scope.anchor is None and len(scope.sessions) else {}),
    }


def _frozen_subset(scope: _Scope) -> FrozenInputs:
    paths = _frozen_paths()
    if _read(paths["study.json"].parent.parent / "status.json").get("status") != "completed":
        raise ExpertSpecError("冻结 v2 基线不可用。")
    cutoff = scope.sessions[-1]
    stocks = scope.roster.index.tolist()
    filters = [("date", "<=", cutoff), ("stock", "in", stocks)]
    panel = paths["panel.parquet"]
    columns = [name for name in pq.read_schema(panel).names if name != "target"]
    frame = pd.read_parquet(panel, columns=columns, filters=filters)
    # Only mature values pass the Arrow predicate, even if later values are
    # already present as labels in the longer historical snapshot.
    labels = pd.read_parquet(panel, columns=[*_KEYS, "target"], filters=[*filters, ("label_end", "<=", cutoff)])
    frame = frame.merge(labels, on=_KEYS, validate="one_to_one", how="left")
    frame["trade_date"] = frame["trade_date"].where(frame["trade_date"].le(cutoff))
    frame["label_end"] = frame["label_end"].where(frame["label_end"].le(cutoff))
    benchmark = pd.read_parquet(paths["benchmark.parquet"], filters=[("date", "<=", cutoff)]).set_index("date")
    if not pd.DatetimeIndex(benchmark.index).equals(scope.sessions) or not frame["date"].isin(scope.sessions).all():
        raise ExpertSpecError("冻结基线与当前可核验交易日历不一致。")
    if not np.isfinite(benchmark[["open", "close", "daily_return"]]).all().all():
        raise ExpertSpecError("冻结基准缺少模拟所需的价格或收益。")
    dates_type = pq.read_schema(paths["market_status.parquet"]).field("date").type
    market = (pd.read_parquet(paths["market_status.parquet"], filters=filters)
              if str(dates_type).startswith("timestamp") else pd.read_parquet(paths["market_status.parquet"]))
    if not market["date"].isin(scope.sessions).all() or not market["stock"].isin(stocks).all():
        raise ExpertSpecError("冻结交易状态超出所选日期或股票池。")
    coverage = _read(paths["coverage.json"])
    coverage.update(source_run_id=BASELINE_ID, frozen_input_sha256={name: _sha(path) for name, path in paths.items()})
    return FrozenInputs(frame.sort_values(["date", "stock"]).reset_index(drop=True), scope.sessions,
                        benchmark["daily_return"], market, scope.roster, coverage, benchmark["close"])


def _feature_batch(bars: pd.DataFrame, scope: _Scope, breadth: pd.Series, history: pd.DataFrame,
                   actual_status: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    parts = [_stock_features(daily, scope.calendar) for _, daily in bars.groupby("stock", sort=False)]
    frame = pd.concat(parts, ignore_index=True)
    frame = frame.loc[frame["date"].isin(scope.sessions)].reset_index(drop=True)
    frame["breadth"] = frame["date"].map(breadth)
    frame = attach_point_in_time_classifications(frame, history, board_reference=scope.roster["board"])
    for column in ("list_date", "delist_date"):
        frame[column] = frame["stock"].map(scope.roster[column])
    frame = frame.merge(actual_status[[*_KEYS, "market_status"]], on=_KEYS, how="left", validate="one_to_one")
    traded = frame["has_trade_price"] & frame["trade_open"].gt(0) & frame["volume"].gt(0)
    frame["suspended"] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    frame.loc[traded, "suspended"] = False
    frame.loc[frame["market_status"].eq("suspended"), "suspended"] = True
    _attach_labels(frame, scope.sessions, None, benchmark, horizon=5)
    frame["signal_eligible"] &= np.isfinite(frame[[*BASE_FEATURES, *EXTRA_FEATURES]]).all(axis=1) & ~frame["suspended"].fillna(False)
    return frame


def _archive_benchmark(scope: _Scope) -> tuple[pd.DataFrame, dict]:
    warm, cutoff = scope.calendar[[0, -1]]
    source, evidence = _bounded_prices(_ARCHIVE["benchmark"], warm, cutoff)
    with source:
        benchmark = _load_benchmark(source).reindex(scope.calendar)
    if not np.isfinite(benchmark[["open", "close"]]).all().all():
        raise ExpertSpecError("沪深 300 基准在请求范围缺少已核验的开盘价或收盘价。")
    benchmark["daily_return"] = benchmark["close"].pct_change(fill_method=None)
    return benchmark.rename_axis("date"), evidence


def _archive_inputs(scope: _Scope, raw: pd.DataFrame) -> FrozenInputs:
    stocks = scope.roster.index.tolist()
    warm, cutoff = scope.calendar[[0, -1]]
    bars = restrict_bars(raw.loc[raw["date"].isin(scope.calendar)], scope.roster)
    if bars.empty:
        raise ExpertSpecError("时点股票池缺少原始日线。")
    for column in (*_TRADE_PRICES, "volume", "amount", "turnover"):
        values = pd.to_numeric(bars[column], errors="coerce")
        bars[column] = values.where(np.isfinite(values) & values.gt(0))
    bars["has_trade_price"] = bars["has_trade_price"].eq(1)
    bars[list(_TRADE_PRICES)] = bars[list(_TRADE_PRICES)].where(bars["has_trade_price"], axis=0)
    bars["prev_trade_close"] = bars.groupby("stock", sort=False)["trade_close"].transform(lambda values: values.ffill().shift())
    hfq, adjusted_evidence = _bounded_prices(_ARCHIVE["adjusted"], warm, cutoff)
    with hfq:
        adjusted, breadth, effective = _market_adjusted(hfq, scope.historical_roster, scope.calendar)
    if not effective:
        raise ExpertSpecError("归档无法证明复权因子的有效时点。")
    bars = bars.merge(adjusted.loc[adjusted["stock"].isin(stocks)], on=_KEYS, how="left", validate="one_to_one")
    del adjusted
    reference, reference_evidence = _bounded_prices(_ARCHIVE["reference"], warm, cutoff)
    with reference:
        bars = _match_execution_reference(bars, reference, stocks)
    history = load_classification_history(_ARCHIVE["classification"]).assign(board="")
    actual_status = _status_evidence([_ARCHIVE[key] for key in ("status_roster", "status_updates", "status_star")], stocks, scope.sessions)
    benchmark, benchmark_evidence = _archive_benchmark(scope)
    if scope.anchor is None:
        batch = bars["stock"].map(pd.Series(np.arange(len(stocks)) // 256, index=stocks))
        frame = pd.concat((_feature_batch(part, scope, breadth, history, actual_status, benchmark)
                           for _, part in bars.groupby(batch, sort=False)), ignore_index=True)
    else:
        frame = _feature_batch(bars, scope, breadth, history, actual_status, benchmark)
    del bars
    coverage = {
        "price_basis": "effective_date_hfq", "breadth_scope": (
            "历史沪深主板及科创板全量档案相邻交易日宽度" if universe_policy(scope.spec) == DEFAULT_UNIVERSE_POLICY
            else f"历史{universe_scope_label(scope.spec)}全量档案相邻交易日宽度"
        ),
        "universe_policy": universe_policy(scope.spec), "scope_label": universe_scope_label(scope.spec),
        "warmup_start": _day(warm), "benchmark": "000300.SH",
        "bounded_price_sources": {"adjusted": adjusted_evidence, "reference": reference_evidence, "benchmark": benchmark_evidence},
        "point_in_time": {"verified": False, "adjusted_effective_dates_verified": True,
                          "historical_publication_timestamps_verified": False, "vendor_revisions_verified": False},
    }
    returns = benchmark["daily_return"].reindex(scope.sessions)
    return FrozenInputs(frame.sort_values(["date", "stock"]).reset_index(drop=True), scope.sessions,
                        returns, actual_status, scope.roster, coverage, benchmark["close"].reindex(scope.sessions))


def prepare_expert_inputs(spec: Mapping | None = None) -> FrozenInputs:
    """Prepare full training history plus the requested historical evaluation dates."""
    scope = _scope(spec)
    if scope.spec["pool"] == "full" and universe_policy(scope.spec) != DEFAULT_UNIVERSE_POLICY:
        require_full_universe_evidence(scope.spec)
    _require_selection(scope)
    if scope.spec["preset"] == ORIGINAL_PRESET:
        return _load_frozen()
    raw = None if scope.use_frozen else _read_raw(scope)
    identity = _identity(scope, raw)
    data = _frozen_subset(scope) if scope.use_frozen else _archive_inputs(scope, raw)
    frame = data.frame
    eligible = frame["signal_eligible"]
    evaluation = frame["date"].isin(scope.evaluation)
    grouped = pd.DataFrame({"stock": frame["stock"], "is_eligible": eligible, "evaluation_eligible": eligible & evaluation})
    by_stock = grouped.groupby("stock").agg(rows=("is_eligible", "size"), eligible_days=("is_eligible", "sum"),
                                            evaluation_eligible_days=("evaluation_eligible", "sum"))
    coverage = data.coverage | {
        "scope": "custom_expert_research", "pool": scope.spec["pool"], "stocks": scope.roster.index.tolist(),
        "loaded_stocks": int(frame["stock"].nunique()), "first_session": _day(scope.sessions[0]),
        "last_session": _day(scope.sessions[-1]), "input_cutoff": _day(scope.sessions[-1]),
        "requested_evaluation_start": scope.spec["start"], "requested_evaluation_end": scope.spec["end"],
        "first_evaluation_session": _day(scope.evaluation[0]), "last_evaluation_session": _day(scope.evaluation[-1]),
        "rows": len(frame), "signal_eligible_rows": int(eligible.sum()),
        "eligible_labeled_rows": int((eligible & frame["target"].notna()).sum()),
        "stock_counts_by_board": scope.roster["board"].value_counts().to_dict(),
        "coverage_by_symbol": by_stock.to_dict("index"),
        "input_identity": identity, "input_version": stable_hash(identity),
    }
    if scope.anchor is None:
        coverage["membership"] = _full_membership(scope)
        coverage["daily_coverage"] = _daily_full_coverage(data)
    if not coverage["signal_eligible_rows"]:
        raise ExpertSpecError("所选股票池缺少同口径、有前置预热的量价特征。")
    return FrozenInputs(data.frame, data.sessions, data.benchmark_returns, data.market_status, data.roster, coverage,
                        data.benchmark_close, data.sector_daily)


def _daily_full_coverage(data: FrozenInputs) -> dict:
    frame, sessions, roster = data.frame, data.sessions, data.roster
    starts = sessions.searchsorted(roster["list_date"].to_numpy())
    ends = sessions.searchsorted(roster["delist_date"].to_numpy())
    active = (np.bincount(starts, minlength=len(sessions) + 1) - np.bincount(ends, minlength=len(sessions) + 1)).cumsum()[:-1]
    raw = frame["has_trade_price"] & frame[list(_TRADE_PRICES)].gt(0).all(axis=1) & frame["volume"].gt(0)
    adjusted = frame[["open", "high", "low", "close"]].gt(0).all(axis=1)
    same_day = pd.DataFrame({
        "observed_rows": 1, "valid_raw": raw, "joint_raw_hfq_reference": raw & adjusted & frame["reference_verified"],
        "signal_eligible": frame["signal_eligible"], "signal_eligible_mature_labels": frame["signal_eligible"] & frame["target"].notna(),
    }).groupby(frame["date"]).sum().reindex(sessions, fill_value=0)
    same_day["active_members"] = active
    return {_day(day): counts for day, counts in same_day.to_dict("index").items()}
