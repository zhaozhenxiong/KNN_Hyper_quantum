"""Next-open portfolio simulation with prior-close order quantities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, fields
from math import floor, isfinite
from typing import Any

import pandas as pd

from my_strategy.execution import BrokerSimulator, CostModel, LedgerEntry, Order
from my_strategy.execution.order import OrderSide

from .market_evidence import MarketEvidence, execution_evidence_identity, load_market_evidence


@dataclass(frozen=True)
class StudyCostModel(CostModel):
    """Research costs with the A-share transfer levy on both sides."""

    transfer_fee: float = 0.00001

    def fees(self, side: OrderSide, notional: float) -> float:
        return super().fees(side, notional) + notional * self.transfer_fee

    def affordable_buy_shares(self, budget: float, price: float) -> int:
        return max(0, floor(min(
            (budget - self.min_commission) / (price * (1 + self.transfer_fee)),
            budget / (price * (1 + self.commission + self.transfer_fee)),
        )))


@dataclass(frozen=True)
class Strategy:
    top_n: int = 10
    rebalance_days: int = 5
    max_weight: float = 0.10
    cash_buffer: float = 0.05
    max_industry_weight: float = 0.35
    max_board_weight: float = 0.50
    participation: float = 0.02
    min_score: float | None = 0.0
    selection_mode: str = "absolute_q50"

    def __post_init__(self) -> None:
        if self.selection_mode not in {"absolute_q50", "cross_sectional_rank"}:
            raise ValueError("Unknown score selection mode")
        if self.selection_mode == "absolute_q50":
            if self.min_score is None or not isfinite(float(self.min_score)):
                raise ValueError("Absolute q50 selection needs a finite minimum score")
        elif self.min_score is not None:
            raise ValueError("Cross-sectional rank selection requires min_score=None")


@dataclass(frozen=True)
class Mark:
    price: float
    date: pd.Timestamp
    board: str
    industry: str | None


@dataclass(frozen=True)
class Intent:
    stock: str
    side: OrderSide
    signal_date: pd.Timestamp
    mark: Mark
    signal_equity: float
    target_weight: float
    desired_shares: int
    requested_shares: int
    prior_amount: float | None
    reason: str


@dataclass(frozen=True)
class Opening:
    reason: str = ""
    basis: str = "not_applicable"
    lower: float | None = None
    upper: float | None = None
    ratio: float | None = None
    preclose: float | None = None
    preclose_source: str | None = None
    limit_source: str | None = None
    warning_status: str = "unknown"
    warning_source: str | None = None
    warning_available_at: str | None = None
    warning_effective_from: str | None = None
    possible_warning_boundary: bool = False


_ORDER_COLUMNS = (
    "run_id", "signal_date", "date", "stock", "side", "board", "industry",
    "signal_equity", "reference_close", "reference_date", "prior_amount", "target_weight",
    "desired_shares", "requested_shares", "filled_shares", "open", "execution_price",
    "market_status", "market_status_source", "limit_basis", "status", "reason",
    "limit_lower", "limit_upper", "limit_pct_used", "limit_rule_source", "preclose_used", "preclose_source",
    "risk_warning_status", "risk_warning_source", "risk_warning_available_at",
    "risk_warning_effective_from", "possible_risk_warning_boundary",
    "unclamped_execution_price", "slippage_clamped", "modeled_slippage_paid",
)
_POSITION_COLUMNS = (
    "run_id", "date", "stock", "shares", "mark_price", "mark_date", "market_value",
    "board", "industry", "market_status", "market_status_source", "position_state",
    "listed_market_value", "detached_on", "mark_is_stale",
)
_SELECTION_CANDIDATE_COLUMNS = (
    "run_id", "date", "stock", "industry", "score", "global_rank", "qualified",
    "planned_rebalance", "purchase_candidate", "current_shares", "target_weight", "intent_side",
    "selection_reason", "order_count", "fill_count",
)
_EXECUTION_COLUMNS = (
    "date", "stock", "trade_open", "trade_close", "has_trade_price", "amount", "volume", "board", "industry",
    "signal_eligible", "delist_date", "suspended", "limit_up", "limit_down", "limit_pct",
    "provider_preclose", "prev_trade_close",
)


def _positive(value: Any) -> bool:
    return pd.notna(value) and isfinite(float(value)) and float(value) > 0


def _name(value: Any) -> str | None:
    return str(value) if pd.notna(value) and value not in ("", "unknown", "unclassified") else None


def _lot_quantity(quantity: int, board: str, side: OrderSide, held: int) -> int:
    if side == "SELL" and quantity >= held:
        return held
    if board == "star":
        return quantity if quantity >= 200 else 0
    if board == "bse":
        return quantity if quantity >= 100 else 0
    return max(0, quantity // 100 * 100)


def _qualified(known: pd.DataFrame, strategy: Strategy) -> pd.DataFrame:
    if strategy.selection_mode == "cross_sectional_rank":
        return known
    return known.loc[known["score"].gt(strategy.min_score)]


def _targets(known: pd.DataFrame, marks: dict[str, Mark], day: pd.Timestamp, strategy: Strategy) -> dict[str, float]:
    targets: dict[str, float] = {}
    boards: Counter[str] = Counter()
    industries: Counter[str] = Counter()
    weight = min(strategy.max_weight, (1 - strategy.cash_buffer) / strategy.top_n)
    candidates = _qualified(known, strategy)
    for row in candidates.sort_values(["score", "stock"], ascending=[False, True]).itertuples(index=False):
        mark = marks.get(row.stock)
        if mark is None or mark.date != day:
            continue
        room = min(weight, strategy.max_board_weight - boards[mark.board])
        if mark.industry is not None:
            room = min(room, strategy.max_industry_weight - industries[mark.industry])
        if room > 1e-12:
            targets[row.stock] = room
            boards[mark.board] += room
            if mark.industry is not None:
                industries[mark.industry] += room
        if len(targets) >= strategy.top_n:
            break
    return targets


def _candidate_ranks(known: pd.DataFrame, strategy: Strategy) -> dict[str, int]:
    ranked = _qualified(known, strategy).sort_values(["score", "stock"], ascending=[False, True])
    return {stock: rank for rank, stock in enumerate(ranked["stock"], start=1)}


def _purchase_candidates(
    known: pd.DataFrame, marks: dict[str, Mark], day: pd.Timestamp, strategy: Strategy,
) -> set[str]:
    """Take the global Top-N with current marks before concentration constraints."""
    ranked = _qualified(known, strategy).sort_values(["score", "stock"], ascending=[False, True])
    selected = []
    for stock in ranked["stock"]:
        mark = marks.get(stock)
        if mark is not None and mark.date == day:
            selected.append(stock)
            if len(selected) >= strategy.top_n:
                break
    return set(selected)


def _selection_rows(
    *, run_id: str, day: pd.Timestamp, eligible: pd.DataFrame, known: pd.DataFrame,
    qualified: pd.DataFrame, targets: dict[str, float], pending: list[Intent],
    broker: BrokerSimulator, current: dict[str, Any], marks: dict[str, Mark],
    strategy: Strategy, planned: bool, day_reason: str,
) -> list[dict[str, Any]]:
    ranks = _candidate_ranks(known, strategy) if planned else {}
    purchases = _purchase_candidates(known, marks, day, strategy) if planned else set()
    intents = {intent.stock: intent.side for intent in pending}
    eligible_stocks = set(eligible["stock"])
    qualified_stocks = set(qualified["stock"])
    scores = known.set_index("stock")["score"].to_dict()
    stocks = sorted(eligible_stocks | set(targets) | set(intents)
                    | {stock for stock, shares in broker.positions.items() if shares})
    records = []
    for stock in stocks:
        row = current.get(stock)
        raw_industry = getattr(row, "industry", None) if row is not None else None
        industry = _name(raw_industry) or _name(getattr(marks.get(stock), "industry", None)) or "unknown"
        if not planned:
            reason = day_reason or "NOT_REBALANCE_DAY"
        elif stock in purchases:
            reason = "PURCHASE_CANDIDATE_PRE_CONCENTRATION"
        elif stock in targets:
            reason = "TARGET_AFTER_CONCENTRATION"
        elif stock in intents:
            reason = "EXIT_OR_REBALANCE"
        elif stock in qualified_stocks:
            reason = "QUALIFIED_NOT_TOP_N"
        else:
            reason = day_reason or "NOT_QUALIFIED"
        records.append({
            "run_id": run_id, "date": day.date().isoformat(), "stock": stock, "industry": industry,
            "score": scores.get(stock, float("nan")), "global_rank": ranks.get(stock),
            "qualified": stock in qualified_stocks, "planned_rebalance": planned,
            "purchase_candidate": stock in purchases, "current_shares": int(broker.shares(stock)),
            "target_weight": float(targets.get(stock, 0.0)), "intent_side": intents.get(stock, ""),
            "selection_reason": reason,
        })
    return records


def _plan(
    known: pd.DataFrame,
    current: dict[str, Any],
    marks: dict[str, Mark],
    broker: BrokerSimulator,
    day: pd.Timestamp,
    strategy: Strategy,
    targets: dict[str, float] | None = None,
) -> list[Intent]:
    targets = _targets(known, marks, day, strategy) if targets is None else targets
    equity = broker.equity({stock: marks[stock].price for stock, shares in broker.positions.items() if shares})
    intents = []
    exiting = sorted(stock for stock, shares in broker.positions.items() if shares and stock not in targets)
    for stock in [*targets, *exiting]:
        mark = marks[stock]
        weight = targets.get(stock, 0.0)
        held = broker.shares(stock)
        change = floor(equity * weight / mark.price + 1e-9) - held
        if change == 0:
            continue
        side: OrderSide = "BUY" if change > 0 else "SELL"
        desired = _lot_quantity(abs(change), mark.board, side, held)
        if not desired:
            continue
        amount = getattr(current.get(stock), "amount", None)
        prior_amount = float(amount) if _positive(amount) else None
        if mark.date != day or prior_amount is None:
            requested, reason = 0, "no_previous_session_liquidity"
        else:
            max_shares = floor(prior_amount * strategy.participation / mark.price)
            requested = _lot_quantity(min(desired, max_shares), mark.board, side, held)
            reason = "liquidity_below_lot" if not requested else ""
        intents.append(Intent(stock, side, day, mark, equity, weight, desired, requested, prior_amount, reason))
    return sorted(intents, key=lambda intent: intent.side == "BUY")


def _market_status(
    stock: str,
    day: pd.Timestamp,
    row: Any,
    statuses: dict[tuple[str, pd.Timestamp], tuple[str, str]],
    delisted: dict[str, pd.Timestamp],
    evidence: MarketEvidence,
) -> tuple[str, str]:
    day_date = day.date()
    if official := evidence.event(stock, "delisted", day_date):
        return "delisted", official.source_url
    delist_date = delisted.get(stock, getattr(row, "delist_date", None))
    if pd.notna(delist_date) and day >= pd.Timestamp(delist_date):
        return "delisted", "listing_history"
    if official := evidence.event(stock, "suspended", day_date):
        return "suspended", official.source_url
    explicit = statuses.get((stock, day))
    if explicit is not None:
        return explicit
    suspended = getattr(row, "suspended", None)
    if pd.notna(suspended):
        return ("suspended", "bar_status") if suspended else ("tradable", "bar_status")
    return "unknown", "unavailable"


def _limit_bounds(previous: float, ratio: float) -> tuple[float, float]:
    return tuple(floor(previous * (1 + change) * 100 + 0.500000001) / 100 for change in (-ratio, ratio))


def _opening_check(row: Any, intent: Intent, day: pd.Timestamp, status: str, evidence: MarketEvidence) -> Opening:
    if status in {"suspended", "delisted"}:
        return Opening(status)
    opening = getattr(row, "trade_open", None)
    has_trade = getattr(row, "has_trade_price", True)
    if not _positive(opening) or pd.isna(has_trade) or not has_trade:
        return Opening("no_open", "unavailable")
    volume = getattr(row, "volume", None)
    if pd.notna(volume) and volume <= 0:
        return Opening("zero_volume_bar", "unavailable")
    stock_day = day.date()
    board = intent.mark.board
    warning = evidence.event(intent.stock, "risk_warning", stock_day)
    rule = evidence.rule(board, stock_day)
    provenance = {"warning_status": "confirmed" if warning else "unknown",
                  "warning_source": warning.source_url if warning else None,
                  "warning_available_at": warning.available_at.isoformat() if warning else None,
                  "warning_effective_from": warning.first_day.isoformat() if warning else None}
    flag = getattr(row, "limit_up" if intent.side == "BUY" else "limit_down", None)
    if pd.notna(flag) and bool(flag):
        return Opening("price_limit", "observed_open_limit_flag", limit_source="input_open_limit_flag", **provenance)
    ratio = getattr(row, "limit_pct", None)
    if pd.notna(ratio):
        limit_basis, limit_source = "observed_limit_ratio", "input_limit_ratio"
        if ratio == 0:
            return Opening(basis=limit_basis, ratio=0.0, limit_source=limit_source, **provenance)
        if not 0 < ratio < 1:
            return Opening("invalid_limit_ratio", "unavailable", **provenance)
    elif warning and rule:
        limit_basis, limit_source, ratio = "verified_risk_warning_limit", rule.evidence.source_url, rule.ratio
    else:
        limit_basis, limit_source = "approximate_board_limit", "board_default_without_historical_warning_status"
        ratio = 0.10
        if board == "star" or (board == "chinext" and day >= pd.Timestamp("2020-08-24")):
            ratio = 0.20
        if board == "bse":
            ratio = 0.30
    previous_close = getattr(row, "provider_preclose", None)
    reference_source = "provider_preclose"
    if not _positive(previous_close):
        previous_close = getattr(row, "prev_trade_close", None)
        reference_source = "previous_raw_close_proxy"
    if not _positive(previous_close):
        previous_close = intent.mark.price if intent.mark.date == intent.signal_date else None
        reference_source = "signal_raw_close_proxy"
    if not _positive(previous_close):
        if pd.notna(flag) and not bool(flag):
            return Opening(basis="observed_open_limit_flag", limit_source="input_open_limit_flag", **provenance)
        return Opening("no_limit_reference", "unavailable", limit_source=limit_source, **provenance)
    lower, upper = _limit_bounds(float(previous_close), ratio)
    buy = intent.side == "BUY"
    at_limit = opening >= upper - 1e-8 if buy else opening <= lower + 1e-8
    cause = ""
    possible_warning_boundary = False
    if pd.notna(flag) and not bool(flag):
        limit_basis, limit_source = "observed_open_limit_flag", "input_open_limit_flag"
        if not lower < opening < upper:
            lower = upper = None
    elif limit_basis != "approximate_board_limit" and not lower - 1e-8 <= opening <= upper + 1e-8:
        cause = "open_outside_price_band"
    elif at_limit:
        cause = "approximate_price_limit" if limit_basis == "approximate_board_limit" else "price_limit"
    elif limit_basis == "approximate_board_limit" and rule and rule.ratio < ratio:
        possible_lower, possible_upper = _limit_bounds(float(previous_close), rule.ratio)
        possible_limit = possible_upper if buy else possible_lower
        possible_warning_boundary = abs(opening - possible_limit) <= 1e-8
    if limit_basis == "approximate_board_limit" and not cause and not lower <= opening <= upper:
        lower = upper = None
    return Opening(cause, limit_basis, lower, upper, ratio, float(previous_close), reference_source, limit_source,
                   possible_warning_boundary=possible_warning_boundary, **provenance)


def _risk_room(intent: Intent, broker: BrokerSimulator, marks: dict[str, Mark], strategy: Strategy) -> int:
    board_value = industry_value = 0.0
    for stock, held in broker.positions.items():
        if not held:
            continue
        mark = marks[stock]
        if mark.board == intent.mark.board:
            board_value += mark.price * held
        if intent.mark.industry is not None and mark.industry == intent.mark.industry:
            industry_value += mark.price * held
    room = intent.signal_equity * strategy.max_board_weight - board_value
    if intent.mark.industry is not None:
        room = min(room, intent.signal_equity * strategy.max_industry_weight - industry_value)
    return max(0, floor(room / intent.mark.price + 1e-9))


def _fill(
    intent: Intent,
    row: Any,
    market_state: tuple[str, str],
    marks: dict[str, Mark],
    broker: BrokerSimulator,
    cost: StudyCostModel,
    day: pd.Timestamp,
    strategy: Strategy,
    evidence: MarketEvidence,
) -> dict[str, Any]:
    date = day.date().isoformat()
    signal_date = intent.signal_date.date().isoformat()
    market_status, source = market_state
    opening = getattr(row, "trade_open", None)
    status = "pretrade_rejected"
    reason = market_status if market_status in {"suspended", "delisted"} else intent.reason
    check = Opening()
    shares, price, unclamped = 0, None, None
    if intent.requested_shares:
        status = "rejected"
        check = _opening_check(row, intent, day, market_status, evidence)
        reason = check.reason
        if not reason:
            unclamped = cost.execution_price(intent.side, opening)
            price = unclamped if check.lower is None else min(check.upper, max(check.lower, unclamped))
            held = broker.shares(intent.stock)
            shares = intent.requested_shares
            if intent.side == "BUY":
                budget = broker.cash - intent.signal_equity * strategy.cash_buffer
                affordable = cost.affordable_buy_shares(budget, price)
                headroom = _risk_room(intent, broker, marks, strategy)
                allowed = min(affordable, headroom)
                if not held and sum(quantity > 0 for quantity in broker.positions.values()) >= strategy.top_n:
                    shares, reason = 0, "position_limit"
                elif allowed < shares:
                    reason = "cash_buffer" if affordable <= headroom else "concentration"
                    shares = _lot_quantity(allowed, intent.mark.board, intent.side, held)
            if shares:
                fill = broker.submit(
                    Order(date, intent.stock, intent.side, shares, price, f"signal_date={signal_date}", intent.target_weight),
                    mark_price=float(opening), equity_before=intent.signal_equity,
                )
                if fill is None:
                    shares, reason = 0, "broker_rejected"
                else:
                    status = "filled" if shares == intent.requested_shares else "partial"
    return {
        "run_id": broker.run_id, "signal_date": signal_date, "date": date, "stock": intent.stock,
        "side": intent.side, "board": intent.mark.board, "industry": intent.mark.industry,
        "signal_equity": intent.signal_equity, "reference_close": intent.mark.price,
        "reference_date": intent.mark.date.date().isoformat(), "prior_amount": intent.prior_amount,
        "target_weight": intent.target_weight, "desired_shares": intent.desired_shares,
        "requested_shares": intent.requested_shares, "filled_shares": shares,
        "open": float(opening) if _positive(opening) else None, "execution_price": price if shares else None,
        "market_status": market_status, "market_status_source": source, "limit_basis": check.basis,
        "status": status, "reason": reason,
        "limit_lower": check.lower, "limit_upper": check.upper, "limit_pct_used": check.ratio,
        "limit_rule_source": check.limit_source, "preclose_used": check.preclose, "preclose_source": check.preclose_source,
        "risk_warning_status": check.warning_status, "risk_warning_source": check.warning_source,
        "risk_warning_available_at": check.warning_available_at, "risk_warning_effective_from": check.warning_effective_from,
        "possible_risk_warning_boundary": check.possible_warning_boundary, "unclamped_execution_price": unclamped,
        "slippage_clamped": bool(shares and price != unclamped),
        "modeled_slippage_paid": abs(price - float(opening)) * shares if shares else 0.0,
    }


def scaled_study_cost(cost_multiplier: float, base_cost_model: StudyCostModel | None = None) -> StudyCostModel:
    """Scale commission, its minimum and slippage while preserving statutory fees."""
    defaults = base_cost_model or StudyCostModel()
    return StudyCostModel(
        commission=defaults.commission * cost_multiplier, stamp_tax=defaults.stamp_tax,
        slippage=defaults.slippage * cost_multiplier, min_commission=defaults.min_commission * cost_multiplier,
        transfer_fee=defaults.transfer_fee,
    )


def simulate(
    frame: pd.DataFrame,
    scores: pd.Series,
    sessions: pd.DatetimeIndex,
    benchmark_returns: pd.Series,
    *,
    run_id: str,
    cost_multiplier: float = 1.0,
    initial_cash: float = 1_000_000.0,
    strategy: Strategy = Strategy(),
    market_status: pd.DataFrame | None = None,
    roster: pd.DataFrame | None = None,
    target_exposure: pd.Series | None = None,
    base_cost_model: StudyCostModel | None = None,
    market_evidence: MarketEvidence | None = None,
) -> dict[str, Any]:
    """Simulate a fresh account in one evaluation fold using raw daily prices.

    Scores are indexed like the input frame. Plan orders only after a fold
    session closes and execute once, at the next session's open. Market
    status and price-limit checks use evidence available at that execution.
    """
    if len(sessions) < 2 or not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError("Execution needs at least two unique, ordered evaluation sessions")
    if not _positive(cost_multiplier) or not _positive(initial_cash):
        raise ValueError("Execution needs positive initial cash and cost multiplier")
    if target_exposure is None:
        exposure = pd.Series(1.0, index=sessions)
    else:
        supplied = target_exposure.copy()
        supplied.index = pd.to_datetime(supplied.index)
        if supplied.index.has_duplicates:
            raise ValueError("Target exposure needs unique dates")
        exposure = pd.to_numeric(supplied.reindex(sessions), errors="coerce")
        if exposure.isna().any() or not exposure.map(lambda value: isfinite(value) and 0 <= value <= 1).all():
            raise ValueError("Target exposure must be finite and between zero and one for every session")
    exposure_by_day = exposure.astype(float).to_dict()
    cost = scaled_study_cost(cost_multiplier, base_cost_model)
    broker = BrokerSimulator(initial_cash=initial_cash, cost_model=cost, run_id=run_id)
    evidence = market_evidence if market_evidence is not None else load_market_evidence()
    columns = [column for column in _EXECUTION_COLUMNS if column in frame]
    rows = frame.loc[frame["date"].isin(sessions), columns].copy()
    rows["score"] = scores.reindex(rows.index)
    by_day = {day: group for day, group in rows.groupby("date", sort=False)}
    statuses = {} if market_status is None else {
        (row.stock, pd.Timestamp(row.date)): (row.market_status, row.source)
        for row in market_status.itertuples(index=False)
    }
    delisted = {} if roster is None else pd.to_datetime(roster["delist_date"]).dropna().to_dict()
    marks: dict[str, Mark] = {}
    detached: dict[str, tuple[int, Mark, str, str]] = {}
    pending: list[Intent] = []
    daily, positions, orders, selection, candidate_records = [], [], [], [], []
    benchmark = float(initial_cash)
    for index, day in enumerate(sessions):
        date = day.date().isoformat()
        group = by_day.get(day)
        current = {} if group is None else {row.stock: row for row in group.itertuples(index=False)}
        previous_equity = daily[-1]["equity"] if daily else initial_cash
        fill_start = len(broker.ledger.entries)
        order_start = len(orders)
        transitions = 0
        for stock, held in list(broker.positions.items()):
            if held:
                state, source = _market_status(stock, day, current.get(stock), statuses, delisted, evidence)
                if state == "delisted":
                    detached[stock] = (held, marks[stock], date, source)
                    broker.positions.pop(stock)
                    broker.avg_cost.pop(stock, None)
                    transitions += 1
        for intent in pending:
            row = current.get(intent.stock)
            state = _market_status(intent.stock, day, row, statuses, delisted, evidence)
            orders.append(_fill(intent, row, state, marks, broker, cost, day, strategy, evidence))
        pending = []
        for stock, row in current.items():
            status, _ = _market_status(stock, day, row, statuses, delisted, evidence)
            has_trade = getattr(row, "has_trade_price", True)
            volume = getattr(row, "volume", None)
            if (status not in {"suspended", "delisted"} and pd.notna(has_trade) and has_trade
                    and (pd.isna(volume) or volume > 0) and _positive(row.trade_close)):
                marks[stock] = Mark(
                    float(row.trade_close), day, _name(getattr(row, "board", None)) or "unknown",
                    _name(getattr(row, "industry", None)),
                )
        listed_value = detached_value = 0.0
        stale_count = 0
        for stock, held in sorted(broker.positions.items()):
            if not held:
                continue
            mark = marks[stock]
            state, source = _market_status(stock, day, current.get(stock), statuses, delisted, evidence)
            value = mark.price * held
            listed_value += value
            stale_count += mark.date != day
            positions.append({
                "run_id": run_id, "date": date, "stock": stock, "shares": held, "mark_price": mark.price,
                "mark_date": mark.date.date().isoformat(), "market_value": value, "board": mark.board,
                "industry": mark.industry, "market_status": state, "market_status_source": source,
                "position_state": "listed", "listed_market_value": value, "detached_on": None,
                "mark_is_stale": mark.date != day,
            })
        for stock, (held, mark, changed, source) in sorted(detached.items()):
            value = mark.price * held
            detached_value += value
            stale_count += 1
            positions.append({
                "run_id": run_id, "date": date, "stock": stock, "shares": held, "mark_price": mark.price,
                "mark_date": mark.date.date().isoformat(), "market_value": value, "board": mark.board,
                "industry": mark.industry, "market_status": "delisted", "market_status_source": source,
                "position_state": "stranded_delisted", "listed_market_value": 0.0,
                "detached_on": changed, "mark_is_stale": True,
            })
        fills = broker.ledger.entries[fill_start:]
        notional = sum(fill.price * fill.shares for fill in fills)
        if index:
            index_return = benchmark_returns.get(day, float("nan"))
            benchmark *= 1 + float(index_return) if pd.notna(index_return) else float("nan")
        held_value = listed_value + detached_value
        equity = broker.cash + held_value
        eligible = rows.iloc[0:0] if group is None else group
        if "signal_eligible" in eligible:
            eligible = eligible.loc[eligible["signal_eligible"].eq(True)]
        known = eligible.loc[eligible["score"].map(lambda value: pd.notna(value) and isfinite(value))]
        qualified = _qualified(known, strategy)
        day_exposure = exposure_by_day[day]
        planned = group is not None and index % strategy.rebalance_days == 0 and index + 1 < len(sessions)
        if index + 1 >= len(sessions):
            selection_reason = "END_OF_EVALUATION"
        elif group is None:
            selection_reason = "NO_SESSION_ROWS"
        elif not planned:
            selection_reason = "NOT_REBALANCE_DAY"
        else:
            selection_reason = ""
        selection.append({
            "run_id": run_id, "date": date, "selection_mode": strategy.selection_mode,
            "score_threshold": strategy.min_score, "planned_rebalance": planned, "target_exposure": day_exposure,
            "eligible_rows": len(eligible), "finite_scores": len(known), "qualified_scores": len(qualified),
            "target_count": 0, "intent_count": 0, "order_records": len(orders) - order_start,
            "fill_count": len(fills), "selection_reason": selection_reason,
        })
        daily.append({
            "run_id": run_id, "date": date, "cash": broker.cash, "market_value": held_value,
            "equity": equity, "net_return": equity / previous_equity - 1, "benchmark": benchmark,
            "fees": sum(fill.fee for fill in fills), "cash_flow": sum(fill.cash_flow for fill in fills),
            "traded_notional": notional, "turnover": notional / previous_equity,
            "stale_positions": stale_count, "delisted_positions": len(detached),
            "listed_market_value": listed_value, "stranded_delisted_value": detached_value,
            "zero_recovery_equity": broker.cash + listed_value, "delist_transitions": transitions,
            "modeled_slippage_paid": sum(order["modeled_slippage_paid"] for order in orders[order_start:]),
            "selection_mode": strategy.selection_mode, "score_threshold": strategy.min_score,
            "target_exposure": day_exposure,
            "eligible_rows": len(eligible), "finite_scores": len(known), "qualified_scores": len(qualified),
            "order_records": len(orders) - order_start, "fill_count": len(fills),
        })
        targets: dict[str, float] = {}
        if planned:
            targets = {} if known.empty else _targets(known, marks, day, strategy)
            if day_exposure < 1:
                targets = {stock: weight * day_exposure for stock, weight in targets.items()}
                targets = {stock: weight for stock, weight in targets.items() if weight > 1e-12}
            if not known.empty:
                pending = _plan(known, current, marks, broker, day, strategy, targets=targets)
            if known.empty:
                selection_reason = "NO_FINITE_SCORES" if len(eligible) else "NO_ELIGIBLE_ROWS"
            elif qualified.empty:
                selection_reason = "NO_QUALIFIED_CANDIDATES"
            elif not targets and day_exposure == 0:
                selection_reason = "DEFENSIVE_CASH"
            elif not targets:
                selection_reason = "NO_TARGET_WEIGHTS"
            elif not pending:
                selection_reason = "NO_ORDER_INTENTS"
            selection[-1].update(
                target_count=len(targets), intent_count=len(pending), selection_reason=selection_reason,
            )
        candidate_records.extend(_selection_rows(
            run_id=run_id, day=day, eligible=eligible, known=known, qualified=qualified,
            targets=targets, pending=pending, broker=broker, current=current, marks=marks,
            strategy=strategy, planned=planned, day_reason=selection_reason,
        ))

    daily_frame = pd.DataFrame(daily)
    selection_frame = pd.DataFrame(selection)
    trades = broker.ledger.to_frame()
    trades = trades.reindex(columns=[field.name for field in fields(LedgerEntry)])
    trades["signal_date"] = trades["reason"].astype("string").str.removeprefix("signal_date=")
    trades["transfer_fee_paid"] = trades["price"] * trades["shares"] * cost.transfer_fee
    equity = daily_frame["equity"]
    total_return = float(equity.iloc[-1] / initial_cash - 1)
    benchmark_return = float(benchmark / initial_cash - 1) if isfinite(benchmark) else None
    counts = Counter(order["status"] for order in orders)
    submitted = counts["filled"] + counts["partial"] + counts["rejected"]
    metrics = {
        "total_return": total_return, "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return if benchmark_return is not None else None,
        "max_drawdown": float((equity / equity.cummax() - 1).min()),
        "fees": float(daily_frame["fees"].sum()), "turnover": float(daily_frame["turnover"].sum()),
        "mean_daily_turnover": float(daily_frame["turnover"].mean()), "trade_count": len(trades),
        "order_count": submitted, "rejected_order_count": counts["rejected"],
        "partial_order_count": counts["partial"], "pretrade_rejected_count": counts["pretrade_rejected"],
        "reject_rate": counts["rejected"] / submitted if submitted else None,
        "reasons": dict(Counter(order["reason"] for order in orders if order["reason"])),
        "selection_mode": strategy.selection_mode, "score_threshold": strategy.min_score,
        "eligible_stock_days": int(selection_frame["eligible_rows"].sum()),
        "finite_score_stock_days": int(selection_frame["finite_scores"].sum()),
        "qualified_stock_days": int(selection_frame["qualified_scores"].sum()),
        "planned_rebalance_count": int(selection_frame["planned_rebalance"].sum()),
        "rebalances_with_targets": int(selection_frame["target_count"].gt(0).sum()),
        "rebalances_with_intents": int(selection_frame["intent_count"].gt(0).sum()),
        "no_candidate_rebalances": int(selection_frame["selection_reason"].eq("NO_QUALIFIED_CANDIDATES").sum()),
        "defensive_cash_rebalances": int(selection_frame["selection_reason"].eq("DEFENSIVE_CASH").sum()),
        "mean_target_exposure": float(selection_frame["target_exposure"].mean()),
        "fills_with_unknown_market_status": sum(order["market_status"] == "unknown" and order["filled_shares"] > 0 for order in orders),
        "positions_at_end": sum(shares > 0 for shares in broker.positions.values()) + len(detached),
        "listed_positions_at_end": sum(shares > 0 for shares in broker.positions.values()),
        "stale_positions_at_end": int(daily_frame["stale_positions"].iloc[-1]),
        "delisted_positions_at_end": int(daily_frame["delisted_positions"].iloc[-1]),
        "delist_transition_count": int(daily_frame["delist_transitions"].sum()),
        "stranded_delisted_value_at_end": float(daily_frame["stranded_delisted_value"].iloc[-1]),
        "zero_recovery_return": float(daily_frame["zero_recovery_equity"].iloc[-1] / initial_cash - 1),
        "modeled_slippage_paid": float(daily_frame["modeled_slippage_paid"].sum()),
        "fills_with_unknown_risk_warning_status": sum(order["risk_warning_status"] == "unknown" and order["filled_shares"] > 0 for order in orders),
        "fills_at_possible_risk_warning_boundary": sum(order["possible_risk_warning_boundary"] and order["filled_shares"] > 0 for order in orders),
        "slippage_clamp_count": sum(order["slippage_clamped"] for order in orders),
    }
    order_frame = pd.DataFrame(orders, columns=_ORDER_COLUMNS)
    fill_audit = order_frame.loc[order_frame["filled_shares"].gt(0), [
        "open", "limit_basis", "limit_lower", "limit_upper", "unclamped_execution_price",
        "slippage_clamped", "modeled_slippage_paid", "risk_warning_status", "possible_risk_warning_boundary",
    ]].reset_index(drop=True)
    trades = trades.join(fill_audit)
    if base_cost_model is not None:
        recorded = {
            "commission_rate": cost.commission, "minimum_commission": cost.min_commission,
            "slippage_rate": cost.slippage, "sell_stamp_tax_rate": cost.stamp_tax, "transfer_fee_rate": cost.transfer_fee,
        }
        order_frame = order_frame.assign(**recorded)
        trades = trades.assign(**recorded)
        notional = trades["price"].astype(float) * trades["shares"].astype(float)
        trades["commission_paid"] = (notional * cost.commission).clip(lower=cost.min_commission)
        trades["stamp_tax_paid"] = (notional * cost.stamp_tax).where(trades["action"].isin(("SELL", "REDUCE")), 0.0)
        daily_frame = daily_frame.assign(**recorded)
        metrics["effective_cost"] = recorded
    selection_daily = selection_frame[[
        "run_id", "date", "selection_mode", "score_threshold", "planned_rebalance", "target_exposure", "eligible_rows",
        "finite_scores", "qualified_scores", "target_count", "intent_count", "order_records", "fill_count",
        "selection_reason",
    ]].copy()
    selection_candidates = pd.DataFrame(
        candidate_records, columns=[column for column in _SELECTION_CANDIDATE_COLUMNS if column not in {"order_count", "fill_count"}],
    ).drop_duplicates(["date", "stock"], keep="last")
    execution_counts = order_frame.loc[order_frame["status"].isin(("filled", "partial", "rejected"))].assign(
        filled=lambda frame: frame["filled_shares"].gt(0),
    ).groupby(["signal_date", "stock"], as_index=False).agg(
        order_count=("stock", "size"), fill_count=("filled", "sum"),
    ).rename(columns={"signal_date": "date"})
    selection_candidates = selection_candidates.merge(
        execution_counts, on=["date", "stock"], how="left", validate="one_to_one",
    )
    selection_candidates[["order_count", "fill_count"]] = selection_candidates[["order_count", "fill_count"]].fillna(0)
    selection_candidates = selection_candidates.reindex(columns=_SELECTION_CANDIDATE_COLUMNS)
    selection_candidates["global_rank"] = selection_candidates["global_rank"].astype("Int64")
    for column in ("qualified", "planned_rebalance", "purchase_candidate"):
        selection_candidates[column] = selection_candidates[column].astype(bool)
    for column in ("order_count", "fill_count"):
        selection_candidates[column] = selection_candidates[column].astype(int)
    from .sector_features import summarize_selection_candidates

    sector_candidate_daily = summarize_selection_candidates(
        selection_candidates, sessions=sessions,
        rebalance_dates=selection_frame.loc[selection_frame["planned_rebalance"], "date"],
    )
    return {
        "metrics": metrics, "daily": daily_frame, "trades": trades,
        "orders": order_frame,
        "positions": pd.DataFrame(positions, columns=_POSITION_COLUMNS),
        "selection_daily": selection_daily,
        "selection_candidates": selection_candidates,
        "sector_candidate_daily": sector_candidate_daily,
    }
