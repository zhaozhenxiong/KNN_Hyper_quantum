"""Exchange-dated market-state evidence used by the daily execution simulator."""

from dataclasses import dataclass
from datetime import date, datetime, time
from hashlib import file_digest
import json
from pathlib import Path
from zoneinfo import ZoneInfo


CATALOG = Path(__file__).with_suffix(".json")
SHANGHAI = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Evidence:
    first_day: date
    last_day: date | None
    available_at: datetime
    source_url: str
    availability_basis: str

    def known_at_open(self, day: date) -> bool:
        return (self.first_day <= day and (self.last_day is None or day <= self.last_day)
                and self.available_at <= datetime.combine(day, time(9, 30), tzinfo=SHANGHAI))


@dataclass(frozen=True)
class RiskRule:
    evidence: Evidence
    ratio: float


@dataclass(frozen=True)
class MarketEvidence:
    events: dict[tuple[str, str], tuple[Evidence, ...]]
    rules: dict[str, tuple[RiskRule, ...]]

    def event(self, stock: str, kind: str, day: date) -> Evidence | None:
        return next((record for record in self.events.get((stock, kind), ()) if record.known_at_open(day)), None)

    def rule(self, board: str, day: date) -> RiskRule | None:
        return next((rule for rule in self.rules.get(board, ()) if rule.evidence.known_at_open(day)), None)


def _interval(row: dict) -> Evidence:
    first = date.fromisoformat(row["effective_from"])
    last = date.fromisoformat(row["effective_through"]) if row["effective_through"] is not None else None
    stamp = datetime.fromisoformat(row["available_at"])
    url = row["source_url"]
    if (stamp.tzinfo is None or (last is not None and last < first) or not url.startswith("https://")):
        raise ValueError("Invalid historical market evidence interval")
    return Evidence(first, last, stamp, url, row.get("availability_basis", "exchange_rule_publication_date"))


def load_market_evidence() -> MarketEvidence:
    value = json.loads(CATALOG.read_text(encoding="utf-8"))
    if value["schema_version"] != 1:
        raise ValueError("Unknown market evidence schema")
    rules: dict[str, tuple[RiskRule, ...]] = {}
    for row in value["rules"]:
        ratio = float(row["risk_warning_limit"])
        if not 0 < ratio < 1:
            raise ValueError("Invalid historical risk-warning ratio")
        board = row["board"]
        rules[board] = (*rules.get(board, ()), RiskRule(_interval(row), ratio))
    events: dict[tuple[str, str], tuple[Evidence, ...]] = {}
    for row in value["events"]:
        key = (row["stock"], row["kind"])
        if key[1] not in {"risk_warning", "suspended", "delisted"}:
            raise ValueError("Unknown historical market event")
        events[key] = (*events.get(key, ()), _interval(row))
    return MarketEvidence(events, rules)


def execution_evidence_identity() -> dict[str, str]:
    """Fingerprint dated market evidence and the local decoder for replay cache keys."""
    output = {}
    for path in (Path(__file__), CATALOG):
        with path.open("rb") as handle:
            output[path.name] = file_digest(handle, "sha256").hexdigest()
    return output
