"""Order schema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class Order:
    date: str
    symbol: str
    side: OrderSide
    shares: int
    price: float
    reason: str = ""
    target_weight: float = 0.0
