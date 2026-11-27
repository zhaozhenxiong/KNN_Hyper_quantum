"""Trading cost model."""

from __future__ import annotations

from dataclasses import dataclass

from my_strategy.execution.order import OrderSide


@dataclass(frozen=True)
class CostModel:
    commission: float = 0.0003
    stamp_tax: float = 0.0005
    slippage: float = 0.0005
    min_commission: float = 5.0

    def execution_price(self, side: OrderSide, reference_price: float) -> float:
        if side == "BUY":
            return float(reference_price) * (1 + self.slippage)
        return float(reference_price) * (1 - self.slippage)

    def fees(self, side: OrderSide, notional: float) -> float:
        commission_fee = max(float(notional) * self.commission, self.min_commission)
        tax = float(notional) * self.stamp_tax if side == "SELL" else 0.0
        return commission_fee + tax
