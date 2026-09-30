"""Discounted cash-flow helpers for monthly, assumption-based economics."""
from __future__ import annotations

import math
from collections.abc import Iterable


def annual_to_monthly_discount_rate(annual_rate: float) -> float:
    """Convert an effective annual rate to an effective monthly rate."""
    if not math.isfinite(annual_rate) or annual_rate < 0:
        raise ValueError("annual_discount_rate must be finite and nonnegative")
    return (1.0 + annual_rate) ** (1.0 / 12.0) - 1.0


def npv(
    monthly_cash_flows: Iterable[float],
    monthly_discount_rate: float,
    contribution_margin: float = 1.0,
    offer_cost: float = 0.0,
) -> float:
    """Discount monthly revenues after margin and subtract an immediate cost.

    Cash flow element zero is month 1; ``offer_cost`` is paid at time zero.
    Pass contribution cash flows with the default margin of 1.0.
    """
    if not math.isfinite(monthly_discount_rate) or monthly_discount_rate < 0:
        raise ValueError("monthly_discount_rate must be finite and nonnegative")
    if not math.isfinite(contribution_margin) or not 0 <= contribution_margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1")
    if not math.isfinite(offer_cost) or offer_cost < 0:
        raise ValueError("offer_cost must be finite and nonnegative")
    total = -offer_cost
    for month, cash_flow in enumerate(monthly_cash_flows, start=1):
        if not math.isfinite(float(cash_flow)):
            raise ValueError("monthly cash flows must be finite")
        total += float(cash_flow) * contribution_margin / (1 + monthly_discount_rate) ** month
    return total
