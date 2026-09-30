"""Offer-scenario calculations; probabilities are observational predictions."""
from __future__ import annotations

import math

from pyspark.sql import DataFrame, functions as F


def expected_offer_value(response_probability: float, expected_contribution_if_response: float,
                         offer_cost: float) -> float:
    """Expected modeled contribution under a response scenario, less cost."""
    if not math.isfinite(response_probability) or not 0 <= response_probability <= 1:
        raise ValueError("response_probability must be between 0 and 1")
    if not math.isfinite(expected_contribution_if_response) or expected_contribution_if_response < 0:
        raise ValueError("expected_contribution_if_response must be finite and nonnegative")
    if not math.isfinite(offer_cost) or offer_cost < 0:
        raise ValueError("offer_cost must be finite and nonnegative")
    return response_probability * expected_contribution_if_response - offer_cost


def add_offer_economics(scored_offers: DataFrame, *, contribution_margin: float,
                        default_offer_cost: float) -> DataFrame:
    """Add one-basket response-scenario contribution and break-even cost.

    The response scenario uses the household's historical average basket
    value times the assumption-based margin. This is a scenario, not causal
    campaign contribution.
    """
    if not math.isfinite(contribution_margin) or not 0 <= contribution_margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1")
    if not math.isfinite(default_offer_cost) or default_offer_cost < 0:
        raise ValueError("default_offer_cost must be finite and nonnegative")
    p = F.col("calibrated_response_probability")
    if scored_offers.filter(p.isNull() | (p < 0) | (p > 1)).limit(1).count():
        raise ValueError("Calibrated response probabilities must be present and in [0, 1]")
    result = (scored_offers
        .withColumn("assumed_offer_cost", F.lit(float(default_offer_cost)))
        .withColumn("expected_contribution_if_response",
                    F.col("average_basket_value") * F.lit(float(contribution_margin)))
        .withColumn("break_even_offer_cost",
                    p * F.col("expected_contribution_if_response"))
        .withColumn("modeled_expected_offer_value",
                    F.col("break_even_offer_cost") - F.col("assumed_offer_cost")))
    return result


def sensitivity_scenarios(base: dict, margins: list[float], retentions: list[float],
                          annual_discounts: list[float], offer_costs: list[float]) -> list[dict]:
    """Return a cartesian set of economic assumptions for transparent review."""
    from itertools import product
    from .npv import annual_to_monthly_discount_rate
    for value in margins:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("sensitivity contribution margins must be between 0 and 1")
    for value in retentions:
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("sensitivity retention rates must be between 0 and 1")
    for value in annual_discounts:
        annual_to_monthly_discount_rate(value)
    if any(not math.isfinite(x) or x < 0 for x in offer_costs):
        raise ValueError("sensitivity offer costs must be finite and nonnegative")
    return [
        {"contribution_margin": m, "monthly_retention_rate": r,
         "annual_discount_rate": d, "offer_cost": c}
        for m, r, d, c in product(margins, retentions, annual_discounts, offer_costs)
    ]
