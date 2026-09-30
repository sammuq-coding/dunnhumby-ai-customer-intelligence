"""Assumption-based customer value calculations and Spark transformations."""
from __future__ import annotations

import math

from pyspark.sql import DataFrame, functions as F

from .npv import annual_to_monthly_discount_rate


def _validate_assumptions(margin: float, retention: float, annual_discount: float,
                          horizon: int) -> None:
    if not math.isfinite(margin) or not 0 <= margin <= 1:
        raise ValueError("contribution_margin must be between 0 and 1")
    if not math.isfinite(retention) or not 0 <= retention <= 1:
        raise ValueError("monthly_retention_rate must be between 0 and 1")
    annual_to_monthly_discount_rate(annual_discount)
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon_months must be a positive integer")


def modeled_clv(monthly_revenue: float, contribution_margin: float,
                monthly_retention_rate: float, annual_discount_rate: float,
                horizon_months: int) -> float:
    """PV of monthly contribution, with activity survival r**(t-1)."""
    _validate_assumptions(contribution_margin, monthly_retention_rate,
                          annual_discount_rate, horizon_months)
    if not math.isfinite(monthly_revenue) or monthly_revenue < 0:
        raise ValueError("monthly_revenue must be finite and nonnegative")
    rate = annual_to_monthly_discount_rate(annual_discount_rate)
    return sum(
        monthly_revenue * contribution_margin * monthly_retention_rate ** (month - 1)
        / (1 + rate) ** month
        for month in range(1, horizon_months + 1)
    )


def add_clv_columns(customer_features: DataFrame, *, observed_history_days: int,
                    days_per_month: float, contribution_margin: float,
                    monthly_retention_rate: float, annual_discount_rate: float,
                    short_horizon_months: int, long_horizon_months: int) -> DataFrame:
    """Add source-span monthlyization and assumption-based value columns."""
    if observed_history_days <= 0 or not math.isfinite(days_per_month) or days_per_month <= 0:
        raise ValueError("observed_history_days and days_per_month must be positive")
    _validate_assumptions(contribution_margin, monthly_retention_rate,
                          annual_discount_rate, short_horizon_months)
    _validate_assumptions(contribution_margin, monthly_retention_rate,
                          annual_discount_rate, long_horizon_months)
    monthly_rate = annual_to_monthly_discount_rate(annual_discount_rate)
    observed_months = observed_history_days / days_per_month
    result = (customer_features
        .withColumn("observed_history_months", F.lit(float(observed_months)))
        .withColumn("historical_revenue", F.col("total_sales"))
        .withColumn("basket_count", F.col("transaction_count"))
        .withColumn("historical_quantity", F.col("total_quantity"))
        .withColumn("estimated_monthly_revenue", F.col("historical_revenue") / F.lit(observed_months))
        .withColumn("assumed_contribution_margin", F.lit(float(contribution_margin)))
        .withColumn("estimated_monthly_contribution",
                    F.col("estimated_monthly_revenue") * F.col("assumed_contribution_margin"))
        .withColumn("monthly_discount_rate", F.lit(float(monthly_rate))))
    for label, horizon in (("modeled_12_month_clv", short_horizon_months),
                           ("modeled_24_month_clv", long_horizon_months)):
        # Month t activity probability is retention**(t-1): current next-month
        # activity starts at 1, then decays by the configured monthly rate.
        contribution = F.col("estimated_monthly_contribution")
        expression = F.aggregate(
            F.sequence(F.lit(1), F.lit(horizon)), F.lit(0.0),
            lambda acc, t: acc + contribution * F.pow(F.lit(float(monthly_retention_rate)), t - 1)
            / F.pow(F.lit(1.0 + monthly_rate), t),
        )
        result = result.withColumn(label, expression)
    return result
