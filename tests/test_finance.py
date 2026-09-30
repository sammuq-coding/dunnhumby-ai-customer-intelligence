"""Tests for assumption-based customer and campaign economics."""
import math

import pytest
from pyspark.sql import SparkSession

from src.finance.clv import add_clv_columns, modeled_clv
from src.finance.npv import annual_to_monthly_discount_rate, npv
from src.finance.offer_economics import (
    add_offer_economics, expected_offer_value, sensitivity_scenarios,
)


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[2]").appName("finance-unit-tests").getOrCreate()
    yield session
    session.stop()


def test_monthly_discount_conversion_uses_effective_annual_rate():
    monthly = annual_to_monthly_discount_rate(0.10)
    assert (1 + monthly) ** 12 == pytest.approx(1.10)


def test_monthly_discount_rejects_invalid_rates():
    for rate in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            annual_to_monthly_discount_rate(rate)


def test_npv_matches_hand_calculated_example():
    rate = 0.10
    expected = 25 / 1.1 + 25 / (1.1 ** 2) - 5
    assert npv([100, 100], rate, contribution_margin=0.25, offer_cost=5) == pytest.approx(expected)


def test_clv_matches_hand_calculated_example():
    rate = annual_to_monthly_discount_rate(0.10)
    expected = 20 / (1 + rate) + 20 * 0.9 / (1 + rate) ** 2
    assert modeled_clv(100, 0.20, 0.90, 0.10, 2) == pytest.approx(expected)


def test_clv_retention_changes_later_period_value():
    assert modeled_clv(100, .2, .5, .1, 2) < modeled_clv(100, .2, 1.0, .1, 2)


def test_clv_contribution_margin_scales_value():
    assert modeled_clv(100, .2, .9, .1, 12) == pytest.approx(
        2 * modeled_clv(100, .1, .9, .1, 12))


def test_clv_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        modeled_clv(-1, .2, .9, .1, 12)
    with pytest.raises(ValueError):
        modeled_clv(10, 1.1, .9, .1, 12)
    with pytest.raises(ValueError):
        modeled_clv(10, .2, 1.1, .1, 12)


def test_offer_value_and_break_even_cost():
    p, contribution, cost = .4, 20.0, 5.0
    assert expected_offer_value(p, contribution, cost) == pytest.approx(3.0)
    assert p * contribution == pytest.approx(8.0)


def test_probability_bounds_are_enforced():
    for p in (-.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            expected_offer_value(p, 20, 5)


def test_sensitivity_grid_is_reusable_and_validated():
    scenarios = sensitivity_scenarios({}, [.2, .3], [.8], [.1], [2, 5])
    assert len(scenarios) == 4
    assert {s["offer_cost"] for s in scenarios} == {2, 5}
    with pytest.raises(ValueError):
        sensitivity_scenarios({}, [1.2], [.8], [.1], [2])


def test_spark_clv_columns_and_null_handling(spark):
    source = spark.createDataFrame([
        (1, 100.0, 2, 50.0, 4, .1, 2),
        (2, 0.0, 1, 0.0, 0, 0.0, 3),
    ], ["household_key", "total_sales", "transaction_count", "average_basket_value",
        "total_quantity", "purchase_frequency", "recency_days"])
    result = add_clv_columns(source, observed_history_days=30, days_per_month=30,
                             contribution_margin=.25, monthly_retention_rate=.9,
                             annual_discount_rate=.1, short_horizon_months=12,
                             long_horizon_months=24)
    rows = {r.household_key: r for r in result.collect()}
    assert rows[1].estimated_monthly_revenue == pytest.approx(100)
    assert rows[1].estimated_monthly_contribution == pytest.approx(25)
    assert rows[1].modeled_12_month_clv > 0
    assert rows[1].modeled_24_month_clv > rows[1].modeled_12_month_clv
    assert rows[2].modeled_12_month_clv == 0
    assert all(math.isfinite(r.modeled_24_month_clv) for r in rows.values())


def test_spark_offer_economics_adds_break_even_and_expected_value(spark):
    source = spark.createDataFrame([(0.4, 20.0)],
        ["calibrated_response_probability", "average_basket_value"])
    result = add_offer_economics(source, contribution_margin=.25, default_offer_cost=5).first()
    assert result.expected_contribution_if_response == pytest.approx(5)
    assert result.break_even_offer_cost == pytest.approx(2)
    assert result.modeled_expected_offer_value == pytest.approx(-3)
