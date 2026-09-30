"""Feature tests use tiny synthetic Spark frames, never the full source files."""
from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from src.features.customer_features import build_customer_features
from src.features.marketing_features import build_campaign_features
from src.features.product_features import build_product_features
from src.features.time_windows import filter_feature_period, filter_outcome_period


@pytest.fixture(scope="module")
def spark():
    session = (SparkSession.builder.master("local[1]")
               .appName("feature-unit-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "1")
               .getOrCreate())
    yield session
    session.stop()


def tiny_transactions(spark):
    return spark.createDataFrame([
        (1, 100, 1, 10, 1, 5.0, -0.5, 0.0, 0.0),
        (1, 100, 1, 11, 2, 7.0, -1.0, -1.0, -0.5),
        (1, 101, 3, 10, 1, 3.0, 0.0, 0.0, 0.0),
        (1, 102, 5, 12, 50, 100.0, 0.0, 0.0, 0.0),  # At cutoff: excluded.
        (2, 200, 1, 10, 1, 4.0, 0.0, 0.0, 0.0),
        (2, 201, 2, 11, 1, 6.0, 0.0, 0.0, 0.0),
        (2, 202, 3, 12, 1, 8.0, 0.0, 0.0, 0.0),
        (2, 203, 4, 12, 1, 9.0, 0.0, 0.0, 0.0),
        (2, 204, 5, 12, 1, 200.0, 0.0, 0.0, 0.0),  # At cutoff: excluded.
    ], ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
        "SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"])


def tiny_products(spark):
    return spark.createDataFrame([
        (10, 1, "Produce", "National", "Fresh", "Fruit", None),
        (11, 2, None, "Private", None, None, "1 ea"),
        (12, 3, "Dairy", "National", "Milk", "Fresh Milk", "1 gal"),
        (13, 4, None, "Private", None, None, None),  # Catalog item with no eligible sale.
    ], ["PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC",
        "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT"])


def test_customer_features_are_household_grain_and_baskets_are_not_lines(spark):
    transactions = tiny_transactions(spark)
    product = tiny_products(spark)
    campaign_household = spark.createDataFrame(
        [(1, 1, 1)], ["household_key", "CAMPAIGN", "campaign_exposure"])
    redemptions = spark.createDataFrame(
        [(1, 2, 1, 900), (1, 5, 1, 901)],
        ["household_key", "DAY", "CAMPAIGN", "COUPON_UPC"])

    result = build_customer_features(transactions, product, campaign_household,
                                     redemptions, analysis_cutoff_day=5)
    assert result.count() == 2
    assert result.groupBy("household_key").count().filter(F.col("count") != 1).count() == 0
    row = result.where(F.col("household_key") == 1).first().asDict()
    assert row["transaction_count"] == 2  # Three item lines, two baskets.
    assert row["total_sales"] == pytest.approx(15.0)
    assert row["average_basket_value"] == pytest.approx(7.5)
    assert row["total_quantity"] == 4  # The day-5 quantity of 50 is excluded.
    assert row["unique_products"] == 2
    assert row["unique_departments"] == 1  # Null product category is not fabricated.
    assert row["unique_commodities"] == 1
    assert row["coupon_usage"] == 1  # Discounted basket count, not row count.
    assert row["campaign_exposure_count"] == 1
    assert row["coupon_redemption_count"] == 1  # Day 5 is outside the cutoff.
    assert row["purchase_day_range"] == 2
    assert row["purchase_frequency"] == pytest.approx(2 / 3)
    assert row["recency_days"] == 2


def test_product_features_preserve_catalog_grain_and_null_metadata(spark):
    result = build_product_features(tiny_transactions(spark), tiny_products(spark),
                                    analysis_cutoff_day=5)
    assert result.count() == 4
    assert result.groupBy("PRODUCT_ID").count().filter(F.col("count") != 1).count() == 0
    p10 = result.where(F.col("PRODUCT_ID") == 10).first().asDict()
    assert p10["total_units_sold"] == 3
    assert p10["total_sales"] == pytest.approx(12.0)
    assert p10["number_of_households_purchasing"] == 2
    assert p10["number_of_baskets_containing_product"] == 3
    assert p10["average_sales_value_per_occurrence"] == pytest.approx(4.0)
    assert p10["number_of_repeat_purchasing_households"] == 1
    assert p10["repeat_purchase_rate"] == pytest.approx(0.5)
    assert p10["CURR_SIZE_OF_PRODUCT"] is None
    p11 = result.where(F.col("PRODUCT_ID") == 11).first().asDict()
    assert p11["DEPARTMENT"] is None
    assert p11["total_sales"] == pytest.approx(13.0)
    p13 = result.where(F.col("PRODUCT_ID") == 13).first().asDict()
    assert p13["total_sales"] == 0.0
    assert p13["average_sales_value_per_occurrence"] is None
    assert p13["repeat_purchase_rate"] is None


def test_campaign_features_use_verified_assignments_and_respect_cutoff(spark):
    campaign_desc = spark.createDataFrame([
        (1, "TypeA", 2, 3),
        (2, "TypeB", 5, 6),  # Starts at cutoff: not exposed in historical features.
    ], ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"])
    campaign_table = spark.createDataFrame([
        (1, 1), (2, 1), (1, 2),
    ], ["household_key", "CAMPAIGN"])
    redemptions = spark.createDataFrame([
        (1, 2, 1, 900), (2, 3, 1, 901), (1, 5, 2, 902),
    ], ["household_key", "DAY", "CAMPAIGN", "COUPON_UPC"])
    campaign_transactions = spark.createDataFrame([
        (1, 1, 12.0), (1, 2, 12.0), (1, 3, 3.0), (1, 4, 2.0), (1, 5, 100.0),
        (2, 1, 4.0), (2, 2, 6.0), (2, 3, 8.0), (2, 4, 9.0), (2, 5, 200.0),
    ], ["household_key", "DAY", "SALES_VALUE"])
    campaign_features, household_features = build_campaign_features(
        campaign_desc, campaign_table, campaign_transactions, redemptions,
        analysis_cutoff_day=5, outcome_window_days=1,
    )
    assert campaign_features.count() == 1
    assert household_features.count() == 2
    assert household_features.groupBy("household_key", "CAMPAIGN").count().filter(
        F.col("count") != 1).count() == 0
    h1 = household_features.where(F.col("household_key") == 1).first().asDict()
    assert h1["campaign_exposure"] == 1
    assert h1["coupon_redemption_count"] == 1
    assert h1["spend_before_campaign"] == pytest.approx(12.0)
    assert h1["spend_during_campaign"] == pytest.approx(15.0)
    assert h1["spend_after_campaign"] == pytest.approx(2.0)
    assert h1["post_campaign_observation_complete"] is True
    total = campaign_features.first().asDict()
    assert total["exposed_households"] == 2
    assert total["coupon_redemption_count"] == 2
    assert total["redeeming_households"] == 2
    assert total["spend_after_campaign"] == pytest.approx(11.0)
    assert total["descriptive_redemption_rate"] == pytest.approx(1.0)


def test_time_window_helpers_are_half_open_and_validate_outcome_length(spark):
    frame = spark.createDataFrame([(1,), (2,), (3,), (4,), (5,)], ["DAY"])
    features = filter_feature_period(frame, "DAY", analysis_cutoff_day=4,
                                     feature_lookback_days=2)
    outcomes = filter_outcome_period(frame, "DAY", analysis_cutoff_day=4,
                                     outcome_window_days=2)
    assert [r.DAY for r in features.orderBy("DAY").collect()] == [2, 3]
    assert [r.DAY for r in outcomes.orderBy("DAY").collect()] == [4, 5]
    with pytest.raises(ValueError):
        filter_outcome_period(frame, "DAY", 4, 0)
