"""Household-grain analytical features built with PySpark."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from .time_windows import filter_feature_period


def build_customer_features(
    transaction_data: DataFrame,
    product_data: DataFrame,
    campaign_household_features: DataFrame,
    coupon_redemptions: DataFrame,
    analysis_cutoff_day: int,
    feature_lookback_days: int | None = None,
) -> DataFrame:
    """Build one row per household using purchases strictly before the cutoff.

    Basket value averages are computed after collapsing item rows to
    household+basket. Category cardinalities use the verified unique product
    dimension; null categories remain null and are ignored by countDistinct.
    """
    tx = filter_feature_period(transaction_data, "DAY", analysis_cutoff_day,
                               feature_lookback_days)
    product_categories = product_data.select("PRODUCT_ID", "DEPARTMENT", "COMMODITY_DESC")
    tx_with_categories = tx.join(product_categories, "PRODUCT_ID", "left")

    core = tx_with_categories.groupBy("household_key").agg(
        F.sum("SALES_VALUE").alias("total_sales"),
        F.countDistinct("BASKET_ID").alias("transaction_count"),
        F.sum("QUANTITY").alias("total_quantity"),
        F.countDistinct("PRODUCT_ID").alias("unique_products"),
        F.countDistinct("DEPARTMENT").alias("unique_departments"),
        F.countDistinct("COMMODITY_DESC").alias("unique_commodities"),
        F.sum("RETAIL_DISC").alias("total_retail_discount"),
        (F.sum("COUPON_DISC") + F.sum("COUPON_MATCH_DISC")).alias("total_coupon_discount"),
        F.min("DAY").alias("_first_purchase_day"),
        F.max("DAY").alias("_last_purchase_day"),
    )

    basket_sales = tx.groupBy("household_key", "BASKET_ID").agg(
        F.sum("SALES_VALUE").alias("_basket_sales")
    ).groupBy("household_key").agg(
        F.avg("_basket_sales").alias("average_basket_value")
    )

    # Coupon usage is the number of distinct baskets with a non-zero applied
    # coupon or retailer-match discount, not a count of product rows.
    coupon_usage = (tx.filter(
        (F.coalesce(F.col("COUPON_DISC"), F.lit(0.0)) != 0)
        | (F.coalesce(F.col("COUPON_MATCH_DISC"), F.lit(0.0)) != 0)
    ).select("household_key", "BASKET_ID").distinct()
      .groupBy("household_key").agg(F.count(F.lit(1)).alias("coupon_usage")))

    campaign_counts = campaign_household_features.groupBy("household_key").agg(
        F.sum("campaign_exposure").alias("campaign_exposure_count")
    )
    redemption_period = filter_feature_period(coupon_redemptions, "DAY", analysis_cutoff_day,
                                              feature_lookback_days)
    redemption_counts = redemption_period.groupBy("household_key").agg(
        F.count(F.lit(1)).alias("coupon_redemption_count")
    )

    result = (core.join(basket_sales, "household_key", "left")
              .join(coupon_usage, "household_key", "left")
              .join(campaign_counts, "household_key", "left")
              .join(redemption_counts, "household_key", "left")
              .withColumn("purchase_day_range", F.col("_last_purchase_day") - F.col("_first_purchase_day"))
              .withColumn("purchase_frequency",
                          F.col("transaction_count") / (F.col("purchase_day_range") + F.lit(1)))
              .withColumn("recency_days", F.lit(analysis_cutoff_day) - F.col("_last_purchase_day"))
              .withColumn("coupon_usage", F.coalesce(F.col("coupon_usage"), F.lit(0)))
              .withColumn("campaign_exposure_count", F.coalesce(F.col("campaign_exposure_count"), F.lit(0)))
              .withColumn("coupon_redemption_count", F.coalesce(F.col("coupon_redemption_count"), F.lit(0)))
              .drop("_first_purchase_day", "_last_purchase_day"))

    return result.select(
        "household_key", "total_sales", "transaction_count", "average_basket_value",
        "total_quantity", "unique_products", "unique_departments", "unique_commodities",
        "coupon_usage", "total_retail_discount", "total_coupon_discount",
        "campaign_exposure_count", "coupon_redemption_count", "purchase_day_range",
        "purchase_frequency", "recency_days",
    )
