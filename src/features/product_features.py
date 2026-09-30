"""Product-grain analytical features built with PySpark."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from .time_windows import filter_feature_period


def build_product_features(transaction_data: DataFrame, product_data: DataFrame,
                           analysis_cutoff_day: int,
                           feature_lookback_days: int | None = None) -> DataFrame:
    """Build one row per product, retaining catalog products with no sales.

    A purchase occurrence is a source transaction line. Basket counts first
    reduce to distinct household+basket+product tuples. Repeat purchasing is
    the share of purchasing households with the product in more than one
    distinct basket during the selected feature period.
    """
    tx = filter_feature_period(transaction_data, "DAY", analysis_cutoff_day,
                               feature_lookback_days)
    sales = tx.groupBy("PRODUCT_ID").agg(
        F.sum("QUANTITY").alias("total_units_sold"),
        F.sum("SALES_VALUE").alias("total_sales"),
        F.countDistinct("household_key").alias("number_of_households_purchasing"),
        F.avg("SALES_VALUE").alias("average_sales_value_per_occurrence"),
    )

    basket_keys = tx.select("PRODUCT_ID", "household_key", "BASKET_ID").distinct()
    basket_counts = basket_keys.groupBy("PRODUCT_ID").agg(
        F.count(F.lit(1)).alias("number_of_baskets_containing_product")
    )
    household_baskets = basket_keys.groupBy("PRODUCT_ID", "household_key").agg(
        F.count(F.lit(1)).alias("_basket_count")
    )
    household_counts = household_baskets.groupBy("PRODUCT_ID").agg(
        F.count(F.lit(1)).alias("_purchasing_households")
    )
    repeat_counts = (household_baskets.filter(F.col("_basket_count") > 1)
                     .groupBy("PRODUCT_ID").agg(
                         F.count(F.lit(1)).alias("number_of_repeat_purchasing_households")
                     ))

    product_columns = [
        "PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC",
        "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT",
    ]
    result = (product_data.select(*product_columns)
              .join(sales, "PRODUCT_ID", "left")
              .join(basket_counts, "PRODUCT_ID", "left")
              .join(household_counts, "PRODUCT_ID", "left")
              .join(repeat_counts, "PRODUCT_ID", "left")
              .withColumn("total_units_sold", F.coalesce(F.col("total_units_sold"), F.lit(0)))
              .withColumn("total_sales", F.coalesce(F.col("total_sales"), F.lit(0.0)))
              .withColumn("number_of_households_purchasing",
                          F.coalesce(F.col("number_of_households_purchasing"), F.lit(0)))
              .withColumn("number_of_baskets_containing_product",
                          F.coalesce(F.col("number_of_baskets_containing_product"), F.lit(0)))
              .withColumn("number_of_repeat_purchasing_households",
                          F.coalesce(F.col("number_of_repeat_purchasing_households"), F.lit(0)))
              .withColumn("repeat_purchase_rate",
                          F.when(F.col("_purchasing_households") > 0,
                                 F.col("number_of_repeat_purchasing_households")
                                 / F.col("_purchasing_households")).cast("double"))
              .drop("_purchasing_households"))
    return result
