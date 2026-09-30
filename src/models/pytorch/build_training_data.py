"""Build campaign response examples with strictly pre-campaign history."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import yaml
from pyspark.sql import DataFrame, SparkSession, Window, functions as F


def campaign_time_splits(campaigns: DataFrame, train_fraction: float = 0.60,
                         validation_fraction: float = 0.20) -> DataFrame:
    """Map distinct START_DAY values to ordered splits; ties stay together."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("split fractions must be positive and sum to less than one")
    starts = [r.START_DAY for r in campaigns.select("START_DAY").distinct().orderBy("START_DAY").collect()]
    if len(starts) < 3:
        raise ValueError("at least three distinct campaign start days are required")
    train_end = max(1, math.floor(len(starts) * train_fraction))
    validation_end = max(train_end + 1, math.floor(len(starts) * (train_fraction + validation_fraction)))
    validation_end = min(validation_end, len(starts) - 1)
    train_last, validation_last = starts[train_end - 1], starts[validation_end - 1]
    validation_start = starts[train_end]
    test_start = starts[validation_end]
    split = campaigns.withColumn(
        "_candidate_split",
        F.when(F.col("START_DAY") <= train_last, "train")
         .when(F.col("START_DAY") <= validation_last, "validation")
         .otherwise("test"),
    ).withColumn(
        "data_split",
        F.when((F.col("_candidate_split") == "train") & (F.col("END_DAY") < validation_start), "train")
         .when((F.col("_candidate_split") == "validation") & (F.col("END_DAY") < test_start), "validation")
         .when(F.col("_candidate_split") == "test", "test")
         .otherwise(F.lit(None).cast("string")),
    ).drop("_candidate_split")
    # Purge labels whose campaign outcome window crosses the next split's
    # start boundary, so later-period outcomes never enter an earlier split.
    # Keep purged campaigns in this metadata frame (with null split) so their
    # exposures remain usable as historical information for later campaigns.
    return split


def build_campaign_response_dataset(
    campaigns: DataFrame,
    assignments: DataFrame,
    transactions: DataFrame,
    products: DataFrame,
    redemptions: DataFrame,
    lookback_days: int = 90,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> DataFrame:
    """Create one row per eligible assigned household/campaign.

    Historical transaction features use [START_DAY-lookback_days, START_DAY).
    The label is at least one coupon_redempt row within the inclusive campaign
    interval. Assignments come from campaign_table; this is observational data.
    """
    if isinstance(lookback_days, bool) or not isinstance(lookback_days, int) or lookback_days <= 0:
        raise ValueError("lookback_days must be a positive integer")
    # A negative label needs the entire campaign interval to be observable.
    # DAY 704 is the latest observed coupon redemption day in this source.
    redemption_max_day = redemptions.agg(F.max("DAY").alias("max_day")).first()["max_day"]
    campaign_splits = campaign_time_splits(campaigns.select("CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"),
                                           train_fraction, validation_fraction)
    campaign_splits = campaign_splits.withColumn(
        "data_split", F.when(F.col("END_DAY") <= F.lit(redemption_max_day), F.col("data_split"))
    )
    exposure = (assignments.select("household_key", "CAMPAIGN")
        .join(campaign_splits, "CAMPAIGN", "inner")
        .select("household_key", "CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY", "data_split"))

    # Select narrow source projections before the household/time range join.
    tx = transactions.select("household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY",
                             "SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC")
    historic = (tx.join(exposure.select("household_key", "CAMPAIGN", "START_DAY"), "household_key")
        .filter((F.col("DAY") < F.col("START_DAY"))
                & (F.col("DAY") >= F.col("START_DAY") - F.lit(lookback_days))))
    # Phase 2 verified PRODUCT_ID uniqueness, so the dimension join does not
    # require deduplication and cannot silently discard catalog records.
    prod = products.select("PRODUCT_ID", "DEPARTMENT")
    historic = historic.join(prod, "PRODUCT_ID", "left")

    by_customer = historic.groupBy("household_key", "CAMPAIGN").agg(
        F.sum("SALES_VALUE").alias("historical_total_sales"),
        F.countDistinct("BASKET_ID").alias("historical_basket_count"),
        F.sum("QUANTITY").alias("historical_total_quantity"),
        F.countDistinct("PRODUCT_ID").alias("historical_unique_products"),
        F.countDistinct("DEPARTMENT").alias("historical_unique_departments"),
        F.sum("RETAIL_DISC").alias("historical_retail_discount"),
        (F.sum("COUPON_DISC") + F.sum("COUPON_MATCH_DISC")).alias("historical_coupon_discount"),
        F.max("DAY").alias("_last_day"),
    )
    basket = historic.groupBy("household_key", "CAMPAIGN", "BASKET_ID").agg(
        F.sum("SALES_VALUE").alias("_basket_sales"),
        F.max(F.when((F.coalesce(F.col("COUPON_DISC"), F.lit(0.0)) != 0)
                     | (F.coalesce(F.col("COUPON_MATCH_DISC"), F.lit(0.0)) != 0), 1).otherwise(0)).alias("_coupon_used"),
    ).groupBy("household_key", "CAMPAIGN").agg(
        F.avg("_basket_sales").alias("historical_average_basket_value"),
        F.sum("_coupon_used").alias("historical_coupon_basket_count"),
    )
    features = (by_customer.join(basket, ["household_key", "CAMPAIGN"], "inner")
        .join(exposure.select("household_key", "CAMPAIGN", "START_DAY"), ["household_key", "CAMPAIGN"], "inner")
        .withColumn("historical_purchase_frequency", F.col("historical_basket_count") / F.lit(lookback_days))
        .withColumn("historical_recency_days", F.col("START_DAY") - F.col("_last_day"))
        .drop("_last_day", "START_DAY"))

    hist_redemptions = (redemptions.select("household_key", "DAY")
        .join(exposure.select("household_key", "CAMPAIGN", "START_DAY"), "household_key")
        .filter((F.col("DAY") < F.col("START_DAY"))
                & (F.col("DAY") >= F.col("START_DAY") - F.lit(lookback_days)))
        .groupBy("household_key", "CAMPAIGN").agg(F.count("*").alias("prior_coupon_redemption_count")))
    prior_exposures = (exposure.alias("cur").join(exposure.alias("prior"),
        (F.col("cur.household_key") == F.col("prior.household_key"))
        & (F.col("prior.START_DAY") < F.col("cur.START_DAY"))
        & (F.col("prior.START_DAY") >= F.col("cur.START_DAY") - F.lit(lookback_days)), "left")
        .groupBy(F.col("cur.household_key").alias("household_key"), F.col("cur.CAMPAIGN").alias("CAMPAIGN"))
        .agg(F.count(F.col("prior.CAMPAIGN")).alias("prior_campaign_exposure_count")))
    # Count earlier assigned-campaign redemptions only when the redemption
    # event itself predates this campaign, even if campaign intervals overlap.
    earlier_campaigns = (exposure.alias("cur").join(exposure.alias("prior"),
        (F.col("cur.household_key") == F.col("prior.household_key"))
        & (F.col("prior.START_DAY") < F.col("cur.START_DAY"))
        & (F.col("prior.START_DAY") >= F.col("cur.START_DAY") - F.lit(lookback_days)), "inner")
        .select(F.col("cur.household_key").alias("household_key"), F.col("cur.CAMPAIGN").alias("current_campaign"),
                F.col("cur.START_DAY").alias("current_start"), F.col("prior.CAMPAIGN").alias("prior_campaign")))
    prior_red = (earlier_campaigns.alias("e").join(redemptions.select("household_key", "CAMPAIGN", "DAY").alias("r"),
            (F.col("e.household_key") == F.col("r.household_key"))
            & (F.col("e.prior_campaign") == F.col("r.CAMPAIGN"))
            & (F.col("r.DAY") < F.col("e.current_start"))
            & (F.col("r.DAY") >= F.col("e.current_start") - F.lit(lookback_days)), "left")
        .groupBy(F.col("e.household_key").alias("household_key"), F.col("e.current_campaign").alias("CAMPAIGN"))
        .agg(F.count(F.col("r.DAY")).alias("prior_campaign_redemption_count")))

    # Target redemptions are attributed by verified household+campaign and must
    # occur from the campaign start through its documented end, inclusive.
    target_counts = (redemptions.select("household_key", "CAMPAIGN", "DAY")
        .join(exposure.select("household_key", "CAMPAIGN", "START_DAY", "END_DAY"),
              ["household_key", "CAMPAIGN"], "inner")
        .filter((F.col("DAY") >= F.col("START_DAY")) & (F.col("DAY") <= F.col("END_DAY")))
        .groupBy("household_key", "CAMPAIGN").agg(F.count("*").alias("_target_count")))

    result = (exposure.join(features.drop("START_DAY"), ["household_key", "CAMPAIGN"], "inner")
        .join(hist_redemptions, ["household_key", "CAMPAIGN"], "left")
        .join(prior_exposures, ["household_key", "CAMPAIGN"], "left")
        .join(prior_red, ["household_key", "CAMPAIGN"], "left")
        .join(target_counts, ["household_key", "CAMPAIGN"], "left")
        .withColumn("prior_coupon_redemption_count", F.coalesce("prior_coupon_redemption_count", F.lit(0)))
        .withColumn("prior_campaign_exposure_count", F.coalesce("prior_campaign_exposure_count", F.lit(0)))
        .withColumn("prior_campaign_redemption_count", F.coalesce("prior_campaign_redemption_count", F.lit(0)))
        .withColumn("target_redeemed", (F.coalesce(F.col("_target_count"), F.lit(0)) > 0).cast("int"))
        .drop("_target_count")
        .filter(F.col("data_split").isNotNull()))
    return result


def _read_csv(spark: SparkSession, root: Path, name: str) -> DataFrame:
    return spark.read.option("header", True).option("inferSchema", True).option("mode", "PERMISSIVE").csv(str(root / name))


def run(config_path: str = "configs/modeling.yaml") -> DataFrame:
    config = yaml.safe_load(Path(config_path).read_text())
    builder = SparkSession.builder.appName("campaign-response-training-data")
    for key, value in config.get("spark", {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    raw = Path(config["paths"]["raw"])
    result = build_campaign_response_dataset(
        _read_csv(spark, raw, "campaign_desc.csv"), _read_csv(spark, raw, "campaign_table.csv"),
        _read_csv(spark, raw, "transaction_data.csv"), _read_csv(spark, raw, "product.csv"),
        _read_csv(spark, raw, "coupon_redempt.csv"), int(config["lookback_days"]),
        float(config["split"]["train_fraction"]), float(config["split"]["validation_fraction"]))
    out = Path(config["paths"]["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.write.mode("overwrite").parquet(str(out))
    result.groupBy("data_split").agg(F.count("*").alias("rows"), F.sum("target_redeemed").alias("positives")).show()
    result.printSchema()
    spark.stop()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modeling.yaml")
    run(parser.parse_args().config)
