"""Campaign and campaign-household descriptive features in PySpark."""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F

from .time_windows import feature_period_bounds


def build_campaign_features(
    campaign_desc: DataFrame,
    campaign_table: DataFrame,
    transaction_data: DataFrame,
    coupon_redemptions: DataFrame,
    analysis_cutoff_day: int,
    feature_lookback_days: int | None = None,
    outcome_window_days: int | None = None,
) -> tuple[DataFrame, DataFrame]:
    """Return (campaign_features, campaign_household_features).

    `campaign_table` records campaign assignments and is used as the observed
    exposure source. Campaign starts on/after the cutoff are excluded. The
    guide and Phase 2 value checks support CAMPAIGN and household_key joins;
    coupon-redemption attribution uses the verified household+CAMPAIGN pair.

    Spend before campaign is bounded by feature_lookback_days when supplied.
    During spend uses inclusive START_DAY/END_DAY. Post-campaign spend is
    bounded by outcome_window_days when supplied, otherwise extends through
    the exclusive analysis cutoff and must be compared with its exposure
    duration. Post-period values are null when the period is incomplete.
    """
    lower, _ = feature_period_bounds(analysis_cutoff_day, feature_lookback_days)
    if outcome_window_days is not None and (
        isinstance(outcome_window_days, bool)
        or not isinstance(outcome_window_days, int)
        or outcome_window_days <= 0
    ):
        raise ValueError("outcome_window_days must be a positive integer or None")

    eligible_campaigns = (campaign_desc
        .select("CAMPAIGN", F.col("DESCRIPTION").alias("campaign_type"), "START_DAY", "END_DAY")
        .filter(F.col("START_DAY") < F.lit(analysis_cutoff_day)))
    if lower is not None:
        eligible_campaigns = eligible_campaigns.filter(F.col("START_DAY") >= F.lit(lower))

    exposures = (campaign_table.select("household_key", "CAMPAIGN")
                 .join(eligible_campaigns, "CAMPAIGN", "inner")
                 .withColumn("campaign_exposure", F.lit(1)))

    spend_tx = transaction_data.filter(F.col("DAY") < F.lit(analysis_cutoff_day)).select(
        "household_key", "DAY", "SALES_VALUE"
    )
    assigned_spend = spend_tx.join(F.broadcast(exposures.select(
        "household_key", "CAMPAIGN", "START_DAY", "END_DAY"
    )), "household_key", "inner")

    before = F.col("DAY") < F.col("START_DAY")
    if feature_lookback_days is not None:
        before = before & (F.col("DAY") >= F.col("START_DAY") - F.lit(feature_lookback_days))
    during = (F.col("DAY") >= F.col("START_DAY")) & (F.col("DAY") <= F.col("END_DAY"))
    after = F.col("DAY") > F.col("END_DAY")
    if outcome_window_days is not None:
        after = after & (F.col("DAY") <= F.col("END_DAY") + F.lit(outcome_window_days))

    spend_by_campaign_household = assigned_spend.groupBy("household_key", "CAMPAIGN").agg(
        F.sum(F.when(before, F.col("SALES_VALUE"))).alias("_spend_before"),
        F.sum(F.when(during, F.col("SALES_VALUE"))).alias("_spend_during"),
        F.sum(F.when(after, F.col("SALES_VALUE"))).alias("_spend_after"),
    )

    # Both fields are part of the Phase 2 verified relationship. Restrict
    # redemptions to the campaign's documented day interval and pre-cutoff data.
    redemptions = (coupon_redemptions.select("household_key", "CAMPAIGN", "DAY")
        .join(exposures.select("household_key", "CAMPAIGN", "START_DAY", "END_DAY"),
              ["household_key", "CAMPAIGN"], "inner")
        .filter((F.col("DAY") >= F.col("START_DAY"))
                & (F.col("DAY") <= F.col("END_DAY"))
                & (F.col("DAY") < F.lit(analysis_cutoff_day)))
        .groupBy("household_key", "CAMPAIGN")
        .agg(F.count(F.lit(1)).alias("coupon_redemption_count")))

    post_days = F.greatest(F.lit(0), F.lit(analysis_cutoff_day) - F.col("END_DAY") - F.lit(1))
    if outcome_window_days is not None:
        post_days = F.least(post_days, F.lit(outcome_window_days))

    joined = (exposures
        .join(spend_by_campaign_household, ["household_key", "CAMPAIGN"], "left")
        .join(redemptions, ["household_key", "CAMPAIGN"], "left")
        .withColumn("_post_period_complete",
                    F.col("END_DAY") < F.lit(analysis_cutoff_day)
                    if outcome_window_days is None
                    else F.col("END_DAY") + F.lit(outcome_window_days) < F.lit(analysis_cutoff_day))
        .withColumn("spend_before_campaign", F.coalesce(F.col("_spend_before"), F.lit(0.0)))
        .withColumn("spend_during_campaign", F.coalesce(F.col("_spend_during"), F.lit(0.0)))
        .withColumn("spend_after_campaign",
                    F.when(F.col("_post_period_complete"),
                           F.coalesce(F.col("_spend_after"), F.lit(0.0))).cast("double"))
        .withColumn("coupon_redemption_count", F.coalesce(F.col("coupon_redemption_count"), F.lit(0)))
        .withColumn("post_campaign_days_observed", post_days)
        .withColumn("campaign_window_complete", F.col("END_DAY") < F.lit(analysis_cutoff_day))
        .withColumn("post_campaign_observation_complete", F.col("_post_period_complete"))
        .select("household_key", "CAMPAIGN", "campaign_type", "START_DAY", "END_DAY",
                "campaign_exposure", "coupon_redemption_count", "spend_before_campaign",
                "spend_during_campaign", "spend_after_campaign", "post_campaign_days_observed",
                "campaign_window_complete", "post_campaign_observation_complete"))

    campaign_totals = joined.groupBy("CAMPAIGN").agg(
        F.count(F.lit(1)).alias("exposed_households"),
        F.sum("coupon_redemption_count").alias("coupon_redemption_count"),
        F.sum(F.when(F.col("coupon_redemption_count") > 0, 1).otherwise(0))
         .alias("redeeming_households"),
        F.sum("spend_before_campaign").alias("spend_before_campaign"),
        F.sum("spend_during_campaign").alias("spend_during_campaign"),
        F.sum("spend_after_campaign").alias("spend_after_campaign"),
        F.min(F.col("campaign_window_complete").cast("int"))
         .cast("boolean").alias("campaign_window_complete"),
        F.min(F.col("post_campaign_observation_complete").cast("int"))
         .cast("boolean").alias("post_campaign_observation_complete"),
        F.max("post_campaign_days_observed").alias("post_campaign_days_observed"),
    )
    campaign_features = (eligible_campaigns.join(campaign_totals, "CAMPAIGN", "left")
        .withColumn("exposed_households", F.coalesce(F.col("exposed_households"), F.lit(0)))
        .withColumn("coupon_redemption_count", F.coalesce(F.col("coupon_redemption_count"), F.lit(0)))
        .withColumn("redeeming_households", F.coalesce(F.col("redeeming_households"), F.lit(0)))
        .withColumn("descriptive_redemption_rate",
                    F.when(F.col("exposed_households") > 0,
                           F.col("redeeming_households") / F.col("exposed_households")).cast("double")))

    return campaign_features, joined
