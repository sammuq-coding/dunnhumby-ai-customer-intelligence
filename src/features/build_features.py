"""Build and persist the PySpark analytical feature tables."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from pyspark import StorageLevel
from pyspark.sql import DataFrame

from src.ingestion.spark_ingestion import create_spark_session
from src.features.customer_features import build_customer_features
from src.features.marketing_features import build_campaign_features
from src.features.product_features import build_product_features
from src.features.time_windows import filter_feature_period


def _read_selected(spark: Any, raw_path: Path, filename: str,
                   options: dict[str, Any], columns: list[str]) -> DataFrame:
    """Read a source CSV with Spark and project only feature-required fields."""
    return spark.read.options(**options).csv(str(raw_path / filename)).select(*columns)


def run_feature_pipeline(feature_config_path: str = "configs/feature_windows.yaml",
                         sample_rows: int | None = None) -> dict[str, dict[str, Any]]:
    """Build feature DataFrames, write Parquet outputs, and return output metadata."""
    feature_config = yaml.safe_load(Path(feature_config_path).read_text())
    ingestion_config = yaml.safe_load(Path(feature_config["spark_config"]).read_text())
    raw_path = Path(feature_config["paths"]["raw"])
    output_path = Path(feature_config["paths"]["processed"])
    cutoff = int(feature_config["analysis_cutoff_day"])
    lookback = feature_config.get("feature_lookback_days")
    outcome_window = feature_config.get("outcome_window_days")
    preview_rows = sample_rows if sample_rows is not None else int(feature_config.get("sample_rows", 5))

    spark = create_spark_session("dunnhumby-phase3-features", ingestion_config.get("spark", {}))
    options = ingestion_config["csv"]
    try:
        # causal_data.csv is intentionally not loaded: its display/mailer grain
        # is not needed for these customer, product, or campaign tables.
        transactions = _read_selected(spark, raw_path, "transaction_data.csv", options, [
            "household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE",
            "STORE_ID", "RETAIL_DISC", "TRANS_TIME", "WEEK_NO", "COUPON_DISC", "COUPON_MATCH_DISC",
        ])
        products = _read_selected(spark, raw_path, "product.csv", options, [
            "PRODUCT_ID", "MANUFACTURER", "DEPARTMENT", "BRAND", "COMMODITY_DESC",
            "SUB_COMMODITY_DESC", "CURR_SIZE_OF_PRODUCT",
        ])
        campaign_desc = _read_selected(spark, raw_path, "campaign_desc.csv", options,
                                       ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"])
        campaign_table = _read_selected(spark, raw_path, "campaign_table.csv", options,
                                         ["household_key", "CAMPAIGN"])
        coupon_redemptions = _read_selected(spark, raw_path, "coupon_redempt.csv", options,
                                            ["household_key", "DAY", "CAMPAIGN", "COUPON_UPC"])
        household_demographics = _read_selected(spark, raw_path, "hh_demographic.csv", options,
                                                ["household_key", "classification_1"])
        # Reused by all three analytical domains. Persist the projected Spark
        # relation once to avoid repeatedly parsing the multi-million-row CSV.
        transactions.persist(StorageLevel.MEMORY_AND_DISK)
        transactions.count()

        campaign_features, campaign_household_features = build_campaign_features(
            campaign_desc, campaign_table, transactions, coupon_redemptions,
            cutoff, lookback, outcome_window,
        )
        customer_features = build_customer_features(
            transactions, products, campaign_household_features,
            coupon_redemptions, cutoff, lookback,
        )
        product_features = build_product_features(transactions, products, cutoff, lookback)

        tables = {
            "campaign_household_features": campaign_household_features,
            "campaign_features": campaign_features,
            "customer_features": customer_features,
            "product_features": product_features,
        }
        for name, frame in tables.items():
            frame.createOrReplaceTempView(name)
        feature_transactions = filter_feature_period(transactions, "DAY", cutoff, lookback)
        feature_transactions.createOrReplaceTempView("feature_transactions")
        products.createOrReplaceTempView("product_source")
        household_demographics.createOrReplaceTempView("household_demographics")

        results: dict[str, dict[str, Any]] = {}
        for name, frame in tables.items():
            frame.persist(StorageLevel.MEMORY_AND_DISK)
            parquet_path = output_path / name
            frame.write.mode("overwrite").parquet(str(parquet_path))
            row_count = frame.count()
            print(f"\n{name}: {row_count} rows")
            frame.printSchema()
            frame.show(preview_rows, truncate=False)
            results[name] = {
                "row_count": row_count,
                "schema": frame.schema.jsonValue(),
                "parquet_path": str(parquet_path),
            }
            frame.unpersist()
        return results
    finally:
        if "transactions" in locals():
            transactions.unpersist()
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/feature_windows.yaml")
    parser.add_argument("--sample-rows", type=int)
    args = parser.parse_args()
    run_feature_pipeline(args.config, args.sample_rows)


if __name__ == "__main__":
    main()
