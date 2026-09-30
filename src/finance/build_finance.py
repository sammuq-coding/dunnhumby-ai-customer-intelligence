"""Build household CLV and held-out campaign offer-economics tables."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
import yaml
from pyspark.sql import SparkSession, functions as F, types as T

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.finance.clv import add_clv_columns
from src.finance.npv import annual_to_monthly_discount_rate
from src.finance.offer_economics import add_offer_economics, sensitivity_scenarios
from src.models.pytorch.train_response_model import FEATURES, ResponseNetwork


def _validate_config(cfg: dict) -> None:
    f = cfg["finance"]
    for key in ("observed_history_days", "days_per_month", "horizon_months", "annual_discount_rate",
                "contribution_margin", "monthly_retention_rate", "default_offer_cost"):
        if key not in f:
            raise ValueError(f"Missing finance configuration: {key}")
    if int(f["observed_history_days"]) <= 0 or float(f["days_per_month"]) <= 0:
        raise ValueError("History days and days_per_month must be positive")
    if not 0 <= float(f["contribution_margin"]) <= 1:
        raise ValueError("contribution_margin must be in [0, 1]")
    if not 0 <= float(f["monthly_retention_rate"]) <= 1:
        raise ValueError("monthly_retention_rate must be in [0, 1]")
    if float(f["default_offer_cost"]) < 0:
        raise ValueError("default_offer_cost must be nonnegative")
    annual_to_monthly_discount_rate(float(f["annual_discount_rate"]))
    if f.get("response_scenario") != "one_average_basket":
        raise ValueError("Only the documented one_average_basket response scenario is implemented")


def _assert_customer_quality(df) -> dict:
    count = df.count()
    distinct = df.select("household_key").distinct().count()
    if count != distinct:
        raise ValueError(f"customer_features has duplicate household keys: rows={count}, unique={distinct}")
    required = ["household_key", "total_sales", "transaction_count", "average_basket_value",
                "total_quantity", "purchase_frequency", "recency_days"]
    missing_exprs = [F.sum(F.col(c).isNull().cast("long")).alias(c) for c in df.columns]
    missing_all = df.select(*missing_exprs).first().asDict()
    missing_required = {c: missing_all[c] for c in required}
    if any(missing_required.values()):
        raise ValueError(f"Required customer economics values contain nulls: {missing_required}")
    numeric = [c for c in required if c != "household_key"]
    nonfinite_conditions = [F.isnan(F.col(c)) | (F.abs(F.col(c)) == F.lit(float("inf"))) for c in numeric]
    bad_nonfinite = df.filter(F.greatest(*[x.cast("int") for x in nonfinite_conditions]) == 1).count()
    if bad_nonfinite:
        raise ValueError(f"Found {bad_nonfinite} rows with infinite/NaN economic inputs")
    impossible = df.filter(
        (F.col("total_sales") < 0) | (F.col("average_basket_value") < 0)
        | (F.col("transaction_count") < 0) | (F.col("total_quantity") < 0)
        | (F.col("purchase_frequency") < 0) | (F.col("recency_days") < 0)
    ).count()
    if impossible:
        raise ValueError(f"Found {impossible} rows with negative values in nonnegative economic inputs")
    return {"rows": count, "unique_households": distinct,
            "missing_values_by_column": missing_all,
            "nonfinite_rows": bad_nonfinite, "impossible_negative_rows": impossible}


def _score_held_out_test(spark, cfg: dict):
    """Score test feature rows without selecting or reading their labels."""
    model_artifact = torch.load(cfg["paths"]["response_model"], map_location="cpu", weights_only=False)
    calibrator = torch.load(cfg["paths"]["response_calibrator"], map_location="cpu", weights_only=False)
    if model_artifact["feature_names"] != FEATURES:
        raise ValueError("Saved model feature order does not match the existing response model")
    if calibrator.get("fit_split") != "validation" or calibrator.get("method") != "platt_scaling":
        raise ValueError("Expected the saved validation-only Platt calibrator")
    model = ResponseNetwork(len(FEATURES), int(model_artifact["architecture"]["layers"][1]))
    model.load_state_dict(model_artifact["state_dict"])
    model.eval()
    means = np.asarray(model_artifact["imputation_means"], dtype=np.float32)
    stds = np.asarray(model_artifact["normalization_stds"], dtype=np.float32)

    response = spark.read.parquet(cfg["paths"]["response_dataset"])
    # `target_redeemed` is intentionally excluded. Only held-out test rows are
    # used to represent out-of-sample offer-score coverage for this report.
    selected = response.filter(F.col("data_split") == "test").select(
        "household_key", "CAMPAIGN", "START_DAY", *FEATURES)
    records = list(selected.toLocalIterator())  # test cohort is 2,213 rows in this project
    if not records:
        raise ValueError("No held-out test rows found in campaign response dataset")
    ids = [(int(r.household_key), int(r.CAMPAIGN), int(r.START_DAY)) for r in records]
    x = np.asarray([[np.nan if r[c] is None else float(r[c]) for c in FEATURES] for r in records], dtype=np.float32)
    if np.isinf(x).any():
        raise ValueError("Non-finite feature encountered when scoring the held-out test cohort")
    x = np.where(np.isnan(x), means, x)
    x = (x - means) / stds
    with torch.no_grad():
        logits = model(torch.tensor(x)).numpy().astype(np.float64)
    p = 1.0 / (1.0 + np.exp(-np.clip(float(calibrator["slope"]) * logits + float(calibrator["intercept"]), -80, 80)))
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Model produced invalid calibrated probabilities")
    rows = [(h, c, day, float(prob)) for (h, c, day), prob in zip(ids, p)]
    schema = T.StructType([
        T.StructField("household_key", T.IntegerType(), False),
        T.StructField("CAMPAIGN", T.IntegerType(), False),
        T.StructField("START_DAY", T.IntegerType(), False),
        T.StructField("calibrated_response_probability", T.DoubleType(), False),
    ])
    return spark.createDataFrame(rows, schema), len(rows)


def _sensitivity_report(clv_df, offers, cfg: dict) -> list[dict]:
    base = cfg["finance"]
    sens = cfg["sensitivity"]
    scenarios = sensitivity_scenarios(
        base, list(map(float, sens["contribution_margins"])),
        list(map(float, sens["monthly_retention_rates"])),
        list(map(float, sens["annual_discount_rates"])),
        list(map(float, sens["offer_costs"])),
    )
    days = float(base["observed_history_days"])
    month_days = float(base["days_per_month"])
    observed_months = days / month_days
    base_sales = clv_df.select("historical_revenue").persist()
    med_revenue = base_sales.select(F.expr("percentile_approx(historical_revenue, 0.5)").alias("m")).first().m
    # Generate one-factor-at-a-time sensitivity around the configured base.
    one_factor = []
    for dimension, values in (("contribution_margin", sens["contribution_margins"]),
                              ("monthly_retention_rate", sens["monthly_retention_rates"]),
                              ("annual_discount_rate", sens["annual_discount_rates"])):
        for value in values:
            m = float(value) if dimension == "contribution_margin" else float(base["contribution_margin"])
            ret = float(value) if dimension == "monthly_retention_rate" else float(base["monthly_retention_rate"])
            disc = float(value) if dimension == "annual_discount_rate" else float(base["annual_discount_rate"])
            rate = annual_to_monthly_discount_rate(disc)
            horizon = int(base["horizon_months"]["short"])
            factors = sum(ret ** (t - 1) / (1 + rate) ** t for t in range(1, horizon + 1))
            median_clv = float(med_revenue) / observed_months * m * factors
            one_factor.append({"varied_assumption": dimension, "value": float(value),
                               "median_modeled_12_month_clv": median_clv})
    base_sales.unpersist()
    offer_unit = offers.agg(F.avg(
        F.col("calibrated_response_probability") * F.col("average_basket_value")
    ).alias("mean_probability_times_basket")).first()[0]
    grid = []
    median_monthly_revenue = float(med_revenue) / observed_months
    for scenario in scenarios:
        rate = annual_to_monthly_discount_rate(scenario["annual_discount_rate"])
        horizon = int(base["horizon_months"]["short"])
        factor = sum(scenario["monthly_retention_rate"] ** (t - 1) / (1 + rate) ** t
                     for t in range(1, horizon + 1))
        grid.append({
            **scenario,
            "median_modeled_12_month_clv": median_monthly_revenue
                * scenario["contribution_margin"] * factor,
            "mean_modeled_expected_offer_value": offer_unit
                * scenario["contribution_margin"] - scenario["offer_cost"],
        })
    return [{"type": "one_factor_clv", **x} for x in one_factor] + [
        {"type": "full_sensitivity_grid", **x} for x in grid]


def run(config_path: str = "configs/finance.yaml") -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    _validate_config(cfg)
    builder = SparkSession.builder.appName("dunnhumby-finance-economics")
    for key, value in cfg.get("spark", {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    try:
        source = spark.read.parquet(cfg["paths"]["customer_features"])
        dqc = _assert_customer_quality(source)
        f = cfg["finance"]
        economics = add_clv_columns(
            source, observed_history_days=int(f["observed_history_days"]),
            days_per_month=float(f["days_per_month"]),
            contribution_margin=float(f["contribution_margin"]),
            monthly_retention_rate=float(f["monthly_retention_rate"]),
            annual_discount_rate=float(f["annual_discount_rate"]),
            short_horizon_months=int(f["horizon_months"]["short"]),
            long_horizon_months=int(f["horizon_months"]["long"]),
        )
        dqc["usable_households"] = economics.filter(
            F.col("modeled_12_month_clv").isNotNull() & F.col("modeled_24_month_clv").isNotNull()).count()
        out_customer = Path(cfg["paths"]["output_customer_economics"])
        out_customer.parent.mkdir(parents=True, exist_ok=True)
        economics.write.mode("overwrite").parquet(str(out_customer))
        clv_summary = economics.agg(
            F.mean("modeled_12_month_clv").alias("mean_12"),
            F.expr("percentile_approx(modeled_12_month_clv, 0.5)").alias("median_12"),
            F.min("modeled_12_month_clv").alias("min_12"), F.max("modeled_12_month_clv").alias("max_12"),
            F.mean("modeled_24_month_clv").alias("mean_24"),
            F.expr("percentile_approx(modeled_24_month_clv, 0.5)").alias("median_24"),
            F.min("modeled_24_month_clv").alias("min_24"), F.max("modeled_24_month_clv").alias("max_24"),
        ).first().asDict()

        scored, scored_count = _score_held_out_test(spark, cfg)
        selected_customers = economics.select("household_key", "average_basket_value", "modeled_12_month_clv")
        offers = scored.join(selected_customers, "household_key", "inner")
        offers = add_offer_economics(offers,
            contribution_margin=float(f["contribution_margin"]),
            default_offer_cost=float(f["default_offer_cost"]))
        offer_rows = offers.count()
        if offer_rows != scored_count:
            raise ValueError(f"Offer scoring join changed row count: {scored_count} scored, {offer_rows} joined")
        out_offers = Path(cfg["paths"]["output_campaign_offer_economics"])
        out_offers.parent.mkdir(parents=True, exist_ok=True)
        offers.write.mode("overwrite").parquet(str(out_offers))
        offer_summary = offers.agg(
            F.mean("calibrated_response_probability").alias("mean_probability"),
            F.expr("percentile_approx(calibrated_response_probability, 0.5)").alias("median_probability"),
            F.mean("modeled_expected_offer_value").alias("mean_expected_value"),
            F.expr("percentile_approx(modeled_expected_offer_value, 0.5)").alias("median_expected_value"),
            F.sum((F.col("modeled_expected_offer_value") > 0).cast("long")).alias("above_zero"),
            F.sum((F.col("modeled_expected_offer_value") < 0).cast("long")).alias("below_zero"),
            F.mean("break_even_offer_cost").alias("mean_break_even_offer_cost"),
        ).first().asDict()
        sensitivity = _sensitivity_report(economics, offers, cfg)
        report = {
            "customer_data_quality": dqc,
            "customer_economics_rows": economics.count(),
            "customer_economics_path": str(out_customer),
            "campaign_offer_economics_rows": offer_rows,
            "campaign_offer_economics_path": str(out_offers),
            "clv_summary": clv_summary,
            "offer_summary": offer_summary,
            "assumptions": f,
            "monthly_discount_rate": annual_to_monthly_discount_rate(float(f["annual_discount_rate"])),
            "response_probability_scope": "Held-out test campaign-household rows only; target column was not read. Saved PyTorch model and validation-only Platt calibrator used without refitting.",
            "sensitivity": sensitivity,
            "warnings": ["Offer value is a one-average-basket response scenario, not a treatment effect or incremental value.",
                         "Revenue is annualized over the configured full dataset DAY span, not observed household tenure.",
                         "Customer quantity outliers are preserved in source features and do not enter CLV calculations."],
        }
        report_path = Path(cfg["paths"]["output_report"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(report, indent=2, allow_nan=False))
        print("\nCustomer economics schema:")
        economics.printSchema()
        print("Campaign offer economics schema:")
        offers.printSchema()
        return report
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/finance.yaml")
    args = parser.parse_args()
    run(args.config)
