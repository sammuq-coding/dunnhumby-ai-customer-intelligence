"""Cached artifact reads and integrity checks for the Streamlit application."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATHS = {
    "customers": "data/processed/features/customer_features",
    "products": "data/processed/features/product_features",
    "campaign_households": "data/processed/features/campaign_household_features",
    "campaigns": "data/processed/features/campaign_features",
    "customer_economics": "data/processed/finance/customer_economics.parquet",
    "offers": "data/processed/finance/campaign_offer_economics.parquet",
    "finance_report": "data/processed/finance/finance_report.json",
    "model_metrics": "data/processed/modeling/response_model_metrics.json",
    "calibration_metrics": "data/processed/modeling/response_model_calibration_metrics.json",
    "dataset_inventory": "data/processed/dataset_inventory.json",
    "model_rows": "data/processed/modeling/campaign_response_dataset.parquet",
    "model_config": "configs/modeling.yaml",
    "finance_config": "configs/finance.yaml",
    "demographics": "data/raw/hh_demographic.csv",
}


def artifact_path(name: str, root: str | Path | None = None) -> Path:
    """Resolve a known project artifact path, useful for tests and CLI use."""
    if name not in ARTIFACT_PATHS:
        raise KeyError(f"Unknown dashboard artifact: {name}")
    return Path(root or PROJECT_ROOT) / ARTIFACT_PATHS[name]


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required dashboard Parquet artifact not found: {file_path}")
    # These Spark 4.2 Parquet outputs are not currently readable by the local
    # PyArrow 19 runtime (it raises a repetition-level histogram error), so
    # use the project's Spark reader and convert only the small analytical
    # tables to pandas for presentation. The 36M-row causal table is excluded.
    frame = _spark_session().read.parquet(str(file_path))
    if columns:
        frame = frame.select(*columns)
    return frame.toPandas()


@st.cache_resource(show_spinner=False)
def _spark_session():
    """Share a lightweight local Spark session across Streamlit reruns."""
    from pyspark.sql import SparkSession
    return (SparkSession.builder.appName("dunnhumby-customer-intelligence-dashboard")
            .master("local[2]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.session.timeZone", "UTC")
            .getOrCreate())


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required dashboard JSON artifact not found: {file_path}")
    return json.loads(file_path.read_text())


@st.cache_data(show_spinner=False)
def _read_yaml(path: str) -> dict[str, Any]:
    import yaml
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Required dashboard YAML configuration not found: {file_path}")
    return yaml.safe_load(file_path.read_text())


@st.cache_data(show_spinner=False)
def _read_demographics(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        # Demographics are supplemental; the core dashboard remains available.
        return pd.DataFrame(columns=["household_key"])
    return pd.read_csv(file_path)


def load_dashboard_data(root: str | Path | None = None) -> dict[str, Any]:
    """Read only processed analytical outputs plus the small demographic file."""
    root_path = Path(root or PROJECT_ROOT)
    def get(name: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
        return _read_parquet(str(artifact_path(name, root_path)), columns)

    data: dict[str, Any] = {
        "customers": get("customers"),
        "products": get("products"),
        "campaign_households": get("campaign_households"),
        "campaigns": get("campaigns"),
        "customer_economics": get("customer_economics"),
        "offers": get("offers"),
        "finance_report": _read_json(str(artifact_path("finance_report", root_path))),
        "model_metrics": _read_json(str(artifact_path("model_metrics", root_path))),
        "calibration_metrics": _read_json(str(artifact_path("calibration_metrics", root_path))),
        "dataset_inventory": _read_json(str(artifact_path("dataset_inventory", root_path))),
        "model_rows": get("model_rows", ("household_key", "CAMPAIGN", "START_DAY", "data_split")),
        "model_config": _read_yaml(str(artifact_path("model_config", root_path))),
        "finance_config": _read_yaml(str(artifact_path("finance_config", root_path))),
        "demographics": _read_demographics(str(artifact_path("demographics", root_path))),
    }
    validate_dashboard_data(data)
    return data


REQUIRED_COLUMNS = {
    "customers": {"household_key", "total_sales", "transaction_count", "average_basket_value",
                  "total_quantity", "unique_products", "unique_departments", "coupon_usage",
                  "total_retail_discount", "total_coupon_discount", "purchase_frequency", "recency_days"},
    "products": {"PRODUCT_ID", "DEPARTMENT", "BRAND", "total_units_sold", "total_sales",
                  "number_of_households_purchasing"},
    "campaign_households": {"household_key", "CAMPAIGN", "campaign_type", "START_DAY", "END_DAY",
                            "coupon_redemption_count", "spend_before_campaign", "spend_during_campaign",
                            "spend_after_campaign", "campaign_window_complete",
                            "post_campaign_observation_complete", "post_campaign_days_observed"},
    "campaigns": {"CAMPAIGN", "campaign_type", "START_DAY", "END_DAY", "exposed_households",
                   "redeeming_households", "descriptive_redemption_rate", "spend_before_campaign",
                   "spend_during_campaign", "spend_after_campaign", "campaign_window_complete",
                   "post_campaign_observation_complete", "post_campaign_days_observed"},
    "customer_economics": {"household_key", "modeled_12_month_clv", "modeled_24_month_clv",
                            "assumed_contribution_margin"},
    "offers": {"household_key", "CAMPAIGN", "START_DAY", "calibrated_response_probability",
               "modeled_expected_offer_value", "assumed_offer_cost", "break_even_offer_cost"},
    "model_rows": {"household_key", "CAMPAIGN", "START_DAY", "data_split"},
}


def validate_dashboard_data(data: dict[str, Any]) -> None:
    """Fail early when saved artifacts violate expected schema/grain/ranges."""
    for name, required in REQUIRED_COLUMNS.items():
        if name not in data:
            raise ValueError(f"Missing dashboard dataset: {name}")
        missing = required - set(data[name].columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
    for name, key in (("customers", "household_key"), ("customer_economics", "household_key"),
                      ("products", "PRODUCT_ID"), ("campaigns", "CAMPAIGN")):
        frame = data[name]
        if frame[key].isna().any() or frame[key].duplicated().any():
            raise ValueError(f"{name}.{key} must be present and unique")
    assignments = data["campaign_households"]
    if assignments[["household_key", "CAMPAIGN"]].duplicated().any():
        raise ValueError("campaign_households must be unique at household + campaign grain")
    valid_campaigns = set(data["campaigns"]["CAMPAIGN"].dropna().tolist())
    for name in ("campaign_households", "offers", "model_rows"):
        invalid = set(data[name]["CAMPAIGN"].dropna().tolist()) - valid_campaigns
        if invalid:
            raise ValueError(f"{name} contains campaign IDs absent from campaign_features: {sorted(invalid)}")
    valid_households = set(data["customers"]["household_key"].dropna().tolist())
    if not set(data["offers"]["household_key"].dropna().tolist()).issubset(valid_households):
        raise ValueError("Offer rows contain household keys absent from customer_features")
    if data["offers"][["household_key", "CAMPAIGN"]].duplicated().any():
        raise ValueError("offers must be unique at household + campaign grain")
    probabilities = pd.to_numeric(data["offers"]["calibrated_response_probability"], errors="coerce")
    if probabilities.isna().any() or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Calibrated response probabilities must be finite and in [0, 1]")
    for col in ("modeled_12_month_clv", "modeled_24_month_clv"):
        values = pd.to_numeric(data["customer_economics"][col], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{col} must contain only finite nonnegative values")
    if len(data["campaigns"]) == 0 or len(data["customers"]) == 0:
        raise ValueError("Customer and campaign tables must not be empty")
    metrics = data.get("model_metrics", {})
    if not {"row_counts", "baselines", "model_metrics", "validation_permutation_importance"}.issubset(metrics):
        raise ValueError("Saved model metrics JSON is missing expected report sections")
    calibration = data.get("calibration_metrics", {})
    if not {"calibration_method", "validation", "test", "calibrator_fit_split"}.issubset(calibration):
        raise ValueError("Saved calibration metrics JSON is incomplete")
    fit_split = str(data["calibration_metrics"]["calibrator_fit_split"]).lower()
    if not fit_split.startswith("validation"):
        raise ValueError("The dashboard expects a validation-fitted frozen calibrator")


def join_offers_with_campaigns(offers: pd.DataFrame, campaigns: pd.DataFrame) -> pd.DataFrame:
    """Attach campaign descriptors without multiplying scored observations."""
    before = len(offers)
    result = offers.merge(
        campaigns[["CAMPAIGN", "campaign_type", "END_DAY"]],
        on="CAMPAIGN", how="left", validate="many_to_one", suffixes=("", "_campaign"),
        indicator=True,
    )
    if len(result) != before or (result["_merge"] != "both").any():
        raise ValueError("Campaign metadata join changed or failed to cover the scored population")
    return result.drop(columns="_merge")


def observed_redemption_day_max(inventory: dict[str, Any]) -> int | None:
    """Read the observed coupon_redempt.DAY maximum from the verified inventory."""
    datasets = inventory.get("datasets", [])
    record = next((x for x in datasets if x.get("filename") == "coupon_redempt.csv"), None)
    if not record:
        return None
    for summary in record.get("numeric_summary", []):
        if summary.get("summary") == "max" and summary.get("DAY") is not None:
            return int(float(summary["DAY"]))
    return None
