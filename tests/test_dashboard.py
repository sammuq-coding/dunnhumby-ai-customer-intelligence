"""Unit checks for dashboard artifact contracts (without UI rendering tests)."""
import pandas as pd
import pytest

from src.dashboard.data_loader import (
    ARTIFACT_PATHS,
    artifact_path,
    join_offers_with_campaigns,
    observed_redemption_day_max,
    validate_dashboard_data,
)


def minimal_dashboard_data():
    customers = pd.DataFrame([{
        "household_key": 10, "total_sales": 100.0, "transaction_count": 2,
        "average_basket_value": 50.0, "total_quantity": 4, "unique_products": 3,
        "unique_departments": 2, "coupon_usage": 1, "total_retail_discount": -2.0,
        "total_coupon_discount": -1.0, "purchase_frequency": 0.1, "recency_days": 3,
    }])
    products = pd.DataFrame([{
        "PRODUCT_ID": 100, "DEPARTMENT": "GROCERY", "BRAND": "National",
        "total_units_sold": 3, "total_sales": 12.0, "number_of_households_purchasing": 1,
    }])
    campaign_households = pd.DataFrame([{
        "household_key": 10, "CAMPAIGN": 1, "campaign_type": "TypeA", "START_DAY": 10,
        "END_DAY": 12, "coupon_redemption_count": 1, "spend_before_campaign": 5.0,
        "spend_during_campaign": 10.0, "spend_after_campaign": 2.0,
        "campaign_window_complete": True, "post_campaign_observation_complete": True,
        "post_campaign_days_observed": 5,
    }])
    campaigns = pd.DataFrame([{
        "CAMPAIGN": 1, "campaign_type": "TypeA", "START_DAY": 10, "END_DAY": 12,
        "exposed_households": 1, "redeeming_households": 1, "descriptive_redemption_rate": 1.0,
        "spend_before_campaign": 5.0, "spend_during_campaign": 10.0,
        "spend_after_campaign": 2.0, "campaign_window_complete": True,
        "post_campaign_observation_complete": True, "post_campaign_days_observed": 5,
    }])
    economics = pd.DataFrame([{
        "household_key": 10, "modeled_12_month_clv": 12.0, "modeled_24_month_clv": 20.0,
        "assumed_contribution_margin": 0.25,
    }])
    offers = pd.DataFrame([{
        "household_key": 10, "CAMPAIGN": 1, "START_DAY": 10,
        "calibrated_response_probability": 0.3, "modeled_expected_offer_value": -2.0,
        "assumed_offer_cost": 5.0, "break_even_offer_cost": 3.0,
    }])
    model_rows = pd.DataFrame([{
        "household_key": 10, "CAMPAIGN": 1, "START_DAY": 10, "data_split": "test",
    }])
    return {
        "customers": customers, "products": products,
        "campaign_households": campaign_households, "campaigns": campaigns,
        "customer_economics": economics, "offers": offers, "model_rows": model_rows,
        "model_metrics": {"row_counts": {}, "baselines": {}, "model_metrics": {},
                          "validation_permutation_importance": []},
        "calibration_metrics": {"calibration_method": "platt", "validation": {}, "test": {},
                                "calibrator_fit_split": "validation"},
    }


def test_required_artifact_paths_are_defined_and_existing_workspace_outputs_exist():
    assert {"customers", "products", "campaigns", "offers", "model_metrics"}.issubset(ARTIFACT_PATHS)
    required = ["customers", "products", "campaign_households", "campaigns", "customer_economics",
                "offers", "finance_report", "model_metrics", "calibration_metrics"]
    missing = [name for name in required if not artifact_path(name).exists()]
    if missing:
        pytest.skip(f"Processed artifacts not built in this checkout: {missing}")
    assert not missing


def test_dashboard_schema_grains_and_metrics_validate():
    validate_dashboard_data(minimal_dashboard_data())


def test_duplicate_household_key_fails_validation():
    data = minimal_dashboard_data()
    data["customers"] = pd.concat([data["customers"], data["customers"]], ignore_index=True)
    with pytest.raises(ValueError, match="must be present and unique"):
        validate_dashboard_data(data)


def test_probability_bounds_are_validated():
    data = minimal_dashboard_data()
    data["offers"].loc[0, "calibrated_response_probability"] = 1.01
    with pytest.raises(ValueError, match="probabilities"):
        validate_dashboard_data(data)


def test_clv_must_be_finite_and_nonnegative():
    data = minimal_dashboard_data()
    data["customer_economics"].loc[0, "modeled_12_month_clv"] = float("nan")
    with pytest.raises(ValueError, match="modeled_12_month_clv"):
        validate_dashboard_data(data)


def test_campaign_ids_must_exist_in_campaign_features():
    data = minimal_dashboard_data()
    data["offers"].loc[0, "CAMPAIGN"] = 9
    with pytest.raises(ValueError, match="campaign IDs"):
        validate_dashboard_data(data)


def test_offer_campaign_join_preserves_scored_population():
    data = minimal_dashboard_data()
    before = len(data["offers"])
    merged = join_offers_with_campaigns(data["offers"], data["campaigns"])
    assert len(merged) == before
    assert merged.loc[0, "campaign_type"] == "TypeA"
    assert merged.loc[0, "END_DAY"] == 12


def test_redemption_coverage_is_read_from_verified_inventory():
    inventory = {"datasets": [{"filename": "coupon_redempt.csv", "numeric_summary": [
        {"summary": "min", "DAY": "225"}, {"summary": "max", "DAY": "704"},
    ]}]}
    assert observed_redemption_day_max(inventory) == 704
    assert observed_redemption_day_max({"datasets": []}) is None
