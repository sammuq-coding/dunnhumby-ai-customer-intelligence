"""RAG and assistant tests use synthetic data and a deterministic mock provider."""
import copy
import json

import pandas as pd
import pytest

from src.dashboard.data_loader import validate_dashboard_data
from src.rag.analytics_tools import (
    campaign_lookup,
    clv_assumptions,
    customer_lookup,
    deterministic_tool,
    model_metrics,
    offer_economics_lookup,
    rank_campaigns,
    rank_customers,
)
from src.rag.assistant import MockLLMProvider, answer_question
from src.rag.context_builder import build_knowledge_base
from src.rag.retriever import BM25Retriever


def sample_data():
    return {
        "products": pd.DataFrame([{"PRODUCT_ID": 1, "DEPARTMENT": "GROCERY", "BRAND": "National",
                                  "total_units_sold": 2, "total_sales": 3.0,
                                  "number_of_households_purchasing": 1}]),
        "customers": pd.DataFrame([
            {"household_key": 123, "total_sales": 500., "transaction_count": 12,
             "average_basket_value": 41.67, "purchase_frequency": .2, "recency_days": 3,
             "total_quantity": 50, "unique_products": 15, "unique_departments": 5,
             "coupon_usage": 2, "total_retail_discount": -10., "total_coupon_discount": -3.,
             "coupon_redemption_count": 1},
            {"household_key": 124, "total_sales": 250., "transaction_count": 5,
             "average_basket_value": 50., "purchase_frequency": .1, "recency_days": 10,
             "total_quantity": 20, "unique_products": 7, "unique_departments": 3,
             "coupon_usage": 0, "total_retail_discount": -2., "total_coupon_discount": 0.,
             "coupon_redemption_count": 0},
        ]),
        "customer_economics": pd.DataFrame([
            {"household_key": 123, "historical_revenue": 500., "basket_count": 12,
             "modeled_12_month_clv": 100., "modeled_24_month_clv": 160., "assumed_contribution_margin": .25},
            {"household_key": 124, "historical_revenue": 250., "basket_count": 5,
             "modeled_12_month_clv": 80., "modeled_24_month_clv": 120., "assumed_contribution_margin": .25},
        ]),
        "campaigns": pd.DataFrame([
            {"CAMPAIGN": 1, "campaign_type": "TypeA", "START_DAY": 20, "END_DAY": 30,
             "exposed_households": 100, "coupon_redemption_count": 15, "redeeming_households": 12,
             "descriptive_redemption_rate": .12, "spend_before_campaign": 1000.,
             "spend_during_campaign": 1200., "spend_after_campaign": 900.,
             "campaign_window_complete": True, "post_campaign_observation_complete": True,
             "post_campaign_days_observed": 20},
            {"CAMPAIGN": 2, "campaign_type": "TypeB", "START_DAY": 40, "END_DAY": 50,
             "exposed_households": 80, "coupon_redemption_count": 20, "redeeming_households": 18,
             "descriptive_redemption_rate": .225, "spend_before_campaign": 800.,
             "spend_during_campaign": 900., "spend_after_campaign": 950.,
             "campaign_window_complete": True, "post_campaign_observation_complete": True,
             "post_campaign_days_observed": 10},
        ]),
        "campaign_households": pd.DataFrame([
            {"household_key": 123, "CAMPAIGN": 1, "campaign_type": "TypeA", "START_DAY": 20,
             "END_DAY": 30, "coupon_redemption_count": 1, "spend_before_campaign": 50.,
             "spend_during_campaign": 60., "spend_after_campaign": 40.,
             "campaign_window_complete": True, "post_campaign_observation_complete": True,
             "post_campaign_days_observed": 20},
        ]),
        "offers": pd.DataFrame([
            {"household_key": 123, "CAMPAIGN": 1, "START_DAY": 20,
             "calibrated_response_probability": .4, "expected_contribution_if_response": 10.,
             "assumed_offer_cost": 5., "break_even_offer_cost": 4., "modeled_expected_offer_value": -1.},
            {"household_key": 124, "CAMPAIGN": 2, "START_DAY": 40,
             "calibrated_response_probability": .2, "expected_contribution_if_response": 12.,
             "assumed_offer_cost": 5., "break_even_offer_cost": 2.4, "modeled_expected_offer_value": -2.6},
        ]),
        "model_rows": pd.DataFrame([
            {"household_key": 123, "CAMPAIGN": 1, "START_DAY": 20, "data_split": "train"},
            {"household_key": 124, "CAMPAIGN": 2, "START_DAY": 40, "data_split": "test"},
        ]),
        "model_metrics": {
            "row_counts": {"train": 10, "validation": 4, "test": 5},
            "baselines": {"test": {"majority_class": {"roc_auc": .5},
                                    "constant_training_rate": {"roc_auc": .5},
                                    "prior_coupon_redemption_indicator": {"roc_auc": .6}}},
            "model_metrics": {s: {"roc_auc": .8, "pr_auc_average_precision": .4,
                                  "actual_positive_rate": .2, "precision": .5, "recall": .4,
                                  "f1": .44, "brier_score": .16} for s in ("train", "validation", "test")},
            "validation_permutation_importance": [], "threshold": .5,
            "interpretation": "observational prediction, not a treatment effect",
        },
        "calibration_metrics": {
            "calibration_method": "validation-only Platt scaling",
            "calibrator_fit_split": "validation only",
            "validation": {"raw": {"brier_score": .2}, "calibrated": {"brier_score": .15}},
            "test": {"raw": {"brier_score": .2}, "calibrated": {"brier_score": .16}},
        },
        "finance_report": {
            "assumptions": {"contribution_margin": .25, "monthly_retention_rate": .9,
                            "annual_discount_rate": .1, "observed_history_days": 711},
            "monthly_discount_rate": .008,
        },
        "dataset_inventory": {"datasets": [{"filename": "coupon_redempt.csv", "numeric_summary": [
            {"summary": "max", "DAY": "60"}]}]},
    }


@pytest.fixture
def docs():
    return [
        {"id": "clv", "title": "CLV methodology", "source": "src/finance/README.md",
         "category": "methodology", "text": "Modeled CLV uses revenue, contribution margin, monthly activity retention, effective monthly discounting, and a dataset-span assumption. It is not observed future value."},
        {"id": "causal", "title": "Campaign limitations", "source": "data/README.md",
         "category": "limitations", "text": "Campaign exposure is observational. Redemption and spend comparisons are descriptive and do not establish causal lift or treatment effects."},
    ]


def test_customer_lookup_returns_observed_and_separate_modeled_data():
    result = customer_lookup(sample_data(), 123)
    assert result["status"] == "found"
    assert result["observed_historical_behavior"]["total_sales"] == 500
    assert result["modeled_financial_estimates"]["modeled_12_month_clv"] == 100
    assert result["held_out_campaign_response_predictions"][0]["calibrated_response_probability"] == .4


def test_campaign_lookup_returns_saved_descriptive_summary():
    result = campaign_lookup(sample_data(), 2)
    assert result["status"] == "found"
    assert result["campaign"]["descriptive_redemption_rate"] == .225
    assert result["campaign"]["post_campaign_observation_complete"] is True
    assert "not a causal" in result["interpretation"]


def test_customer_ranking_is_deterministic_and_top_n():
    data = sample_data()
    first = rank_customers(data, limit=1)
    second = rank_customers(data, limit=1)
    assert first == second
    assert first["rows"][0]["household_key"] == 123


def test_campaign_ranking_uses_observed_coverage():
    data = sample_data()
    data["campaigns"].loc[len(data["campaigns"])] = {
        **data["campaigns"].iloc[0].to_dict(), "CAMPAIGN": 3, "END_DAY": 70,
        "descriptive_redemption_rate": .99,
    }
    result = rank_campaigns(data, limit=1)
    assert result["rows"][0]["CAMPAIGN"] == 2
    assert "<= observed coupon_redempt.DAY maximum (60)" in result["coverage_rule"]


def test_saved_model_metrics_include_calibration_and_baselines():
    result = model_metrics(sample_data(), "test")
    assert result["observations"] == 5
    assert result["raw_model"]["roc_auc"] == .8
    assert result["calibrated_metrics"]["brier_score"] == .16
    assert result["baselines"]["majority_class"]["roc_auc"] == .5


def test_clv_assumptions_are_returned_without_recalculation():
    result = clv_assumptions(sample_data())
    assert result["assumptions"]["monthly_retention_rate"] == .9
    assert result["monthly_discount_rate"] == .008


def test_offer_economics_reads_saved_outputs():
    result = offer_economics_lookup(sample_data(), household_key=123)
    assert result["rows"][0]["modeled_expected_offer_value"] == -1
    assert result["rows"][0]["break_even_offer_cost"] == 4


@pytest.mark.parametrize("key", [99999, -1])
def test_missing_or_invalid_customer_id_is_unavailable(key):
    assert customer_lookup(sample_data(), key)["status"] == "unavailable"


@pytest.mark.parametrize("key", [999, -1])
def test_missing_or_invalid_campaign_id_is_unavailable(key):
    assert campaign_lookup(sample_data(), key)["status"] == "unavailable"


def test_ranking_invalid_limit_and_metric_are_rejected():
    with pytest.raises(ValueError):
        rank_customers(sample_data(), 101)
    with pytest.raises(ValueError):
        rank_customers(sample_data(), metric="total_sales")
    with pytest.raises(ValueError):
        rank_campaigns(sample_data(), 31)


def test_probabilities_out_of_bounds_fail_dashboard_integrity_validation():
    data = sample_data()
    data["offers"].loc[0, "calibrated_response_probability"] = 1.5
    with pytest.raises(ValueError, match="probabilities"):
        validate_dashboard_data(data)


def test_mock_provider_and_assistant_are_deterministic(docs):
    data = sample_data()
    mock = MockLLMProvider(responses=[
        json.dumps({"intent": "rank_customers", "limit": 1}),
        "Household 123 has the highest modeled CLV in the returned ranking.",
    ])
    first = answer_question("Which customer has the highest CLV?", data, docs, mock)
    assert first["answer"].startswith("Household 123")
    assert first["tool"] == "rank_customers"
    assert first["mode"] == "llm_grounded"
    assert len(mock.calls) == 2

    without_llm_a = answer_question("Which five campaigns had the highest redemption rates?", data, docs)
    without_llm_b = answer_question("Which five campaigns had the highest redemption rates?", data, docs)
    assert without_llm_a == without_llm_b
    assert without_llm_a["tool"] == "rank_campaigns"


def test_context_retrieval_and_building_from_docs(tmp_path, docs):
    results = BM25Retriever(docs).search("why is campaign performance observational rather than causal")
    assert results and results[0]["id"] == "causal"
    (tmp_path / "README.md").write_text("# Project\n\nAnalytics platform notes.")
    built = build_knowledge_base(sample_data(), tmp_path)
    assert any(d["source"] == "README.md" for d in built)
    assert any(d["id"] == "finance-summary" for d in built)


def test_question_router_and_unsupported_request_behavior(docs):
    data = sample_data()
    routed = deterministic_tool("What is household 123's recent purchasing behavior?", data)
    assert routed["tool"] == "customer_lookup"
    top_five = deterministic_tool("Show the 10 customers with highest modeled CLV", data)
    assert top_five["result"]["limit"] == 10
    cause = answer_question("Did campaign 18 cause sales to increase?", data, docs)
    assert "does not establish a causal effect" in cause["answer"]
    unsupported = answer_question("What was the weather on Neptune in 1942?", data, docs)
    assert "unavailable" in unsupported["answer"].lower()


def test_llm_rejected_question_is_not_answered_from_adjacent_retrieval(docs):
    mock = MockLLMProvider(responses=[
        json.dumps({"intent": "unsupported"}),
        "Invented unsupported answer that must never be returned.",
    ])
    result = answer_question("What p-value proves campaign lift?", sample_data(), docs, mock)
    assert "unavailable" in result["answer"].lower()
    assert "Invented unsupported" not in result["answer"]
    assert len(mock.calls) == 1  # Classification only; no synthesis request is sent.


def test_quantitative_answers_include_source_transparency(docs):
    result = answer_question("Tell me about household 123", sample_data(), docs)
    assert "customer_features (Parquet)" in result["source"]
    assert result["evidence_type"].startswith("Observed household behavior")
