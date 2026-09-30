"""Small synthetic tests for temporal campaign-response modeling logic."""
import numpy as np
import pytest
from pyspark.sql import SparkSession

from src.models.pytorch.build_training_data import build_campaign_response_dataset, campaign_time_splits
from src.models.pytorch.train_response_model import _auc_metrics, classification_metrics, train_arrays


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.master("local[2]").appName("modeling-unit-tests").getOrCreate()
    yield session
    session.stop()


def test_features_are_strictly_pre_start_and_target_is_campaign_window(spark):
    campaigns = spark.createDataFrame([
        (1, "TypeA", 100, 102), (2, "TypeB", 110, 112), (3, "TypeC", 120, 122),
    ], ["CAMPAIGN", "DESCRIPTION", "START_DAY", "END_DAY"])
    assignments = spark.createDataFrame([(10, 1), (10, 2), (10, 3)], ["household_key", "CAMPAIGN"])
    tx = spark.createDataFrame([
        (10, 1001, 99, 1, 2, 10.0, -1.0, 0.0, 0.0),
        (10, 1002, 100, 1, 100, 1000.0, 0.0, -5.0, 0.0), # current-day row must be excluded
        (10, 1003, 110, 2, 1, 7.0, 0.0, 0.0, 0.0),
    ], ["household_key", "BASKET_ID", "DAY", "PRODUCT_ID", "QUANTITY", "SALES_VALUE", "RETAIL_DISC", "COUPON_DISC", "COUPON_MATCH_DISC"])
    products = spark.createDataFrame([(1, "A"), (2, None)], ["PRODUCT_ID", "DEPARTMENT"])
    redemptions = spark.createDataFrame([(10, 1, 101), (10, 2, 109), (10, 3, 122)],
                                        ["household_key", "CAMPAIGN", "DAY"])
    result = build_campaign_response_dataset(campaigns, assignments, tx, products, redemptions, lookback_days=10)
    rows = {r.CAMPAIGN: r for r in result.collect()}
    # DAY == START_DAY is excluded, while the day-110 transaction is valid
    # pre-start history for campaign 3 (START_DAY 120).
    assert set(rows) == {1, 2, 3}
    assert rows[1].historical_total_sales == 10.0
    assert rows[1].historical_basket_count == 1  # item rows/baskets aren't conflated
    assert rows[1].historical_recency_days == 1
    assert rows[1].target_redeemed == 1
    assert rows[1].prior_coupon_redemption_count == 0  # day 101 is after campaign 1 starts
    assert rows[2].historical_total_sales == 1000.0  # day 100 is pre-start; day 110 is not
    assert rows[2].historical_recency_days == 10
    assert rows[2].target_redeemed == 0  # day 109 is before campaign 2
    assert rows[2].prior_coupon_redemption_count == 2  # days 101 and 109, both pre-start
    assert rows[3].historical_total_sales == 7.0  # day 110 is pre-start for campaign 3
    assert rows[3].historical_recency_days == 10
    assert rows[3].target_redeemed == 1
    assert rows[3].prior_coupon_redemption_count == 0  # earlier events are outside this 10-day window


def test_chronological_splits_keep_start_day_ties(spark):
    campaigns = spark.createDataFrame([(1, 10, 11), (2, 20, 21), (3, 20, 22), (4, 30, 31), (5, 40, 41)],
                                      ["CAMPAIGN", "START_DAY", "END_DAY"])
    splits = campaign_time_splits(campaigns, .4, .2).collect()
    grouped = {}
    for row in splits:
        grouped.setdefault(row.START_DAY, set()).add(row.data_split)
    assert all(len(v) == 1 for v in grouped.values())
    assert grouped[10] == {"train"}
    assert grouped[40] == {"test"}


def test_tiny_torch_fit_and_metrics_are_finite():
    x = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([0, 0, 1, 1], dtype=np.float32)
    model, _, _, fit = train_arrays(x, y, x, y, max_epochs=4, patience=2, seed=3, hidden_units=4)
    import torch
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.tensor(x))).numpy()
    report = classification_metrics(y, probabilities)
    assert fit["best_epoch"] > 0
    assert report["roc_auc"] >= 0.5
    assert report["confusion_matrix"]["true_positive"] + report["confusion_matrix"]["false_negative"] == 2


def test_average_precision_is_invariant_to_order_when_scores_are_tied():
    y = np.array([1, 0, 0, 1], dtype=np.int64)
    _, ap = _auc_metrics(y, np.full(4, 0.25))
    assert ap == pytest.approx(0.5)
