"""Deterministic analytics tools over already-built project artifacts."""
from __future__ import annotations

from typing import Any

import pandas as pd

from src.dashboard.data_loader import observed_redemption_day_max


def _python(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _record(row: pd.Series, fields: list[str] | None = None) -> dict[str, Any]:
    selected = row if fields is None else row[fields]
    return {str(k): _python(v) for k, v in selected.items()}


def customer_lookup(data: dict, household_key: int) -> dict[str, Any]:
    """Return verified historical behavior, modeled value and existing scores."""
    customers = data["customers"]
    found = customers.loc[customers["household_key"] == int(household_key)]
    if found.empty:
        return {"status": "unavailable", "reason": f"household_key {household_key} is not in customer_features"}
    customer = _record(found.iloc[0])
    economics = data["customer_economics"].loc[
        data["customer_economics"]["household_key"] == int(household_key)]
    economics_record = _record(economics.iloc[0]) if not economics.empty else None
    score = data["offers"].loc[data["offers"]["household_key"] == int(household_key)]
    score_fields = ["CAMPAIGN", "START_DAY", "calibrated_response_probability",
                    "modeled_expected_offer_value", "assumed_offer_cost", "break_even_offer_cost"]
    predictions = [_record(row, score_fields) for _, row in score.sort_values(["START_DAY", "CAMPAIGN"]).iterrows()]
    return {
        "status": "found",
        "household_key": int(household_key),
        "observed_historical_behavior": customer,
        "modeled_financial_estimates": economics_record,
        "held_out_campaign_response_predictions": predictions,
        "prediction_scope": "Existing held-out test campaign-household rows only; probability predicts observed redemption among assigned households.",
    }


def campaign_lookup(data: dict, campaign_id: int) -> dict[str, Any]:
    """Return saved campaign summary and completeness flags."""
    campaigns = data["campaigns"]
    found = campaigns.loc[campaigns["CAMPAIGN"] == int(campaign_id)]
    if found.empty:
        return {"status": "unavailable", "reason": f"CAMPAIGN {campaign_id} is not in campaign_features"}
    fields = ["CAMPAIGN", "campaign_type", "START_DAY", "END_DAY", "exposed_households",
              "coupon_redemption_count", "redeeming_households", "descriptive_redemption_rate",
              "spend_before_campaign", "spend_during_campaign", "spend_after_campaign",
              "campaign_window_complete", "post_campaign_observation_complete", "post_campaign_days_observed"]
    return {"status": "found", "campaign": _record(found.iloc[0], fields),
            "interpretation": "Descriptive summary of observed assignment/redemption; not a causal effect."}


def rank_customers(data: dict, limit: int = 10, metric: str = "modeled_12_month_clv") -> dict[str, Any]:
    """Rank existing household economics rows deterministically."""
    allowed = {"modeled_12_month_clv", "modeled_24_month_clv"}
    if metric not in allowed:
        raise ValueError(f"metric must be one of {sorted(allowed)}")
    if not 1 <= int(limit) <= 100:
        raise ValueError("limit must be between 1 and 100")
    cols = ["household_key", metric, "historical_revenue", "basket_count"]
    frame = data["customer_economics"]
    cols = [c for c in cols if c in frame]
    ranked = frame.sort_values([metric, "household_key"], ascending=[False, True]).head(int(limit))
    return {"status": "found", "metric": metric, "limit": int(limit),
            "classification": "Modeled financial estimate based on configured assumptions.",
            "rows": [_record(row, cols) for _, row in ranked.iterrows()]}


def rank_campaigns(data: dict, limit: int = 10) -> dict[str, Any]:
    """Rank campaigns by descriptive rate among intervals fully covered by observed redemption days."""
    if not 1 <= int(limit) <= 30:
        raise ValueError("limit must be between 1 and 30")
    frame = data["campaigns"].copy()
    max_day = observed_redemption_day_max(data.get("dataset_inventory", {}))
    if max_day is not None:
        frame = frame[frame["END_DAY"] <= max_day]
    frame = frame.dropna(subset=["descriptive_redemption_rate"])
    cols = ["CAMPAIGN", "campaign_type", "START_DAY", "END_DAY", "exposed_households",
            "redeeming_households", "coupon_redemption_count", "descriptive_redemption_rate",
            "campaign_window_complete", "post_campaign_observation_complete"]
    frame = frame.sort_values(["descriptive_redemption_rate", "CAMPAIGN"], ascending=[False, True]).head(int(limit))
    return {"status": "found", "metric": "descriptive_redemption_rate",
            "coverage_rule": f"Campaign END_DAY <= observed coupon_redempt.DAY maximum ({max_day})" if max_day else "No coverage cutoff available; source campaign rows shown.",
            "interpretation": "Descriptive observed redemption rates do not estimate causal campaign lift.",
            "rows": [_record(row, cols) for _, row in frame.iterrows()]}


def model_metrics(data: dict, split: str = "test") -> dict[str, Any]:
    """Return saved raw/calibrated scores and test baselines without recomputation."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    saved = data["model_metrics"]
    result = {
        "status": "found", "split": split,
        "observations": saved["row_counts"][split],
        "raw_model": saved["model_metrics"][split],
        "actual_positive_rate": saved["model_metrics"][split].get("actual_positive_rate"),
    }
    calibration = data["calibration_metrics"]
    if split in {"validation", "test"}:
        result["raw_calibration_metrics"] = calibration[split]["raw"]
        result["calibrated_metrics"] = calibration[split]["calibrated"]
    if split == "test":
        result["baselines"] = saved["baselines"]["test"]
    result["calibration_method"] = calibration["calibration_method"]
    result["calibrator_fit_split"] = calibration["calibrator_fit_split"]
    result["threshold"] = saved["threshold"]
    result["interpretation"] = saved["interpretation"]
    return result


def clv_assumptions(data: dict) -> dict[str, Any]:
    """Return the finance report's unchanged configured assumptions."""
    return {"status": "found", "assumptions": data["finance_report"]["assumptions"],
            "monthly_discount_rate": data["finance_report"]["monthly_discount_rate"],
            "interpretation": "These are business assumptions for modeled CLV and offer scenarios, not learned outcomes."}


def offer_economics_lookup(data: dict, household_key: int | None = None,
                           campaign_id: int | None = None, limit: int = 20) -> dict[str, Any]:
    """Read existing offer value outputs; never recalculate their economics."""
    frame = data["offers"]
    if household_key is not None:
        frame = frame[frame["household_key"] == int(household_key)]
    if campaign_id is not None:
        frame = frame[frame["CAMPAIGN"] == int(campaign_id)]
    cols = ["household_key", "CAMPAIGN", "START_DAY", "calibrated_response_probability",
            "expected_contribution_if_response", "assumed_offer_cost", "break_even_offer_cost",
            "modeled_expected_offer_value"]
    cols = [c for c in cols if c in frame]
    frame = frame.sort_values(["modeled_expected_offer_value", "household_key"], ascending=[False, True]).head(limit)
    return {"status": "found" if not frame.empty else "unavailable",
            "reason": None if not frame.empty else "No saved held-out test offer-economics row matches this filter.",
            "row_count_returned": len(frame),
            "rows": [_record(row, cols) for _, row in frame.iterrows()],
            "interpretation": "Modeled expected offer value under the stored response scenario; not incremental profit."}


def deterministic_tool(question: str, data: dict) -> dict[str, Any] | None:
    """Resolve common quantitative natural-language asks to deterministic tools."""
    import re
    text = question.lower()
    household_match = re.search(r"\b(?:household|customer|hh)(?:\s+key)?\s*#?\s*(\d+)\b", text)
    campaign_match = re.search(r"\bcampaign(?:\s+id)?\s*#?\s*(\d+)\b", text)
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
    limit_match = re.search(r"\b(?:top|highest|first|show|list)?\s*(\d{1,3})\s+(?:customers|households|campaigns)\b", text)
    if not limit_match:
        word_match = re.search(r"\b(" + "|".join(number_words) + r")\s+(?:customers|households|campaigns)\b", text)
        limit = number_words[word_match.group(1)] if word_match else 10
    else:
        limit = int(limit_match.group(1))

    if household_match:
        key = int(household_match.group(1))
        if any(w in text for w in ("offer", "economics", "break-even", "break even")):
            return {"tool": "offer_economics_lookup", "result": offer_economics_lookup(data, household_key=key)}
        return {"tool": "customer_lookup", "result": customer_lookup(data, key)}
    if campaign_match and any(w in text for w in ("lookup", "details", "performance", "redemption", "spend", "campaign type")):
        cid = int(campaign_match.group(1))
        return {"tool": "campaign_lookup", "result": campaign_lookup(data, cid)}
    if ("campaign" in text and any(w in text for w in ("highest", "top", "rank", "best"))
            and any(w in text for w in ("redemption", "redeeming", "rate"))):
        return {"tool": "rank_campaigns", "result": rank_campaigns(data, min(limit, 30))}
    if any(w in text for w in ("highest", "top", "rank", "customers with most")) and any(w in text for w in ("clv", "customer value")):
        metric = "modeled_24_month_clv" if "24" in text else "modeled_12_month_clv"
        return {"tool": "rank_customers", "result": rank_customers(data, limit, metric)}
    if any(w in text for w in ("assumption", "retention rate", "contribution margin", "discount rate")) and any(w in text for w in ("clv", "finance", "offer", "used", "configured")):
        return {"tool": "clv_assumptions", "result": clv_assumptions(data)}
    calibration_explanation = "calibrat" in text and any(w in text for w in ("what does", "what is", "explain", "how does", "how is"))
    if not calibration_explanation and any(w in text for w in ("model", "roc-auc", "average precision", "calibration", "brier", "baseline", "evaluated")):
        return {"tool": "model_metrics", "result": model_metrics(data, "test")}
    if "offer" in text and any(w in text for w in ("value", "economics", "break-even", "break even")):
        return {"tool": "offer_economics_lookup", "result": offer_economics_lookup(data, campaign_id=int(campaign_match.group(1)) if campaign_match else None)}
    return None
