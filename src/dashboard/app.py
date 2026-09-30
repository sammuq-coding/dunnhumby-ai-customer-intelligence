"""Streamlit interface for the existing AI Customer Intelligence artifacts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.dashboard.data_loader import (
    join_offers_with_campaigns,
    load_dashboard_data,
    observed_redemption_day_max,
)
from src.rag.assistant_ui import render_assistant_page


st.set_page_config(
    page_title="AI Customer Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVIGATION = [
    "Executive Overview", "Customer Explorer", "Campaign Analytics",
    "Customer Value / CLV", "Offer Economics", "Model Performance",
    "Methodology & Assumptions", "AI Analytics Assistant",
]


def _format_number(value, digits: int = 1) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _format_rate(value) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _chart(fig, height: int = 360):
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=12, r=12, t=42, b=12),
        font=dict(family="Inter, Arial, sans-serif", color="#1E2A36"),
        title_font=dict(size=17, color="#19324A"),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def _observed_campaigns(data: dict, campaigns: pd.DataFrame | None = None) -> pd.DataFrame:
    """Filter campaign intervals to the verified coupon redemption coverage."""
    frame = (campaigns if campaigns is not None else data["campaigns"]).copy()
    max_day = observed_redemption_day_max(data["dataset_inventory"])
    if max_day is not None:
        frame = frame[frame["END_DAY"] <= max_day]
    return frame


def _top_kpis(data: dict):
    customers = data["customers"]
    econ = data["customer_economics"]
    campaigns = data["campaigns"]
    scored = data["offers"]
    observed = _observed_campaigns(data)
    exp = int(observed["exposed_households"].sum()) if not observed.empty else 0
    red = int(observed["redeeming_households"].sum()) if not observed.empty else 0
    overall_rate = red / exp if exp else np.nan
    kpis = [
        ("Households", f"{len(customers):,}", "One row per household"),
        ("Historical sales", _format_number(customers["total_sales"].sum()), "Source SALES_VALUE units"),
        ("Observed baskets", f"{int(customers['transaction_count'].sum()):,}", "Household basket counts summed"),
        ("Average basket value", _format_number(customers["average_basket_value"].mean()), "Unweighted mean across households"),
        ("Mean 12-month modeled CLV", _format_number(econ["modeled_12_month_clv"].mean()), "Assumption-based estimate"),
        ("Median 12-month modeled CLV", _format_number(econ["modeled_12_month_clv"].median()), "Assumption-based estimate"),
        ("Descriptive redemption rate", _format_rate(overall_rate), f"{len(observed)} campaign intervals within observed redemption coverage"),
        ("Mean calibrated response probability", _format_rate(scored["calibrated_response_probability"].mean()), f"{len(scored):,} held-out test campaign-household rows"),
        ("Mean modeled offer value", _format_number(scored["modeled_expected_offer_value"].mean()), "Scenario-based estimate"),
    ]
    cols = st.columns(3)
    for i, (label, value, help_text) in enumerate(kpis):
        cols[i % 3].metric(label, value, help=help_text)


def _page_overview(data: dict):
    st.title("AI Customer Intelligence Platform")
    st.caption("A decision-support view over verified Complete Journey analytics, predictive scores, and assumption-based economics.")
    _top_kpis(data)
    left, right = st.columns(2)
    with left:
        st.subheader("Household modeled CLV")
        _chart(px.histogram(data["customer_economics"], x="modeled_12_month_clv", nbins=40,
                            labels={"modeled_12_month_clv": "Modeled 12-month CLV (source value units)"}), 330)
    with right:
        campaigns = _observed_campaigns(data).sort_values("CAMPAIGN")
        st.subheader("Descriptive campaign redemption rate")
        _chart(px.bar(campaigns, x="CAMPAIGN", y="descriptive_redemption_rate", color="campaign_type",
                      labels={"CAMPAIGN": "Campaign", "descriptive_redemption_rate": "Redeeming assigned households / exposed households",
                              "campaign_type": "Campaign type"}), 330)
        st.caption("Only campaign intervals within the verified observed coupon-redemption day coverage are shown.")
    left, right = st.columns(2)
    with left:
        st.subheader("Calibrated response scores")
        _chart(px.histogram(data["offers"], x="calibrated_response_probability", nbins=30,
                            labels={"calibrated_response_probability": "Predicted probability of observed redemption"}), 320)
    with right:
        st.subheader("Modeled offer value scenarios")
        _chart(px.histogram(data["offers"], x="modeled_expected_offer_value", nbins=35,
                            labels={"modeled_expected_offer_value": "Modeled expected offer value (source value units)"}), 320)
    st.subheader("Highest-revenue products")
    products = data["products"].nlargest(10, "total_sales").sort_values("total_sales")
    _chart(px.bar(products, x="total_sales", y="PRODUCT_ID", orientation="h", color="DEPARTMENT",
                  labels={"total_sales": "Historical sales (source value units)", "PRODUCT_ID": "Product ID",
                          "DEPARTMENT": "Department"}), 360)
    st.info("Campaign rates are descriptive. Response scores predict observed redemption among assigned households. Offer values and CLV are modeled under explicit assumptions; neither is a causal or incremental effect.")


def _page_customer(data: dict):
    st.title("Customer Explorer")
    customers = data["customers"]
    econ = data["customer_economics"]
    household_ids = sorted(int(x) for x in customers["household_key"].dropna().unique())
    household = st.selectbox("Household key", household_ids, index=0)
    row = customers.loc[customers["household_key"] == household].iloc[0]
    value = econ.loc[econ["household_key"] == household].iloc[0]
    demos = data["demographics"]
    demo = demos.loc[demos["household_key"] == household] if "household_key" in demos else pd.DataFrame()

    st.subheader("Customer profile")
    if not demo.empty:
        st.caption("Demographic classifications are shown using source labels; generic classification codes are not interpreted.")
        st.dataframe(demo.drop(columns=["household_key"]).T.rename(columns={demo.index[0]: "Observed source value"}),
                     use_container_width=True)
    else:
        st.caption("No demographic row is available for this household in the partial demographic file.")

    st.subheader("Historical behavior")
    cols = st.columns(4)
    history_metrics = [
        ("Historical sales", row.total_sales), ("Basket count", row.transaction_count),
        ("Average basket value", row.average_basket_value), ("Total quantity", row.total_quantity),
        ("Unique products", row.unique_products), ("Unique departments", row.unique_departments),
        ("Coupon-using baskets", row.coupon_usage), ("Retail discount", row.total_retail_discount),
        ("Coupon discount", row.total_coupon_discount), ("Purchase frequency", row.purchase_frequency),
        ("Recency (DAY indices)", row.recency_days), ("Purchase day range", row.purchase_day_range),
    ]
    for i, (label, v) in enumerate(history_metrics):
        cols[i % 4].metric(label, _format_number(v, 3 if label == "Purchase frequency" else 1))
    st.caption("Historical sales and discounts are source metrics, not profit. DAY values are ordered indices, not dates.")

    st.subheader("Customer economics")
    st.caption("Modeled estimates under the finance configuration; not observed future value.")
    finance = data["finance_report"]["assumptions"]
    cols = st.columns(4)
    cols[0].metric("12-month modeled CLV", _format_number(value.modeled_12_month_clv))
    cols[1].metric("24-month modeled CLV", _format_number(value.modeled_24_month_clv))
    cols[2].metric("Contribution margin", _format_rate(finance["contribution_margin"]))
    cols[3].metric("Monthly activity assumption", _format_rate(finance["monthly_retention_rate"]))
    st.caption(f"Annual discount rate: {_format_rate(finance['annual_discount_rate'])} · Revenue denominator: {finance['observed_history_days']} dataset DAY indices ({finance['days_per_month']} DAY indices/month).")

    st.subheader("Held-out campaign response scores")
    offers = data["offers"].loc[data["offers"]["household_key"] == household]
    if offers.empty:
        st.info("This household has no score in the held-out test campaign-household population. No probability is inferred for it.")
    else:
        offer_details = join_offers_with_campaigns(offers, data["campaigns"])
        display = offer_details[["CAMPAIGN", "campaign_type", "START_DAY", "END_DAY",
                                 "calibrated_response_probability", "modeled_expected_offer_value",
                                 "assumed_offer_cost", "break_even_offer_cost"]].copy()
        display["calibrated_response_probability"] = display["calibrated_response_probability"].map(_format_rate)
        st.dataframe(display.sort_values("START_DAY"), use_container_width=True, hide_index=True)
        st.caption("Scores use the frozen validation-fitted Platt calibrator. Redemption outcome is not shown as a model input. Offer values use the one-average-basket response scenario.")


def _page_campaigns(data: dict):
    st.title("Campaign Analytics")
    st.caption("Observational campaign assignment and redemption summaries. No randomized treatment is assumed.")
    source = data["campaigns"].copy()
    day_max = observed_redemption_day_max(data["dataset_inventory"])
    types = sorted(source["campaign_type"].dropna().unique())
    selected_types = st.multiselect("Campaign type", types, default=types)
    within_coverage = st.toggle("Limit to campaigns ending within observed coupon redemption coverage", value=True,
                                help=f"Verified maximum coupon_redempt.DAY: {day_max}.")
    filtered = source[source["campaign_type"].isin(selected_types)]
    if within_coverage and day_max is not None:
        filtered = filtered[filtered["END_DAY"] <= day_max]
    if filtered.empty:
        st.warning("No campaigns match these filters.")
        return
    cols = st.columns(4)
    exposure = int(filtered["exposed_households"].sum())
    redeeming = int(filtered["redeeming_households"].sum())
    cols[0].metric("Campaigns", len(filtered))
    cols[1].metric("Assigned household-campaigns", f"{exposure:,}")
    cols[2].metric("Redeeming household-campaigns", f"{redeeming:,}")
    cols[3].metric("Pooled descriptive redemption rate", _format_rate(redeeming / exposure if exposure else np.nan))
    st.caption("Counts sum campaign-level assignments; a household exposed to multiple campaigns appears once in each campaign. Redemption rates are descriptive, not causal.")

    selected_campaigns = st.multiselect("Campaigns to display", sorted(filtered["CAMPAIGN"].astype(int)),
                                        default=sorted(filtered["CAMPAIGN"].astype(int)))
    view = filtered[filtered["CAMPAIGN"].isin(selected_campaigns)].copy()
    left, right = st.columns(2)
    with left:
        st.subheader("Descriptive redemption rate")
        _chart(px.bar(view.sort_values("CAMPAIGN"), x="CAMPAIGN", y="descriptive_redemption_rate",
                      color="campaign_type", hover_data=["exposed_households", "redeeming_households"],
                      labels={"descriptive_redemption_rate": "Descriptive redemption rate", "CAMPAIGN": "Campaign"}), 340)
    with right:
        st.subheader("Exposed households")
        _chart(px.bar(view.sort_values("exposed_households"), x="exposed_households", y="CAMPAIGN",
                      orientation="h", color="campaign_type",
                      labels={"exposed_households": "Assigned households", "CAMPAIGN": "Campaign"}), 340)

    st.subheader("Observed spend before, during, and after campaign")
    spend = view.melt(id_vars=["CAMPAIGN", "campaign_type"],
                      value_vars=["spend_before_campaign", "spend_during_campaign", "spend_after_campaign"],
                      var_name="campaign_period", value_name="summed_spend")
    spend["campaign_period"] = spend["campaign_period"].str.replace("_campaign", "", regex=False).str.replace("_", " ").str.title()
    _chart(px.bar(spend, x="CAMPAIGN", y="summed_spend", color="campaign_period", barmode="group",
                  hover_data=["campaign_type"], labels={"summed_spend": "Summed spend (source value units)",
                                                        "CAMPAIGN": "Campaign", "campaign_period": "Period"}), 370)
    st.caption("Post-campaign spend can be missing or observed for different durations. Completeness flags are preserved below; comparisons do not establish campaign impact.")

    offer_campaigns = data["offers"].groupby("CAMPAIGN", as_index=False).agg(
        mean_response_probability=("calibrated_response_probability", "mean"),
        mean_modeled_offer_value=("modeled_expected_offer_value", "mean"),
        scored_rows=("household_key", "size"),
    )
    offer_campaigns = offer_campaigns[offer_campaigns["CAMPAIGN"].isin(selected_campaigns)]
    if not offer_campaigns.empty:
        st.subheader("Test-population response scores and offer scenarios")
        _chart(px.bar(offer_campaigns, x="CAMPAIGN", y="mean_response_probability",
                      hover_data=["scored_rows", "mean_modeled_offer_value"],
                      labels={"mean_response_probability": "Mean calibrated response probability", "CAMPAIGN": "Campaign"}), 320)
        st.caption("These scores exist only for held-out test household-campaign rows.")
    display_columns = ["CAMPAIGN", "campaign_type", "START_DAY", "END_DAY", "exposed_households",
                       "coupon_redemption_count", "redeeming_households", "descriptive_redemption_rate",
                       "spend_before_campaign", "spend_during_campaign", "spend_after_campaign",
                       "campaign_window_complete", "post_campaign_observation_complete", "post_campaign_days_observed"]
    st.dataframe(view[display_columns].sort_values("CAMPAIGN"), use_container_width=True, hide_index=True)


def _clv_segments(economics: pd.DataFrame) -> pd.DataFrame:
    frame = economics.copy()
    pct = frame["modeled_12_month_clv"].rank(method="first", pct=True)
    frame["value_segment"] = np.select([pct <= 1 / 3, pct <= 2 / 3],
                                       ["Lower modeled CLV", "Middle modeled CLV"],
                                       default="Higher modeled CLV")
    return frame


def _page_clv(data: dict):
    st.title("Customer Value / CLV")
    st.caption("Assumption-based modeled value. These estimates are not observed lifetime value.")
    economics = data["customer_economics"]
    report = data["finance_report"]
    assumptions = report["assumptions"]
    summary = report["clv_summary"]
    cols = st.columns(4)
    cols[0].metric("Mean 12-month modeled CLV", _format_number(summary["mean_12"]))
    cols[1].metric("Median 12-month modeled CLV", _format_number(summary["median_12"]))
    cols[2].metric("Mean 24-month modeled CLV", _format_number(summary["mean_24"]))
    cols[3].metric("Median 24-month modeled CLV", _format_number(summary["median_24"]))
    left, right = st.columns(2)
    with left:
        st.subheader("Modeled 12-month CLV distribution")
        _chart(px.histogram(economics, x="modeled_12_month_clv", nbins=40,
                            labels={"modeled_12_month_clv": "Modeled 12-month CLV (source value units)"}), 350)
    with right:
        st.subheader("Descriptive value groups")
        segmented = _clv_segments(economics)
        order = ["Lower modeled CLV", "Middle modeled CLV", "Higher modeled CLV"]
        _chart(px.box(segmented, x="value_segment", y="modeled_12_month_clv", category_orders={"value_segment": order},
                      labels={"value_segment": "Quantile group", "modeled_12_month_clv": "Modeled 12-month CLV"}), 350)
        st.caption("Groups split households into approximate thirds by modeled 12-month CLV; labels are descriptive, not quality judgments.")

    st.subheader("Current finance assumptions")
    assumption_table = pd.DataFrame([
        ("Contribution margin", _format_rate(assumptions["contribution_margin"])),
        ("Monthly activity / retention", _format_rate(assumptions["monthly_retention_rate"])),
        ("Effective annual discount rate", _format_rate(assumptions["annual_discount_rate"])),
        ("Observed dataset span", f"{assumptions['observed_history_days']} ordered DAY indices"),
        ("Days per modeled month", _format_number(assumptions["days_per_month"], 4)),
        ("Revenue denominator", f"{assumptions['observed_history_days']} / {assumptions['days_per_month']} = {_format_number(assumptions['observed_history_days'] / assumptions['days_per_month'], 2)} months"),
    ], columns=["Assumption", "Current setting"])
    st.dataframe(assumption_table, use_container_width=True, hide_index=True)
    st.caption("Revenue is divided by the full dataset DAY span, not household tenure. Effective monthly discounting uses compounding: (1 + annual rate)^(1/12) − 1.")

    sensitivity = pd.DataFrame([x for x in report["sensitivity"] if x.get("type") == "one_factor_clv"])
    if not sensitivity.empty:
        st.subheader("Median 12-month modeled CLV sensitivity")
        sensitivity["setting"] = sensitivity.apply(lambda r: f"{r['varied_assumption']} = {r['value']:.2f}", axis=1)
        _chart(px.bar(sensitivity, x="setting", y="median_modeled_12_month_clv", color="varied_assumption",
                      labels={"setting": "Assumption scenario", "median_modeled_12_month_clv": "Median modeled 12-month CLV"}), 350)
    with st.expander("Method and limitations"):
        st.write("Monthly revenue is observed historical sales divided by the configured dataset-span months. Monthly contribution applies an assumed margin; future activity decays geometrically by the monthly retention assumption, and each month is discounted. This short observational dataset does not identify household tenure or future survival.")


def _page_offer(data: dict):
    st.title("Offer Economics")
    st.caption("Modeled expected offer value for the scored held-out test population; scenario-based, not incremental profit.")
    offers = join_offers_with_campaigns(data["offers"], data["campaigns"])
    campaigns = sorted(offers["CAMPAIGN"].unique().tolist())
    selected = st.multiselect("Campaign", campaigns, default=campaigns)
    view = offers[offers["CAMPAIGN"].isin(selected)].copy()
    if view.empty:
        st.info("No scored rows match the campaign selection.")
        return
    positive = int((view["modeled_expected_offer_value"] > 0).sum())
    negative = int((view["modeled_expected_offer_value"] < 0).sum())
    cols = st.columns(5)
    cols[0].metric("Scored campaign-households", f"{len(view):,}")
    cols[1].metric("Unique households", f"{view['household_key'].nunique():,}")
    cols[2].metric("Mean calibrated response probability", _format_rate(view["calibrated_response_probability"].mean()))
    cols[3].metric("Above zero modeled value", f"{positive:,} ({_format_rate(positive / len(view))})")
    cols[4].metric("Below zero modeled value", f"{negative:,} ({_format_rate(negative / len(view))})")
    left, right = st.columns(2)
    with left:
        _chart(px.histogram(view, x="modeled_expected_offer_value", nbins=35,
                            labels={"modeled_expected_offer_value": "Modeled expected offer value (source value units)"}), 350)
    with right:
        _chart(px.scatter(view, x="calibrated_response_probability", y="break_even_offer_cost",
                          color="campaign_type", hover_data=["household_key", "CAMPAIGN", "assumed_offer_cost"],
                          labels={"calibrated_response_probability": "Calibrated response probability",
                                  "break_even_offer_cost": "Break-even offer cost (source value units)"}), 350)
    st.subheader("Scored household-campaign rows")
    display = view[["household_key", "CAMPAIGN", "campaign_type", "START_DAY", "calibrated_response_probability",
                    "expected_contribution_if_response", "assumed_offer_cost", "break_even_offer_cost",
                    "modeled_expected_offer_value"]].copy()
    st.dataframe(display.sort_values("calibrated_response_probability", ascending=False),
                 use_container_width=True, hide_index=True)
    assumptions = data["finance_report"]["assumptions"]
    st.info(f"Response probability means predicted observed redemption among campaign-assigned households. The response scenario assumes one average historical basket × {assumptions['contribution_margin']:.0%} margin; assumed offer cost is {assumptions['default_offer_cost']:.2f}. The model does not estimate the causal effect of assignment.")


def _metric_table(metric_groups: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for label, metrics in metric_groups.items():
        rows.append({"Evaluation": label,
                     "ROC-AUC": metrics.get("roc_auc"),
                     "Average Precision": metrics.get("pr_auc_average_precision"),
                     "Accuracy": metrics.get("accuracy"),
                     "Precision": metrics.get("precision"),
                     "Recall": metrics.get("recall"),
                     "F1": metrics.get("f1"),
                     "Brier": metrics.get("brier_score"),
                     "Log loss": metrics.get("log_loss"),
                     "Threshold": metrics.get("threshold")})
    return pd.DataFrame(rows)


def _page_model(data: dict):
    st.title("Model Performance & Calibration")
    metrics = data["model_metrics"]
    calibration = data["calibration_metrics"]
    rows = data["model_rows"]
    counts = metrics["row_counts"]
    cols = st.columns(4)
    for i, split in enumerate(["train", "validation", "test"]):
        split_rows = rows[rows["data_split"] == split]
        positive_rate = metrics["model_metrics"][split].get("actual_positive_rate")
        starts = split_rows["START_DAY"]
        campaigns = split_rows["CAMPAIGN"].nunique()
        span = f"{int(starts.min())}–{int(starts.max())}" if not split_rows.empty else "—"
        cols[i].metric(f"{split.title()} observations", f"{counts[split]:,}", help=f"{campaigns} campaigns; START_DAY {span}")
        cols[i].caption(f"Actual positive rate: {_format_rate(positive_rate)} · Campaigns: {campaigns} · START_DAY: {span}")
    st.caption("Chronological splits grouped by campaign START_DAY; test observations are later campaigns than train and validation.")

    st.subheader("Ranking and classification metrics")
    all_groups = {f"{split.title()} raw model": metrics["model_metrics"][split] for split in ["train", "validation", "test"]}
    all_groups.update({f"{split.title()} calibrated": calibration[split]["calibrated"]
                       for split in ["validation", "test"]})
    baseline_test = metrics["baselines"]["test"]
    all_groups.update({
        "Test majority baseline": baseline_test["majority_class"],
        "Test constant train-rate baseline": baseline_test["constant_training_rate"],
        "Test prior-redemption indicator": baseline_test["prior_coupon_redemption_indicator"],
    })
    full_table = _metric_table(all_groups)
    st.dataframe(full_table, use_container_width=True, hide_index=True)
    st.caption("The saved classification threshold is 0.5. The calibrator does not materially change ranking; the test ranking scores remain ROC-AUC 0.806 and AP 0.431.")

    st.subheader("Test calibration reliability")
    raw_bins = pd.DataFrame(calibration["test"]["raw"]["reliability_bins"])
    calibrated_bins = pd.DataFrame(calibration["test"]["calibrated"]["reliability_bins"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#8996A3"), name="Perfect calibration"))
    fig.add_trace(go.Scatter(x=raw_bins["mean_predicted_probability"], y=raw_bins["observed_redemption_rate"],
                             mode="lines+markers", name="Raw", customdata=raw_bins["count"],
                             hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>Rows %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=calibrated_bins["mean_predicted_probability"], y=calibrated_bins["observed_redemption_rate"],
                             mode="lines+markers", name="Platt calibrated", customdata=calibrated_bins["count"],
                             hovertemplate="Predicted %{x:.2f}<br>Observed %{y:.2f}<br>Rows %{customdata}<extra></extra>"))
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Observed redemption rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
    _chart(fig, 430)
    cal_cols = st.columns(4)
    cal_cols[0].metric("Test Brier · raw", _format_number(calibration["test"]["raw"]["brier_score"], 4))
    cal_cols[1].metric("Test Brier · calibrated", _format_number(calibration["test"]["calibrated"]["brier_score"], 4))
    cal_cols[2].metric("Test ECE · raw", _format_number(calibration["test"]["raw"]["expected_calibration_error"], 4))
    cal_cols[3].metric("Test ECE · calibrated", _format_number(calibration["test"]["calibrated"]["expected_calibration_error"], 4))
    st.caption(f"Platt scaling was fitted on {calibration['calibrator_fit_rows']:,} validation rows and frozen before test evaluation. Test labels were not used to fit it. Mean prediction: {_format_rate(calibration['test']['calibrated']['mean_predicted_probability'])}; actual rate: {_format_rate(calibration['test']['calibrated']['actual_positive_rate'])}.")

    confusion = calibration["test"]["calibrated"]["confusion_matrix"]
    matrix = np.array([[confusion["true_negative"], confusion["false_positive"]],
                       [confusion["false_negative"], confusion["true_positive"]]])
    fig = px.imshow(matrix, text_auto=True, x=["Predicted no redemption", "Predicted redemption"],
                    y=["Observed no redemption", "Observed redemption"], color_continuous_scale="Blues",
                    labels={"x": "Prediction at 0.5", "y": "Observed target", "color": "Rows"})
    st.subheader("Calibrated test confusion matrix at fixed 0.5 threshold")
    _chart(fig, 340)

    st.subheader("Validation predictive sensitivity")
    importance = pd.DataFrame(metrics["validation_permutation_importance"])
    if not importance.empty:
        importance = importance.sort_values("mean_average_precision_decrease", ascending=True)
        _chart(px.bar(importance, x="mean_average_precision_decrease", y="feature", orientation="h",
                      error_x="std_average_precision_decrease",
                      labels={"mean_average_precision_decrease": "Validation AP decrease after permutation", "feature": "Feature"}), 470)
        st.caption("Permutation sensitivity describes predictive association in validation data. It does not show that a feature causes redemption.")

    with st.expander("Training diagnostics"):
        training = metrics["training"]
        st.write(f"Architecture: {metrics['architecture']['layers']} layers, {metrics['architecture']['trainable_parameters']:,} trainable parameters. Best validation epoch: {training['best_epoch']}; epochs run: {training['epochs_run']}; early stopped: {training['early_stopped']}; training-only positive weight: {training['positive_weight']:.3f}.")
        history = pd.DataFrame(training["history"])
        _chart(px.line(history, x="epoch", y=["training_loss", "validation_loss"],
                       labels={"epoch": "Epoch", "value": "Loss", "variable": "Series"}), 300)


def _page_methodology():
    st.title("Methodology & Assumptions")
    st.caption("Scope, source definitions, and limits for interpreting this portfolio project.")
    with st.expander("Data and engineering", expanded=True):
        st.markdown("""
        **Data:** dunnhumby Complete Journey household, product, campaign, coupon, and transaction files.

        **Engineering:** PySpark ingestion, source schema and quality checks, verified-key joins, household/product/campaign feature tables, and Spark SQL examples. The dashboard reads processed Parquet and saved JSON metrics; it does not load the 36M-row `causal_data.csv`.
        """)
    with st.expander("Predictive model and temporal controls", expanded=True):
        st.markdown("""
        A small PyTorch binary classifier predicts observed coupon redemption among assigned household-campaign rows. Its 14 historical numeric features use a configurable 90-day lookback and strict `DAY < START_DAY`. The target covers the current campaign interval. Campaign start days define chronological train/validation/test splits; train-only weighting, preprocessing, and early stopping are used.

        The score is observational prediction, not a treatment effect. Assignment is not assumed randomized. The saved model is evaluated on train, validation, and later test periods; a 0.5 classification threshold is shown as configured.
        """)
    with st.expander("Calibration", expanded=True):
        st.markdown("""
        Platt scaling was fitted using validation predictions and labels only. The calibrator was frozen before held-out test evaluation. The test reliability chart compares raw and calibrated probabilities using the saved probability bins. The validation set and campaign periods are limited, so future calibration stability is unknown.
        """)
    with st.expander("Finance and offer scenarios", expanded=True):
        st.markdown("""
        CLV applies explicit margin, monthly activity/retention, discount, and dataset-span assumptions to historical revenue. Revenue is not profit and the estimates are not observed lifetime value. Monthly discounting uses an effective annual-to-monthly conversion.

        Offer economics uses the frozen calibrated redemption probability and a one-average-basket response scenario. The model predicts observed redemption; it does not estimate incremental redemption, incremental revenue, or campaign profit. Offer cost and margin are assumptions.
        """)
    with st.expander("Important limitations", expanded=True):
        st.markdown("""
        - Complete Journey is observational; campaign exposure is not automatically randomized.
        - Campaign redemption and spend summaries are descriptive and can have incomplete observation windows.
        - Response modeling predicts observed redemption, not causal campaign effect.
        - CLV and offer economics are modeled estimates based on assumptions.
        - DAY and WEEK_NO are relative integer indices; the supplied data does not establish calendar dates.
        - `causal_data.csv` covers only a subset of transaction stores and is not loaded into this dashboard.
        - Model/calibration validation uses a limited chronological campaign population; generalization should be interpreted cautiously.
        - Product quantity includes an extreme observed upper tail and is displayed as reported.
        """)


def main():
    st.sidebar.title("Customer Intelligence")
    st.sidebar.caption("Complete Journey · PySpark · PyTorch · modeled economics")
    page = st.sidebar.radio("Navigate", NAVIGATION, label_visibility="collapsed")
    try:
        with st.spinner("Loading verified project artifacts…"):
            data = load_dashboard_data()
    except Exception as exc:
        st.error(f"Dashboard artifacts could not be loaded: {exc}")
        st.info("Build Phase 3 features and Phase 4/finance outputs first, then relaunch this app.")
        st.stop()
    st.sidebar.divider()
    st.sidebar.caption(f"{len(data['customers']):,} households · {len(data['campaigns']):,} campaign rows · {len(data['offers']):,} held-out response scores")
    st.sidebar.caption("Observational data · modeled finance estimates")
    pages = {
        "Executive Overview": _page_overview,
        "Customer Explorer": _page_customer,
        "Campaign Analytics": _page_campaigns,
        "Customer Value / CLV": _page_clv,
        "Offer Economics": _page_offer,
        "Model Performance": _page_model,
    }
    if page == "Methodology & Assumptions":
        _page_methodology()
    elif page == "AI Analytics Assistant":
        render_assistant_page(data)
    else:
        pages[page](data)


# Streamlit executes the script top-to-bottom; keeping the entrypoint at module
# scope also lets the built-in Streamlit smoke runner exercise the full page.
main()
