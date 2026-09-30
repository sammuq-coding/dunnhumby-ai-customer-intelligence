# PySpark analytical feature layer

## Run

From the repository root, run:

```bash
python -m src.features.build_features
```

The pipeline writes Parquet directories under `data/processed/features/` and prints each output's row count, Spark schema, and five-row preview. It reads only transaction, product, campaign description/assignment, and coupon redemption inputs. `causal_data.csv` is not loaded because display/mailer observations are not needed for the current customer, product, or campaign tables.

Analysis-window settings are in `configs/feature_windows.yaml`. The configured cutoff is day 712, exclusive, based on Phase 2's observed maximum transaction `DAY` of 711. `feature_lookback_days: null` means all available history before cutoff. `outcome_window_days: null` means campaign post-period spend extends only through the analysis cutoff and has a variable observed duration. No calendar dates or fixed retention target are assumed.

## Tables and grains

### `customer_features`

**Grain: one row per `household_key` with at least one transaction in the selected feature period.**

Sources: `transaction_data`, `product`, `campaign_desc` + `campaign_table` via `CAMPAIGN`, and `coupon_redempt` via household. Product metadata is joined on Phase 2-verified `PRODUCT_ID`; product ID is unique in `product.csv`. Campaign exposure counts are derived from observed campaign assignment rows whose campaign start is before cutoff. Coupon redemption counts use redemption rows before cutoff. All event inputs obey the configured lower lookback bound when one is set.

Definitions:

| Feature | Definition |
| --- | --- |
| `household_key` | Verified household identifier; table key. |
| `total_sales` | Sum of source `SALES_VALUE` across eligible item rows. Per the guide, this is retailer receipts after stated discounts, not necessarily customer-paid value. |
| `transaction_count` | Distinct `BASKET_ID` count within household; item rows are not counted as separate transactions. |
| `average_basket_value` | Mean of household+basket `SALES_VALUE` sums. |
| `total_quantity` | Sum of source `QUANTITY`; no clipping, trimming, or correction. |
| `unique_products` | Distinct purchased `PRODUCT_ID` count. |
| `unique_departments`, `unique_commodities` | Distinct non-null joined `DEPARTMENT` and `COMMODITY_DESC` values. Null category labels remain missing and do not add to the count. |
| `coupon_usage` | Distinct household baskets with a non-zero `COUPON_DISC` or `COUPON_MATCH_DISC`; not a transaction-row count or a redemption count. |
| `total_retail_discount` | Signed sum of `RETAIL_DISC` source values. |
| `total_coupon_discount` | Signed sum of `COUPON_DISC` plus `COUPON_MATCH_DISC`. |
| `campaign_exposure_count` | Count of observed household-campaign assignment rows whose `START_DAY` is before cutoff. `START_DAY` is the available exposure-time proxy; the table does not provide a contact timestamp. |
| `coupon_redemption_count` | Number of observed redemption rows for the household in the feature period. |
| `purchase_day_range` | Latest eligible `DAY` minus earliest eligible `DAY`. |
| `purchase_frequency` | Distinct baskets divided by the inclusive index span (`purchase_day_range + 1`); baskets per observed day-index interval. |
| `recency_days` | `analysis_cutoff_day - latest eligible DAY`; cutoff is exclusive, so a transaction on cutoff-1 has recency 1. |

### `product_features`

**Grain: one row per `PRODUCT_ID` in `product.csv`, including catalog products with no eligible transactions.**

Sources: `product` left-joined to transaction aggregates by the Phase 2-verified unique `PRODUCT_ID`.

The output carries the observed catalog fields `MANUFACTURER`, `DEPARTMENT`, `BRAND`, `COMMODITY_DESC`, `SUB_COMMODITY_DESC`, and `CURR_SIZE_OF_PRODUCT` unchanged, including null category/size values. `total_units_sold` and `total_sales` sum the transaction source fields without altering their outliers. `number_of_households_purchasing` is a distinct household count. `number_of_baskets_containing_product` counts distinct household+basket+product tuples. `average_sales_value_per_occurrence` is the average line-level `SALES_VALUE` per transaction occurrence. `number_of_repeat_purchasing_households` counts households buying the product in more than one distinct basket; `repeat_purchase_rate` divides that by purchasing households. For a catalog product with no eligible sales, count/sum metrics are zero and average/rate are null where no denominator exists.

### `campaign_household_features`

**Grain: one row per observed `(household_key, CAMPAIGN)` assignment whose campaign `START_DAY` is before cutoff and within the optional lookback.**

Sources: `campaign_table` joined to `campaign_desc` by verified `CAMPAIGN`; `transaction_data` by household; `coupon_redempt` by the verified household+campaign pair and restricted to the documented inclusive campaign day interval. The pair has 889 distinct redemption keys and zero unmatched assignments. `campaign_exposure` is 1 because each row represents an assignment in `campaign_table`.

`spend_before_campaign` sums transactions before `START_DAY`, bounded by `feature_lookback_days` when supplied. `spend_during_campaign` sums transactions from `START_DAY` through `END_DAY`, inclusive, but never at/after the global cutoff. `spend_after_campaign` sums after `END_DAY` and before cutoff (or within the optional explicit outcome window). It is null until the requested post-period is complete. `post_campaign_days_observed` gives the available post-period length; with a configured outcome window it is capped at that length. `campaign_window_complete` identifies campaigns whose end is before cutoff, and `post_campaign_observation_complete` indicates whether a full post-period is observed. `coupon_redemption_count` counts matching redemption rows within the campaign interval before cutoff.

### `campaign_features`

**Grain: one row per eligible `CAMPAIGN` in `campaign_desc` whose start is before cutoff and within the optional lookback.**

Aggregates campaign-household exposure rows: `exposed_households`, `redeeming_households`, `coupon_redemption_count`, and sums of before/during/after spend. `descriptive_redemption_rate` is redeeming assigned households divided by exposed households. It is descriptive only: this observational dataset provides no evidence of random assignment, so the rate is not causal lift or incrementality. During spend can be partial when a campaign is active at cutoff; use `campaign_window_complete` and post-period completeness before comparing campaigns.

## Temporal leakage controls

- Customer and product inputs keep only `DAY < analysis_cutoff_day`; a configured lookback additionally imposes `DAY >= cutoff - lookback`.
- Recency uses the explicit exclusive cutoff. No after-cutoff record can enter historical customer/product features.
- Campaign assignments starting on/after cutoff are excluded. Spend is split relative to each observed campaign's integer start/end days, and every transaction also remains strictly before the global cutoff.
- Post-campaign spend is an outcome/descriptive field, not a model input. With no configured outcome window it extends through cutoff and has a varying duration; the output includes observed days and completeness flags. A fixed outcome period can be set without choosing a retention-specific horizon.
- `time_windows.py` also exposes explicit, half-open outcome filtering `[cutoff, cutoff + outcome_window_days)` for future predictive labels. The requested outcome window must be supplied; no default is fabricated.

## Data-quality decisions

- `coupon.csv` has 5,164 exact duplicate rows (Phase 2). The current feature calculations do not join `coupon.csv`, so these duplicates cannot multiply customer/product/campaign metrics. They are preserved in raw data and no deduplication is applied.
- `product.CURR_SIZE_OF_PRODUCT` stays null where absent; Phase 2 found 30,607 missing values. The size field is carried as metadata but is not used in current aggregates. Missing department/commodity values remain null and are excluded only from distinct non-null category counts; source records are preserved.
- `transaction_data.QUANTITY` is summed as supplied, including the observed zero and extreme upper tail (maximum 89,638, Phase 2). No clipping, filtering, or imputation is performed. Totals can therefore be strongly influenced by extreme source values; review before interpreting them.
- Campaign/redemption and product/transaction joins rely on Phase 2-verified identifiers and cardinalities. Current config and tests guard the expected schemas and feature grains; source rows are never silently removed to repair quality issues.

## Spark SQL examples

Examples are in `src/sql/` and query the feature views: highest-value customers, highest-revenue products, campaign performance, customer spend by department, and coupon usage by the observed `classification_1` code. The last two examples also require the pipeline's `feature_transactions` and `household_demographics` source views. `classification_1` is used as a code; no undocumented segment meaning is inferred.
