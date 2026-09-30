# Campaign response modeling dataset

Generated from the raw CSVs by `src/models/pytorch/build_training_data.py`.
The parquet output is at `data/processed/modeling/campaign_response_dataset.parquet`.

## Grain and target

One row per assigned `household_key` + `CAMPAIGN`, sourced from
`campaign_table.csv`, provided that at least one transaction is observed for
that household in the configured pre-campaign history window. `target_redeemed`
is 1 when `coupon_redempt.csv` contains one or more rows for that household and
campaign with `START_DAY <= DAY <= END_DAY`; otherwise it is 0. This is an
observational prediction label. Campaign assignment is not treated as random
or as evidence of causal treatment.

## Features

Every feature below uses transaction rows with
`START_DAY - lookback_days <= DAY < START_DAY`, or prior redemption/exposure
events in the same pre-start window. The default lookback is 90 integer DAY
indices and is configurable in `configs/modeling.yaml`.

| Feature | Definition |
| --- | --- |
| `historical_total_sales` | Sum of `SALES_VALUE` over eligible historical item rows. |
| `historical_basket_count` | Distinct `BASKET_ID` values, not transaction rows. |
| `historical_average_basket_value` | Mean of per-household/per-basket sums of `SALES_VALUE`. |
| `historical_total_quantity` | Sum of source `QUANTITY` without clipping; known extreme values remain visible. |
| `historical_unique_products` | Distinct historical `PRODUCT_ID` count. |
| `historical_unique_departments` | Distinct non-null `product.csv.DEPARTMENT` values joined by `PRODUCT_ID`; source nulls are not replaced. |
| `historical_retail_discount` | Sum of `RETAIL_DISC`. |
| `historical_coupon_discount` | Sum of `COUPON_DISC` plus `COUPON_MATCH_DISC`. |
| `historical_coupon_basket_count` | Distinct historical baskets with either coupon discount nonzero. |
| `historical_purchase_frequency` | Distinct historical baskets divided by configured lookback day count. |
| `historical_recency_days` | Campaign `START_DAY` minus the latest historical transaction `DAY`. |
| `prior_coupon_redemption_count` | Number of coupon redemption rows for this household in the pre-start lookback window. |
| `prior_campaign_exposure_count` | Earlier assigned household campaigns whose `START_DAY` falls in the lookback window. |
| `prior_campaign_redemption_count` | Redemption rows belonging to earlier assigned campaigns, with redemption `DAY` in the current campaign's pre-start lookback. |

`CAMPAIGN`, campaign description, start/end day, and split name are retained as
identifiers/metadata and are excluded from the PyTorch numeric feature vector.
The target count is reduced to a binary indicator. Missing numerical inputs in
model training are imputed using the training split's column mean; normalization
uses the training split's mean and standard deviation. Zero-variance columns use
scale 1. Historical aggregates with no event are zero; the modeling cohort
requires at least one historical transaction. Null product departments are
ignored in distinct department counts. Product package size and demographics
are not used. Quantity outliers are preserved, not silently clipped.

## Leakage controls and splits

The code uses integer `DAY` as an ordered index and does not create calendar
dates. Current and future transaction rows are excluded with the strict bound
`DAY < START_DAY`; prior redemptions are also checked against the current start
day, including when campaign intervals overlap. The current-campaign redemption
data is used only to construct the target. Only households present in
`campaign_table.csv` are eligible; no non-assigned household is relabeled as a
negative. A campaign label is retained only when `END_DAY` is no later than
the maximum observed `coupon_redempt.DAY` (704), so an unobserved tail is not
mistaken for a non-redemption. Campaigns 15 and 24 end on days 708 and 719 and
are excluded from the labeled modeling cohort; their assignment rows remain
available as historical exposure context.

Splits are chronological by distinct campaign `START_DAY`, using the configured
60%/20% fractions to choose day groups. Tied starts stay together. In the
verified metadata there are 27 distinct starts: training candidates start on
days 224–477, validation candidates on days 504–575, and test candidates on
days 587–659. Campaigns 11 and 12 (both start day 477) are purged from training
because they end on days 523 and 509, at or after validation begins on day 504.
Campaigns 14–17 (start days 531–575) are purged from validation because their
end days cross the test boundary at day 587. Campaign 13 (start 504, end 551)
remains in validation. Purged campaigns remain in exposure history for later
feature calculations but do not contribute labeled examples. The actual
household row and positive counts require executing the PySpark builder; they
are emitted by that run and are not fabricated here.

## Source quality notes

`coupon.csv` has 5,164 exact duplicate rows and is not joined into this table,
avoiding duplicate-driven row multiplication. `product.csv` has missing package
size and 15 missing department/category values in each named category field;
package size is not used, and null departments are preserved. Transaction
quantity has an extreme upper tail; the source values are retained without
automatic correction. The join to product metadata uses the verified unique
`PRODUCT_ID` dimension. The target uses the verified household+campaign
redemption attribution and campaign day interval.
