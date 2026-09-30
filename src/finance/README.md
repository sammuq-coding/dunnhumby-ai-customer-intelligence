# Customer value and offer economics

This phase adds an assumption-based finance layer over the existing
`data/processed/features/customer_features` table. It does not rebuild
transaction aggregations or change any modeling or calibration artifacts.

## Tables and grain

* `customer_economics.parquet`: one row per `household_key`, based on the
  existing household-grain customer features. It carries the source behavior
  columns and adds source history months, monthly revenue/contribution, and
  modeled 12- and 24-month CLV.
* `campaign_offer_economics.parquet`: one row per held-out test
  `household_key + CAMPAIGN`, with campaign start day and calibrated response
  score. It repeats relevant customer-level values for that campaign row.

The second table is deliberately not joined into the customer table, because
households may have multiple campaign observations. It only covers rows in the
existing chronological `test` split. No probability is fabricated for other
households or campaigns.

## Revenue monthlyization and modeled CLV

`total_sales` is treated as historical revenue in the source's `SALES_VALUE`
units; it is not profit. In `configs/finance.yaml`, the observed dataset DAY
span is configured as 711 ordered days (the observed transaction index spans
DAY 1–711), and the month length as 30.436875 days. Estimated monthly revenue
is `total_sales / (711 / 30.436875)`. This uses the full data observation
window as a common denominator; it is not household tenure, and households
may have entered or left the observed purchasing history during that window.
No calendar dates are invented.

Monthly contribution is estimated monthly revenue times the configured
contribution margin. The default margin (25%) is an editable assumption, not a
measured dunnhumby value. For month `t`, modeled contribution is multiplied by
`monthly_retention_rate ** (t - 1)`. This defines first-month activity as 1.0;
subsequent activity decays by the monthly rate. The effective monthly discount
rate is `(1 + annual_discount_rate) ** (1/12) - 1`, not annual rate divided by
12. Each month's expected contribution is discounted at month `t`. The 12- and
24-month outputs are sums of those discounted monthly contributions.

These are **modeled 12-/24-month CLV estimates under stated assumptions**, not
observed future outcomes or a learned survival model. Quantity, frequency,
recency, and historical basket value are retained for context; only revenue,
margin, activity, and discount assumptions enter this simple CLV formula.

## Offer scenario and probability meaning

The existing PyTorch model and saved Platt calibrator are loaded without
training or refitting. The calibrator was fitted on validation predictions and
labels in the prior phase. The finance scorer only selects existing held-out
test rows and their 14 feature columns; it does not select or use
`target_redeemed`. The reported score is the calibrated predictive probability
of **observed coupon redemption among households assigned to a campaign**.
It is not a probability of incremental redemption and does not estimate a
treatment effect.

For a modeled response scenario, this implementation assumes one purchase
basket equal to that household's historical average basket value. Conditional
scenario contribution is that basket value times the configured margin. Then:

* break-even offer cost = calibrated response probability × scenario
  contribution;
* modeled expected offer value = break-even offer cost − assumed offer cost.

Offer cost is treated as paid per assigned offer at time zero, whether or not a
redemption occurs. The default cost is an editable 5.00 source-value-unit
assumption. This is **modeled expected value under the stated assumptions**;
it is not incremental revenue, incremental contribution, or campaign ROI.

## Sensitivity and quality controls

`configs/finance.yaml` exposes margins, monthly retention, annual discount
rates, and offer costs. `sensitivity_scenarios` validates and creates the
configured assumption grid; the pipeline also reports one-factor median
12-month CLV sensitivity for margin, retention, and discount rate. These are
business assumption scenarios, not ML hyperparameters. No “best” assumption is
selected from test performance.

The build fails on duplicate/null household keys, required missing values,
non-finite inputs, negative values in nonnegative economic inputs, invalid
assumptions, or response probabilities outside [0, 1]. Source values are not
silently imputed, clipped, or removed. The known extreme transaction quantity
tail remains in the source customer features but is not used in this CLV
calculation. Historical sales and average basket values are verified
nonnegative in the current generated table; a future negative-value source
would stop the pipeline for review.

## Limitations

The data is a short observational retail history with integer DAY indices. The
fixed dataset-window denominator does not measure household tenure, and the
retention, margin, discount, and offer-cost inputs are assumptions. The model
population is assigned household-campaign observations in the held-out test
split only. Campaign assignment is not treated as randomized. These results
are suitable as a documented portfolio demonstration of finance mechanics,
not a decision-grade financial forecast or causal campaign evaluation.

Run with Java 17 available to PySpark:

```bash
python -m src.finance.build_finance --config configs/finance.yaml
```
