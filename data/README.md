# Complete Journey data inventory and verified schema

## Dataset access

The Complete Journey is published for research/educational use on the [official dunnhumby source-files page](https://www.dunnhumby.com/source-files/), which also links the dataset user guide. Obtain the source from the publisher and follow the accompanying usage terms. Place the original CSV files in this directory (`data/raw/`) without renaming or editing them. Raw files are excluded from Git.

## Inspection method and source

Inspected on 2026-09-28 with the repository's PySpark inventory (`inferSchema=true`, CSV header enabled). Row counts, null-like values, exact duplicate rows, and configured numeric parse checks were calculated from Spark DataFrames. Spark's inferred type is an observation of these CSVs, not a guarantee of the source system's intended type. The source files in `raw/` were read only and were not modified.

Interpretations below are distinguished as **documented** (the dunnhumby Complete Journey User Guide supplied in the ZIP, 2023, pp. 2-6) or **observed/inferred** from column labels and values. The guide describes a two-year household transaction sample; it does not provide a calendar epoch for the integer day values. Full summaries, sample rows, and parser-quality details are in the generated, Git-ignored [`data/processed/dataset_inventory.json`](../data/processed/dataset_inventory.json).

## Dataset inventory

| CSV / table | Rows | Columns | Exact duplicate rows | Missing values | Invalid configured numeric values |
| --- | ---: | ---: | ---: | ---: | ---: |
| `campaign_desc.csv` | 30 | 4 | 0 | 0 | 0 |
| `campaign_table.csv` | 7,208 | 3 | 0 | 0 | 0 |
| `causal_data.csv` | 36,786,524 | 5 | 0 | 0 | 0 |
| `coupon.csv` | 124,548 | 3 | 5,164 | 0 | 0 |
| `coupon_redempt.csv` | 2,318 | 4 | 0 | 0 | 0 |
| `hh_demographic.csv` | 801 | 8 | 0 | 0 | 0 |
| `product.csv` | 92,353 | 7 | 0 | 30,652 | 0 |
| `transaction_data.csv` | 2,595,732 | 12 | 0 | 0 | 0 |

No inferred row counts changed between the string-preserving and schema-inferred reads. No required columns were absent and no unexpected columns were found under the dataset-specific rules in `configs/ingestion.yaml`.

## Verified column schemas

“Missing” counts null or blank source fields and gives count/rate. The duplicate count applies to complete rows in that table; it is repeated in the table heading because duplicates are a row-level observation, not a property of one column.

### `campaign_desc.csv` — 30 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `DESCRIPTION` | string | 0 / 0% | Campaign type; documented values are TypeA, TypeB, TypeC. |
| `CAMPAIGN` | int | 0 / 0% | Campaign identifier; guide says values 1-30. |
| `START_DAY` | int | 0 / 0% | Campaign start day index, not a calendar date. |
| `END_DAY` | int | 0 / 0% | Campaign end day index, not a calendar date. |

### `campaign_table.csv` — 7,208 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `DESCRIPTION` | string | 0 / 0% | Campaign type; documented as TypeA/TypeB/TypeC. |
| `household_key` | int | 0 / 0% | Household identifier; documented by the guide. |
| `CAMPAIGN` | int | 0 / 0% | Campaign identifier; documented by the guide. |

### `causal_data.csv` — 36,786,524 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `PRODUCT_ID` | int | 0 / 0% | Product identifier; documented. |
| `STORE_ID` | int | 0 / 0% | Store identifier; documented. |
| `WEEK_NO` | int | 0 / 0% | Week number; guide defines the corresponding transaction week scale as 1-102. |
| `display` | string | 0 / 0% | In-store display location code; documented in the guide. |
| `mailer` | string | 0 / 0% | Weekly mailer placement code; documented in the guide. |

### `coupon.csv` — 124,548 rows, 5,164 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `COUPON_UPC` | bigint | 0 / 0% | Coupon identifier; documented. |
| `PRODUCT_ID` | int | 0 / 0% | Product for which the coupon is redeemable; documented. |
| `CAMPAIGN` | int | 0 / 0% | Campaign identifier; documented. |

### `coupon_redempt.csv` — 2,318 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `household_key` | int | 0 / 0% | Household identifier; documented. |
| `DAY` | int | 0 / 0% | Day index when redemption occurred; guide calls it a day, not a calendar date. |
| `COUPON_UPC` | bigint | 0 / 0% | Coupon identifier; documented. |
| `CAMPAIGN` | int | 0 / 0% | Campaign identifier; documented. |

### `hh_demographic.csv` — 801 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `classification_1` | string | 0 / 0% | Coded demographic classification. Sample values include `Age GroupN`; exact category definition is not given in the guide. |
| `classification_2` | string | 0 / 0% | Coded demographic classification; observed values include X/Y/Z, exact meaning undocumented. |
| `classification_3` | string | 0 / 0% | Coded demographic classification; observed values include `LevelN`, exact meaning undocumented. |
| `HOMEOWNER_DESC` | string | 0 / 0% | Homeowner description/status is supported by the column label and observed values. |
| `classification_5` | string | 0 / 0% | Coded demographic classification; exact meaning undocumented. |
| `classification_4` | string | 0 / 0% | Coded demographic classification; exact meaning undocumented. |
| `KID_CATEGORY_DESC` | string | 0 / 0% | Kid-category description is supported by the column label and observed values. |
| `household_key` | int | 0 / 0% | Household identifier; documented. |

The guide says these are demographic fields for a portion of households and that the generic classification values were chosen with meaningful ordering. It does not map each `classification_N` field to a specific demographic attribute. The labels and sample values above do not establish those missing definitions.

### `product.csv` — 92,353 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `PRODUCT_ID` | int | 0 / 0% | Product identifier; documented. |
| `MANUFACTURER` | int | 0 / 0% | Manufacturer code; documented. |
| `DEPARTMENT` | string | 15 / 0.0162% | Product grouping; documented. |
| `BRAND` | string | 0 / 0% | Brand indicator; guide says it distinguishes private/national label. |
| `COMMODITY_DESC` | string | 15 / 0.0162% | Product category description; documented. |
| `SUB_COMMODITY_DESC` | string | 15 / 0.0162% | Lower-level product category description; documented. |
| `CURR_SIZE_OF_PRODUCT` | string | 30,607 / 33.1413% | Package size; guide explicitly says it is not available for all products. |

### `transaction_data.csv` — 2,595,732 rows, 0 exact duplicate rows

| Column | Spark type | Missing (count / rate) | Meaning and evidence |
| --- | --- | ---: | --- |
| `household_key` | int | 0 / 0% | Household identifier; documented. |
| `BASKET_ID` | bigint | 0 / 0% | Purchase occasion / basket identifier; documented. |
| `DAY` | int | 0 / 0% | Day index on which transaction occurred; documented, but no calendar epoch is supplied. |
| `PRODUCT_ID` | int | 0 / 0% | Product identifier; documented. |
| `QUANTITY` | int | 0 / 0% | Number of products purchased on the trip; documented. |
| `SALES_VALUE` | double | 0 / 0% | Amount received by retailer after specified discounts; the guide cautions this is not necessarily the customer's actual price. |
| `STORE_ID` | int | 0 / 0% | Store identifier; documented. |
| `RETAIL_DISC` | double | 0 / 0% | Retailer loyalty-card discount; documented. |
| `TRANS_TIME` | int | 0 / 0% | Time of day; documented. |
| `WEEK_NO` | int | 0 / 0% | Transaction week number; documented range is 1-102. |
| `COUPON_DISC` | double | 0 / 0% | Manufacturer coupon discount; documented. |
| `COUPON_MATCH_DISC` | double | 0 / 0% | Retailer's match of a manufacturer coupon; documented. |

## Day and week ranges

No column was inferred as a date/timestamp. These are numeric day/week indices; they cannot be converted to calendar dates from the supplied files alone. The source guide describes the overall transactions as covering two years.

| Dataset / field | Observed minimum | Observed maximum |
| --- | ---: | ---: |
| `campaign_desc.START_DAY` | 224 | 659 |
| `campaign_desc.END_DAY` | 264 | 719 |
| `coupon_redempt.DAY` | 225 | 704 |
| `transaction_data.DAY` | 1 | 711 |
| `causal_data.WEEK_NO` | 9 | 101 |
| `transaction_data.WEEK_NO` | 1 | 102 |

## Numeric summaries

Spark inferred numeric columns and calculated count, mean, standard deviation, quartiles, minimum, and maximum. The compact table below reports minimum / mean / maximum, rounded to two decimals. Identifier statistics are included because those columns were inferred numeric, but their means should not be interpreted as business measures. Full precision and quartiles are in `dataset_inventory.json`.

| Dataset | Numeric columns: min / mean / max |
| --- | --- |
| `campaign_desc.csv` | `CAMPAIGN` 1 / 15.50 / 30; `START_DAY` 224 / 463.87 / 659; `END_DAY` 264 / 510.47 / 719 |
| `campaign_table.csv` | `household_key` 1 / 1268.70 / 2500; `CAMPAIGN` 1 / 15.66 / 30 |
| `causal_data.csv` | `PRODUCT_ID` 26190 / 3512236.99 / 18244453; `STORE_ID` 286 / 3234.15 / 34280; `WEEK_NO` 9 / 55.30 / 101 |
| `coupon.csv` | `COUPON_UPC` 10000085189 / 21982253584.33 / 59986600074; `PRODUCT_ID` 25671 / 4651276.62 / 18148540; `CAMPAIGN` 1 / 15.86 / 30 |
| `coupon_redempt.csv` | `household_key` 1 / 1302.82 / 2500; `DAY` 225 / 528.22 / 704; `COUPON_UPC` 10000085320 / 41230488063.41 / 58978500076; `CAMPAIGN` 1 / 15.55 / 30 |
| `hh_demographic.csv` | `household_key` 1 / 1235.18 / 2499 |
| `product.csv` | `PRODUCT_ID` 25671 / 5328352.84 / 18316298; `MANUFACTURER` 1 / 1739.23 / 6477 |
| `transaction_data.csv` | `household_key` 1 / 1271.95 / 2500; `BASKET_ID` 26984851472 / 34026199138.88 / 42305362535; `DAY` 1 / 388.76 / 711; `PRODUCT_ID` 25671 / 2891435.16 / 18316298; `QUANTITY` 0 / 100.43 / 89638; `SALES_VALUE` 0 / 3.10 / 840; `STORE_ID` 1 / 3142.67 / 34280; `RETAIL_DISC` -180 / -0.54 / 3.99; `TRANS_TIME` 0 / 1561.59 / 2359; `WEEK_NO` 1 / 56.22 / 102; `COUPON_DISC` -55.93 / -0.02 / 0; `COUPON_MATCH_DISC` -7.7 / 0 / 0 |

## Data-quality findings

- `coupon.csv` contains 5,164 excess exact full-row duplicates. This is an observed duplicate count; records have not been removed.
- `product.csv` has 30,607 missing `CURR_SIZE_OF_PRODUCT` values (33.1413%). The guide says this field is not available for all products. `DEPARTMENT`, `COMMODITY_DESC`, and `SUB_COMMODITY_DESC` each have 15 missing values (0.0162%).
- No other missing values, exact duplicate rows, invalid configured numeric values, absent required columns, or unexpected columns were found.
- In `transaction_data.QUANTITY`, the observed 75th percentile is 1 while the maximum is 89,638 and the minimum is 0. This is a pronounced tail worth reviewing; no records were declared invalid because no supported business threshold was established.
- The `(PRODUCT_ID, STORE_ID, WEEK_NO)` combination repeats in 15,245 groups in `causal_data.csv`, despite there being no exact duplicate rows. Treating those three fields alone as a unique row key is not supported by the data.

## Relationships verified from observed values

The following matches use same-named identifiers supported by the user guide. Counts are distinct child keys/pairs; all listed relationships have zero unmatched values unless stated otherwise.

| Observed relationship | Coverage / key observation |
| --- | --- |
| `campaign_table.CAMPAIGN` -> `campaign_desc.CAMPAIGN` | 30 / 30 campaign IDs match; `campaign_desc.CAMPAIGN` is unique (30 rows). |
| `coupon.CAMPAIGN` -> `campaign_desc.CAMPAIGN` | 30 / 30 campaign IDs match. |
| `coupon_redempt.CAMPAIGN` -> `campaign_desc.CAMPAIGN` | 30 / 30 campaign IDs match. |
| `coupon.PRODUCT_ID` -> `product.PRODUCT_ID` | 44,133 / 44,133 product IDs match. |
| `transaction_data.PRODUCT_ID` -> `product.PRODUCT_ID` | 92,339 / 92,339 product IDs match; `product.PRODUCT_ID` is unique (92,353 rows). |
| `causal_data.PRODUCT_ID` -> `product.PRODUCT_ID` | 68,377 / 68,377 product IDs match. |
| `hh_demographic.household_key` -> `transaction_data.household_key` | 801 / 801 household IDs match; demographic key is unique (801 rows). |
| `campaign_table.household_key` -> `transaction_data.household_key` | 1,584 / 1,584 household IDs match; `(household_key, CAMPAIGN)` is unique in `campaign_table`. |
| `coupon_redempt.household_key` -> `transaction_data.household_key` | 434 / 434 household IDs match. |
| `coupon_redempt.(COUPON_UPC, CAMPAIGN)` -> `coupon.(COUPON_UPC, CAMPAIGN)` | 643 / 643 distinct pairs match. |
| `coupon_redempt.(household_key, CAMPAIGN)` -> `campaign_table.(household_key, CAMPAIGN)` | 889 / 889 distinct redemption assignment pairs match; pair was directly checked before Phase 3 attribution. |
| Transaction store IDs observed in `causal_data.STORE_ID` | Only 115 of 582 transaction store IDs overlap; 467 transaction store IDs do not appear in `causal_data`. The guide describes `causal_data` as records of products featured in a mailer/display, so this is partial coverage rather than evidence of a broken key. |

The exact row duplicate count in `coupon.csv` means duplicate coupon-product-campaign records exist; the verified overlap above establishes referential coverage, not uniqueness. Candidate keys are not inferred beyond uniqueness explicitly reported from observed values.

## Configuration and reproducibility

`configs/ingestion.yaml` now lists the exact observed columns as required/allowed for each CSV and configures numeric parse checks only for fields Spark inferred as numeric. No date-format rules are set because the temporal fields are integer day/week indices. The Spark local master is limited to two workers with a 4 GiB driver heap and 32 shuffle partitions; the initial `local[*]` run exhausted heap on exact duplicate detection for the wide, 664 MiB `causal_data.csv`.

Rerun the inventory from the repository root with `python -m src.ingestion.inspect_dataset`. The generated `data/processed/dataset_inventory.json` is ignored by Git. Raw CSV files remain unchanged.
