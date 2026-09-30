# Defensible portfolio and interview claims

This project is strongest as an end-to-end analytics engineering and applied
predictive modeling demonstration. Claims below describe what the repository
actually implements; they do not imply production deployment or causal proof.

| Area | Defensible claim | Repository evidence | Interview explanation | Do not claim |
| --- | --- | --- | --- | --- |
| PySpark ingestion and validation | Built configurable PySpark CSV inspection and quality checks for schema, missing values, duplicates, numeric parsing, and configured relationships. | `src/ingestion/`, `configs/ingestion.yaml`, `data/README.md` | Explain why raw CSVs are read without mutation, how schema inference differs from source truth, and which observed quality issues remain visible. | That every source defect is automatically repaired, or that `causal_data.csv` powers the feature/model pipeline. |
| Scalable feature engineering | Used Spark DataFrames and aggregations to create customer-, product-, and campaign-grain tables, with basket-aware counts and verified-key joins. | `src/features/`, `src/sql/`, `src/features/README.md` | Discuss table grain, avoiding item-row/basket confusion, and handling missing product labels and extreme quantity values. | That all metrics have been validated as business KPIs or that all possible sources feed each table. |
| PyTorch classification | Trained a small 14-input feed-forward network to predict observed coupon redemption for campaign-assigned household/campaign rows. | `src/models/pytorch/train_response_model.py`, `data/processed/modeling/response_model_metrics.json` (generated locally) | Describe the target population, network `[14, 32, 1]`, class imbalance, baselines, and the test-period ranking metrics. | Causal response, incremental lift, guaranteed redemption, or a deployed production model. |
| Leakage controls and time-aware evaluation | Built features from a configurable 90 `DAY`-index lookback with strict `DAY < START_DAY`, then split by campaign start day chronologically and purged overlapping campaign windows. | `src/models/pytorch/build_training_data.py`, `configs/modeling.yaml`, `data/processed/modeling/README.md`, `tests/test_modeling.py` | Walk through current-campaign labels versus historical inputs and explain why `DAY` is not a calendar date. | A randomized evaluation, leakage-free guarantee for every future deployment setting, or generalization beyond the observed campaigns. |
| Model evaluation and baselines | Reported ranking, classification, and probability metrics on train/validation/test and compared with majority, train-rate, and prior-redemption baselines. | `src/models/pytorch/train_response_model.py`, saved model metrics, dashboard Model Performance section | Note that test AP is about 0.431 versus 0.267 for the prior-redemption indicator, while fixed-threshold F1 is slightly lower for the network (about 0.429 versus 0.441). | That the neural network wins on every metric or that high accuracy alone indicates useful performance under class imbalance. |
| Probability calibration | Fit Platt scaling from validation predictions and labels, froze it before test scoring, and reported reliability, Brier score, log loss, and ECE. | `src/models/pytorch/calibrate_response_model.py`, calibration JSON, `tests/test_calibration.py` | Explain why ranking metrics remain unchanged and how test calibration is assessed after fitting only on validation. | That calibration is guaranteed to remain stable on future populations or that the validation calibration scores are held-out estimates. |
| CLV / NPV | Implemented explicit discounted contribution calculations for modeled 12-/24-month CLV using configured margin, retention, discount rate, and a common 711-index-day dataset-span revenue assumption. | `src/finance/clv.py`, `src/finance/npv.py`, `configs/finance.yaml`, `src/finance/README.md` | Derive the monthlyization assumption and discount formula; emphasize that historical revenue is not profit and the denominator is not household tenure. | Observed lifetime value, learned customer survival, decision-grade forecasts, or validated profit. |
| Offer economics | Calculated a held-out response scenario using calibrated observed-redemption probability, one historical average basket, an assumed margin, and configured offer cost. | `src/finance/offer_economics.py`, `src/finance/build_finance.py`, `src/finance/README.md` | Explain the break-even cost formula and why expected value is conditional on explicit scenario assumptions. | Incremental profit, campaign ROI, response uplift from changing the offer, or a causal offer effect. |
| Statistical experimentation | Performed observational/descriptive campaign and before/during/after spend analysis with completeness indicators. | `src/features/marketing_features.py`, `src/dashboard/app.py`, `data/README.md` | Explain why assigned-versus-unassigned or before-versus-after comparisons do not identify causal effects here. | Designed or analyzed a randomized A/B test, identified incrementality, or estimated treatment effects. |
| Streamlit analytics | Built an eight-section dashboard over saved analytics/model/finance outputs, with validation checks for table grain, references, and probability/value bounds. | `src/dashboard/app.py`, `src/dashboard/data_loader.py`, `tests/test_dashboard.py` | Describe the pages, artifact loading, and checks that fail early on inconsistent outputs. | A hosted, monitored, multi-user production application; the dashboard is locally run and needs generated artifacts. |
| BM25 retrieval and assistant | Added BM25 retrieval over project documentation plus deterministic Python tools for numeric customer, campaign, model, CLV-assumption, and offer lookups; an OpenAI-compatible adapter is optional. | `src/rag/`, `tests/test_rag.py`, AI Analytics Assistant dashboard section | Explain that deterministic tools provide numbers, retrieved passages provide methodology, and the optional LLM classifies/summarizes bounded evidence. | A vector database, embeddings pipeline, autonomous agent, fine-tuned model, or guaranteed hallucination-free LLM. |
| Docker/containerization | Added a Python/Java Docker build recipe as an optional environment scaffold. | `Dockerfile`, `.dockerignore` | State that Docker can package dependencies, but the project was verified locally rather than through a full container pipeline. | That all ingestion/model/dashboard workflows were tested in Docker or that the image contains the dataset/artifacts. |

## Resume bullet examples

* Built a PySpark pipeline that validates the Complete Journey source files and
  produces customer-, product-, and campaign-grain analytical tables with
  documented quality and temporal assumptions.
* Trained and evaluated a PyTorch coupon-redemption prediction model using
  strict pre-campaign features, chronological campaign splits, simple
  baselines, and validation-only Platt calibration.
* Implemented assumption-driven CLV/NPV and offer-value scenarios and a
  Streamlit dashboard that distinguishes observed metrics, predictions, and
  modeled estimates.
* Added deterministic analytics tools and BM25 retrieval over project
  documentation, with optional OpenAI-compatible synthesis and an offline
  test mode.

Use these as prototype/project bullets. Do not imply business deployment,
measured revenue impact, or causal marketing lift.
