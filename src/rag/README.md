# Customer Intelligence Analytics Assistant

The assistant is an additive eighth section in the existing Streamlit
dashboard. It sits on top of the verified analytics, saved model metrics, and
finance outputs; it does not replace or mutate those layers.

This is a deliberately small retrieval-augmented analytics assistant: BM25
retrieves project documentation, while deterministic Python tools answer
quantitative questions. Optional LLM synthesis uses only those bounded results
and retrieved excerpts. Offline mode uses deterministic answer templates.

## Architecture

* `analytics_tools.py` provides deterministic lookups and rankings over the
  existing dashboard data objects. Quantitative answers are read from saved
  Parquet/JSON artifacts; CLV, offer economics, model scores, and labels are
  not recomputed here.
* `context_builder.py` creates small retrieval documents from project
  documentation, YAML configuration, and selected concise metric/report
  summaries. It does not add Parquet rows to the knowledge base.
* `retriever.py` implements a dependency-free BM25 ranker for this small
  documentation collection. The index is built in memory and no vector store
  or embedding service is required.
* `assistant.py` routes known quantitative requests to an allowlisted Python
  tool, retrieves relevant documentation, and composes a grounded response.
  It exposes a provider-neutral `LLMProvider` interface, an API-compatible
  HTTP implementation, and a deterministic mock.
* `assistant_ui.py` supplies chat history, source/evidence disclosure, and the
  no-LLM experience in the dashboard.

## Sources and evidence labels

Knowledge sources include the main README, `data/README.md` (dataset and
ingestion facts), feature documentation, response-model dataset documentation,
finance documentation, dashboard documentation, and the feature/model/finance
configs. Compact generated documents summarize the saved response metrics,
calibration report, and finance report. The assistant does not load
`causal_data.csv` and does not ingest complete Parquet tables into an LLM
prompt.

Quantitative tools include household lookup, campaign lookup, customer and
campaign ranking, saved model metric retrieval, exact finance assumptions,
and saved campaign offer economics. Source indicators identify the relevant
table or report. Customer history, model predictions, financial estimates,
and assumptions are separately identified. Campaign rates remain descriptive;
the model predicts observed redemption among campaign-assigned households;
CLV and offer value remain modeled scenarios.

If a requested entity is absent or a documentation query has no relevant
source, the assistant reports that the information is unavailable. An LLM is
instructed to refuse unsupported claims, but generated language is not
formally verified and should be checked against the disclosed evidence. Campaign
causality and offer-response counterfactuals are not inferred from these
artifacts. Campaign ranking uses only intervals ending within the verified
observed coupon-redemption DAY coverage.

## LLM provider and deterministic mode

No third-party LLM framework or SDK is required. Without configuration, the
app uses deterministic intent routing, BM25 retrieval, and extractive/factual
response templates. Tests use `MockLLMProvider`; it never makes a network call.

To enable an OpenAI-compatible Chat Completions endpoint, set environment
variables in the shell that launches Streamlit:

```bash
export LLM_BASE_URL="https://your-provider.example/v1"
export LLM_MODEL="your-model-name"
export LLM_API_KEY="your-secret"
```

`LLM_API_KEY` is optional for a local unauthenticated endpoint. The configured
endpoint receives the user's question, the selected deterministic result, and
up to four retrieved document chunks. It does not receive raw CSV files or
whole Parquet tables. Do not commit credentials. If the endpoint is absent or
fails, deterministic responses remain available. LLM output is instructed to
use supplied evidence only; quantitative results still come from Python.

The generic HTTP adapter expects a Chat Completions-compatible endpoint at
`<LLM_BASE_URL>/chat/completions` (or a base URL that already ends in that
path). Other provider APIs can implement `LLMProvider.generate()` without
changing analytics tools or the UI.

## Testing

`tests/test_rag.py` exercises each deterministic tool, source coverage,
rankings, missing entities, probability validation, retrieval, mock-provider
responses, and unsupported/counterfactual questions using synthetic fixtures.
No live LLM API is called by the test suite.

Run the full project suite from the repository root:

```bash
python -m pytest -q
```
