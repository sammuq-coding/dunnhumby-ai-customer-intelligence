"""Build compact RAG documents from project docs and saved metric summaries."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENT_SOURCES = [
    ("README.md", "project"),
    ("data/README.md", "dataset and ingestion"),
    ("src/features/README.md", "feature engineering"),
    ("data/processed/modeling/README.md", "response modeling"),
    ("src/finance/README.md", "finance methodology"),
    ("src/dashboard/README.md", "dashboard"),
    ("configs/feature_windows.yaml", "feature configuration"),
    ("configs/modeling.yaml", "model configuration"),
    ("configs/finance.yaml", "finance assumptions"),
]


def _split_markdown(text: str, max_chars: int = 1400) -> list[str]:
    """Keep heading sections together when practical, then bound chunk size."""
    blocks = []
    current = []
    for line in text.splitlines():
        if line.startswith("#") and current:
            blocks.append("\n".join(current).strip())
            current = []
        current.append(line)
    if current:
        blocks.append("\n".join(current).strip())
    chunks: list[str] = []
    for block in blocks:
        if len(block) <= max_chars:
            if block:
                chunks.append(block)
            continue
        paragraphs = block.split("\n\n")
        buffer = ""
        for paragraph in paragraphs:
            if buffer and len(buffer) + len(paragraph) + 2 > max_chars:
                chunks.append(buffer)
                buffer = ""
            if len(paragraph) > max_chars:
                chunks.extend(paragraph[i:i + max_chars] for i in range(0, len(paragraph), max_chars))
            else:
                buffer = f"{buffer}\n\n{paragraph}".strip()
        if buffer:
            chunks.append(buffer)
    return chunks


def _metric_documents(data: dict[str, Any]) -> list[dict[str, str]]:
    docs = []
    model = data.get("model_metrics", {})
    calibration = data.get("calibration_metrics", {})
    finance = data.get("finance_report", {})
    for split, metric in model.get("model_metrics", {}).items():
        baseline = model.get("baselines", {}).get(split, {})
        content = {
            "observations": model.get("row_counts", {}).get(split),
            "actual_positive_rate": metric.get("actual_positive_rate"),
            "raw_model": {k: metric.get(k) for k in (
                "roc_auc", "pr_auc_average_precision", "accuracy", "precision", "recall",
                "f1", "brier_score", "log_loss") if metric.get(k) is not None},
            "baselines": {
                name: (
                    {k: value.get(k) for k in (
                        "roc_auc", "pr_auc_average_precision", "accuracy", "brier_score"
                    ) if value.get(k) is not None}
                    if isinstance(value, dict) else value
                )
                for name, value in baseline.items()
            },
            "threshold": model.get("threshold"),
        }
        docs.append({"id": f"model-metrics-{split}", "title": f"Response model metrics: {split}",
                     "source": "data/processed/modeling/response_model_metrics.json",
                     "category": "saved model results", "text": json.dumps(content, indent=2)})
    if calibration:
        val = calibration.get("validation", {})
        test = calibration.get("test", {})
        content = {
            "method": calibration.get("calibration_method"),
            "fit_split": calibration.get("calibrator_fit_split"),
            "fit_rows": calibration.get("calibrator_fit_rows"),
            "test_labels_used_during_calibration_fit": calibration.get("test_labels_used_during_calibration_fit"),
            "validation": {kind: _calibration_summary(value) for kind, value in val.items() if isinstance(value, dict)},
            "test": {kind: _calibration_summary(value) for kind, value in test.items() if isinstance(value, dict)},
        }
        docs.append({"id": "calibration-summary", "title": "Probability calibration results",
                     "source": "data/processed/modeling/response_model_calibration_metrics.json",
                     "category": "calibration", "text": json.dumps(content, indent=2)})
    if finance:
        content = {k: finance.get(k) for k in ("assumptions", "monthly_discount_rate", "clv_summary", "offer_summary", "warnings")}
        docs.append({"id": "finance-summary", "title": "Finance outputs and assumptions",
                     "source": "data/processed/finance/finance_report.json",
                     "category": "modeled financial estimates", "text": json.dumps(content, indent=2)})
    return docs


def _calibration_summary(metrics: dict) -> dict:
    result = {}
    for kind, values in metrics.items():
        if not isinstance(values, dict):
            continue
        result[kind] = {key: values.get(key) for key in (
            "roc_auc", "pr_auc_average_precision", "brier_score", "log_loss",
            "expected_calibration_error", "mean_predicted_probability", "actual_positive_rate")}
    return result


def build_knowledge_base(data: dict[str, Any], root: str | Path | None = None) -> list[dict[str, str]]:
    """Create compact searchable docs; never serializes Parquet table rows."""
    root_path = Path(root or PROJECT_ROOT)
    docs: list[dict[str, str]] = []
    for relative, category in DOCUMENT_SOURCES:
        path = root_path / relative
        if not path.is_file():
            continue
        for index, text in enumerate(_split_markdown(path.read_text(encoding="utf-8"))):
            if not text:
                continue
            heading = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), path.name)
            docs.append({"id": f"{path.stem}-{index}", "title": heading,
                         "source": relative, "category": category, "text": text})
    docs.extend(_metric_documents(data))
    return docs
