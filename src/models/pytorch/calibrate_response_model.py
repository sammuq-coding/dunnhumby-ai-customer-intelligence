"""Fit validation-only Platt scaling for the saved campaign response model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import SparkSession
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pytorch.train_response_model import FEATURES, _auc_metrics, _collect_split, classification_metrics


def fit_platt_calibrator(validation_logits: np.ndarray, validation_targets: np.ndarray) -> tuple[float, float]:
    """Fit sigmoid(slope * raw_logit + intercept) using validation data only."""
    logits = torch.tensor(validation_logits.reshape(-1, 1), dtype=torch.float64)
    targets = torch.tensor(validation_targets, dtype=torch.float64)
    if logits.numel() != targets.numel() or logits.numel() == 0:
        raise ValueError("Validation logits and targets must be nonempty arrays of equal length")
    if torch.unique(targets).numel() != 2:
        raise ValueError("Validation targets must contain both classes for Platt calibration")

    slope = nn.Parameter(torch.ones((), dtype=torch.float64))
    intercept = nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([slope, intercept], lr=0.1, max_iter=200,
                                  tolerance_grad=1e-10, tolerance_change=1e-12,
                                  line_search_fn="strong_wolfe")
    loss_fn = nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(logits[:, 0] * slope + intercept, targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    result = float(slope.detach()), float(intercept.detach())
    if not np.all(np.isfinite(result)):
        raise ValueError("Platt calibration produced non-finite parameters")
    return result


def reliability_data(targets: np.ndarray, probabilities: np.ndarray,
                     bin_edges: np.ndarray | None = None) -> tuple[list[dict], float]:
    """Return fixed-width reliability bins and expected calibration error."""
    if bin_edges is None:
        bin_edges = np.linspace(0.0, 1.0, 11)
    indexes = np.searchsorted(bin_edges, probabilities, side="right") - 1
    indexes = np.clip(indexes, 0, len(bin_edges) - 2)
    rows, ece_numerator = [], 0.0
    for i in range(len(bin_edges) - 1):
        mask = indexes == i
        count = int(mask.sum())
        predicted = float(probabilities[mask].mean()) if count else None
        observed = float(targets[mask].mean()) if count else None
        if count:
            ece_numerator += count * abs(predicted - observed)
        rows.append({
            "bin": i,
            "lower_inclusive": float(bin_edges[i]),
            "upper_bound": float(bin_edges[i + 1]),
            "upper_inclusive": i == len(bin_edges) - 2,
            "count": count,
            "mean_predicted_probability": predicted,
            "observed_redemption_rate": observed,
        })
    return rows, ece_numerator / len(targets) if len(targets) else float("nan")


def compare_probabilities(targets: np.ndarray, probabilities: np.ndarray,
                          threshold: float, bin_edges: np.ndarray) -> dict:
    metrics = classification_metrics(targets, probabilities, threshold)
    eps = 1e-12
    clipped = np.clip(probabilities, eps, 1.0 - eps)
    metrics["log_loss"] = float(-np.mean(targets * np.log(clipped) + (1 - targets) * np.log(1 - clipped)))
    metrics["reliability_bins"], metrics["expected_calibration_error"] = reliability_data(
        targets, probabilities, bin_edges)
    return metrics


def _raw_logits(model: nn.Module, x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isnan(x), means, x)
    standardized = torch.tensor(((imputed - means) / stds).astype(np.float32))
    model.eval()
    with torch.no_grad():
        return model(standardized).numpy()


def run(config_path: str = "configs/modeling.yaml", n_bins: int = 10) -> dict:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    model_artifact = torch.load(cfg["paths"]["model"], map_location="cpu", weights_only=False)
    if model_artifact["feature_names"] != FEATURES:
        raise ValueError("Saved model feature order differs from the current trainer feature list")
    hidden_units = int(model_artifact["architecture"]["layers"][1])
    from src.models.pytorch.train_response_model import ResponseNetwork
    model = ResponseNetwork(len(FEATURES), hidden_units)
    model.load_state_dict(model_artifact["state_dict"])
    means = np.asarray(model_artifact["imputation_means"], dtype=np.float32)
    stds = np.asarray(model_artifact["normalization_stds"], dtype=np.float32)
    threshold = float(model_artifact["threshold"])
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    builder = SparkSession.builder.appName("campaign-response-platt-calibration")
    for key, value in cfg.get("spark", {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    try:
        data = spark.read.parquet(cfg["paths"]["output"])
        # Validation predictions and labels are the only inputs to fitting.
        val_x, val_y = _collect_split(data, "validation")
        val_logits = _raw_logits(model, val_x, means, stds)
        slope, intercept = fit_platt_calibrator(val_logits, val_y)
        val_raw = torch.sigmoid(torch.tensor(val_logits)).numpy()
        val_calibrated = torch.sigmoid(torch.tensor(slope * val_logits + intercept)).numpy()

        # Test labels are not collected until after the calibrator is frozen.
        test_x, test_y = _collect_split(data, "test")
        test_logits = _raw_logits(model, test_x, means, stds)
        test_raw = torch.sigmoid(torch.tensor(test_logits)).numpy()
        test_calibrated = torch.sigmoid(torch.tensor(slope * test_logits + intercept)).numpy()

        metrics = {
            "calibration_method": "Platt scaling: sigmoid(slope * validation_model_logit + intercept)",
            "calibrator_fit_split": "validation only",
            "calibrator_fit_rows": int(len(val_y)),
            "calibrator_fit_positive_rows": int(val_y.sum()),
            "calibrator_parameters": {"slope": slope, "intercept": intercept},
            "test_labels_used_during_calibration_fit": False,
            "classification_threshold": threshold,
            "threshold_note": "Fixed at the saved model's configured 0.5 threshold; threshold metrics are secondary to calibration.",
            "probability_bin_edges": edges.tolist(),
            "validation": {
                "raw": compare_probabilities(val_y, val_raw, threshold, edges),
                "calibrated": compare_probabilities(val_y, val_calibrated, threshold, edges),
                "calibrated_metrics_are_in_sample": True,
            },
            "test": {
                "raw": compare_probabilities(test_y, test_raw, threshold, edges),
                "calibrated": compare_probabilities(test_y, test_calibrated, threshold, edges),
                "calibrated_metrics_are_in_sample": False,
            },
            "ranking_change": {
                "validation_roc_auc_delta": _auc_metrics(val_y, val_calibrated)[0] - _auc_metrics(val_y, val_raw)[0],
                "validation_average_precision_delta": _auc_metrics(val_y, val_calibrated)[1] - _auc_metrics(val_y, val_raw)[1],
                "test_roc_auc_delta": _auc_metrics(test_y, test_calibrated)[0] - _auc_metrics(test_y, test_raw)[0],
                "test_average_precision_delta": _auc_metrics(test_y, test_calibrated)[1] - _auc_metrics(test_y, test_raw)[1],
            },
        }
        calibrator_path = Path(cfg["paths"]["calibrator"])
        metrics_path = Path(cfg["paths"]["calibration_metrics"])
        calibrator_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "method": "platt_scaling",
            "slope": slope,
            "intercept": intercept,
            "input": "raw model logit",
            "fit_split": "validation",
            "validation_rows": int(len(val_y)),
            "validation_positive_rows": int(val_y.sum()),
        }, calibrator_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
        return metrics
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modeling.yaml")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.bins), indent=2, allow_nan=True))
