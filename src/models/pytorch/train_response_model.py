"""Train and evaluate a small PyTorch household-campaign response classifier."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from pyspark.sql import SparkSession
from torch import nn


FEATURES = [
    "historical_total_sales", "historical_basket_count", "historical_total_quantity",
    "historical_unique_products", "historical_unique_departments", "historical_retail_discount",
    "historical_coupon_discount", "historical_average_basket_value", "historical_coupon_basket_count",
    "historical_purchase_frequency", "historical_recency_days", "prior_coupon_redemption_count",
    "prior_campaign_exposure_count", "prior_campaign_redemption_count",
]


class ResponseNetwork(nn.Module):
    """14 numeric features -> 32 ReLU units -> one binary-classification logit."""
    def __init__(self, n_features: int, hidden_units: int = 32):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(n_features, hidden_units), nn.ReLU(), nn.Linear(hidden_units, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x).squeeze(-1)


def _auc_metrics(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return tie-adjusted ROC-AUC and average precision (PR-AUC summary)."""
    y = y.astype(np.int64)
    positives, negatives = int(y.sum()), int((1 - y).sum())
    roc = float("nan")
    if positives and negatives:
        order = np.argsort(scores, kind="mergesort")
        sorted_scores = scores[order]
        ranks = np.empty(len(scores), dtype=float)
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            ranks[order[i:j]] = (i + 1 + j) / 2.0
            i = j
        roc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    ap = float("nan")
    if positives:
        order = np.argsort(-scores, kind="mergesort")
        sorted_scores = scores[order]
        sorted_y = y[order]
        cumulative_positives = 0
        ap_sum = 0.0
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and sorted_scores[j] == sorted_scores[i]:
                j += 1
            group_positives = int(sorted_y[i:j].sum())
            cumulative_positives += group_positives
            # Average precision is computed at distinct score thresholds, so
            # tied predictions do not depend on arbitrary row ordering.
            ap_sum += (group_positives / positives) * (cumulative_positives / j)
            i = j
        ap = float(ap_sum)
    return roc, ap


def classification_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Metrics at a fixed, predeclared threshold plus ranking/calibration measures."""
    y = y.astype(np.int64)
    pred = (probs >= threshold).astype(np.int64)
    tp = int(((y == 1) & (pred == 1)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    roc, ap = _auc_metrics(y, probs)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "roc_auc": roc,
        "pr_auc_average_precision": ap,
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": {"true_negative": tn, "false_positive": fp,
                             "false_negative": fn, "true_positive": tp},
        "threshold": float(threshold),
        "positive_prediction_rate": float(pred.mean()) if len(pred) else 0.0,
        "mean_predicted_probability": float(probs.mean()) if len(probs) else float("nan"),
        "actual_positive_rate": float(y.mean()) if len(y) else float("nan"),
        "brier_score": float(np.mean((probs - y) ** 2)) if len(y) else float("nan"),
    }


def _collect_split(df, split: str) -> tuple[np.ndarray, np.ndarray]:
    rows = df.filter(df.data_split == split).select(*FEATURES, "target_redeemed").toLocalIterator()
    x, y = [], []
    for row in rows:
        x.append([np.nan if row[c] is None else float(row[c]) for c in FEATURES])
        y.append(int(row["target_redeemed"]))
    if not x:
        raise ValueError(f"No rows found for {split} split")
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _fit_preprocessing(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit imputation and scale using training values only."""
    means = np.nanmean(train_x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    stds = np.nanstd(train_x, axis=0)
    stds = np.where((stds > 0) & np.isfinite(stds), stds, 1.0)
    return means.astype(np.float32), stds.astype(np.float32)


def _transform(x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    imputed = np.where(np.isnan(x), means, x)
    return ((imputed - means) / stds).astype(np.float32)


def train_arrays(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray,
                 *, seed: int = 20260929, batch_size: int = 256, learning_rate: float = 0.001,
                 max_epochs: int = 100, patience: int = 10, hidden_units: int = 32,
                 use_pos_weight: bool = False):
    """Fit with train-only preprocessing, optional train-only class weighting,
    and early stopping on unweighted validation binary cross-entropy.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    means, stds = _fit_preprocessing(train_x)
    tx = torch.tensor(_transform(train_x, means, stds))
    vx = torch.tensor(_transform(val_x, means, stds))
    ty = torch.tensor(train_y, dtype=torch.float32)
    vy = torch.tensor(val_y, dtype=torch.float32)

    positives = float(train_y.sum())
    negatives = float(len(train_y) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("Training split must contain both target classes")
    pos_weight_value = negatives / positives if use_pos_weight else None
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32) if use_pos_weight else None
    weighted_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    unweighted_loss_fn = nn.BCEWithLogitsLoss()
    model = ResponseNetwork(train_x.shape[1], hidden_units)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_state, best_loss, best_epoch, waiting = None, float("inf"), 0, 0
    history = []
    for epoch in range(max_epochs):
        model.train()
        order = torch.randperm(len(tx))
        total_weighted_loss = 0.0
        for indexes in order.split(batch_size):
            optimizer.zero_grad()
            logits = model(tx[indexes])
            loss = weighted_loss_fn(logits, ty[indexes])
            loss.backward()
            optimizer.step()
            total_weighted_loss += float(loss.detach()) * len(indexes)
        model.eval()
        with torch.no_grad():
            train_bce = float(unweighted_loss_fn(model(tx), ty))
            val_loss = float(unweighted_loss_fn(model(vx), vy))
        history.append({
            "epoch": epoch + 1,
            "training_loss": total_weighted_loss / len(tx),
            "training_unweighted_bce": train_bce,
            "validation_loss": val_loss,
        })
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, waiting = val_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waiting += 1
            if waiting >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    fit = {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_run": len(history),
        "early_stopped": len(history) < max_epochs,
        "early_stopping_patience": patience,
        "positive_weight": pos_weight_value,
        "train_positive_count": int(positives),
        "train_negative_count": int(negatives),
        "history": history,
    }
    return model, means, stds, fit


def _predict(model: nn.Module, x: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(_transform(x, means, stds)))
        return torch.sigmoid(logits).numpy()


def _baseline_report(train_y: np.ndarray, split_x: np.ndarray, split_y: np.ndarray,
                     threshold: float) -> dict:
    train_rate = float(train_y.mean())
    majority_class = int(train_rate >= 0.5)
    majority_probs = np.full(len(split_y), majority_class, dtype=float)
    constant_probs = np.full(len(split_y), train_rate, dtype=float)
    # A transparent historical heuristic: any prior coupon redemption predicts
    # response. The indicator is also its 0/1 probability score.
    historical_probs = (split_x[:, FEATURES.index("prior_coupon_redemption_count")] > 0).astype(float)
    return {
        "training_positive_rate": train_rate,
        "majority_class": classification_metrics(split_y, majority_probs, threshold),
        "constant_training_rate": classification_metrics(split_y, constant_probs, threshold),
        "prior_coupon_redemption_indicator": classification_metrics(split_y, historical_probs, threshold),
    }


def _permutation_importance(model: nn.Module, val_x: np.ndarray, val_y: np.ndarray,
                           means: np.ndarray, stds: np.ndarray, repeats: int, seed: int) -> list[dict]:
    """Validation-set AP and log-loss sensitivity; not a causal interpretation."""
    baseline_probs = _predict(model, val_x, means, stds)
    _, baseline_ap = _auc_metrics(val_y, baseline_probs)
    eps = 1e-7

    def log_loss(y, p):
        p = np.clip(p, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    baseline_loss = log_loss(val_y, baseline_probs)
    rng = np.random.default_rng(seed)
    reports = []
    for index, name in enumerate(FEATURES):
        ap_decreases, loss_increases = [], []
        for _ in range(repeats):
            permuted = val_x.copy()
            permuted[:, index] = permuted[rng.permutation(len(permuted)), index]
            probs = _predict(model, permuted, means, stds)
            _, permuted_ap = _auc_metrics(val_y, probs)
            ap_decreases.append(baseline_ap - permuted_ap)
            loss_increases.append(log_loss(val_y, probs) - baseline_loss)
        reports.append({
            "feature": name,
            "mean_average_precision_decrease": float(np.mean(ap_decreases)),
            "std_average_precision_decrease": float(np.std(ap_decreases)),
            "mean_log_loss_increase": float(np.mean(loss_increases)),
        })
    return sorted(reports, key=lambda row: row["mean_average_precision_decrease"], reverse=True)


def run(config_path: str = "configs/modeling.yaml") -> dict:
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg["training"]["seed"])
    builder = SparkSession.builder.appName("campaign-response-model-training")
    for key, value in cfg.get("spark", {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    try:
        data = spark.read.parquet(cfg["paths"]["output"])
        train_x, train_y = _collect_split(data, "train")
        val_x, val_y = _collect_split(data, "validation")
        test_x, test_y = _collect_split(data, "test")
        train_cfg = cfg["training"]
        threshold = float(train_cfg["threshold"])
        model, means, stds, fit = train_arrays(
            train_x, train_y, val_x, val_y, seed=seed,
            batch_size=int(train_cfg["batch_size"]), learning_rate=float(train_cfg["learning_rate"]),
            max_epochs=int(train_cfg["max_epochs"]), patience=int(train_cfg["early_stopping_patience"]),
            hidden_units=int(train_cfg["hidden_units"]),
            use_pos_weight=bool(train_cfg.get("use_pos_weight", False)),
        )
        train_probs = _predict(model, train_x, means, stds)
        val_probs = _predict(model, val_x, means, stds)
        test_probs = _predict(model, test_x, means, stds)
        model_metrics = {
            "train": classification_metrics(train_y, train_probs, threshold),
            "validation": classification_metrics(val_y, val_probs, threshold),
            "test": classification_metrics(test_y, test_probs, threshold),
        }
        baselines = {
            split: _baseline_report(train_y, x, y, threshold)
            for split, x, y in (("train", train_x, train_y), ("validation", val_x, val_y), ("test", test_x, test_y))
        }
        importance = _permutation_importance(
            model, val_x, val_y, means, stds,
            int(train_cfg.get("permutation_repeats", 10)), seed,
        )
        metrics = {
            "interpretation": "observational prediction among campaign-assigned households; not a treatment-effect estimate",
            "architecture": {
                "type": "feed_forward_neural_network",
                "layers": [len(FEATURES), int(train_cfg["hidden_units"]), 1],
                "activation": "ReLU",
                "output": "logit; sigmoid for probabilities",
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "loss": "BCEWithLogitsLoss",
            },
            "feature_names": FEATURES,
            "threshold": threshold,
            "threshold_selection": "fixed at the configured 0.5; no test-set threshold selection",
            "preprocessing": "training-only column-mean imputation and z-score scaling; applied unchanged to validation/test",
            "class_weighting": "training-only positive weight (negative_count / positive_count)" if fit["positive_weight"] else "disabled",
            "row_counts": {"train": int(len(train_y)), "validation": int(len(val_y)), "test": int(len(test_y))},
            "training": fit,
            "baselines": baselines,
            "model_metrics": model_metrics,
            "validation_permutation_importance": importance,
            "split_column": "data_split",
            "seed": seed,
        }
        model_path = Path(cfg["paths"]["model"])
        metrics_path = Path(cfg["paths"]["metrics"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "architecture": metrics["architecture"],
            "feature_names": FEATURES,
            "imputation_means": means.tolist(),
            "normalization_stds": stds.tolist(),
            "seed": seed,
            "threshold": threshold,
            "class_weighting_positive_weight": fit["positive_weight"],
        }, model_path)
        metrics_path.write_text(json.dumps(metrics, indent=2, allow_nan=True) + "\n")
        return metrics
    finally:
        spark.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/modeling.yaml")
    print(json.dumps(run(parser.parse_args().config), indent=2, allow_nan=True))
