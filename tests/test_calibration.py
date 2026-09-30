"""Focused tests for Platt calibration and reliability summaries."""
import numpy as np
import pytest

from src.models.pytorch.calibrate_response_model import fit_platt_calibrator, reliability_data


def test_platt_calibrator_fits_validation_logit_target_pairs():
    logits = np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0])
    targets = np.array([0, 0, 1, 0, 0, 1, 1, 1])
    slope, intercept = fit_platt_calibrator(logits, targets)
    assert np.isfinite(slope)
    assert np.isfinite(intercept)
    assert slope > 0


def test_reliability_bins_cover_each_probability_once_and_compute_ece():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.25, 0.75, 0.95])
    bins, ece = reliability_data(targets, probabilities, np.array([0.0, 0.5, 1.0]))
    assert sum(row["count"] for row in bins) == len(targets)
    assert bins[0]["count"] == bins[1]["count"] == 2
    assert ece == pytest.approx(0.15)
