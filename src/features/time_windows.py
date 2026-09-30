"""Leakage-safe filters for Complete Journey's integer DAY index.

DAY is treated as an ordered integer. No calendar conversion is performed.
Feature intervals are half-open [cutoff - lookback, cutoff); outcome intervals
are [cutoff, cutoff + outcome_window). A missing feature lookback means all
available history before the cutoff. An outcome window must be explicit.
"""
from __future__ import annotations

from pyspark.sql import DataFrame, functions as F


def _positive_days(name: str, value: int | None) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
        raise ValueError(f"{name} must be a positive integer or None")


def feature_period_bounds(analysis_cutoff_day: int,
                          feature_lookback_days: int | None = None) -> tuple[int | None, int]:
    """Return the inclusive lower and exclusive upper feature bounds."""
    _positive_days("feature_lookback_days", feature_lookback_days)
    if isinstance(analysis_cutoff_day, bool) or not isinstance(analysis_cutoff_day, int):
        raise ValueError("analysis_cutoff_day must be an integer DAY index")
    lower = None if feature_lookback_days is None else analysis_cutoff_day - feature_lookback_days
    return lower, analysis_cutoff_day


def outcome_period_bounds(analysis_cutoff_day: int,
                          outcome_window_days: int) -> tuple[int, int]:
    """Return the half-open outcome interval [cutoff, cutoff + window)."""
    _positive_days("outcome_window_days", outcome_window_days)
    if isinstance(analysis_cutoff_day, bool) or not isinstance(analysis_cutoff_day, int):
        raise ValueError("analysis_cutoff_day must be an integer DAY index")
    return analysis_cutoff_day, analysis_cutoff_day + outcome_window_days


def filter_feature_period(df: DataFrame, day_column: str, analysis_cutoff_day: int,
                          feature_lookback_days: int | None = None) -> DataFrame:
    """Keep only pre-cutoff rows, optionally limited to a lookback window."""
    lower, upper = feature_period_bounds(analysis_cutoff_day, feature_lookback_days)
    result = df.filter(F.col(day_column) < F.lit(upper))
    if lower is not None:
        result = result.filter(F.col(day_column) >= F.lit(lower))
    return result


def filter_outcome_period(df: DataFrame, day_column: str, analysis_cutoff_day: int,
                          outcome_window_days: int) -> DataFrame:
    """Keep rows in [cutoff, cutoff + outcome window); requires a stated window."""
    lower, upper = outcome_period_bounds(analysis_cutoff_day, outcome_window_days)
    return df.filter((F.col(day_column) >= F.lit(lower)) & (F.col(day_column) < F.lit(upper)))
