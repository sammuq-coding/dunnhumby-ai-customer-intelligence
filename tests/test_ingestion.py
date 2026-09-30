"""Small Spark fixture tests; no Complete Journey data is required."""
from __future__ import annotations

from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, StructField, StructType

from src.ingestion.spark_ingestion import discover_csv_files, read_csv
from src.ingestion.validation import validate_dataframe


@pytest.fixture(scope="session")
def spark():
    session = (SparkSession.builder.master("local[1]")
               .appName("ingestion-unit-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "1")
               .getOrCreate())
    yield session
    session.stop()


def test_discovery_finds_csv_recursively_and_ignores_other_files(tmp_path: Path):
    (tmp_path / "one.csv").write_text("a\n1\n")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "TWO.CSV").write_text("b\n2\n")
    (tmp_path / "notes.txt").write_text("ignore")
    assert [p.name for p in discover_csv_files(tmp_path)] == ["one.csv", "TWO.CSV"]


def test_expected_datasets_match_configured_datasets(tmp_path: Path):
    """Configured expected files, when source expectations exist, must be present."""
    (tmp_path / "known.csv").write_text("x\n1\n")
    expected_datasets = ["known.csv"]
    found = {p.name for p in discover_csv_files(tmp_path)}
    assert set(expected_datasets) <= found


def test_ingestion_preserves_row_count_and_headers(spark, tmp_path: Path):
    source = tmp_path / "fixture.csv"
    source.write_text("alpha,beta\n1,yes\n2,no\n3,yes\n")
    frame = read_csv(spark, source, {"header": "true", "inferSchema": "true"})
    assert frame.columns == ["alpha", "beta"]
    assert frame.count() == 3
    assert source.read_text().startswith("alpha,beta\n")  # The raw fixture was not rewritten.


def test_quality_checks_report_schema_nulls_duplicates_and_invalid_values(spark):
    schema = StructType([
        StructField("key", StringType(), True),
        StructField("amount", StringType(), True),
        StructField("event_day", StringType(), True),
        StructField("extra", StringType(), True),
    ])
    rows = [("a", "10", "2024-01-01", "x"),
            ("a", "10", "2024-01-01", "x"),
            (None, "bad", "not-a-date", "y")]
    frame = spark.createDataFrame(rows, schema)
    result = validate_dataframe(
        frame,
        {"required_columns": ["key", "missing"], "allowed_columns": ["key", "amount", "event_day"],
         "numeric_columns": ["amount"], "date_columns": {"event_day": "yyyy-MM-dd"}},
        {"max_null_rate": 0.2},
    )
    assert result["missing_required_columns"] == ["missing"]
    assert result["unexpected_columns"] == ["extra"]
    assert result["duplicate_row_count"] == 1
    assert result["null_counts"]["key"] == 1
    assert result["invalid_numeric_counts"]["amount"] == 1
    assert result["invalid_date_counts"]["event_day"] == 1
    assert "key" in result["null_rate_violations"]


def test_no_column_assumptions_when_no_rules_are_configured(spark):
    frame = spark.createDataFrame([("v",)], ["observed_field"])
    result = validate_dataframe(frame)
    assert result["missing_required_columns"] == []
    assert result["unexpected_columns"] == []
    assert result["relationship_findings"] == []
