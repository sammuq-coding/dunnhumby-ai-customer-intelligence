"""Reusable CSV discovery and ingestion with PySpark.

Schema inference is enabled for initial discovery. Once the real source schema
has been inspected, a versioned explicit schema can be supplied per dataset.
The reader never writes to or modifies source files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pyspark.sql import DataFrame, SparkSession


def discover_csv_files(raw_path: str | Path) -> list[Path]:
    """Return CSV files below ``raw_path`` in deterministic path order."""
    root = Path(raw_path)
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory does not exist: {root}")
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".csv"), key=str)


def create_spark_session(app_name: str, settings: dict[str, Any] | None = None) -> SparkSession:
    """Create or retrieve a SparkSession using configuration-provided settings."""
    builder = SparkSession.builder.appName(app_name)
    for key, value in (settings or {}).items():
        builder = builder.config(key, value)
    return builder.getOrCreate()


def read_csv(spark: SparkSession, path: str | Path, options: dict[str, Any] | None = None,
             schema: Any | None = None) -> DataFrame:
    """Read one CSV while preserving values and reporting malformed records.

    Inference is used only when no inspected, explicit schema is provided. A
    corrupt-record field is reserved so malformed lines are not silently lost.
    """
    reader = spark.read.options(**(options or {}))
    if schema is not None:
        reader = reader.schema(schema)
    return reader.csv(str(path))


def read_csv_as_strings(spark: SparkSession, path: str | Path,
                        options: dict[str, Any] | None = None) -> DataFrame:
    """Read CSV fields as strings for loss-aware validation of source values.

    Spark type inference can turn malformed numeric tokens into nulls. A
    parallel string-typed read lets configured numeric/date checks distinguish
    malformed source text from actual missing fields.
    """
    string_options = dict(options or {})
    string_options["inferSchema"] = "false"
    return spark.read.options(**string_options).csv(str(path))


def read_datasets(spark: SparkSession, raw_path: str | Path,
                  options: dict[str, Any] | None = None,
                  schemas: dict[str, Any] | None = None) -> dict[str, DataFrame]:
    """Read all discovered CSV files, keyed by filename stem.

    Duplicate filename stems in separate subdirectories are rejected to avoid
    silently overwriting one DataFrame with another.
    """
    files = discover_csv_files(raw_path)
    keys = [p.stem for p in files]
    if len(keys) != len(set(keys)):
        raise ValueError("CSV filenames must have unique stems across raw data subdirectories")
    return {p.stem: read_csv(spark, p, options, (schemas or {}).get(p.name)) for p in files}
