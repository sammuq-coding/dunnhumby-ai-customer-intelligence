"""Inspect raw CSVs with Spark and emit a JSON dataset inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pyspark.sql.types import NumericType

try:
    from .spark_ingestion import create_spark_session, discover_csv_files, read_csv, read_csv_as_strings
    from .validation import schema_description, validate_dataframe
except ImportError:  # Supports ``python src/ingestion/inspect_dataset.py``.
    from spark_ingestion import create_spark_session, discover_csv_files, read_csv, read_csv_as_strings
    from validation import schema_description, validate_dataframe


def inspect_file(spark: Any, path: Path, options: dict[str, Any], sample_size: int,
                 rules: dict[str, Any] | None = None,
                 thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Build an inventory entry for one CSV without changing its source."""
    df = read_csv(spark, path, options)
    # Validate source text separately so inference cannot mask malformed
    # numeric/date tokens by converting them to nulls.
    raw_df = read_csv_as_strings(spark, path, options)
    quality = validate_dataframe(raw_df, rules, thresholds)
    if quality["row_count"] != df.count():
        quality["inferred_read_row_count"] = df.count()
        quality["row_count_changed_by_inference"] = True
    else:
        quality["row_count_changed_by_inference"] = False
    numeric_columns = [f.name for f in df.schema.fields if isinstance(f.dataType, NumericType)]
    summary = df.select(numeric_columns).summary().toJSON().map(json.loads).collect() if numeric_columns else []
    sample = [json.loads(row) for row in df.limit(sample_size).toJSON().collect()]
    return {
        "filename": path.name,
        "relative_path": str(path),
        "row_count": quality["row_count"],
        "column_count": quality["column_count"],
        "columns": schema_description(df),
        "missing_value_counts": quality["null_counts"],
        "duplicate_row_count": quality["duplicate_row_count"],
        "sample_records": sample,
        "numeric_summary": summary,
        "quality": quality,
    }


def inventory(raw_path: str | Path, spark_settings: dict[str, Any] | None = None,
             csv_options: dict[str, Any] | None = None, sample_size: int = 5,
             rules_by_file: dict[str, Any] | None = None,
             thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Inspect all discovered CSV files and return a JSON-safe report."""
    files = discover_csv_files(raw_path)
    spark = create_spark_session("dunnhumby-dataset-inventory", spark_settings)
    try:
        return {
            "raw_path": str(raw_path),
            "csv_file_count": len(files),
            "datasets": [inspect_file(spark, p, csv_options or {}, sample_size,
                                      (rules_by_file or {}).get(p.name, {}), thresholds)
                         for p in files],
        }
    finally:
        spark.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ingestion.yaml")
    parser.add_argument("--output", default="data/processed/dataset_inventory.json")
    args = parser.parse_args()
    # PyYAML is a declared project dependency; keep config out of source code.
    import yaml
    config = yaml.safe_load(Path(args.config).read_text())
    report = inventory(config["paths"]["raw"], config["spark"], config["csv"],
                       config["inspection"]["sample_size"], config.get("datasets", {}),
                       config["validation"]["thresholds"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
