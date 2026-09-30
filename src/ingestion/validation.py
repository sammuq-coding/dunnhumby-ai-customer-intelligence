"""Non-destructive, configurable data-quality checks for Spark DataFrames."""
from __future__ import annotations

from typing import Any

from pyspark.sql import DataFrame, functions as F


def validate_dataframe(df: DataFrame, rules: dict[str, Any] | None = None,
                       thresholds: dict[str, float] | None = None) -> dict[str, Any]:
    """Collect quality findings without dropping or correcting any rows.

    Column-specific rules are deliberately configuration-driven: no source
    columns, numeric fields, dates, or keys are assumed by this module.
    """
    rules = rules or {}
    thresholds = thresholds or {}
    columns = df.columns
    present = set(columns)
    required = set(rules.get("required_columns", []))
    allowed = set(rules.get("allowed_columns", []))
    missing_required = sorted(required - present)
    unexpected = sorted(present - allowed) if allowed else []
    # Aggregate row/null/format counts in one pass. Full-row duplicate
    # detection remains a separate shuffle, as Spark must compare records.
    count_exprs = [F.count(F.lit(1)).alias("__row_count")]
    null_aliases = {}
    for i, c in enumerate(columns):
        alias = f"__null_{i}"
        null_aliases[c] = alias
        is_missing = F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")
        count_exprs.append(F.sum(F.when(is_missing, 1).otherwise(0)).alias(alias))
    numeric_aliases = {}
    for i, c in enumerate(rules.get("numeric_columns", [])):
        if c in present:
            alias = f"__invalid_numeric_{i}"
            numeric_aliases[c] = alias
            quoted = "`" + c.replace("`", "``") + "`"
            safely_cast = F.expr(f"try_cast({quoted} as double)")
            present_value = F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) != "")
            count_exprs.append(F.sum(F.when(present_value & safely_cast.isNull(), 1).otherwise(0)).alias(alias))
    date_aliases = {}
    for i, (c, pattern) in enumerate(rules.get("date_columns", {}).items()):
        if c in present:
            alias = f"__invalid_date_{i}"
            date_aliases[c] = alias
            parsed = F.try_to_timestamp(F.col(c), F.lit(pattern))
            present_value = F.col(c).isNotNull() & (F.trim(F.col(c).cast("string")) != "")
            count_exprs.append(F.sum(F.when(present_value & parsed.isNull(), 1).otherwise(0)).alias(alias))
    aggregate = df.agg(*count_exprs).first().asDict()
    row_count = int(aggregate["__row_count"] or 0)
    null_counts = {c: int(aggregate[alias] or 0) for c, alias in null_aliases.items()}
    duplicate_count = row_count - df.dropDuplicates().count()
    null_rates = {c: (n / row_count if row_count else 0.0) for c, n in null_counts.items()}
    invalid_numeric = {c: int(aggregate[alias] or 0) for c, alias in numeric_aliases.items()}
    invalid_dates = {c: int(aggregate[alias] or 0) for c, alias in date_aliases.items()}
    null_rate_violations = {
        c: rate for c, rate in null_rates.items()
        if rate > float(thresholds.get("max_null_rate", 1.0))
    }
    reference_findings = []
    for relation in rules.get("relationships", []):
        # Relationships are only evaluated when explicitly configured and the
        # referenced key DataFrame is provided by the caller.
        if relation.get("reference_values") is None:
            reference_findings.append({"relationship": relation, "status": "not_evaluated_no_reference"})
            continue
        key, values = relation["column"], relation["reference_values"]
        if key in present:
            unknown = df.select(key).where(F.col(key).isNotNull()).distinct().join(
                values.select(F.col(relation["reference_column"])).distinct(),
                df[key] == values[relation["reference_column"]], "left_anti"
            ).count()
            reference_findings.append({"column": key, "unmatched_values": unknown})
    return {
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "missing_required_columns": missing_required,
        "unexpected_columns": unexpected,
        "duplicate_row_count": duplicate_count,
        "null_counts": null_counts,
        "null_rates": null_rates,
        "null_rate_violations": null_rate_violations,
        "invalid_numeric_counts": invalid_numeric,
        "invalid_date_counts": invalid_dates,
        "relationship_findings": reference_findings,
    }


def schema_description(df: DataFrame) -> list[dict[str, str]]:
    """Return observed Spark field names and data types."""
    return [{"column": f.name, "data_type": f.dataType.simpleString(), "nullable": str(f.nullable).lower()}
            for f in df.schema.fields]
