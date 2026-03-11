from __future__ import annotations

from typing import Any

import pandas as pd

from app.stats.columns import is_technical_id_column


def dataframe_quality_warnings(df: pd.DataFrame) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    n_rows = max(len(df), 1)

    for col in df.columns:
        if is_technical_id_column(str(col)):
            continue

        if str(col).lower() == "week" and int(df[col].nunique(dropna=True)) <= 1:
            # Keep only the specialized week warning to avoid redundant constant/near-constant duplicates.
            continue

        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1:
            warnings.append(
                {
                    "code": "constant_column",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column {col} is constant (nunique={nunique}).",
                    "suggestion": "Consider dropping constant columns from models.",
                }
            )
            continue

        value_counts = series.value_counts(dropna=True)
        if not value_counts.empty:
            top_freq_ratio = float(value_counts.iloc[0] / n_rows)
            if top_freq_ratio >= 0.98:
                warnings.append(
                    {
                        "code": "near_constant_column",
                        "severity": "warning",
                        "column": col,
                        "message": f"Column {col} is near-constant (top value {top_freq_ratio:.2%}).",
                        "suggestion": "Use binning or remove if it adds little signal.",
                    }
                )

        unique_ratio = nunique / n_rows
        is_numeric = pd.api.types.is_numeric_dtype(series)
        if not is_numeric and (nunique > 50 or unique_ratio > 0.30):
            suggestion = "Apply top-k + other grouping or target encoding in later iterations."
            if str(col).lower().startswith("owner"):
                suggestion = (
                    "Owner appears high-cardinality/ID-like; consider grouping, per-owner target mean, "
                    "or random-effects modeling in future iterations."
                )
            warnings.append(
                {
                    "code": "high_cardinality",
                    "severity": "warning",
                    "column": col,
                    "message": f"Column {col} has high cardinality (nunique={nunique}, ratio={unique_ratio:.2%}).",
                    "suggestion": suggestion,
                }
            )

    if "week" in df.columns and int(df["week"].nunique(dropna=True)) <= 1:
        warnings.append(
            {
                "code": "week_constant",
                "severity": "warning",
                "column": "week",
                "message": "Column week is constant or nearly constant.",
                "suggestion": "Review temporal segmentation or drop week from feature relevance analysis.",
            }
        )

    return warnings
