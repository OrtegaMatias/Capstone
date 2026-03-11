from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.stats.columns import analytical_columns


def shannon_entropy(series: pd.Series) -> float:
    counts = series.dropna().astype("string").value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def gini_impurity(series: pd.Series) -> float:
    probs = series.dropna().astype("string").value_counts(normalize=True)
    if probs.empty:
        return 0.0
    return float(1.0 - np.square(probs).sum())


def coefficient_variation(series: pd.Series) -> tuple[float | None, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None, warnings
    mean_val = float(values.mean())
    if np.isclose(mean_val, 0.0):
        warnings.append(
            {
                "code": "cv_mean_near_zero",
                "severity": "warning",
                "column": series.name,
                "message": "Coefficient of variation undefined when mean is close to zero.",
                "suggestion": "Use std or IQR for scale instead of CV.",
            }
        )
        return None, warnings
    return float(values.std(ddof=1) / mean_val), warnings


def custom_variability_index(
    series: pd.Series,
    custom_mode: str = "freq_only",
    ordinal_strategy: str = "frequency",
) -> tuple[float | None, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    clean = series.dropna()
    total = len(clean)
    if total == 0:
        return None, warnings

    numeric = pd.to_numeric(clean, errors="coerce")
    numeric_ratio = float(numeric.notna().mean())

    if numeric_ratio >= 0.9:
        vc = numeric.value_counts()
        score = float(((vc.index.to_numpy(dtype=float) * vc.to_numpy(dtype=float)) / total).sum())
        return score, warnings

    if custom_mode == "freq_only":
        warnings.append(
            {
                "code": "custom_non_informative",
                "severity": "info",
                "column": series.name,
                "message": "Custom index in freq_only mode is always 1.0 for categorical variables.",
                "suggestion": "Use Entropy/Gini for categorical variability.",
            }
        )
        return 1.0, warnings

    labels = clean.astype("string")
    if ordinal_strategy == "alphabetical":
        ordered_labels = sorted(labels.unique().tolist())
    else:
        ordered_labels = labels.value_counts().index.tolist()

    mapping = {label: idx + 1 for idx, label in enumerate(ordered_labels)}
    mapped = labels.map(mapping)
    vc = mapped.value_counts()
    score = float(((vc.index.to_numpy(dtype=float) * vc.to_numpy(dtype=float)) / total).sum())

    warnings.append(
        {
            "code": "custom_ordinal_map_warning",
            "severity": "warning",
            "column": series.name,
            "message": "Ordinal map introduces arbitrary numeric structure for categories.",
            "suggestion": "Interpret this index carefully; prefer Entropy/Gini.",
        }
    )
    return score, warnings


def is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def compute_variability_scores(
    df: pd.DataFrame,
    custom_mode: str = "freq_only",
    ordinal_strategy: str = "frequency",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    analysis_cols = analytical_columns(df)

    for col in analysis_cols:
        series = df[col]
        row_warnings: list[dict[str, Any]] = []
        dtype_group = "numeric" if is_numeric(series) else "categorical"

        entropy_val = shannon_entropy(series) if dtype_group == "categorical" else None
        gini_val = gini_impurity(series) if dtype_group == "categorical" else None

        cv_val, cv_warnings = coefficient_variation(series) if dtype_group == "numeric" else (None, [])
        row_warnings.extend(cv_warnings)

        custom_val, custom_warnings = custom_variability_index(
            series,
            custom_mode=custom_mode,
            ordinal_strategy=ordinal_strategy,
        )
        row_warnings.extend(custom_warnings)

        recommendation = (
            "Use Entropy/Gini for categorical columns."
            if dtype_group == "categorical"
            else "Use CV with mean-level checks for numeric columns."
        )

        rows.append(
            {
                "column": col,
                "dtype_group": dtype_group,
                "entropy": entropy_val,
                "gini_impurity": gini_val,
                "coefficient_variation": cv_val,
                "custom_index": custom_val,
                "custom_mode": custom_mode,
                "recommendation": recommendation,
                "warnings": row_warnings,
            }
        )

    return {"rows": rows}
