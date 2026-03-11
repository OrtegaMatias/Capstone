from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.stats.columns import analytical_columns
from app.stats.warnings import dataframe_quality_warnings


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if np.isnan(value):
            return None
        return float(value)
    return None


def compute_eda(df: pd.DataFrame) -> dict[str, Any]:
    analysis_cols = analytical_columns(df)
    analysis_df = df[analysis_cols].copy()
    excluded_cols = [str(col) for col in df.columns if str(col) not in analysis_cols]

    dtypes = {col: str(dtype) for col, dtype in analysis_df.dtypes.items()}

    missingness: dict[str, dict[str, float]] = {}
    cardinality: dict[str, int] = {}
    top_values: dict[str, list[dict[str, Any]]] = {}
    numeric_stats: dict[str, dict[str, float | None]] = {}
    numeric_histograms: dict[str, list[dict[str, float | int]]] = {}

    numeric_cols = [col for col in analysis_df.columns if pd.api.types.is_numeric_dtype(analysis_df[col])]
    categorical_cols = [col for col in analysis_df.columns if col not in numeric_cols]

    for col in analysis_df.columns:
        missing_count = int(analysis_df[col].isna().sum())
        missing_pct = (missing_count / max(len(analysis_df), 1)) * 100
        missingness[col] = {"count": missing_count, "pct": round(float(missing_pct), 4)}
        cardinality[col] = int(analysis_df[col].nunique(dropna=True))

    for col in categorical_cols:
        vc = analysis_df[col].astype("string").value_counts(dropna=True).head(10)
        top_values[col] = [
            {"value": str(idx), "count": int(count), "pct": round(float(count / max(len(analysis_df), 1) * 100), 4)}
            for idx, count in vc.items()
        ]

    for col in numeric_cols:
        series = pd.to_numeric(analysis_df[col], errors="coerce").dropna()
        if series.empty:
            continue
        desc = series.describe(percentiles=[0.25, 0.5, 0.75])
        numeric_stats[col] = {
            "mean": _safe_float(desc.get("mean")),
            "std": _safe_float(desc.get("std")),
            "min": _safe_float(desc.get("min")),
            "p25": _safe_float(desc.get("25%")),
            "p50": _safe_float(desc.get("50%")),
            "p75": _safe_float(desc.get("75%")),
            "max": _safe_float(desc.get("max")),
        }

        hist_counts, hist_edges = np.histogram(series, bins="auto")
        bins: list[dict[str, float | int]] = []
        for i, count in enumerate(hist_counts):
            bins.append(
                {
                    "left": float(hist_edges[i]),
                    "right": float(hist_edges[i + 1]),
                    "count": int(count),
                }
            )
        numeric_histograms[col] = bins

    total_missing = int(analysis_df.isna().sum().sum())
    total_cells = int(analysis_df.shape[0] * analysis_df.shape[1]) if analysis_df.shape[0] and analysis_df.shape[1] else 0
    global_metrics = {
        "rows": int(analysis_df.shape[0]),
        "columns": int(analysis_df.shape[1]),
        "missing_total": total_missing,
        "missing_pct_total": round(float((total_missing / total_cells) * 100), 4) if total_cells else 0.0,
        "categorical_columns": int(len(categorical_cols)),
        "numeric_columns": int(len(numeric_cols)),
    }

    warnings = dataframe_quality_warnings(analysis_df)
    if excluded_cols:
        warnings.append(
            {
                "code": "technical_columns_excluded",
                "severity": "info",
                "column": None,
                "message": f"Excluded technical columns from analytics: {', '.join(excluded_cols)}.",
                "suggestion": "This is expected for ID-like columns used only for joins/traceability.",
            }
        )

    return {
        "shape": (int(analysis_df.shape[0]), int(analysis_df.shape[1])),
        "columns": [str(c) for c in analysis_df.columns],
        "dtypes": dtypes,
        "missingness": missingness,
        "cardinality": cardinality,
        "top_values": top_values,
        "numeric_stats": numeric_stats,
        "numeric_histograms": numeric_histograms,
        "global_metrics": global_metrics,
        "warnings": warnings,
    }
