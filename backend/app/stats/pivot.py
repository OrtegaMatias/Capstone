from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from app.stats.columns import analytical_columns, is_technical_id_column
from app.stats.warnings import dataframe_quality_warnings


BLANK_LABEL = "(en blanco)"
OTHER_LABEL = "Other"
AGG_FUNCTIONS_BASE = ["count", "sum", "mean", "median"]
RATE_THRESHOLDS: dict[str, int] = {
    "rate_gt_7": 7,
    "rate_gt_14": 14,
    "rate_gt_30": 30,
}
MAX_FILTER_OPTIONS_PER_DIM = 200


@dataclass
class PivotRequest:
    source: str
    row_dim: str
    col_dim: str
    value_field: str
    agg_func: str
    filters: dict[str, list[str]]
    include_blank: bool
    top_k: int
    small_n_threshold: int


def _label_value(value: Any) -> str:
    if pd.isna(value):
        return BLANK_LABEL

    if isinstance(value, str):
        normalized = value.strip()
        return BLANK_LABEL if normalized == "" else normalized

    if isinstance(value, (np.integer, int)):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return BLANK_LABEL
        if float(value).is_integer():
            return str(int(value))
        return str(float(value))

    return str(value)


def _series_to_labels(series: pd.Series) -> pd.Series:
    return series.map(_label_value)


def _sort_keys(keys: list[str]) -> list[str]:
    def as_num(label: str) -> tuple[int, float]:
        try:
            return (0, float(label))
        except Exception:
            return (1, float("inf"))

    specials = [label for label in [OTHER_LABEL, BLANK_LABEL] if label in keys]
    regular = [k for k in keys if k not in specials]
    regular_sorted = sorted(regular, key=lambda k: (as_num(k), k))
    return regular_sorted + [s for s in [OTHER_LABEL, BLANK_LABEL] if s in specials]


def _apply_top_k(labels: pd.Series, top_k: int) -> tuple[pd.Series, bool]:
    if top_k <= 0:
        return labels, False

    non_blank = labels[labels != BLANK_LABEL]
    distinct = int(non_blank.nunique(dropna=True))
    if distinct <= top_k:
        return labels, False

    keep = set(non_blank.value_counts().head(top_k).index.tolist())
    grouped = labels.map(lambda x: x if x == BLANK_LABEL or x in keep else OTHER_LABEL)
    return grouped, True


def _numeric_series(df: pd.DataFrame, value_field: str) -> pd.Series:
    return pd.to_numeric(df[value_field], errors="coerce")


def _is_numeric_eligible(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return bool(series.notna().any())
    parsed = pd.to_numeric(series, errors="coerce")
    return bool(parsed.notna().any())


def _validate_request(df: pd.DataFrame, request: PivotRequest) -> None:
    for col in [request.row_dim, request.col_dim, request.value_field]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in source '{request.source}'")
        if is_technical_id_column(col):
            raise ValueError(f"Column '{col}' is technical/id-like and excluded from pivot analytics")

    agg_supported = AGG_FUNCTIONS_BASE + list(RATE_THRESHOLDS.keys())
    if request.agg_func not in agg_supported:
        raise ValueError(f"agg_func '{request.agg_func}' is not supported")

    if request.agg_func in RATE_THRESHOLDS and request.value_field != "DaysInDeposit":
        raise ValueError("rate_gt_* aggregations require value_field='DaysInDeposit'")

    if request.agg_func in {"sum", "mean", "median"}:
        series = pd.to_numeric(df[request.value_field], errors="coerce")
        if not _is_numeric_eligible(series):
            raise ValueError(
                f"Aggregation '{request.agg_func}' requires numeric value_field. "
                f"Column '{request.value_field}' is not numeric in the current source."
            )


def _apply_filters(df: pd.DataFrame, filters: dict[str, list[str]]) -> pd.DataFrame:
    if not filters:
        return df

    filtered = df.copy()
    mask = pd.Series(True, index=filtered.index)

    for column, accepted_values in filters.items():
        if column not in filtered.columns:
            raise ValueError(f"Filter column '{column}' not found")
        labels = _series_to_labels(filtered[column])
        mask &= labels.isin(accepted_values)

    return filtered[mask].copy()


def _aggregate_value(df_subset: pd.DataFrame, value_field: str, agg_func: str) -> tuple[float | int | None, int]:
    row_count = int(len(df_subset))

    if row_count == 0:
        return None, 0

    if agg_func == "count":
        return row_count, row_count

    numeric = pd.to_numeric(df_subset[value_field], errors="coerce")
    valid = numeric.dropna()
    valid_count = int(valid.shape[0])

    if valid_count == 0:
        return None, row_count

    if agg_func == "sum":
        return float(valid.sum()), row_count

    if agg_func == "mean":
        return float(valid.mean()), row_count

    if agg_func == "median":
        return float(valid.median()), row_count

    if agg_func in RATE_THRESHOLDS:
        threshold = RATE_THRESHOLDS[agg_func]
        return float((valid > threshold).mean() * 100.0), row_count

    raise ValueError(f"Unsupported agg_func '{agg_func}'")


def pivot_agg_functions_for_source(df: pd.DataFrame) -> list[str]:
    agg = list(AGG_FUNCTIONS_BASE)
    if "DaysInDeposit" in df.columns:
        agg.extend(RATE_THRESHOLDS.keys())
    return agg


def _field_agg_functions(df: pd.DataFrame, columns: list[str]) -> dict[str, list[str]]:
    field_map: dict[str, list[str]] = {}
    for column in columns:
        allowed = ["count"]
        if _is_numeric_eligible(df[column]):
            allowed.extend(["sum", "mean", "median"])
        if str(column) == "DaysInDeposit":
            allowed.extend(list(RATE_THRESHOLDS.keys()))
        # preserve insertion order and avoid duplicates
        deduped = list(dict.fromkeys(allowed))
        field_map[str(column)] = deduped
    return field_map


def build_pivot_metadata(source: str, df: pd.DataFrame) -> dict[str, Any]:
    analysis_cols = analytical_columns(df)
    if not analysis_cols:
        raise ValueError("No eligible analytical columns found after excluding technical/id columns")

    dimensions = analysis_cols
    value_fields = analysis_cols
    agg_functions = pivot_agg_functions_for_source(df)
    field_agg_functions = _field_agg_functions(df, value_fields)
    filter_options: dict[str, list[str]] = {}

    if source == "in":
        default_row = "Owner" if "Owner" in df.columns else dimensions[0]
        default_col = "Quality" if "Quality" in df.columns else dimensions[min(1, len(dimensions) - 1)]
        default_value = "Size" if "Size" in df.columns else value_fields[0]
        default_agg = "sum" if "sum" in field_agg_functions.get(default_value, []) else field_agg_functions[default_value][0]
    else:
        default_row = "Owner" if "Owner" in df.columns else dimensions[0]
        default_col = "Size" if "Size" in df.columns else dimensions[min(1, len(dimensions) - 1)]
        default_value = "DaysInDeposit" if "DaysInDeposit" in df.columns else value_fields[0]
        default_agg = "mean" if "mean" in field_agg_functions.get(default_value, []) else field_agg_functions[default_value][0]

    warnings = [
        warning
        for warning in dataframe_quality_warnings(df)
        if warning.get("code") in {"week_constant", "high_cardinality", "constant_column"}
    ]

    for dimension in dimensions:
        values = _series_to_labels(df[dimension]).dropna().astype(str).unique().tolist()
        values_sorted = _sort_keys(values)
        if len(values_sorted) > MAX_FILTER_OPTIONS_PER_DIM:
            warnings.append(
                {
                    "code": "filter_options_truncated",
                    "severity": "warning",
                    "column": dimension,
                    "message": (
                        f"Filter options for '{dimension}' were truncated to "
                        f"{MAX_FILTER_OPTIONS_PER_DIM} values."
                    ),
                    "suggestion": "Use row/column dimensions plus top_k to refine high-cardinality fields.",
                }
            )
            values_sorted = values_sorted[:MAX_FILTER_OPTIONS_PER_DIM]
        filter_options[dimension] = values_sorted

    return {
        "source": source,
        "dimensions": dimensions,
        "value_fields": value_fields,
        "agg_functions": agg_functions,
        "field_agg_functions": field_agg_functions,
        "filter_options": filter_options,
        "defaults": {
            "row_dim": default_row,
            "col_dim": default_col,
            "value_field": default_value,
            "agg_func": default_agg,
            "include_blank": True,
            "top_k": 10,
            "small_n_threshold": 5,
        },
        "warnings": warnings,
    }


def run_pivot_query(df: pd.DataFrame, request: PivotRequest) -> dict[str, Any]:
    _validate_request(df, request)

    warnings: list[dict[str, Any]] = []

    working = _apply_filters(df, request.filters)

    if working.empty:
        return {
            "source": request.source,
            "row_dim": request.row_dim,
            "col_dim": request.col_dim,
            "value_field": request.value_field,
            "agg_func": request.agg_func,
            "matrix": {
                "columns": [],
                "rows": [],
                "column_totals": [],
                "grand_total": {"value": None, "count": 0},
            },
            "warnings": warnings,
        }

    row_labels = _series_to_labels(working[request.row_dim])
    col_labels = _series_to_labels(working[request.col_dim])

    if not request.include_blank:
        mask = (row_labels != BLANK_LABEL) & (col_labels != BLANK_LABEL)
        working = working[mask].copy()
        row_labels = row_labels[mask]
        col_labels = col_labels[mask]

    if working.empty:
        return {
            "source": request.source,
            "row_dim": request.row_dim,
            "col_dim": request.col_dim,
            "value_field": request.value_field,
            "agg_func": request.agg_func,
            "matrix": {
                "columns": [],
                "rows": [],
                "column_totals": [],
                "grand_total": {"value": None, "count": 0},
            },
            "warnings": warnings,
        }

    row_labels, row_grouped = _apply_top_k(row_labels, request.top_k)
    col_labels, col_grouped = _apply_top_k(col_labels, request.top_k)

    if row_grouped:
        warnings.append(
            {
                "code": "top_k_applied",
                "severity": "info",
                "column": request.row_dim,
                "message": f"Applied Top-{request.top_k} grouping on row dimension '{request.row_dim}'.",
                "suggestion": "Increase top_k to show more categories.",
            }
        )

    if col_grouped:
        warnings.append(
            {
                "code": "top_k_applied",
                "severity": "info",
                "column": request.col_dim,
                "message": f"Applied Top-{request.top_k} grouping on column dimension '{request.col_dim}'.",
                "suggestion": "Increase top_k to show more categories.",
            }
        )

    working = working.copy()
    working["__row_key"] = row_labels.values
    working["__col_key"] = col_labels.values

    row_keys = _sort_keys(working["__row_key"].astype(str).unique().tolist())
    col_keys = _sort_keys(working["__col_key"].astype(str).unique().tolist())

    grouped = {
        (str(row_key), str(col_key)): cell_df
        for (row_key, col_key), cell_df in working.groupby(["__row_key", "__col_key"], dropna=False)
    }

    rows_payload: list[dict[str, Any]] = []
    for row_key in row_keys:
        row_cells: list[dict[str, Any]] = []

        row_subset = working[working["__row_key"] == row_key]
        row_total_value, row_total_count = _aggregate_value(row_subset, request.value_field, request.agg_func)

        for col_key in col_keys:
            subset = grouped.get((row_key, col_key))
            if subset is None:
                value = None
                count = 0
            else:
                value, count = _aggregate_value(subset, request.value_field, request.agg_func)

            row_cells.append(
                {
                    "col_key": col_key,
                    "value": value,
                    "count": count,
                    "low_sample": count < request.small_n_threshold,
                }
            )

        rows_payload.append(
            {
                "row_key": row_key,
                "cells": row_cells,
                "row_total": {
                    "value": row_total_value,
                    "count": row_total_count,
                },
            }
        )

    col_totals: list[dict[str, Any]] = []
    for col_key in col_keys:
        subset = working[working["__col_key"] == col_key]
        total_value, total_count = _aggregate_value(subset, request.value_field, request.agg_func)
        col_totals.append(
            {
                "col_key": col_key,
                "value": total_value,
                "count": total_count,
            }
        )

    grand_value, grand_count = _aggregate_value(working, request.value_field, request.agg_func)

    return {
        "source": request.source,
        "row_dim": request.row_dim,
        "col_dim": request.col_dim,
        "value_field": request.value_field,
        "agg_func": request.agg_func,
        "matrix": {
            "columns": col_keys,
            "rows": rows_payload,
            "column_totals": col_totals,
            "grand_total": {
                "value": grand_value,
                "count": grand_count,
            },
        },
        "warnings": warnings,
    }
