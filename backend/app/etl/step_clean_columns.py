from __future__ import annotations

import re
from typing import Any

import pandas as pd

from app.etl.types import PipelineContext, PipelineStep


INDEX_CANDIDATES = {"unnamed: 0", "index", "id", "record_id", "row_id"}


def _normalize_column_name(name: str) -> str:
    name = name.replace("\ufeff", "").strip()
    name = re.sub(r"\s+", " ", name)
    return name


def _find_index_column(df: pd.DataFrame) -> str | None:
    for col in df.columns:
        if _normalize_column_name(str(col)).lower() in INDEX_CANDIDATES:
            return col

    for col in df.columns:
        series = df[col]
        non_na = series.dropna()
        if non_na.empty:
            continue
        is_unique = non_na.nunique(dropna=True) == len(non_na)
        uniqueness_ratio = non_na.nunique(dropna=True) / max(len(df), 1)
        if is_unique and uniqueness_ratio > 0.95:
            return col

    return None


def _clean_dataframe(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    original_columns = list(df.columns)

    rename_map = {col: _normalize_column_name(str(col)) for col in df.columns}
    df = df.rename(columns=rename_map)

    empty_columns = [col for col in df.columns if df[col].isna().all()]
    if empty_columns:
        df = df.drop(columns=empty_columns)

    index_col = _find_index_column(df)
    if index_col is None:
        df["Unnamed: 0"] = range(len(df))
        warnings.append(
            {
                "code": "index_inferred",
                "severity": "warning",
                "column": "Unnamed: 0",
                "message": f"No index-like column found in {source}; generated synthetic Unnamed: 0",
                "suggestion": "Provide stable unique id column for robust merging.",
            }
        )
    elif index_col != "Unnamed: 0":
        df = df.rename(columns={index_col: "Unnamed: 0"})

    return (
        df,
        {
            "source": source,
            "original_columns": [str(c) for c in original_columns],
            "clean_columns": [str(c) for c in df.columns],
            "dropped_empty_columns": empty_columns,
        },
        warnings,
    )


class CleanColumnsStep(PipelineStep):
    name = "step_clean_columns"

    def run(self, context: PipelineContext) -> PipelineContext:
        metadata: dict[str, Any] = {}

        if context.in_df is not None:
            clean_df, clean_meta, warnings = _clean_dataframe(context.in_df, "in_file")
            context.in_df = clean_df
            metadata["in_file"] = clean_meta
            context.warnings.extend(warnings)

        if context.out_df is not None:
            clean_df, clean_meta, warnings = _clean_dataframe(context.out_df, "out_file")
            context.out_df = clean_df
            metadata["out_file"] = clean_meta
            context.warnings.extend(warnings)

        context.add_step_metadata(self.name, metadata)
        return context
