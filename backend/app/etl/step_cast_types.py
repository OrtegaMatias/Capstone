from __future__ import annotations

from typing import Any

import pandas as pd

from app.etl.types import PipelineContext, PipelineStep


CATEGORICAL_COLUMNS = {"Owner", "Condition", "Type", "Quality"}


def _normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            df[col] = df[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return df


def _size_should_be_categorical(size_numeric: pd.Series, total_rows: int) -> bool:
    """Treat Size as categorical when it behaves like a discrete class code."""
    clean = size_numeric.dropna()
    if clean.empty:
        return True

    nunique = int(clean.nunique(dropna=True))
    unique_ratio = float(nunique / max(total_rows, 1))
    all_integer_like = bool(((clean % 1).abs() < 1e-9).all())
    value_set = {float(v) for v in clean.unique().tolist()}
    common_container_codes = {1.0, 2.0, 20.0, 40.0, 45.0}

    if value_set.issubset(common_container_codes):
        return True

    # Very low-cardinality integer features usually encode classes, not linear magnitude.
    if all_integer_like and nunique <= 12 and unique_ratio <= 0.1:
        return True

    return False


def _cast_single_dataframe(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    semantic_types: dict[str, str] = {}
    df = _normalize_strings(df.copy())

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string")
            semantic_types[col] = "categorical"

    if "Size" in df.columns:
        size_numeric = pd.to_numeric(df["Size"], errors="coerce")
        ratio = float(size_numeric.notna().mean())
        if ratio >= 0.9:
            if _size_should_be_categorical(size_numeric, total_rows=len(df)):
                # Keep canonical integer-like label and analyze as categorical (container class).
                normalized = size_numeric.map(lambda x: int(x) if pd.notna(x) else pd.NA)
                df["Size"] = normalized.astype("string")
                semantic_types["Size"] = "categorical_discrete"
                warnings.append(
                    {
                        "code": "size_casted_to_categorical_discrete",
                        "severity": "info",
                        "column": "Size",
                        "message": f"Size in {source} has low-cardinality discrete values; treated as categorical.",
                        "suggestion": "Use class-wise analysis (ANOVA/boxplots). Use numeric Size only if continuous.",
                    }
                )
            else:
                df["Size"] = size_numeric
                semantic_types["Size"] = "numeric"
        else:
            df["Size"] = df["Size"].astype("string")
            semantic_types["Size"] = "categorical"
            warnings.append(
                {
                    "code": "size_casted_to_categorical",
                    "severity": "warning",
                    "column": "Size",
                    "message": f"Size in {source} was not >=90% numeric parseable; treated as categorical.",
                    "suggestion": "Clean Size values or standardize units for numeric modeling.",
                }
            )

    if "DaysInDeposit" in df.columns:
        df["DaysInDeposit"] = pd.to_numeric(df["DaysInDeposit"], errors="coerce")
        semantic_types["DaysInDeposit"] = "numeric_target"

    if "week" in df.columns:
        week_numeric = pd.to_numeric(df["week"], errors="coerce")
        parse_ratio = float(week_numeric.notna().mean())
        if parse_ratio >= 0.9:
            df["week"] = week_numeric
            unique_ratio = float(week_numeric.nunique(dropna=True) / max(len(df), 1))
            if week_numeric.nunique(dropna=True) <= 53 and unique_ratio < 0.1:
                semantic_types["week"] = "discrete_numeric"
            else:
                semantic_types["week"] = "numeric"
        else:
            df["week"] = df["week"].astype("string")
            semantic_types["week"] = "categorical"

    for col in df.columns:
        if col not in semantic_types:
            if pd.api.types.is_numeric_dtype(df[col]):
                semantic_types[col] = "numeric"
            else:
                semantic_types[col] = "categorical"

    return df, {"source": source, "semantic_types": semantic_types}, warnings


class CastTypesStep(PipelineStep):
    name = "step_cast_types"

    def run(self, context: PipelineContext) -> PipelineContext:
        metadata: dict[str, Any] = {}

        if context.in_df is not None:
            in_df, in_meta, warnings = _cast_single_dataframe(context.in_df, "in_file")
            context.in_df = in_df
            metadata["in_file"] = in_meta
            context.warnings.extend(warnings)

        if context.out_df is not None:
            out_df, out_meta, warnings = _cast_single_dataframe(context.out_df, "out_file")
            context.out_df = out_df
            metadata["out_file"] = out_meta
            context.warnings.extend(warnings)

        context.add_step_metadata(self.name, metadata)
        return context
