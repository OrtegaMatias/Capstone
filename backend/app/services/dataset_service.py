from __future__ import annotations

from typing import Any
from uuid import uuid4

import pandas as pd
from fastapi import HTTPException, UploadFile

from app.core.config import get_settings
from app.etl.pipeline import ETLPipeline
from app.etl.step_cast_types import _cast_single_dataframe
from app.etl.step_clean_columns import _clean_dataframe
from app.etl.step_read_csv import robust_read_csv
from app.etl.types import DatasetInput, PipelineContext
from app.stats.columns import is_technical_id_column
from app.stats.eda import compute_eda
from app.stats.pivot import PivotRequest, build_pivot_metadata, run_pivot_query as compute_pivot_query
from app.stats.supervised import compute_anova, compute_multiple_regression_out, compute_supervised_overview
from app.stats.variability import compute_variability_scores
from app.storage.file_store import DatasetFileStore
from app.storage.memory_cache import CachedDataset, DatasetMemoryCache


EXPECTED_IN_COLUMNS = ["Unnamed: 0", "Condition", "Owner", "Size", "Type", "Quality", "week"]
EXPECTED_OUT_COLUMNS = ["Unnamed: 0", "Owner", "Size", "Type", "Quality", "DaysInDeposit", "week"]


class DatasetService:
    def __init__(self):
        settings = get_settings()
        self.file_store = DatasetFileStore(settings.datasets_dir)
        self.cache = DatasetMemoryCache(settings.cache_max_items)
        self.pipeline = ETLPipeline()

    async def upload_dataset(self, in_file: UploadFile | None, out_file: UploadFile | None) -> dict[str, Any]:
        if in_file is None and out_file is None:
            raise HTTPException(status_code=400, detail="At least one file must be uploaded")

        in_input = None
        out_input = None
        in_content = None
        out_content = None

        if in_file is not None:
            in_content = await in_file.read()
            in_input = DatasetInput(filename=in_file.filename or "in_file.csv", content=in_content)

        if out_file is not None:
            out_content = await out_file.read()
            out_input = DatasetInput(filename=out_file.filename or "out_file.csv", content=out_content)

        context = PipelineContext(in_input=in_input, out_input=out_input)

        try:
            context = self.pipeline.run(context)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Upload/ETL failed: {exc}") from exc

        if context.merged_df is None:
            raise HTTPException(status_code=400, detail="No dataset could be produced from the provided files")

        dataset_id = str(uuid4())
        df = context.merged_df

        schema_detected = {
            "in_file": [str(c) for c in context.in_df.columns] if context.in_df is not None else [],
            "out_file": [str(c) for c in context.out_df.columns] if context.out_df is not None else [],
            "canonical": [str(c) for c in df.columns],
        }

        warnings = context.warnings + self._schema_warnings(schema_detected)

        metadata = {
            "dataset_id": dataset_id,
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [str(c) for c in df.columns],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "has_target": "DaysInDeposit" in df.columns,
            "schema": schema_detected,
            "warnings": warnings,
            "step_metadata": context.step_metadata,
        }

        self.file_store.save_dataset(
            dataset_id=dataset_id,
            canonical_df=df,
            metadata=metadata,
            in_bytes=in_content,
            out_bytes=out_content,
        )

        self.cache.set(CachedDataset(dataset_id=dataset_id, dataframe=df, metadata=metadata))

        return {
            "dataset_id": dataset_id,
            "has_in": in_file is not None,
            "has_out": out_file is not None,
            "has_target": bool(metadata["has_target"]),
            "schema": schema_detected,
            "preview": self._preview_rows(df, limit=20),
            "dtype_summary": metadata["dtypes"],
            "warnings": warnings,
        }

    def _schema_warnings(self, schema: dict[str, list[str]]) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []

        in_columns = set(schema.get("in_file", []))
        out_columns = set(schema.get("out_file", []))

        if in_columns:
            missing_in = [col for col in EXPECTED_IN_COLUMNS if col not in in_columns]
            if missing_in:
                warnings.append(
                    {
                        "code": "missing_expected_columns_in",
                        "severity": "warning",
                        "column": None,
                        "message": f"in_file missing expected columns: {', '.join(missing_in)}",
                        "suggestion": "Verify input schema or provide mapping in future iterations.",
                    }
                )

        if out_columns:
            missing_out = [col for col in EXPECTED_OUT_COLUMNS if col not in out_columns]
            if missing_out:
                warnings.append(
                    {
                        "code": "missing_expected_columns_out",
                        "severity": "warning",
                        "column": None,
                        "message": f"out_file missing expected columns: {', '.join(missing_out)}",
                        "suggestion": "Verify output schema or provide mapping in future iterations.",
                    }
                )

        return warnings

    def _load_dataset(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        cached = self.cache.get(dataset_id)
        if cached is not None:
            return cached.dataframe.copy(), cached.metadata

        try:
            df, metadata = self.file_store.load_dataset(dataset_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_id} not found") from exc

        self.cache.set(CachedDataset(dataset_id=dataset_id, dataframe=df, metadata=metadata))
        return df.copy(), metadata

    @staticmethod
    def _preview_rows(df: pd.DataFrame, limit: int = 20) -> list[dict[str, Any]]:
        preview_df = df.head(limit).copy()
        preview_df = preview_df.where(pd.notna(preview_df), None)
        return preview_df.to_dict(orient="records")

    def get_preview(self, dataset_id: str, limit: int = 20) -> dict[str, Any]:
        df, _ = self._load_dataset(dataset_id)
        return {
            "columns": [str(c) for c in df.columns],
            "rows": self._preview_rows(df, limit=limit),
            "total_rows": int(df.shape[0]),
        }

    def get_eda(self, dataset_id: str) -> dict[str, Any]:
        df, metadata = self._load_dataset(dataset_id)
        eda_payload = compute_eda(df)
        merged_warnings = metadata.get("warnings", []) + eda_payload.get("warnings", [])
        deduped = self._dedupe_warnings(merged_warnings)
        eda_payload["warnings"] = self._normalize_eda_warnings(deduped, df)
        return eda_payload

    def get_variability(self, dataset_id: str, custom_mode: str, ordinal_strategy: str) -> dict[str, Any]:
        df, _ = self._load_dataset(dataset_id)
        return compute_variability_scores(df, custom_mode=custom_mode, ordinal_strategy=ordinal_strategy)

    def get_supervised_overview(self, dataset_id: str) -> dict[str, Any]:
        df, _ = self._load_dataset(dataset_id)
        return compute_supervised_overview(df)

    def get_anova(self, dataset_id: str) -> dict[str, Any]:
        df, _ = self._load_dataset(dataset_id)
        return compute_anova(df)

    def get_multiple_regression(self, dataset_id: str) -> dict[str, Any]:
        # Validate dataset exists first.
        self._load_dataset(dataset_id)
        try:
            out_df = self._load_source_dataframe(dataset_id, "out")
        except ValueError:
            return {
                "source": "out",
                "target_present": False,
                "model_built": False,
                "formula": None,
                "n_obs": 0,
                "n_features": 0,
                "r_squared": None,
                "adj_r_squared": None,
                "f_statistic": None,
                "f_p_value": None,
                "aic": None,
                "bic": None,
                "coefficients": [],
                "anova_rows": [],
                "conclusions": [],
                "warnings": [
                    {
                        "code": "missing_out_source",
                        "severity": "warning",
                        "column": None,
                        "message": "OUT source is not available for this dataset.",
                        "suggestion": "Upload Grupo1_out.csv to enable multiple regression for DaysInDeposit.",
                    }
                ],
            }
        return compute_multiple_regression_out(out_df)

    def _load_source_dataframe(self, dataset_id: str, source: str) -> pd.DataFrame:
        if source not in {"in", "out"}:
            raise ValueError("source must be 'in' or 'out'")

        if not self.file_store.source_exists(dataset_id, source):
            raise ValueError(f"source '{source}' is not available for dataset {dataset_id}")

        raw_bytes = self.file_store.read_source_bytes(dataset_id, source)
        raw_df, _ = robust_read_csv(raw_bytes, filename=f"raw_{source}.csv")
        clean_df, _, _ = _clean_dataframe(raw_df, source=f"{source}_file")
        cast_df, _, _ = _cast_single_dataframe(clean_df, source=f"{source}_file")
        return cast_df

    def get_pivot_sources(self, dataset_id: str) -> dict[str, Any]:
        # Validate dataset exists.
        self._load_dataset(dataset_id)

        return {
            "sources": [
                {"source": "in", "available": self.file_store.source_exists(dataset_id, "in")},
                {"source": "out", "available": self.file_store.source_exists(dataset_id, "out")},
            ]
        }

    def get_pivot_metadata(self, dataset_id: str, source: str) -> dict[str, Any]:
        df = self._load_source_dataframe(dataset_id, source)
        return build_pivot_metadata(source=source, df=df)

    def run_pivot_query(self, dataset_id: str, request_payload: dict[str, Any]) -> dict[str, Any]:
        request = PivotRequest(
            source=request_payload["source"],
            row_dim=request_payload["row_dim"],
            col_dim=request_payload["col_dim"],
            value_field=request_payload["value_field"],
            agg_func=request_payload["agg_func"],
            filters=request_payload.get("filters", {}),
            include_blank=bool(request_payload.get("include_blank", True)),
            top_k=int(request_payload.get("top_k", 10)),
            small_n_threshold=int(request_payload.get("small_n_threshold", 5)),
        )

        df = self._load_source_dataframe(dataset_id, request.source)
        return compute_pivot_query(df=df, request=request)

    @staticmethod
    def _dedupe_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[tuple[Any, ...]] = set()
        deduped: list[dict[str, Any]] = []
        for warning in warnings:
            key = (
                warning.get("code"),
                warning.get("column"),
                warning.get("message"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(warning)
        return deduped

    @staticmethod
    def _normalize_eda_warnings(warnings: list[dict[str, Any]], df: pd.DataFrame) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        overlap_columns: list[str] = []
        has_week_constant = any(w.get("code") == "week_constant" for w in warnings)

        for warning in warnings:
            code = warning.get("code")
            column = warning.get("column")

            if isinstance(column, str) and is_technical_id_column(column):
                continue

            if code == "overlap_column_conflict":
                if isinstance(column, str):
                    overlap_columns.append(column)
                continue

            if has_week_constant and column == "week" and code in {"constant_column", "near_constant_column"}:
                continue

            if code == "high_cardinality" and isinstance(column, str) and column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    continue

            cleaned.append(warning)

        if overlap_columns:
            unique_cols = sorted(set(overlap_columns))
            cleaned.append(
                {
                    "code": "overlap_column_conflict",
                    "severity": "info",
                    "column": None,
                    "message": (
                        "Columns differ between in/out and were kept with suffixes _in/_out: "
                        f"{', '.join(unique_cols)}."
                    ),
                    "suggestion": (
                        "This is expected when IN/OUT represent different stages. "
                        "Analyze *_in and *_out separately."
                    ),
                }
            )

        return cleaned
