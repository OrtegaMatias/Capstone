from __future__ import annotations

from typing import Any

import pandas as pd

from app.etl.types import PipelineContext, PipelineStep


OVERLAP_COLUMNS = ["Owner", "Size", "Type", "Quality", "week"]
MERGE_KEY = "Unnamed: 0"


def _series_equal(left: pd.Series, right: pd.Series) -> bool:
    left_cmp = left.astype("string").fillna("<NA>")
    right_cmp = right.astype("string").fillna("<NA>")
    return left_cmp.equals(right_cmp)


class MergeStep(PipelineStep):
    name = "step_merge"

    def run(self, context: PipelineContext) -> PipelineContext:
        metadata: dict[str, Any] = {}

        if context.in_df is None and context.out_df is None:
            raise ValueError("At least one dataframe should be available before merge step")

        if context.in_df is not None and context.out_df is not None:
            in_df = context.in_df.copy()
            out_df = context.out_df.copy()

            if MERGE_KEY not in in_df.columns or MERGE_KEY not in out_df.columns:
                raise ValueError("Merge key Unnamed: 0 not available in both files")

            merged = in_df.merge(out_df, on=MERGE_KEY, how="inner", suffixes=("_in", "_out"))
            conflict_columns: list[str] = []

            for col in OVERLAP_COLUMNS:
                in_col = f"{col}_in"
                out_col = f"{col}_out"
                if in_col in merged.columns and out_col in merged.columns:
                    if _series_equal(merged[in_col], merged[out_col]):
                        merged[col] = merged[in_col]
                        merged = merged.drop(columns=[in_col, out_col])
                    else:
                        conflict_columns.append(col)

            if conflict_columns:
                context.warnings.append(
                    {
                        "code": "overlap_column_conflict",
                        "severity": "info",
                        "column": None,
                        "message": (
                            "Columns differ between in/out and were kept with suffixes _in/_out: "
                            f"{', '.join(conflict_columns)}."
                        ),
                        "suggestion": (
                            "This is expected when IN/OUT represent different stages. "
                            "Analyze *_in and *_out separately."
                        ),
                    }
                )

            in_unmatched = int(len(in_df) - in_df[MERGE_KEY].isin(merged[MERGE_KEY]).sum())
            out_unmatched = int(len(out_df) - out_df[MERGE_KEY].isin(merged[MERGE_KEY]).sum())

            metadata = {
                "mode": "merged",
                "merge_key": MERGE_KEY,
                "input_in_rows": int(len(in_df)),
                "input_out_rows": int(len(out_df)),
                "merged_rows": int(len(merged)),
                "unmatched_in_rows": in_unmatched,
                "unmatched_out_rows": out_unmatched,
            }
            context.merged_df = merged
        else:
            passthrough = context.in_df if context.in_df is not None else context.out_df
            context.merged_df = passthrough.copy() if passthrough is not None else None
            metadata = {
                "mode": "passthrough",
                "rows": int(context.merged_df.shape[0]) if context.merged_df is not None else 0,
                "columns": int(context.merged_df.shape[1]) if context.merged_df is not None else 0,
            }

        context.add_step_metadata(self.name, metadata)
        return context
