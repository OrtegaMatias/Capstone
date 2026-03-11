from __future__ import annotations

from typing import Any

from app.etl.types import PipelineContext, PipelineStep


class MissingStep(PipelineStep):
    name = "step_missing"

    def run(self, context: PipelineContext) -> PipelineContext:
        if context.merged_df is None:
            raise ValueError("No merged dataframe available for missing step")

        df = context.merged_df
        report: dict[str, dict[str, float]] = {}

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_pct = float((missing_count / max(len(df), 1)) * 100)
            report[col] = {"count": missing_count, "pct": round(missing_pct, 4)}
            if missing_count > 0:
                context.warnings.append(
                    {
                        "code": "missing_values_detected",
                        "severity": "warning",
                        "column": col,
                        "message": f"Column {col} has {missing_count} missing values ({missing_pct:.2f}%).",
                        "suggestion": "No imputation applied in Iteration 1. Decide strategy before modeling.",
                    }
                )

        context.add_step_metadata(self.name, {"missing_report": report})
        return context
