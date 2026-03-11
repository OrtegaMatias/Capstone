from __future__ import annotations

import io
from typing import Any

import pandas as pd

from app.etl.types import PipelineContext, PipelineStep


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "latin-1"]
SEPARATOR_CANDIDATES = [",", ";", "\t", "|"]


def robust_read_csv(content: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    best_df: pd.DataFrame | None = None
    best_meta: dict[str, Any] | None = None
    best_score = -1

    for encoding in ENCODING_CANDIDATES:
        for sep in SEPARATOR_CANDIDATES:
            try:
                df = pd.read_csv(
                    io.BytesIO(content),
                    sep=sep,
                    encoding=encoding,
                    engine="python",
                    on_bad_lines="skip",
                    dtype=str,
                )
            except Exception:
                continue

            if df.empty and len(df.columns) <= 1:
                continue

            unnamed_count = sum(str(c).lower().startswith("unnamed") for c in df.columns)
            score = len(df.columns) * 10 - unnamed_count
            if score > best_score:
                best_df = df
                best_meta = {
                    "filename": filename,
                    "encoding": encoding,
                    "separator": sep,
                    "rows": int(df.shape[0]),
                    "cols": int(df.shape[1]),
                }
                best_score = score

    if best_df is None or best_meta is None:
        raise ValueError(f"Could not parse CSV file '{filename}' with robust reader")

    return best_df, best_meta


class ReadCSVStep(PipelineStep):
    name = "step_read_csv"

    def run(self, context: PipelineContext) -> PipelineContext:
        metadata: dict[str, Any] = {}

        if context.in_input is not None:
            in_df, in_meta = robust_read_csv(context.in_input.content, context.in_input.filename)
            context.in_df = in_df
            metadata["in_file"] = in_meta

        if context.out_input is not None:
            out_df, out_meta = robust_read_csv(context.out_input.content, context.out_input.filename)
            context.out_df = out_df
            metadata["out_file"] = out_meta

        context.add_step_metadata(self.name, metadata)
        return context
