from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetInput:
    filename: str
    content: bytes


@dataclass
class PipelineContext:
    in_input: DatasetInput | None = None
    out_input: DatasetInput | None = None
    in_df: pd.DataFrame | None = None
    out_df: pd.DataFrame | None = None
    merged_df: pd.DataFrame | None = None
    step_metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)

    def add_step_metadata(self, step_name: str, metadata: dict[str, Any]) -> None:
        self.step_metadata[step_name] = metadata


class PipelineStep:
    name = "base_step"

    def run(self, context: PipelineContext) -> PipelineContext:
        raise NotImplementedError
