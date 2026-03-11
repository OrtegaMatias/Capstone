from __future__ import annotations

from app.etl.step_cast_types import CastTypesStep
from app.etl.step_clean_columns import CleanColumnsStep
from app.etl.step_merge import MergeStep
from app.etl.step_missing import MissingStep
from app.etl.step_read_csv import ReadCSVStep
from app.etl.types import PipelineContext, PipelineStep


class ETLPipeline:
    def __init__(self, steps: list[PipelineStep] | None = None):
        self.steps = steps or [
            ReadCSVStep(),
            CleanColumnsStep(),
            CastTypesStep(),
            MergeStep(),
            MissingStep(),
        ]

    def run(self, context: PipelineContext) -> PipelineContext:
        for step in self.steps:
            context = step.run(context)
        return context
