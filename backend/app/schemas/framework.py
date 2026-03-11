from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from app.schemas.academic_eda import AcademicContextSummary
from app.schemas.common import WarningItem
from app.schemas.supervised import AnovaResponse, MultipleRegressionResponse, SupervisedOverviewResponse


WeekStatus = Literal["active", "scaffolded"]


class WeekSeedPaths(BaseModel):
    in_file: str | None = None
    out_file: str | None = None


class WeekArtifact(BaseModel):
    kind: str
    label: str
    path: str
    available: bool


class WeekSummary(BaseModel):
    week_id: str
    week_number: int
    title: str
    stage_name: str
    status: WeekStatus
    summary: str
    seed_paths: WeekSeedPaths
    analysis_available: list[str]
    artifacts: list[WeekArtifact]
    notes_updated_at: str | None = None
    report_available: bool = False


class WeekConfig(WeekSummary):
    description: str
    expected_inputs: list[str]
    checklist: list[str]
    deliverables: list[str]
    academic_context: AcademicContextSummary | None = None


class FrameworkSummary(BaseModel):
    framework_name: str
    summary: str
    workspace_root: str
    generated_at: str
    weeks: list[WeekSummary]


class WeekReportSummary(BaseModel):
    week_id: str
    stage_name: str
    markdown_content: str
    html_content: str
    updated_at: str | None = None
    artifacts: list[WeekArtifact]


class MlSplitSummary(BaseModel):
    train_weeks: list[str]
    test_weeks: list[str]
    train_rows: int
    test_rows: int


class MlMetricSummary(BaseModel):
    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None
    baseline_mae: float | None = None


class MlPredictionSample(BaseModel):
    week: str
    actual: float
    predicted: float


class MlFeatureEffect(BaseModel):
    feature: str
    coefficient: float


class MlModelResult(BaseModel):
    model_name: str
    metrics: MlMetricSummary
    predictions: list[MlPredictionSample]
    feature_effects: list[MlFeatureEffect]


class MlEvaluationSummary(BaseModel):
    week_id: str
    stage_name: str
    target_present: bool
    model_built: bool
    target_column: str
    split: MlSplitSummary
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    models: list[MlModelResult]
    warnings: list[WarningItem]
    supervised_overview: SupervisedOverviewResponse
    anova: AnovaResponse
    multiple_regression: MultipleRegressionResponse
    artifacts: list[WeekArtifact]
