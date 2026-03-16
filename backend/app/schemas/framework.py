from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.academic_eda import AcademicContextSummary
from app.schemas.common import WarningItem
from app.schemas.supervised import AnovaResponse, BoxplotSeries, MultipleRegressionResponse, SupervisedOverviewResponse


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
    medae: float | None = None
    baseline_mae: float | None = None
    msle: float | None = None
    mape: float | None = None


class MlPredictionSample(BaseModel):
    row_id: int
    week: str
    actual: float
    predicted: float


class MlFeatureEffect(BaseModel):
    feature: str
    coefficient: float


class MlModelResult(BaseModel):
    model_name: str
    strategy_name: str | None = None
    strategy_label: str | None = None
    metrics: MlMetricSummary
    train_metrics: MlMetricSummary | None = None
    predictions: list[MlPredictionSample]
    feature_effects: list[MlFeatureEffect]
    tree_structure: dict[str, Any] | None = None
    notes: list[str] = []


class MlBenchmarkRow(BaseModel):
    model_name: str
    strategy_name: str
    strategy_label: str
    metrics: MlMetricSummary
    available: bool = True
    notes: list[str] = []


class MlSegmentRow(BaseModel):
    segment: str
    train_count: int
    test_count: int
    actual_mean: float | None = None
    actual_median: float | None = None
    regression_mae: float | None = None
    heuristic_mae: float | None = None
    baseline_mae: float | None = None


class MlSegmentReport(BaseModel):
    family_key: str
    family_label: str
    grouping_type: str
    rows: list[MlSegmentRow]


class MlTierUsage(BaseModel):
    source: str
    count: int


class HeuristicModelResult(BaseModel):
    model_name: str
    family_key: str
    family_label: str
    rule_summary: str
    train_metrics: MlMetricSummary
    metrics: MlMetricSummary
    predictions: list[MlPredictionSample]
    tier_usage: list[MlTierUsage]
    segment_medians: list[SegmentMedianEntry] = []


class StrategyComparisonEntry(BaseModel):
    model_name: str
    strategy_name: str | None = None
    strategy_label: str | None = None
    metrics: MlMetricSummary
    notes: list[str] = []


class StrategyComparison(BaseModel):
    winner: str
    mae_gap: float
    best_regression: StrategyComparisonEntry
    best_heuristic: StrategyComparisonEntry
    narrative: str


class SegmentMedianEntry(BaseModel):
    segment: str
    median: float
    count: int


class TargetTransformationStats(BaseModel):
    count: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None
    skew: float | None = None
    iqr: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    outlier_count: int
    outlier_ratio: float


class TargetTransformationStep(BaseModel):
    step_key: str
    step_label: str
    scale: str
    stats: TargetTransformationStats
    notes: list[str] = []


class TargetTransformationDiagnostics(BaseModel):
    scope: str
    boxplot_data: list[BoxplotSeries]
    steps: list[TargetTransformationStep]


class PriorityBand(BaseModel):
    label: str
    min_days: int
    max_days: int
    count_train: int
    count_test: int


class ClassificationPerClass(BaseModel):
    band: str
    precision: float
    recall: float
    f1: float
    support: int


class ClassificationPredictionPoint(BaseModel):
    row_id: int
    week: str
    actual_days: float
    actual_band: str
    predicted_band: str
    correct: bool
    adjacent_hit: bool
    predicted_confidence: float | None = None
    priority_score: float | None = None
    band_probabilities: dict[str, float] = {}


class ClassificationModelResult(BaseModel):
    model_name: str
    accuracy: float
    adjacent_accuracy: float
    band_mae: float
    f1_weighted: float
    per_class: list[ClassificationPerClass]
    confusion_matrix: list[list[int]]
    available: bool = True
    notes: list[str] = []


class MethodologyStep(BaseModel):
    step: int
    title: str
    rationale: str
    decision: str
    evidence: str | None = None


class PriorityClassificationResult(BaseModel):
    bands: list[PriorityBand]
    band_distribution: dict[str, dict[str, int]]
    models: list[ClassificationModelResult]
    best_model_predictions: list[ClassificationPredictionPoint] = []
    best_model: str
    baseline_accuracy: float
    narrative: str
    methodology: list[MethodologyStep] = []
    feature_names: list[str] = []
    priority_score_corr: float | None = None
    priority_score_stats: dict[str, float] = {}


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
    preprocessing_benchmarks: list[MlBenchmarkRow]
    segment_reports: list[MlSegmentReport]
    heuristic_models: list[HeuristicModelResult]
    strategy_comparison: StrategyComparison | None = None
    target_transformation_diagnostics: TargetTransformationDiagnostics | None = None
    priority_classification: PriorityClassificationResult | None = None
    supervised_overview: SupervisedOverviewResponse
    anova: AnovaResponse
    multiple_regression: MultipleRegressionResponse
    artifacts: list[WeekArtifact]
