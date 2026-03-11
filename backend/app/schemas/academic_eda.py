from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

from app.schemas.common import WarningItem
from app.schemas.eda import HistogramBin
from app.schemas.supervised import AnovaRow, BoxplotSeries


SectionStatus = Literal["available", "not_applicable"]


class VariableDefinition(BaseModel):
    name: str
    logical_type: str
    observed_type: str | None = None
    business_description: str
    analytical_role: str
    notes: str | None = None
    missing_pct: float | None = None
    unique_values: int | None = None


class WeekAcademicMetadata(BaseModel):
    objective: str
    analytical_goal: str
    domain_context: str
    unit_of_observation: str
    target_variable: str | None = None
    explanatory_variables: list[str]
    initial_hypotheses: list[str]
    variable_dictionary: list[VariableDefinition]


class AcademicContextSummary(BaseModel):
    objective: str
    analytical_goal: str
    unit_of_observation: str
    target_variable: str | None = None
    initial_hypotheses: list[str]


class InsightItem(BaseModel):
    title: str
    evidence: str
    implication: str
    next_step: str


class QualityFinding(BaseModel):
    kind: str
    severity: str
    hallazgo: str
    riesgo: str
    decision: str
    affected_columns: list[str]


class MissingSummary(BaseModel):
    count: int
    pct: float


class SourceOverviewMetrics(BaseModel):
    row_count: int
    column_count: int
    missing_cells: int
    completeness_pct: float
    duplicate_rows_exact: int
    numeric_variables: int
    categorical_variables: int


class DatasetAuditSection(BaseModel):
    shape: tuple[int, int]
    variable_type_counts: dict[str, int]
    duplicate_rows_exact: int
    unique_identifier_columns: list[str]
    completeness_pct: float
    missingness: dict[str, MissingSummary]
    cardinality: dict[str, int]
    variable_dictionary: list[VariableDefinition]


class DataQualitySection(BaseModel):
    findings: list[QualityFinding]


class NumericDistributionProfile(BaseModel):
    column: str
    stats: dict[str, float | None]
    histogram_bins: list[HistogramBin]
    values: list[float]


class NumericUnivariateSection(BaseModel):
    status: SectionStatus
    message: str | None = None
    variables: list[NumericDistributionProfile]


class CategoryFrequencyItem(BaseModel):
    value: str
    count: int
    pct: float


class CategoricalDistributionProfile(BaseModel):
    column: str
    total_categories: int
    top_categories: list[CategoryFrequencyItem]
    other_count: int
    other_pct: float


class CategoricalUnivariateSection(BaseModel):
    status: SectionStatus
    message: str | None = None
    variables: list[CategoricalDistributionProfile]


class TargetCorrelationRow(BaseModel):
    feature: str
    pearson: float | None = None
    spearman: float | None = None


class NumericNumericBivariateSection(BaseModel):
    status: SectionStatus
    message: str | None = None
    labels: list[str]
    pearson_matrix: list[list[float | None]]
    spearman_matrix: list[list[float | None]]
    target_rankings: list[TargetCorrelationRow]


class CategoricalNumericBivariateSection(BaseModel):
    status: SectionStatus
    message: str | None = None
    rows: list[AnovaRow]
    boxplot_data: list[BoxplotSeries]


class ContingencyPreview(BaseModel):
    row_label: str
    column_label: str
    columns: list[str]
    rows: list[dict[str, int | str]]


class CategoricalAssociationRow(BaseModel):
    feature_x: str
    feature_y: str
    chi2: float | None = None
    p_value: float | None = None
    cramers_v: float | None = None
    sample_size: int


class CategoricalCategoricalBivariateSection(BaseModel):
    status: SectionStatus
    message: str | None = None
    rows: list[CategoricalAssociationRow]
    contingency_preview: ContingencyPreview | None = None


class TemporalCountPoint(BaseModel):
    period: str
    count: int


class TemporalDiagnosticsSection(BaseModel):
    status: SectionStatus
    message: str
    unique_periods: list[str]
    counts: list[TemporalCountPoint]


class Week1SourceEdaSection(BaseModel):
    source: str
    title: str
    target_variable: str | None = None
    overview_metrics: SourceOverviewMetrics
    dataset_audit: DatasetAuditSection
    data_quality: DataQualitySection
    univariate_numeric: NumericUnivariateSection
    univariate_categorical: CategoricalUnivariateSection
    bivariate_numeric_numeric: NumericNumericBivariateSection
    bivariate_categorical_numeric: CategoricalNumericBivariateSection
    bivariate_categorical_categorical: CategoricalCategoricalBivariateSection
    temporal_diagnostics: TemporalDiagnosticsSection
    sample_rows: list[dict[str, Any]]
    columns: list[str]
    warnings: list[WarningItem]


class ComparisonCategoryRow(BaseModel):
    category: str
    in_count: int
    in_pct: float
    out_count: int
    out_pct: float


class ComparisonCategoricalSection(BaseModel):
    column: str
    categories: list[ComparisonCategoryRow]
    note: str | None = None


class NumericComparisonSummary(BaseModel):
    mean: float | None = None
    median: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None


class ComparisonNumericSection(BaseModel):
    column: str
    in_stats: NumericComparisonSummary
    out_stats: NumericComparisonSummary


class Week1ComparisonSection(BaseModel):
    shared_columns: list[str]
    categorical_comparisons: list[ComparisonCategoricalSection]
    numeric_comparisons: list[ComparisonNumericSection]
    notes: list[str]


class Week1ImputationSection(BaseModel):
    raw_missing_summary: dict[str, dict[str, MissingSummary]]
    imputation_applied: bool
    strategy_by_column: dict[str, dict[str, str]]
    imputed_counts: dict[str, dict[str, int]]
    analysis_dataset_paths: dict[str, str]
    notes: list[str]


class OutlierColumnSummary(BaseModel):
    column: str
    flagged_count: int
    flagged_ratio: float
    lower_bound: float | None = None
    upper_bound: float | None = None
    max_abs_robust_z: float | None = None


class Week1OutlierSourceSection(BaseModel):
    status: SectionStatus
    methods: list[str]
    columns: list[OutlierColumnSummary]
    flagged_counts: int
    flagged_ratio: float
    interpretation: str


class Week1OutlierSection(BaseModel):
    policy: str
    sources: dict[str, Week1OutlierSourceSection]


class Week1OpticsSourceSummary(BaseModel):
    status: SectionStatus
    cluster_count: int
    noise_ratio: float
    artifact_path: str


class Week1OpticsSummaryReference(BaseModel):
    endpoint: str
    sources: dict[str, Week1OpticsSourceSummary]


class WeekAcademicEDAResponse(BaseModel):
    problem_definition: WeekAcademicMetadata
    sources: dict[str, Week1SourceEdaSection]
    comparison: Week1ComparisonSection
    imputation: Week1ImputationSection
    outliers: Week1OutlierSection
    optics_summary: Week1OpticsSummaryReference
    insights: list[InsightItem]
    warnings: list[WarningItem]


class OpticsClusterSummary(BaseModel):
    cluster_id: int | None = None
    cluster_label: str
    description: str
    size: int
    pct: float
    top_categories: dict[str, str]
    numeric_means: dict[str, float]


class OpticsProjectionPoint(BaseModel):
    x: float
    y: float
    cluster_id: int | None = None
    cluster_label: str
    is_noise: bool = False
    display_weight: int = 1


class ReachabilityPoint(BaseModel):
    order: int
    reachability: float | None = None
    cluster_id: int | None = None
    cluster_label: str
    is_noise: bool = False


class ClusterRange(BaseModel):
    cluster_id: int | None = None
    cluster_label: str
    start_order: int
    end_order: int


class OverlapStats(BaseModel):
    overlap_pct: float
    unique_coordinates_raw: int
    unique_coordinates_display: int
    jitter_applied: bool


class EmbeddingQuality(BaseModel):
    trustworthiness: float | None = None
    overlap_stats: OverlapStats
    pca_explained_variance_2d: float | None = None


class OpticsCandidateSummary(BaseModel):
    min_samples: int
    min_cluster_size: int
    xi: float
    cluster_count: int
    noise_ratio: float
    non_noise_coverage: float
    silhouette_non_noise: float | None = None
    rejected: bool = False
    rejection_reason: str | None = None
    selected: bool = False


class OpticsSourceResult(BaseModel):
    source: str
    status: SectionStatus
    feature_columns: list[str]
    preprocessing: list[str]
    parameters: dict[str, float | int | str]
    selected_optics_parameters: dict[str, float | int | str]
    candidate_search_summary: list[OpticsCandidateSummary]
    embedding_method: str
    embedding_parameters: dict[str, float | int | str]
    embedding_quality: EmbeddingQuality
    cluster_count: int
    noise_ratio: float
    cluster_summary: list[OpticsClusterSummary]
    embedding_points: list[OpticsProjectionPoint]
    pca_points: list[OpticsProjectionPoint]
    reachability: list[ReachabilityPoint]
    overlap_stats: OverlapStats
    cluster_ranges: list[ClusterRange]
    artifacts: dict[str, str]
    warnings: list[WarningItem]
    interpretation: str


class WeekClusteringResponse(BaseModel):
    week_id: str
    stage_name: str
    sources: dict[str, OpticsSourceResult]
    warnings: list[WarningItem]
