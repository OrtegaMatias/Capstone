export type WarningItem = {
  code: string;
  severity: string;
  column?: string | null;
  message: string;
  suggestion?: string | null;
};

export type UploadResponse = {
  dataset_id: string;
  has_in: boolean;
  has_out: boolean;
  has_target: boolean;
  schema: Record<string, string[]>;
  preview: Array<Record<string, unknown>>;
  dtype_summary: Record<string, string>;
  warnings: WarningItem[];
};

export type PreviewResponse = {
  columns: string[];
  rows: Array<Record<string, unknown>>;
  total_rows: number;
};

export type EDAResponse = {
  shape: [number, number];
  columns: string[];
  dtypes: Record<string, string>;
  missingness: Record<string, { count: number; pct: number }>;
  cardinality: Record<string, number>;
  top_values: Record<string, Array<{ value: string; count: number; pct: number }>>;
  numeric_stats: Record<
    string,
    {
      mean: number | null;
      std: number | null;
      min: number | null;
      p25: number | null;
      p50: number | null;
      p75: number | null;
      max: number | null;
    }
  >;
  numeric_histograms: Record<string, Array<{ left: number; right: number; count: number }>>;
  global_metrics: Record<string, number>;
  warnings: WarningItem[];
};

export type VariableDefinition = {
  name: string;
  logical_type: string;
  observed_type?: string | null;
  business_description: string;
  analytical_role: string;
  notes?: string | null;
  missing_pct?: number | null;
  unique_values?: number | null;
};

export type WeekAcademicMetadata = {
  objective: string;
  analytical_goal: string;
  domain_context: string;
  unit_of_observation: string;
  target_variable?: string | null;
  explanatory_variables: string[];
  initial_hypotheses: string[];
  variable_dictionary: VariableDefinition[];
};

export type AcademicContextSummary = {
  objective: string;
  analytical_goal: string;
  unit_of_observation: string;
  target_variable?: string | null;
  initial_hypotheses: string[];
};

export type InsightItem = {
  title: string;
  evidence: string;
  implication: string;
  next_step: string;
};

export type QualityFinding = {
  kind: string;
  severity: string;
  hallazgo: string;
  riesgo: string;
  decision: string;
  affected_columns: string[];
};

export type MissingSummary = {
  count: number;
  pct: number;
};

export type SourceOverviewMetrics = {
  row_count: number;
  column_count: number;
  missing_cells: number;
  completeness_pct: number;
  duplicate_rows_exact: number;
  numeric_variables: number;
  categorical_variables: number;
};

export type DatasetAuditSection = {
  shape: [number, number];
  variable_type_counts: Record<string, number>;
  duplicate_rows_exact: number;
  unique_identifier_columns: string[];
  completeness_pct: number;
  missingness: Record<string, MissingSummary>;
  cardinality: Record<string, number>;
  variable_dictionary: VariableDefinition[];
};

export type DataQualitySection = {
  findings: QualityFinding[];
};

export type NumericDistributionProfile = {
  column: string;
  stats: Record<string, number | null>;
  histogram_bins: Array<{ left: number; right: number; count: number }>;
  values: number[];
};

export type NumericUnivariateSection = {
  status: 'available' | 'not_applicable';
  message?: string | null;
  variables: NumericDistributionProfile[];
};

export type CategoryFrequencyItem = {
  value: string;
  count: number;
  pct: number;
};

export type CategoricalDistributionProfile = {
  column: string;
  total_categories: number;
  top_categories: CategoryFrequencyItem[];
  other_count: number;
  other_pct: number;
};

export type CategoricalUnivariateSection = {
  status: 'available' | 'not_applicable';
  message?: string | null;
  variables: CategoricalDistributionProfile[];
};

export type TargetCorrelationRow = {
  feature: string;
  pearson?: number | null;
  spearman?: number | null;
};

export type NumericNumericBivariateSection = {
  status: 'available' | 'not_applicable';
  message?: string | null;
  labels: string[];
  pearson_matrix: Array<Array<number | null>>;
  spearman_matrix: Array<Array<number | null>>;
  target_rankings: TargetCorrelationRow[];
};

export type CategoricalNumericBivariateSection = {
  status: 'available' | 'not_applicable';
  message?: string | null;
  rows: AnovaRow[];
  boxplot_data: BoxplotSeries[];
};

export type ContingencyPreview = {
  row_label: string;
  column_label: string;
  columns: string[];
  rows: Array<Record<string, string | number>>;
};

export type CategoricalAssociationRow = {
  feature_x: string;
  feature_y: string;
  chi2?: number | null;
  p_value?: number | null;
  cramers_v?: number | null;
  sample_size: number;
};

export type CategoricalCategoricalBivariateSection = {
  status: 'available' | 'not_applicable';
  message?: string | null;
  rows: CategoricalAssociationRow[];
  contingency_preview?: ContingencyPreview | null;
};

export type TemporalDiagnosticsSection = {
  status: 'available' | 'not_applicable';
  message: string;
  unique_periods: string[];
  counts: Array<{ period: string; count: number }>;
};

export type Week1SourceEdaSection = {
  source: string;
  title: string;
  target_variable?: string | null;
  overview_metrics: SourceOverviewMetrics;
  dataset_audit: DatasetAuditSection;
  data_quality: DataQualitySection;
  univariate_numeric: NumericUnivariateSection;
  univariate_categorical: CategoricalUnivariateSection;
  bivariate_numeric_numeric: NumericNumericBivariateSection;
  bivariate_categorical_numeric: CategoricalNumericBivariateSection;
  bivariate_categorical_categorical: CategoricalCategoricalBivariateSection;
  temporal_diagnostics: TemporalDiagnosticsSection;
  sample_rows: Array<Record<string, unknown>>;
  columns: string[];
  warnings: WarningItem[];
};

export type ComparisonCategoryRow = {
  category: string;
  in_count: number;
  in_pct: number;
  out_count: number;
  out_pct: number;
};

export type ComparisonCategoricalSection = {
  column: string;
  categories: ComparisonCategoryRow[];
  note?: string | null;
};

export type NumericComparisonSummary = {
  mean?: number | null;
  median?: number | null;
  std?: number | null;
  min?: number | null;
  max?: number | null;
};

export type ComparisonNumericSection = {
  column: string;
  in_stats: NumericComparisonSummary;
  out_stats: NumericComparisonSummary;
};

export type Week1ComparisonSection = {
  shared_columns: string[];
  categorical_comparisons: ComparisonCategoricalSection[];
  numeric_comparisons: ComparisonNumericSection[];
  notes: string[];
};

export type Week1ImputationSection = {
  raw_missing_summary: Record<string, Record<string, MissingSummary>>;
  imputation_applied: boolean;
  strategy_by_column: Record<string, Record<string, string>>;
  imputed_counts: Record<string, Record<string, number>>;
  analysis_dataset_paths: Record<string, string>;
  notes: string[];
};

export type OutlierColumnSummary = {
  column: string;
  flagged_count: number;
  flagged_ratio: number;
  lower_bound?: number | null;
  upper_bound?: number | null;
  max_abs_robust_z?: number | null;
};

export type Week1OutlierSourceSection = {
  status: 'available' | 'not_applicable';
  methods: string[];
  columns: OutlierColumnSummary[];
  flagged_counts: number;
  flagged_ratio: number;
  interpretation: string;
};

export type Week1OutlierSection = {
  policy: string;
  sources: Record<string, Week1OutlierSourceSection>;
};

export type Week1OpticsSourceSummary = {
  status: 'available' | 'not_applicable';
  cluster_count: number;
  noise_ratio: number;
  artifact_path: string;
};

export type Week1OpticsSummaryReference = {
  endpoint: string;
  sources: Record<string, Week1OpticsSourceSummary>;
};

export type WeekAcademicEDAResponse = {
  problem_definition: WeekAcademicMetadata;
  sources: Record<string, Week1SourceEdaSection>;
  comparison: Week1ComparisonSection;
  imputation: Week1ImputationSection;
  outliers: Week1OutlierSection;
  optics_summary: Week1OpticsSummaryReference;
  insights: InsightItem[];
  warnings: WarningItem[];
};

export type OpticsClusterSummary = {
  cluster_id?: number | null;
  cluster_label: string;
  description: string;
  size: number;
  pct: number;
  top_categories: Record<string, string>;
  numeric_means: Record<string, number>;
};

export type OpticsProjectionPoint = {
  x: number;
  y: number;
  cluster_id?: number | null;
  cluster_label: string;
  is_noise: boolean;
  display_weight: number;
};

export type ReachabilityPoint = {
  order: number;
  reachability?: number | null;
  cluster_label: string;
  cluster_id?: number | null;
  is_noise: boolean;
};

export type ClusterRange = {
  cluster_id?: number | null;
  cluster_label: string;
  start_order: number;
  end_order: number;
};

export type OverlapStats = {
  overlap_pct: number;
  unique_coordinates_raw: number;
  unique_coordinates_display: number;
  jitter_applied: boolean;
};

export type EmbeddingQuality = {
  trustworthiness?: number | null;
  overlap_stats: OverlapStats;
  pca_explained_variance_2d?: number | null;
};

export type OpticsCandidateSummary = {
  min_samples: number;
  min_cluster_size: number;
  xi: number;
  cluster_count: number;
  noise_ratio: number;
  non_noise_coverage: number;
  silhouette_non_noise?: number | null;
  rejected: boolean;
  rejection_reason?: string | null;
  selected: boolean;
};

export type OpticsSourceResult = {
  source: string;
  status: 'available' | 'not_applicable';
  feature_columns: string[];
  preprocessing: string[];
  parameters: Record<string, string | number>;
  selected_optics_parameters: Record<string, string | number>;
  candidate_search_summary: OpticsCandidateSummary[];
  embedding_method: string;
  embedding_parameters: Record<string, string | number>;
  embedding_quality: EmbeddingQuality;
  cluster_count: number;
  noise_ratio: number;
  cluster_summary: OpticsClusterSummary[];
  embedding_points: OpticsProjectionPoint[];
  pca_points: OpticsProjectionPoint[];
  reachability: ReachabilityPoint[];
  overlap_stats: OverlapStats;
  cluster_ranges: ClusterRange[];
  artifacts: Record<string, string>;
  warnings: WarningItem[];
  interpretation: string;
};

export type WeekClusteringResponse = {
  week_id: string;
  stage_name: string;
  sources: Record<string, OpticsSourceResult>;
  warnings: WarningItem[];
};

export type VariabilityRow = {
  column: string;
  dtype_group: string;
  entropy: number | null;
  gini_impurity: number | null;
  coefficient_variation: number | null;
  custom_index: number | null;
  custom_mode: string;
  recommendation: string;
  warnings: WarningItem[];
};

export type VariabilityResponse = {
  rows: VariabilityRow[];
};

export type SupervisedOverviewResponse = {
  target_present: boolean;
  target_stats: Record<string, number | null>;
  hist_bins: Array<{ left: number; right: number; count: number }>;
  pearson_correlations: Array<{ feature: string; pearson: number }>;
  mutual_information: Array<{ feature: string; mi: number }>;
  warnings: WarningItem[];
};

export type MultipleRegressionCoefficient = {
  term: string;
  estimate: number | null;
  std_error: number | null;
  t_value: number | null;
  p_value: number | null;
  ci_low: number | null;
  ci_high: number | null;
};

export type MultipleRegressionAnovaRow = {
  feature: string;
  feature_type: string;
  df: number | null;
  sum_sq: number | null;
  mean_sq: number | null;
  f_value: number | null;
  p_value: number | null;
  partial_eta_squared: number | null;
};

export type MultipleRegressionResponse = {
  source: string;
  target_present: boolean;
  model_built: boolean;
  formula: string | null;
  n_obs: number;
  n_features: number;
  r_squared: number | null;
  adj_r_squared: number | null;
  f_statistic: number | null;
  f_p_value: number | null;
  aic: number | null;
  bic: number | null;
  coefficients: MultipleRegressionCoefficient[];
  anova_rows: MultipleRegressionAnovaRow[];
  conclusions: string[];
  warnings: WarningItem[];
};

export type AnovaRow = {
  feature: string;
  feature_type: string;
  test_used: string;
  statistic: number | null;
  p_value: number | null;
  effect_size: number | null;
  n_groups: number | null;
  kruskal_statistic: number | null;
  kruskal_p_value: number | null;
  warnings: WarningItem[];
};

export type BoxplotSeries = {
  feature: string;
  groups: Array<{ group: string; values: number[] }>;
};

export type AnovaResponse = {
  rows: AnovaRow[];
  warnings: WarningItem[];
  boxplot_data: BoxplotSeries[];
};

export type NotesResponse = {
  content: string;
  updated_at: string | null;
};

export type NotesSaveResponse = {
  ok: boolean;
  updated_at: string;
};

export type PivotSourceItem = {
  source: 'in' | 'out';
  available: boolean;
};

export type PivotSourcesResponse = {
  sources: PivotSourceItem[];
};

export type PivotMetadataDefaults = {
  row_dim: string;
  col_dim: string;
  value_field: string;
  agg_func: PivotAggFunc;
  include_blank: boolean;
  top_k: number;
  small_n_threshold: number;
};

export type PivotAggFunc =
  | 'count'
  | 'sum'
  | 'mean'
  | 'median'
  | 'rate_gt_7'
  | 'rate_gt_14'
  | 'rate_gt_30';

export type PivotMetadataResponse = {
  source: 'in' | 'out';
  dimensions: string[];
  value_fields: string[];
  agg_functions: PivotAggFunc[];
  field_agg_functions: Record<string, PivotAggFunc[]>;
  filter_options: Record<string, string[]>;
  defaults: PivotMetadataDefaults;
  warnings: WarningItem[];
};

export type PivotQueryRequest = {
  source: 'in' | 'out';
  row_dim: string;
  col_dim: string;
  value_field: string;
  agg_func: PivotAggFunc;
  filters?: Record<string, string[]>;
  include_blank?: boolean;
  top_k?: number;
  small_n_threshold?: number;
};

export type PivotCell = {
  col_key: string;
  value: number | null;
  count: number;
  low_sample: boolean;
};

export type PivotRow = {
  row_key: string;
  cells: PivotCell[];
  row_total: {
    value: number | null;
    count: number;
  };
};

export type PivotColumnTotal = {
  col_key: string;
  value: number | null;
  count: number;
};

export type PivotQueryResponse = {
  source: 'in' | 'out';
  row_dim: string;
  col_dim: string;
  value_field: string;
  agg_func: PivotAggFunc;
  matrix: {
    columns: string[];
    rows: PivotRow[];
    column_totals: PivotColumnTotal[];
    grand_total: {
      value: number | null;
      count: number;
    };
  };
  warnings: WarningItem[];
};

export type WeekStatus = 'active' | 'scaffolded';

export type WeekSeedPaths = {
  in_file?: string | null;
  out_file?: string | null;
};

export type WeekArtifact = {
  kind: string;
  label: string;
  path: string;
  available: boolean;
};

export type WeekSummary = {
  week_id: string;
  week_number: number;
  title: string;
  stage_name: string;
  status: WeekStatus;
  summary: string;
  seed_paths: WeekSeedPaths;
  analysis_available: string[];
  artifacts: WeekArtifact[];
  notes_updated_at?: string | null;
  report_available: boolean;
};

export type WeekConfig = WeekSummary & {
  description: string;
  expected_inputs: string[];
  checklist: string[];
  deliverables: string[];
  academic_context?: AcademicContextSummary | null;
};

export type FrameworkSummary = {
  framework_name: string;
  summary: string;
  workspace_root: string;
  generated_at: string;
  weeks: WeekSummary[];
};

export type WeekReportSummary = {
  week_id: string;
  stage_name: string;
  markdown_content: string;
  html_content: string;
  updated_at?: string | null;
  artifacts: WeekArtifact[];
};

export type MlSplitSummary = {
  train_weeks: string[];
  test_weeks: string[];
  train_rows: number;
  test_rows: number;
};

export type MlMetricSummary = {
  mae?: number | null;
  rmse?: number | null;
  r2?: number | null;
  baseline_mae?: number | null;
};

export type MlPredictionSample = {
  week: string;
  actual: number;
  predicted: number;
};

export type MlFeatureEffect = {
  feature: string;
  coefficient: number;
};

export type TreeNode = {
  type: 'split' | 'leaf';
  feature?: string;
  threshold?: number;
  value: number;
  samples: number;
  left?: TreeNode | null;
  right?: TreeNode | null;
};

export type MlModelResult = {
  model_name: string;
  metrics: MlMetricSummary;
  predictions: MlPredictionSample[];
  feature_effects: MlFeatureEffect[];
  tree_structure?: TreeNode | null;
};

export type MlEvaluationSummary = {
  week_id: string;
  stage_name: string;
  target_present: boolean;
  model_built: boolean;
  target_column: string;
  split: MlSplitSummary;
  feature_columns: string[];
  numeric_features: string[];
  categorical_features: string[];
  models: MlModelResult[];
  warnings: WarningItem[];
  supervised_overview: SupervisedOverviewResponse;
  anova: AnovaResponse;
  multiple_regression: MultipleRegressionResponse;
  artifacts: WeekArtifact[];
};
