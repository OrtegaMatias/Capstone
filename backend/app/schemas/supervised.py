from typing import Any

from pydantic import BaseModel

from app.schemas.common import WarningItem


class SupervisedOverviewResponse(BaseModel):
    target_present: bool
    target_stats: dict[str, float | int | None]
    hist_bins: list[dict[str, float | int]]
    pearson_correlations: list[dict[str, float | str]]
    mutual_information: list[dict[str, float | str]]
    warnings: list[WarningItem]


class AnovaRow(BaseModel):
    feature: str
    feature_type: str
    test_used: str
    statistic: float | None = None
    p_value: float | None = None
    effect_size: float | None = None
    n_groups: int | None = None
    kruskal_statistic: float | None = None
    kruskal_p_value: float | None = None
    warnings: list[WarningItem]


class BoxplotSeries(BaseModel):
    feature: str
    groups: list[dict[str, Any]]


class AnovaResponse(BaseModel):
    rows: list[AnovaRow]
    warnings: list[WarningItem]
    boxplot_data: list[BoxplotSeries]


class RegressionCoefficient(BaseModel):
    term: str
    estimate: float | None = None
    std_error: float | None = None
    t_value: float | None = None
    p_value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None


class RegressionAnovaRow(BaseModel):
    feature: str
    feature_type: str
    df: float | None = None
    sum_sq: float | None = None
    mean_sq: float | None = None
    f_value: float | None = None
    p_value: float | None = None
    partial_eta_squared: float | None = None


class MultipleRegressionResponse(BaseModel):
    source: str
    target_present: bool
    model_built: bool
    formula: str | None = None
    n_obs: int = 0
    n_features: int = 0
    r_squared: float | None = None
    adj_r_squared: float | None = None
    f_statistic: float | None = None
    f_p_value: float | None = None
    aic: float | None = None
    bic: float | None = None
    coefficients: list[RegressionCoefficient]
    anova_rows: list[RegressionAnovaRow]
    conclusions: list[str]
    warnings: list[WarningItem]
