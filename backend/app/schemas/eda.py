from typing import Any

from pydantic import BaseModel

from app.schemas.common import WarningItem


class HistogramBin(BaseModel):
    left: float
    right: float
    count: int


class NumericStats(BaseModel):
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    p25: float | None = None
    p50: float | None = None
    p75: float | None = None
    max: float | None = None


class EDAResponse(BaseModel):
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    missingness: dict[str, dict[str, float]]
    cardinality: dict[str, int]
    top_values: dict[str, list[dict[str, Any]]]
    numeric_stats: dict[str, NumericStats]
    numeric_histograms: dict[str, list[HistogramBin]]
    global_metrics: dict[str, float | int]
    warnings: list[WarningItem]
