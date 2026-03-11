from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.common import WarningItem


class PivotSourceItem(BaseModel):
    source: str
    available: bool


class PivotSourcesResponse(BaseModel):
    sources: list[PivotSourceItem]


class PivotMetadataDefaults(BaseModel):
    row_dim: str
    col_dim: str
    value_field: str
    agg_func: str
    include_blank: bool = True
    top_k: int = 10
    small_n_threshold: int = 5


class PivotMetadataResponse(BaseModel):
    source: str
    dimensions: list[str]
    value_fields: list[str]
    agg_functions: list[str]
    field_agg_functions: dict[str, list[str]] = Field(default_factory=dict)
    filter_options: dict[str, list[str]]
    defaults: PivotMetadataDefaults
    warnings: list[WarningItem]


class PivotQueryRequest(BaseModel):
    source: str
    row_dim: str
    col_dim: str
    value_field: str
    agg_func: str
    filters: dict[str, list[str]] = Field(default_factory=dict)
    include_blank: bool = True
    top_k: int = 10
    small_n_threshold: int = 5


class PivotCell(BaseModel):
    col_key: str
    value: float | int | None
    count: int
    low_sample: bool


class PivotTotal(BaseModel):
    value: float | int | None
    count: int


class PivotRow(BaseModel):
    row_key: str
    cells: list[PivotCell]
    row_total: PivotTotal


class PivotColumnTotal(BaseModel):
    col_key: str
    value: float | int | None
    count: int


class PivotMatrix(BaseModel):
    columns: list[str]
    rows: list[PivotRow]
    column_totals: list[PivotColumnTotal]
    grand_total: PivotTotal


class PivotQueryResponse(BaseModel):
    source: str
    row_dim: str
    col_dim: str
    value_field: str
    agg_func: str
    matrix: PivotMatrix
    warnings: list[WarningItem]
