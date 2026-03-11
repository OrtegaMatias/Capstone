from typing import Any

from pydantic import BaseModel

from app.schemas.common import WarningItem


class PreviewResponse(BaseModel):
    columns: list[str]
    rows: list[dict[str, Any]]
    total_rows: int


class DatasetMetadataResponse(BaseModel):
    dataset_id: str
    shape: tuple[int, int]
    columns: list[str]
    dtype_summary: dict[str, str]
    has_target: bool
    warnings: list[WarningItem]
