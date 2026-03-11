from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import WarningItem


class UploadResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    dataset_id: str
    has_in: bool
    has_out: bool
    has_target: bool
    schema_detected: dict[str, list[str]] = Field(alias="schema")
    preview: list[dict[str, Any]]
    dtype_summary: dict[str, str]
    warnings: list[WarningItem]
