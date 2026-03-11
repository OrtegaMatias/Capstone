from pydantic import BaseModel

from app.schemas.common import WarningItem


class VariabilityRow(BaseModel):
    column: str
    dtype_group: str
    entropy: float | None = None
    gini_impurity: float | None = None
    coefficient_variation: float | None = None
    custom_index: float | None = None
    custom_mode: str
    recommendation: str
    warnings: list[WarningItem]


class VariabilityResponse(BaseModel):
    rows: list[VariabilityRow]
