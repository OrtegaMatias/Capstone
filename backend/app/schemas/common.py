from typing import Any

from pydantic import BaseModel


class WarningItem(BaseModel):
    code: str
    severity: str = "warning"
    column: str | None = None
    message: str
    suggestion: str | None = None


class GenericMessage(BaseModel):
    ok: bool
    message: str
    details: dict[str, Any] | None = None
