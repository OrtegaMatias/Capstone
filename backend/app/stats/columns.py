from __future__ import annotations

from typing import Iterable

import pandas as pd


TECHNICAL_ID_NORMALIZED = {
    "unnamed:0",
    "unnamed:0.1",
    "index",
    "row_id",
    "rowid",
    "record_id",
}


def _normalize_column_name(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if not ch.isspace())


def is_technical_id_column(name: str) -> bool:
    normalized = _normalize_column_name(name)
    if normalized in TECHNICAL_ID_NORMALIZED:
        return True
    if normalized.startswith("unnamed:"):
        return True
    return False


def analytical_columns(df: pd.DataFrame, keep_columns: Iterable[str] | None = None) -> list[str]:
    keep = set(keep_columns or [])
    return [str(col) for col in df.columns if str(col) in keep or not is_technical_id_column(str(col))]

