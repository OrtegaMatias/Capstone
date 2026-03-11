from __future__ import annotations

import pandas as pd

from app.etl.step_cast_types import _cast_single_dataframe


def test_size_low_cardinality_discrete_is_categorical() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": list(range(1, 13)),
            "Owner": ["A"] * 6 + ["B"] * 6,
            "Size": [1, 2] * 6,
            "Type": ["DRY", "RF"] * 6,
            "Quality": ["A", "B"] * 6,
            "DaysInDeposit": [8, 12, 9, 13, 10, 14, 11, 15, 10, 16, 9, 17],
            "week": [5] * 12,
        }
    )

    cast_df, meta, warnings = _cast_single_dataframe(df, source="out_file")

    assert meta["semantic_types"]["Size"] == "categorical_discrete"
    assert pd.api.types.is_string_dtype(cast_df["Size"])
    assert set(cast_df["Size"].dropna().unique().tolist()) == {"1", "2"}
    assert any(w["code"] == "size_casted_to_categorical_discrete" for w in warnings)


def test_size_continuous_stays_numeric() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": list(range(1, 21)),
            "Owner": ["A"] * 10 + ["B"] * 10,
            "Size": [float(v) for v in range(10, 30)],
            "Type": ["DRY"] * 20,
            "Quality": ["A"] * 20,
            "DaysInDeposit": [float(v) for v in range(5, 25)],
            "week": [5] * 20,
        }
    )

    cast_df, meta, warnings = _cast_single_dataframe(df, source="out_file")

    assert meta["semantic_types"]["Size"] == "numeric"
    assert pd.api.types.is_numeric_dtype(cast_df["Size"])
    assert not any(w["code"] == "size_casted_to_categorical_discrete" for w in warnings)
