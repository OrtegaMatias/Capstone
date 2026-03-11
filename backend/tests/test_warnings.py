from __future__ import annotations

import pandas as pd

from app.stats.warnings import dataframe_quality_warnings


def test_warnings_exclude_technical_id_and_reduce_week_noise() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3],
            "week": [5, 5, 5, 5],
            "Owner": ["A", "B", "C", "D"],
        }
    )

    warnings = dataframe_quality_warnings(df)
    warning_columns = [w.get("column") for w in warnings]
    warning_codes = [w.get("code") for w in warnings]

    assert "Unnamed: 0" not in warning_columns
    assert "week_constant" in warning_codes
    assert not any(w.get("code") == "constant_column" and w.get("column") == "week" for w in warnings)
    assert not any(w.get("code") == "near_constant_column" and w.get("column") == "week" for w in warnings)


def test_high_cardinality_only_for_non_numeric() -> None:
    df = pd.DataFrame(
        {
            "numeric_many_values": list(range(120)),
            "categorical_many_values": [f"cat_{i}" for i in range(120)],
        }
    )

    warnings = dataframe_quality_warnings(df)
    numeric_warn = [w for w in warnings if w.get("column") == "numeric_many_values" and w.get("code") == "high_cardinality"]
    categorical_warn = [
        w for w in warnings if w.get("column") == "categorical_many_values" and w.get("code") == "high_cardinality"
    ]

    assert numeric_warn == []
    assert len(categorical_warn) == 1

