from __future__ import annotations

import pandas as pd

from app.etl.step_cast_types import _cast_single_dataframe
from app.stats.supervised import compute_multiple_regression_out


def test_multiple_regression_out_builds_model() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": list(range(12)),
            "Owner": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C"],
            "Size": [1, 2, 3, 4, 5, 6, 2, 3, 4, 1, 5, 6],
            "Type": ["DRY", "RF", "DRY", "RF", "DRY", "RF", "RF", "DRY", "RF", "DRY", "RF", "DRY"],
            "Quality": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A", "B", "C"],
            "week": [5] * 12,
            "DaysInDeposit": [8, 13, 14, 20, 20, 27, 12, 15, 21, 9, 24, 23],
        }
    )

    payload = compute_multiple_regression_out(df)

    assert payload["target_present"] is True
    assert payload["model_built"] is True
    assert payload["n_obs"] > 0
    assert payload["n_features"] >= 2
    assert payload["formula"] is not None
    assert len(payload["coefficients"]) > 0
    assert len(payload["anova_rows"]) > 0
    assert len(payload["conclusions"]) > 0


def test_multiple_regression_out_without_target() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "Owner": ["A", "B", "C"],
            "Size": [1, 2, 3],
        }
    )

    payload = compute_multiple_regression_out(df)
    assert payload["target_present"] is False
    assert payload["model_built"] is False
    assert any(w["code"] == "missing_target" for w in payload["warnings"])


def test_multiple_regression_out_without_usable_features() -> None:
    df = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2, 3],
            "week": [5, 5, 5, 5],
            "DaysInDeposit": [10, 12, 11, 13],
        }
    )

    payload = compute_multiple_regression_out(df)
    assert payload["target_present"] is True
    assert payload["model_built"] is False
    assert any(w["code"] in {"no_features_available", "no_features_after_cleaning"} for w in payload["warnings"])


def test_multiple_regression_treats_discrete_size_as_categorical_after_cast() -> None:
    base = pd.DataFrame(
        {
            "Unnamed: 0": list(range(1, 25)),
            "Owner": (["A", "B", "C", "A"] * 6),
            "Size": ([1, 2, 1, 2] * 6),
            "Type": (["DRY", "RF", "DRY", "RF"] * 6),
            "Quality": (["A", "A", "B", "B"] * 6),
            "week": [5] * 24,
            "DaysInDeposit": [9, 15, 11, 18, 10, 16, 12, 19, 11, 17, 13, 20, 9, 14, 12, 18, 10, 15, 13, 19, 11, 16, 12, 20],
        }
    )

    cast_df, _, _ = _cast_single_dataframe(base, source="out_file")
    payload = compute_multiple_regression_out(cast_df)

    assert payload["target_present"] is True
    assert payload["model_built"] is True
    assert payload["formula"] is not None
    assert 'C(Q("Size"))' in payload["formula"]
