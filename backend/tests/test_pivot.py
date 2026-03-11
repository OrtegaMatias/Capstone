from __future__ import annotations

import pandas as pd
import pytest

from app.stats.pivot import PivotRequest, build_pivot_metadata, run_pivot_query


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Owner": [1, 1, 2, 2, 3, 4, 5],
            "Size": [1, 2, 2, 2, 2, 2, 2],
            "Quality": ["A", "B", "B", "B", "C", "", None],
            "DaysInDeposit": [10, 20, 30, 40, 5, 2, 50],
            "week": [5, 5, 5, 5, 5, 5, 5],
        }
    )


def test_build_pivot_metadata_defaults_in() -> None:
    df = _sample_df().drop(columns=["DaysInDeposit"])
    payload = build_pivot_metadata(source="in", df=df)

    assert payload["defaults"]["row_dim"] == "Owner"
    assert payload["defaults"]["col_dim"] == "Quality"
    assert payload["defaults"]["value_field"] == "Size"
    assert "sum" in payload["agg_functions"]
    assert payload["field_agg_functions"]["Quality"] == ["count"]
    assert "sum" in payload["field_agg_functions"]["Size"]


def test_run_pivot_sum_size_owner_quality() -> None:
    df = _sample_df().drop(columns=["DaysInDeposit"])
    request = PivotRequest(
        source="in",
        row_dim="Owner",
        col_dim="Quality",
        value_field="Size",
        agg_func="sum",
        filters={},
        include_blank=True,
        top_k=10,
        small_n_threshold=5,
    )

    payload = run_pivot_query(df=df, request=request)
    matrix = payload["matrix"]

    assert matrix["grand_total"]["value"] == pytest.approx(13.0)
    owner_1 = next(row for row in matrix["rows"] if row["row_key"] == "1")
    assert owner_1["row_total"]["value"] == pytest.approx(3.0)


def test_run_pivot_rate_thresholds() -> None:
    df = _sample_df()
    request = PivotRequest(
        source="out",
        row_dim="Owner",
        col_dim="Size",
        value_field="DaysInDeposit",
        agg_func="rate_gt_30",
        filters={},
        include_blank=True,
        top_k=10,
        small_n_threshold=2,
    )

    payload = run_pivot_query(df=df, request=request)
    grand = payload["matrix"]["grand_total"]["value"]

    # Values > 30 are 40 and 50 over 7 rows.
    assert grand == pytest.approx((2 / 7) * 100.0)


def test_run_pivot_top_k_groups_tail() -> None:
    df = _sample_df()
    request = PivotRequest(
        source="out",
        row_dim="Owner",
        col_dim="Quality",
        value_field="DaysInDeposit",
        agg_func="mean",
        filters={},
        include_blank=True,
        top_k=2,
        small_n_threshold=5,
    )

    payload = run_pivot_query(df=df, request=request)
    row_keys = [row["row_key"] for row in payload["matrix"]["rows"]]
    assert "Other" in row_keys


def test_run_pivot_rejects_non_numeric_sum() -> None:
    df = _sample_df().drop(columns=["DaysInDeposit"])
    request = PivotRequest(
        source="in",
        row_dim="Owner",
        col_dim="Quality",
        value_field="Quality",
        agg_func="sum",
        filters={},
        include_blank=True,
        top_k=10,
        small_n_threshold=5,
    )

    with pytest.raises(ValueError, match="requires numeric value_field"):
        run_pivot_query(df=df, request=request)
