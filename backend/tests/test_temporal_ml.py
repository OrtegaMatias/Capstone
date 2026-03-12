from __future__ import annotations

import pandas as pd

from app.stats import ml as ml_module
from app.stats.ml import compute_temporal_ml_overview


def _build_temporal_df() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_id = 0

    def add_block(
        *,
        week: int,
        count: int,
        owner: str,
        size: str,
        container_type: str,
        quality: str,
        base_target: int,
    ) -> None:
        nonlocal row_id
        for offset in range(count):
            rows.append(
                {
                    "Unnamed: 0": row_id,
                    "Owner": owner,
                    "Size": size,
                    "Type": container_type,
                    "Quality": quality,
                    "week": week,
                    "DaysInDeposit": base_target + (offset % 5),
                }
            )
            row_id += 1

    for week in (1, 2, 3, 4):
        add_block(week=week, count=35, owner="7", size="2", container_type="DRY", quality="CLASE B-C", base_target=6)
        add_block(week=week, count=30, owner="2", size="2", container_type="DRY", quality="CLASE B-C", base_target=18)
        add_block(week=week, count=28, owner="1", size="1", container_type="RF", quality="INSPECTION", base_target=24)
        add_block(week=week, count=8, owner="9", size="1", container_type="RF", quality="CLASE D", base_target=120)

    add_block(week=5, count=40, owner="7", size="2", container_type="DRY", quality="CLASE B-C", base_target=8)
    add_block(week=5, count=35, owner="2", size="2", container_type="DRY", quality="CLASE B-C", base_target=20)
    add_block(week=5, count=32, owner="1", size="1", container_type="RF", quality="INSPECTION", base_target=26)
    add_block(week=5, count=5, owner="9", size="1", container_type="RF", quality="CLASE D", base_target=180)

    return pd.DataFrame(rows)


def test_temporal_ml_overview_reports_optional_boosters_as_warnings(monkeypatch) -> None:
    monkeypatch.setattr(ml_module, "CatBoostRegressor", None)
    monkeypatch.setattr(ml_module, "LGBMRegressor", None)
    monkeypatch.setattr(ml_module, "XGBRegressor", None)
    monkeypatch.setattr(ml_module, "CATBOOST_IMPORT_ERROR", ImportError("catboost missing"))
    monkeypatch.setattr(ml_module, "LIGHTGBM_IMPORT_ERROR", ImportError("lightgbm missing"))
    monkeypatch.setattr(ml_module, "XGBOOST_IMPORT_ERROR", ImportError("xgboost missing"))

    payload = compute_temporal_ml_overview(_build_temporal_df())

    warning_codes = {warning["code"] for warning in payload["warnings"]}
    assert payload["model_built"] is True
    assert "catboost_unavailable" in warning_codes
    assert "lightgbm_unavailable" in warning_codes
    assert "xgboost_unavailable" in warning_codes
    assert payload["models"]


def test_temporal_ml_overview_exposes_target_strategies_and_learning_sections() -> None:
    payload = compute_temporal_ml_overview(_build_temporal_df())

    strategy_names = {(row["model_name"], row["strategy_name"]) for row in payload["preprocessing_benchmarks"]}
    assert ("Decision Tree", "raw") in strategy_names
    assert ("Decision Tree", "log1p") in strategy_names
    assert ("Decision Tree", "log1p_outlier_norm") in strategy_names
    assert ("Decision Tree", "winsor_iqr") in strategy_names
    assert any(section["slug"] == "transformacion-log" for section in payload["learning_sections"])
    diagnostics = payload["target_transformation_diagnostics"]
    assert diagnostics["scope"] == "train_only"
    assert len(diagnostics["steps"]) >= 4
    assert diagnostics["boxplot_data"]
    assert payload["strategy_comparison"] is not None

    raw_step = next(step for step in diagnostics["steps"] if step["step_key"] == "raw")
    normalized_step = next(step for step in diagnostics["steps"] if step["step_key"] == "log1p_outlier_norm")
    assert normalized_step["stats"]["outlier_ratio"] <= raw_step["stats"]["outlier_ratio"]


def test_temporal_ml_overview_groups_small_segments_into_other_and_builds_hierarchical_backoff() -> None:
    payload = compute_temporal_ml_overview(_build_temporal_df())

    owner_report = next(report for report in payload["segment_reports"] if report["family_key"] == "owner")
    owner_segments = {row["segment"] for row in owner_report["rows"]}
    assert "Other" in owner_segments
    assert "9" not in owner_segments

    hierarchical = next(model for model in payload["heuristic_models"] if model["model_name"] == "Hierarchical Backoff")
    assert hierarchical["tier_usage"]
    assert sum(item["count"] for item in hierarchical["tier_usage"]) == payload["split"]["test_rows"]
    assert hierarchical["metrics"]["mae"] is not None
