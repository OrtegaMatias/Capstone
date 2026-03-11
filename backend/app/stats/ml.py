from __future__ import annotations

from typing import Any

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from app.stats.columns import analytical_columns
from app.stats.warnings import dataframe_quality_warnings


def _extract_tree_structure(
    tree_model: DecisionTreeRegressor,
    feature_names: list[str],
    max_nodes: int = 80,
) -> dict[str, Any] | None:
    """Walk the sklearn tree and build a nested dict suitable for JSON."""
    try:
        tree = tree_model.tree_
    except AttributeError:
        return None

    counter = {"n": 0}

    def _recurse(node_id: int) -> dict[str, Any] | None:
        if counter["n"] >= max_nodes:
            return None
        counter["n"] += 1

        value = float(tree.value[node_id].flatten()[0])
        samples = int(tree.n_node_samples[node_id])
        is_leaf = tree.children_left[node_id] == -1

        if is_leaf:
            return {
                "type": "leaf",
                "value": round(value, 2),
                "samples": samples,
            }

        feat_idx = int(tree.feature[node_id])
        threshold = float(tree.threshold[node_id])
        feat_name = feature_names[feat_idx] if feat_idx < len(feature_names) else f"x[{feat_idx}]"
        clean_name = (
            feat_name
            .replace("num__", "")
            .replace("cat__", "")
            .replace("encoder__", "")
            .replace("imputer__", "")
        )

        left = _recurse(int(tree.children_left[node_id]))
        right = _recurse(int(tree.children_right[node_id]))

        return {
            "type": "split",
            "feature": clean_name,
            "threshold": round(threshold, 4),
            "value": round(value, 2),
            "samples": samples,
            "left": left,
            "right": right,
        }

    return _recurse(0)


TARGET_COL = "DaysInDeposit"
WEEK_COL = "week"


def _empty_ml_response(*, warnings: list[dict[str, Any]], target_present: bool) -> dict[str, Any]:
    return {
        "target_present": target_present,
        "model_built": False,
        "target_column": TARGET_COL,
        "split": {
            "train_weeks": [],
            "test_weeks": [],
            "train_rows": 0,
            "test_rows": 0,
        },
        "feature_columns": [],
        "numeric_features": [],
        "categorical_features": [],
        "models": [],
        "warnings": warnings,
    }


def compute_temporal_ml_overview(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    week_col: str = WEEK_COL,
) -> dict[str, Any]:
    if target_col not in df.columns:
        return _empty_ml_response(
            target_present=False,
            warnings=[
                {
                    "code": "missing_target",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target DaysInDeposit is not available for temporal ML evaluation.",
                    "suggestion": "Provide an OUT dataset with DaysInDeposit.",
                }
            ],
        )

    if week_col not in df.columns:
        return _empty_ml_response(
            target_present=True,
            warnings=[
                {
                    "code": "missing_week",
                    "severity": "warning",
                    "column": week_col,
                    "message": "Column week is required for temporal train/test split.",
                    "suggestion": "Include a valid week column before running ML evaluation.",
                }
            ],
        )

    data = df.copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data[week_col] = pd.to_numeric(data[week_col], errors="coerce")
    data = data.dropna(subset=[target_col, week_col])

    if data.empty:
        return _empty_ml_response(
            target_present=True,
            warnings=[
                {
                    "code": "temporal_ml_empty",
                    "severity": "warning",
                    "column": None,
                    "message": "No valid rows remained after coercing target/week to numeric values.",
                    "suggestion": "Review week and target formatting.",
                }
            ],
        )

    unique_weeks = sorted({int(value) for value in data[week_col].dropna().tolist()})
    if len(unique_weeks) < 2:
        return _empty_ml_response(
            target_present=True,
            warnings=[
                {
                    "code": "insufficient_temporal_history",
                    "severity": "warning",
                    "column": week_col,
                    "message": "At least two different week values are required for temporal validation.",
                    "suggestion": "Use a historical dataset with multiple weeks.",
                }
            ],
        )

    holdout_week = unique_weeks[-1]
    train_df = data[data[week_col] < holdout_week].copy()
    test_df = data[data[week_col] == holdout_week].copy()

    warnings = dataframe_quality_warnings(data)
    if train_df.empty or test_df.empty:
        warnings.append(
            {
                "code": "invalid_temporal_split",
                "severity": "warning",
                "column": week_col,
                "message": "Temporal split produced an empty train or test partition.",
                "suggestion": "Review the week distribution before training the model.",
            }
        )
        return _empty_ml_response(target_present=True, warnings=warnings)

    candidate_features = [col for col in analytical_columns(data, keep_columns={target_col}) if col != target_col]
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    train_model_df = pd.DataFrame(index=train_df.index)
    test_model_df = pd.DataFrame(index=test_df.index)

    for col in candidate_features:
        series = train_df[col]
        if int(series.astype("string").nunique(dropna=True)) <= 1:
            warnings.append(
                {
                    "code": "feature_excluded_constant",
                    "severity": "info",
                    "column": col,
                    "message": f"Feature {col} was excluded because it is constant in training data.",
                    "suggestion": "Use features with variation across weeks for predictive models.",
                }
            )
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_features.append(col)
            train_model_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            test_model_df[col] = pd.to_numeric(test_df[col], errors="coerce")
        else:
            categorical_features.append(col)
            train_model_df[col] = train_df[col].astype("string")
            test_model_df[col] = test_df[col].astype("string")

    feature_columns = numeric_features + categorical_features
    if not feature_columns:
        warnings.append(
            {
                "code": "no_temporal_features",
                "severity": "warning",
                "column": None,
                "message": "No eligible features remained after filtering constants.",
                "suggestion": "Review preprocessing or include richer signals in the dataset.",
            }
        )
        return _empty_ml_response(target_present=True, warnings=warnings)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    train_target = train_df[target_col].to_numpy(dtype=float)
    test_target = test_df[target_col].to_numpy(dtype=float)
    
    train_model_df_features = train_model_df[feature_columns]
    test_model_df_features = test_model_df[feature_columns]
    baseline_prediction = float(np.mean(train_target))

    estimators = [
        ("Decision Tree", DecisionTreeRegressor(max_depth=5, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)),
        ("CatBoost", CatBoostRegressor(iterations=100, depth=5, random_seed=42, verbose=False)),
    ]

    models_out = []

    for model_name, regressor in estimators:
        model = Pipeline([("preprocess", preprocess), ("regressor", regressor)])
        model.fit(train_model_df_features, train_target)
        predictions = model.predict(test_model_df_features)

        metrics = {
            "mae": float(mean_absolute_error(test_target, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(test_target, predictions))),
            "r2": float(r2_score(test_target, predictions)) if len(test_target) > 1 else None,
            "baseline_mae": float(mean_absolute_error(test_target, np.full_like(test_target, baseline_prediction))),
        }

        feature_effects: list[dict[str, float | str]] = []
        preprocessor = model.named_steps["preprocess"]
        fitted_regressor = model.named_steps["regressor"]
        feature_names = preprocessor.get_feature_names_out()
        
        importances = getattr(fitted_regressor, "feature_importances_", [])
        if len(importances) == len(feature_names):
            for name, importance in sorted(
                zip(feature_names.tolist(), importances.tolist(), strict=False),
                key=lambda item: abs(float(item[1])),
                reverse=True,
            )[:12]:
                clean_name = (
                    str(name)
                    .replace("num__", "")
                    .replace("cat__", "")
                    .replace("encoder__", "")
                    .replace("imputer__", "")
                )
                feature_effects.append({"feature": clean_name, "coefficient": float(importance)})

        prediction_rows = []
        test_weeks = test_df[week_col].astype(int).astype(str).tolist()
        for week_value, actual, predicted in zip(test_weeks, test_target.tolist(), predictions.tolist(), strict=False):
            prediction_rows.append(
                {
                    "week": str(week_value),
                    "actual": float(actual),
                    "predicted": float(predicted),
                }
            )

        tree_structure = None
        if isinstance(regressor, DecisionTreeRegressor):
            feature_names_list = preprocessor.get_feature_names_out().tolist()
            tree_structure = _extract_tree_structure(fitted_regressor, feature_names_list)

        models_out.append(
            {
                "model_name": model_name,
                "metrics": metrics,
                "predictions": prediction_rows,
                "feature_effects": feature_effects,
                "tree_structure": tree_structure,
            }
        )

    warnings.append(
        {
            "code": "temporal_holdout",
            "severity": "info",
            "column": week_col,
            "message": f"Temporal holdout uses week {holdout_week} as test set.",
            "suggestion": "Keep the last observed week as untouched validation data for comparisons.",
        }
    )

    return {
        "target_present": True,
        "model_built": True,
        "target_column": target_col,
        "split": {
            "train_weeks": [str(value) for value in unique_weeks[:-1]],
            "test_weeks": [str(holdout_week)],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": models_out,
        "warnings": warnings,
    }
