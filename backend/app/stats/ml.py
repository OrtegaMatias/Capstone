from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from app.stats.columns import analytical_columns
from app.stats.warnings import dataframe_quality_warnings

try:
    from catboost import CatBoostRegressor
except Exception as exc:  # pragma: no cover - depends on local install
    CatBoostRegressor = None
    CATBOOST_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on local install
    CATBOOST_IMPORT_ERROR = None

try:
    from lightgbm import LGBMRegressor
except Exception as exc:  # pragma: no cover - depends on local install
    LGBMRegressor = None
    LIGHTGBM_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on local install
    LIGHTGBM_IMPORT_ERROR = None

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - depends on local install
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = exc
else:  # pragma: no cover - depends on local install
    XGBOOST_IMPORT_ERROR = None


TARGET_COL = "DaysInDeposit"
WEEK_COL = "week"
REPRESENTATIVE_TRAIN_MIN = 100
REPRESENTATIVE_TEST_MIN = 30

SIMPLE_SEGMENT_SPECS: list[tuple[tuple[str, ...], str, str, str]] = [
    (("Type",), "type", "Type", "Mediana por Tipo de contenedor"),
    (("Quality",), "quality", "Quality", "Mediana por Calidad"),
    (("Owner",), "owner", "Owner", "Mediana por Propietario"),
    (("Size",), "size", "Size", "Mediana por Tamaño"),
]

COMBO_SEGMENT_SPECS: list[tuple[tuple[str, ...], str, str, str]] = [
    (("Type", "Quality"), "type_quality", "Type x Quality", "Mediana por Tipo x Calidad"),
    (("Owner", "Size"), "owner_size", "Owner x Size", "Mediana por Propietario x Tamaño"),
    (("Owner", "Type"), "owner_type", "Owner x Type", "Mediana por Propietario x Tipo"),
]

STRATEGY_LABELS = {
    "raw": "Raw target",
    "log1p": "Log1p target",
    "log1p_drop_outliers": "Log1p + winsorizado",
    "baseline": "Baseline",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    builder: Callable[[], Any] | None
    dependency_name: str | None = None
    dependency_error: Exception | None = None


def _extract_tree_structure(
    tree_model: DecisionTreeRegressor,
    feature_names: list[str],
    max_nodes: int = 80,
) -> dict[str, Any] | None:
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
            feat_name.replace("num__", "")
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
        "preprocessing_benchmarks": [],
        "segment_reports": [],
        "heuristic_models": [],
        "strategy_comparison": None,
        "target_transformation_diagnostics": None,
    }


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _metric_summary(actual: np.ndarray, predicted: np.ndarray, baseline_prediction: np.ndarray) -> dict[str, Any]:
    mae = mean_absolute_error(actual, predicted)
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    medae = median_absolute_error(actual, predicted)
    r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else None
    baseline_mae = mean_absolute_error(actual, baseline_prediction)

    safe_actual = np.clip(actual, 0, None)
    safe_predicted = np.clip(predicted, 0, None)
    msle = float(np.mean((np.log1p(safe_actual) - np.log1p(safe_predicted)) ** 2))

    nonzero_mask = actual > 0
    if nonzero_mask.sum() > 0:
        mape = float(np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])))
    else:
        mape = None

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": r2,
        "medae": float(medae),
        "baseline_mae": float(baseline_mae),
        "msle": msle,
        "mape": mape,
    }


def _empty_metric_summary() -> dict[str, Any]:
    return {
        "mae": None,
        "rmse": None,
        "r2": None,
        "medae": None,
        "baseline_mae": None,
        "msle": None,
        "mape": None,
    }


def _iqr_bounds(values: np.ndarray) -> tuple[float | None, float | None]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return None, None
    q1 = float(np.quantile(clean, 0.25))
    q3 = float(np.quantile(clean, 0.75))
    iqr = q3 - q1
    if np.isclose(iqr, 0.0):
        return None, None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _boxplot_stats(values: np.ndarray) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "max": None,
            "skew": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "outlier_count": 0,
            "outlier_ratio": 0.0,
        }

    lower, upper = _iqr_bounds(clean)
    if lower is None or upper is None:
        outlier_count = 0
        outlier_ratio = 0.0
    else:
        outlier_mask = (clean < lower) | (clean > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_ratio = float(outlier_mask.mean())

    return {
        "count": int(clean.size),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        "min": float(np.min(clean)),
        "p25": float(np.quantile(clean, 0.25)),
        "p50": float(np.quantile(clean, 0.50)),
        "p75": float(np.quantile(clean, 0.75)),
        "max": float(np.max(clean)),
        "skew": float(pd.Series(clean).skew()) if clean.size > 2 else 0.0,
        "iqr": float(np.quantile(clean, 0.75) - np.quantile(clean, 0.25)),
        "lower_bound": lower,
        "upper_bound": upper,
        "outlier_count": outlier_count,
        "outlier_ratio": outlier_ratio,
    }


def _build_target_transformation_diagnostics(train_target: np.ndarray) -> dict[str, Any]:
    raw_values = np.asarray(train_target, dtype=float)
    log_values = np.log1p(np.clip(raw_values, a_min=0.0, a_max=None))
    log_lower, log_upper = _iqr_bounds(log_values)
    if log_lower is None or log_upper is None:
        log_winsorized_values = log_values.copy()
        clipped_count = 0
    else:
        outlier_mask = (log_values < log_lower) | (log_values > log_upper)
        clipped_count = int(outlier_mask.sum())
        log_winsorized_values = np.clip(log_values, log_lower, log_upper)

    steps = [
        {
            "step_key": "raw",
            "step_label": "Raw target",
            "scale": "days",
            "stats": _boxplot_stats(raw_values),
            "notes": ["Distribución original del target en días sobre train."],
        },
        {
            "step_key": "log1p",
            "step_label": "Log1p target",
            "scale": "log",
            "stats": _boxplot_stats(log_values),
            "notes": ["Compresión logarítmica del target."],
        },
        {
            "step_key": "log1p_drop_outliers",
            "step_label": "Log1p + winsorizado",
            "scale": "log",
            "stats": _boxplot_stats(log_winsorized_values),
            "notes": [
                (
                    f"Se winsorizaron {clipped_count} valores outlier al rango IQR en escala "
                    f"logarítmica ({clipped_count / max(len(log_values), 1) * 100:.1f}%). "
                    "Los valores extremos se recortan a los límites, sin eliminar filas."
                )
            ],
        },
    ]

    return {
        "scope": "train_only",
        "boxplot_data": [
            {
                "feature": "Escala original vs log",
                "groups": [
                    {"group": "Raw target", "values": raw_values.tolist()},
                    {"group": "Log1p target", "values": log_values.tolist()},
                ],
            },
            {
                "feature": "Log1p: todos vs winsorizado",
                "groups": [
                    {"group": "Log1p target", "values": log_values.tolist()},
                    {"group": "Log1p + winsorizado", "values": log_winsorized_values.tolist()},
                ],
            },
        ],
        "steps": steps,
    }


def _prediction_rows(df: pd.DataFrame, actual: np.ndarray, predicted: np.ndarray, week_col: str) -> list[dict[str, Any]]:
    week_values = df[week_col].astype(int).astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for row_id, week_value, actual_value, predicted_value in zip(
        range(len(actual)),
        week_values,
        actual.tolist(),
        predicted.tolist(),
        strict=False,
    ):
        rows.append(
            {
                "row_id": row_id,
                "week": str(week_value),
                "actual": float(actual_value),
                "predicted": float(predicted_value),
            }
        )
    return rows


def _dependency_warning(spec: ModelSpec) -> dict[str, Any]:
    detail = str(spec.dependency_error) if spec.dependency_error is not None else "Dependency not available."
    dependency_name = spec.dependency_name or spec.name
    return {
        "code": f"{spec.name.lower().replace(' ', '_')}_unavailable",
        "severity": "info",
        "column": None,
        "message": f"{spec.name} benchmark skipped because optional dependency {dependency_name} is not installed.",
        "suggestion": detail,
    }


def _unavailable_benchmark_row(
    spec: ModelSpec,
    *,
    strategy_name: str,
    strategy_label: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "model_name": spec.name,
        "strategy_name": strategy_name,
        "strategy_label": strategy_label,
        "metrics": _empty_metric_summary(),
        "available": False,
        "notes": notes,
    }


def _unavailable_model_result(
    spec: ModelSpec,
    *,
    strategy_name: str,
    strategy_label: str,
    notes: list[str],
) -> dict[str, Any]:
    return {
        "model_name": spec.name,
        "strategy_name": strategy_name,
        "strategy_label": strategy_label,
        "metrics": _empty_metric_summary(),
        "train_metrics": None,
        "predictions": [],
        "feature_effects": [],
        "tree_structure": None,
        "notes": notes,
    }


def _regression_model_specs(strategy_name: str = "raw") -> list[ModelSpec]:
    use_mae = strategy_name in {"log1p", "log1p_drop_outliers"}
    tree_criterion = "absolute_error" if use_mae else "squared_error"

    specs = [
        ModelSpec(
            name="Decision Tree",
            builder=lambda _crit=tree_criterion: DecisionTreeRegressor(
                max_depth=5, random_state=42, criterion=_crit,
            ),
        ),
        ModelSpec(
            name="Random Forest",
            builder=lambda _crit=tree_criterion: RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                criterion=_crit,
            ),
        ),
        ModelSpec(
            name="CatBoost",
            builder=(
                None
                if CatBoostRegressor is None
                else lambda _mae=use_mae: CatBoostRegressor(
                    iterations=100,
                    depth=5,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    loss_function="MAE" if _mae else "RMSE",
                )
            ),
            dependency_name="catboost",
            dependency_error=CATBOOST_IMPORT_ERROR,
        ),
        ModelSpec(
            name="XGBoost",
            builder=(
                None
                if XGBRegressor is None
                else lambda _mae=use_mae: XGBRegressor(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:absoluteerror" if _mae else "reg:squarederror",
                    random_state=42,
                    n_jobs=1,
                    verbosity=0,
                )
            ),
            dependency_name="xgboost",
            dependency_error=XGBOOST_IMPORT_ERROR,
        ),
        ModelSpec(
            name="LightGBM",
            builder=(
                None
                if LGBMRegressor is None
                else lambda _mae=use_mae: LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=5,
                    random_state=42,
                    verbose=-1,
                    objective="regression_l1" if _mae else "regression",
                )
            ),
            dependency_name="lightgbm",
            dependency_error=LIGHTGBM_IMPORT_ERROR,
        ),
    ]
    return specs


def _build_preprocessor(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[ColumnTransformer | None, pd.DataFrame, pd.DataFrame, list[str], list[str], list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    numeric_features: list[str] = []
    categorical_features: list[str] = []
    train_model_df = pd.DataFrame(index=train_df.index)
    test_model_df = pd.DataFrame(index=test_df.index)

    for col in feature_columns:
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

    selected_features = numeric_features + categorical_features
    if not selected_features:
        return None, train_model_df, test_model_df, numeric_features, categorical_features, warnings

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_features,
            )
        )
    return (
        ColumnTransformer(transformers=transformers),
        train_model_df[selected_features],
        test_model_df[selected_features],
        numeric_features,
        categorical_features,
        warnings,
    )


def _prepare_target(strategy_name: str, train_target: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n = len(train_target)
    metadata: dict[str, Any] = {"strategy_name": strategy_name}
    mask_all = np.ones(n, dtype=bool)

    if strategy_name == "raw":
        return train_target.copy(), mask_all, metadata

    if strategy_name == "log1p":
        return np.log1p(np.clip(train_target, a_min=0.0, a_max=None)), mask_all, metadata

    if strategy_name == "log1p_drop_outliers":
        logged_target = np.log1p(np.clip(train_target, a_min=0.0, a_max=None))
        lower, upper = _iqr_bounds(logged_target)
        metadata.update({"lower_bound": lower, "upper_bound": upper})
        if lower is None or upper is None:
            return logged_target, mask_all, metadata
        outlier_mask = (logged_target < lower) | (logged_target > upper)
        clipped_count = int(outlier_mask.sum())
        metadata.update({
            "clipped_count": clipped_count,
            "clipped_ratio": float(clipped_count / n) if n > 0 else 0.0,
        })
        winsorized = np.clip(logged_target, lower, upper)
        return winsorized, mask_all, metadata

    raise ValueError(f"Unknown target strategy '{strategy_name}'")


def _restore_predictions(strategy_name: str, predictions: np.ndarray) -> np.ndarray:
    restored = predictions.copy()
    if strategy_name in {"log1p", "log1p_drop_outliers"}:
        restored = np.expm1(restored)
    return np.clip(restored, a_min=0.0, a_max=None)


def _clean_feature_name(name: str) -> str:
    cleaned = (
        str(name)
        .replace("num__", "")
        .replace("cat__", "")
        .replace("encoder__", "")
        .replace("imputer__", "")
    )
    if "_" in cleaned and cleaned.split("_", 1)[0] in {"Owner", "Type", "Quality", "Size"}:
        prefix, suffix = cleaned.split("_", 1)
        return f"{prefix}={suffix}"
    return cleaned


def _extract_feature_effects(model: Pipeline) -> list[dict[str, Any]]:
    preprocessor = model.named_steps["preprocess"]
    fitted_regressor = model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = getattr(fitted_regressor, "feature_importances_", None)
    if importances is None or len(importances) != len(feature_names):
        return []

    effects: list[dict[str, Any]] = []
    pairs = zip(feature_names.tolist(), importances.tolist(), strict=False)
    for name, importance in sorted(pairs, key=lambda item: abs(float(item[1])), reverse=True)[:12]:
        effects.append({"feature": _clean_feature_name(name), "coefficient": float(importance)})
    return effects


def _segment_key_series(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    normalized = pd.DataFrame(index=df.index)
    for column in columns:
        normalized[column] = df[column].astype("string").fillna("<MISSING>")
    if len(columns) == 1:
        return normalized[columns[0]]
    return normalized.apply(lambda row: tuple(row[col] for col in columns), axis=1)


def _segment_label(value: Any, columns: tuple[str, ...]) -> str:
    if isinstance(value, tuple):
        return " | ".join(str(item) for item in value)
    if len(columns) == 1:
        return str(value)
    return str(value)


def _representative_segments(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: tuple[str, ...],
) -> set[Any]:
    train_counts = _segment_key_series(train_df, columns).value_counts(dropna=False)
    test_counts = _segment_key_series(test_df, columns).value_counts(dropna=False)
    representative = set()
    for key in set(train_counts.index.tolist()) | set(test_counts.index.tolist()):
        if int(train_counts.get(key, 0)) >= REPRESENTATIVE_TRAIN_MIN and int(test_counts.get(key, 0)) >= REPRESENTATIVE_TEST_MIN:
            representative.add(key)
    return representative


def _group_segment_labels(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    representative: set[Any],
) -> pd.Series:
    raw_keys = _segment_key_series(df, columns)
    labels = raw_keys.map(lambda value: _segment_label(value, columns) if value in representative else "Other")
    return labels.astype("string")


def _build_segment_heuristic(
    name: str,
    family_key: str,
    family_label: str,
    columns: tuple[str, ...],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_target: np.ndarray,
    test_target: np.ndarray,
    baseline_prediction: float,
    baseline_test_vector: np.ndarray,
) -> tuple[dict[str, Any], pd.Series, pd.Series, dict[str, float]]:
    representative = _representative_segments(train_df, test_df, columns)
    grouped_train = _group_segment_labels(train_df, columns, representative)
    grouped_test = _group_segment_labels(test_df, columns, representative)
    train_with_target = train_df.copy()
    train_with_target[TARGET_COL] = train_target
    train_with_target["_segment_label"] = grouped_train
    medians = train_with_target.groupby("_segment_label", dropna=False)[TARGET_COL].median().to_dict()

    if "Other" not in medians:
        medians["Other"] = float(np.median(train_target))

    train_predictions = grouped_train.map(lambda label: float(medians.get(str(label), baseline_prediction))).to_numpy(dtype=float)
    test_predictions = grouped_test.map(lambda label: float(medians.get(str(label), baseline_prediction))).to_numpy(dtype=float)

    segment_medians = [
        {"segment": str(seg), "median": float(med), "count": int(train_with_target[train_with_target["_segment_label"] == seg].shape[0])}
        for seg, med in sorted(medians.items(), key=lambda item: item[0])
    ]

    payload = {
        "model_name": name,
        "family_key": family_key,
        "family_label": family_label,
        "rule_summary": (
            f"En lugar de un modelo complejo, se predice la mediana histórica de DaysInDeposit "
            f"por segmento de {family_label}. Segmentos con pocas observaciones se agrupan como 'Otros'."
        ),
        "train_metrics": _metric_summary(train_target, train_predictions, np.full_like(train_target, baseline_prediction)),
        "metrics": _metric_summary(test_target, test_predictions, baseline_test_vector),
        "predictions": _prediction_rows(test_df, test_target, test_predictions, WEEK_COL),
        "tier_usage": [{"source": family_label, "count": int(len(test_predictions))}],
        "segment_medians": segment_medians,
    }
    return payload, grouped_train, grouped_test, medians


def _segment_rows(
    family_key: str,
    family_label: str,
    grouping_type: str,
    train_labels: pd.Series,
    test_labels: pd.Series,
    test_target: np.ndarray,
    regression_predictions: np.ndarray,
    heuristic_predictions: np.ndarray,
    baseline_predictions: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    train_counts = train_labels.value_counts(dropna=False)
    test_counts = test_labels.value_counts(dropna=False)
    ordered_labels = test_counts.sort_values(ascending=False).index.tolist()
    if "Other" in ordered_labels:
        ordered_labels = [label for label in ordered_labels if label != "Other"] + ["Other"]

    for label in ordered_labels:
        mask = test_labels == label
        if int(mask.sum()) == 0:
            continue
        actual_segment = test_target[mask.to_numpy()]
        regression_segment = regression_predictions[mask.to_numpy()]
        heuristic_segment = heuristic_predictions[mask.to_numpy()]
        baseline_segment = baseline_predictions[mask.to_numpy()]
        rows.append(
            {
                "segment": str(label),
                "train_count": int(train_counts.get(label, 0)),
                "test_count": int(test_counts.get(label, 0)),
                "actual_mean": float(np.mean(actual_segment)),
                "actual_median": float(np.median(actual_segment)),
                "regression_mae": float(mean_absolute_error(actual_segment, regression_segment)),
                "heuristic_mae": float(mean_absolute_error(actual_segment, heuristic_segment)),
                "baseline_mae": float(mean_absolute_error(actual_segment, baseline_segment)),
            }
        )

    return {
        "family_key": family_key,
        "family_label": family_label,
        "grouping_type": grouping_type,
        "rows": rows,
    }


def _top_level_model_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_by_model: dict[str, dict[str, Any]] = {}
    for result in results:
        current = best_by_model.get(result["model_name"])
        if current is None or float(result["metrics"]["mae"]) < float(current["metrics"]["mae"]):
            best_by_model[result["model_name"]] = result
    ordered = sorted(best_by_model.values(), key=lambda item: float(item["metrics"]["mae"]))
    return ordered


def _comparison_entry(result: dict[str, Any], *, include_strategy: bool) -> dict[str, Any]:
    payload = {
        "model_name": result["model_name"],
        "metrics": result["metrics"],
        "notes": result.get("notes", []),
    }
    if include_strategy:
        payload["strategy_name"] = result.get("strategy_name")
        payload["strategy_label"] = result.get("strategy_label")
    return payload


def compute_temporal_ml_overview(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    week_col: str = WEEK_COL,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    # Total steps: 2 setup + 3 strategies × 5 models + 1 heuristics + 1 classification + 1 done = ~21
    _total_steps = 21

    def _progress(step: int, msg: str) -> None:
        if on_progress is not None:
            on_progress(step, _total_steps, msg)

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

    _progress(1, "Split temporal completado")

    train_target = train_df[target_col].to_numpy(dtype=float)
    test_target = test_df[target_col].to_numpy(dtype=float)
    baseline_prediction = float(np.median(train_target))
    baseline_train_vector = np.full_like(train_target, baseline_prediction)
    baseline_test_vector = np.full_like(test_target, baseline_prediction)
    target_transformation_diagnostics = _build_target_transformation_diagnostics(train_target)

    target_skew = float(pd.Series(train_target).skew()) if len(train_target) > 2 else 0.0
    if abs(target_skew) >= 1.0:
        warnings.append(
            {
                "code": "target_high_skew",
                "severity": "info",
                "column": target_col,
                "message": f"Target is highly skewed in training data (skewness={target_skew:.2f}).",
                "suggestion": "Compare raw target against log1p, post-log normalization and robust target strategies.",
            }
        )

    q1 = float(pd.Series(train_target).quantile(0.25))
    q3 = float(pd.Series(train_target).quantile(0.75))
    iqr = q3 - q1
    if not np.isclose(iqr, 0.0):
        upper = q3 + 1.5 * iqr
        outlier_ratio = float((pd.Series(train_target) > upper).mean())
        if outlier_ratio > 0.05:
            warnings.append(
                {
                    "code": "target_outlier_pressure",
                    "severity": "info",
                    "column": target_col,
                    "message": f"Training target has {outlier_ratio:.2%} upper-tail outliers by IQR rule.",
                    "suggestion": "Use winsorization or post-log outlier normalization as benchmark instead of dropping rows by default.",
                }
            )

    candidate_features = [col for col in analytical_columns(data, keep_columns={target_col}) if col != target_col]
    preprocessor, train_features, test_features, numeric_features, categorical_features, prep_warnings = _build_preprocessor(
        train_df,
        test_df,
        candidate_features,
    )
    warnings.extend(prep_warnings)
    _progress(2, "Preprocessor construido")

    if preprocessor is None:
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

    regression_runs: list[dict[str, Any]] = []
    unavailable_model_results: list[dict[str, Any]] = []
    preprocessing_benchmarks: list[dict[str, Any]] = [
        {
            "model_name": "Global Median",
            "strategy_name": "baseline",
            "strategy_label": STRATEGY_LABELS["baseline"],
            "metrics": _metric_summary(test_target, baseline_test_vector, baseline_test_vector),
            "available": True,
            "notes": ["Baseline robusto que siempre predice la mediana historica de train."],
        }
    ]

    # Report unavailable models once (strategy doesn't affect availability).
    for spec in _regression_model_specs("raw"):
        if spec.builder is None:
            warning = _dependency_warning(spec)
            warnings.append(warning)
            notes = [warning["message"], warning["suggestion"]]
            preprocessing_benchmarks.append(
                _unavailable_benchmark_row(
                    spec,
                    strategy_name="unavailable",
                    strategy_label="No disponible",
                    notes=notes,
                )
            )
            unavailable_model_results.append(
                _unavailable_model_result(
                    spec,
                    strategy_name="unavailable",
                    strategy_label="No disponible",
                    notes=notes,
                )
            )

    # Track per-model success across all strategies for failure reporting.
    model_succeeded: dict[str, bool] = {}
    model_failures: dict[str, list[str]] = {}
    _step_counter = 2  # steps 1-2 already done (split + preprocessor)

    for strategy_name in ("raw", "log1p", "log1p_drop_outliers"):
        transformed_target, train_mask, strategy_metadata = _prepare_target(strategy_name, train_target)
        strategy_notes: list[str] = []
        if strategy_name == "log1p":
            strategy_notes.append("Predicciones revertidas a escala original con expm1.")
        if strategy_name == "log1p_drop_outliers":
            clipped = strategy_metadata.get("clipped_count", 0)
            clipped_ratio = strategy_metadata.get("clipped_ratio", 0.0)
            strategy_notes.append(
                f"Se winsorizaron {clipped} valores outlier del train ({clipped_ratio:.1%}) "
                "al rango IQR en escala log. Predicciones revertidas con expm1."
            )

        for spec in _regression_model_specs(strategy_name):
            if spec.builder is None:
                _step_counter += 1
                continue

            model_succeeded.setdefault(spec.name, False)
            model_failures.setdefault(spec.name, [])
            notes = list(strategy_notes)

            try:
                fit_features = train_features[train_mask] if not train_mask.all() else train_features
                model = Pipeline([("preprocess", preprocessor), ("regressor", spec.builder())])
                model.fit(fit_features, transformed_target)

                predicted_test = _restore_predictions(strategy_name, model.predict(test_features))
                predicted_train = _restore_predictions(strategy_name, model.predict(train_features))

                result = {
                    "model_name": spec.name,
                    "strategy_name": strategy_name,
                    "strategy_label": STRATEGY_LABELS[strategy_name],
                    "metrics": _metric_summary(test_target, predicted_test, baseline_test_vector),
                    "train_metrics": _metric_summary(train_target, predicted_train, baseline_train_vector),
                    "predictions": _prediction_rows(test_df, test_target, predicted_test, week_col),
                    "feature_effects": _extract_feature_effects(model),
                    "tree_structure": None,
                    "notes": notes,
                }

                fitted_regressor = model.named_steps["regressor"]
                if isinstance(fitted_regressor, DecisionTreeRegressor):
                    feature_names = model.named_steps["preprocess"].get_feature_names_out().tolist()
                    result["tree_structure"] = _extract_tree_structure(fitted_regressor, feature_names)

                regression_runs.append(result)
                model_succeeded[spec.name] = True
                _step_counter += 1
                _progress(_step_counter, f"Entrenado {spec.name} · {STRATEGY_LABELS[strategy_name]}")
                preprocessing_benchmarks.append(
                    {
                        "model_name": spec.name,
                        "strategy_name": strategy_name,
                        "strategy_label": STRATEGY_LABELS[strategy_name],
                        "metrics": result["metrics"],
                        "available": True,
                        "notes": notes,
                    }
                )
            except Exception as exc:
                warnings.append(
                    {
                        "code": "model_training_failed",
                        "severity": "warning",
                        "column": None,
                        "message": f"{spec.name} with {STRATEGY_LABELS[strategy_name]} failed during training.",
                        "suggestion": str(exc),
                    }
                )
                model_failures[spec.name].append(f"{STRATEGY_LABELS[strategy_name]}: {exc}")
                _step_counter += 1
                _progress(_step_counter, f"Entrenado {spec.name} · {STRATEGY_LABELS[strategy_name]}")

    # Add failure rows for models where no strategy succeeded.
    for spec in _regression_model_specs("raw"):
        if spec.builder is not None and not model_succeeded.get(spec.name, False):
            notes = ["Ninguna estrategia completó entrenamiento para este modelo."] + model_failures.get(spec.name, [])[:3]
            preprocessing_benchmarks.append(
                _unavailable_benchmark_row(
                    spec,
                    strategy_name="failed",
                    strategy_label="Sin resultado",
                    notes=notes,
                )
            )
            unavailable_model_results.append(
                _unavailable_model_result(
                    spec,
                    strategy_name="failed",
                    strategy_label="Sin resultado",
                    notes=notes,
                )
            )

    heuristic_models: list[dict[str, Any]] = []
    segment_artifacts: dict[str, dict[str, Any]] = {}

    baseline_heuristic = {
        "model_name": "Global Median",
        "family_key": "global",
        "family_label": "Global",
        "rule_summary": "Predice siempre la mediana historica de entrenamiento.",
        "train_metrics": _metric_summary(train_target, baseline_train_vector, baseline_train_vector),
        "metrics": _metric_summary(test_target, baseline_test_vector, baseline_test_vector),
        "predictions": _prediction_rows(test_df, test_target, baseline_test_vector, week_col),
        "tier_usage": [{"source": "Global Median", "count": int(len(test_target))}],
    }
    heuristic_models.append(baseline_heuristic)

    simple_heuristics: list[dict[str, Any]] = []
    combo_heuristics: list[dict[str, Any]] = []

    for columns, family_key, family_label, display_label in SIMPLE_SEGMENT_SPECS:
        heuristic, grouped_train, grouped_test, medians = _build_segment_heuristic(
            name=f"Heuristic Median - {family_label}",
            family_key=family_key,
            family_label=display_label,
            columns=columns,
            train_df=train_df,
            test_df=test_df,
            train_target=train_target,
            test_target=test_target,
            baseline_prediction=baseline_prediction,
            baseline_test_vector=baseline_test_vector,
        )
        heuristic_models.append(heuristic)
        simple_heuristics.append(heuristic)
        segment_artifacts[family_key] = {
            "train_labels": grouped_train,
            "test_labels": grouped_test,
            "medians": medians,
            "columns": columns,
            "family_label": family_label,
            "grouping_type": "simple",
        }

    for columns, family_key, family_label, display_label in COMBO_SEGMENT_SPECS:
        heuristic, grouped_train, grouped_test, medians = _build_segment_heuristic(
            name=f"Heuristic Median - {family_label}",
            family_key=family_key,
            family_label=display_label,
            columns=columns,
            train_df=train_df,
            test_df=test_df,
            train_target=train_target,
            test_target=test_target,
            baseline_prediction=baseline_prediction,
            baseline_test_vector=baseline_test_vector,
        )
        heuristic_models.append(heuristic)
        combo_heuristics.append(heuristic)
        segment_artifacts[family_key] = {
            "train_labels": grouped_train,
            "test_labels": grouped_test,
            "medians": medians,
            "columns": columns,
            "family_label": family_label,
            "grouping_type": "combination",
        }

    best_simple = min(simple_heuristics, key=lambda item: float(item["train_metrics"]["mae"]), default=None)
    best_combo = min(combo_heuristics, key=lambda item: float(item["train_metrics"]["mae"]), default=None)

    if best_simple is not None:
        simple_artifact = segment_artifacts[best_simple["family_key"]]
        simple_train_labels = simple_artifact["train_labels"]
        simple_test_labels = simple_artifact["test_labels"]
        simple_medians = simple_artifact["medians"]
    else:
        simple_train_labels = pd.Series(["Other"] * len(train_df), index=train_df.index, dtype="string")
        simple_test_labels = pd.Series(["Other"] * len(test_df), index=test_df.index, dtype="string")
        simple_medians = {"Other": baseline_prediction}

    if best_combo is not None:
        combo_artifact = segment_artifacts[best_combo["family_key"]]
        combo_train_labels = combo_artifact["train_labels"]
        combo_test_labels = combo_artifact["test_labels"]
        combo_medians = combo_artifact["medians"]
    else:
        combo_train_labels = pd.Series(["Other"] * len(train_df), index=train_df.index, dtype="string")
        combo_test_labels = pd.Series(["Other"] * len(test_df), index=test_df.index, dtype="string")
        combo_medians = {"Other": baseline_prediction}

    hierarchical_train_predictions: list[float] = []
    hierarchical_train_usage = {"Mejor combinación": 0, "Mejor segmento simple": 0, "Mediana global": 0}
    for combo_label, simple_label in zip(combo_train_labels.tolist(), simple_train_labels.tolist(), strict=False):
        if combo_label != "Other" and combo_label in combo_medians:
            hierarchical_train_predictions.append(float(combo_medians[combo_label]))
            hierarchical_train_usage["Mejor combinación"] += 1
        elif simple_label != "Other" and simple_label in simple_medians:
            hierarchical_train_predictions.append(float(simple_medians[simple_label]))
            hierarchical_train_usage["Mejor segmento simple"] += 1
        else:
            hierarchical_train_predictions.append(float(baseline_prediction))
            hierarchical_train_usage["Mediana global"] += 1

    hierarchical_test_predictions: list[float] = []
    hierarchical_test_usage = {"Mejor combinación": 0, "Mejor segmento simple": 0, "Mediana global": 0}
    for combo_label, simple_label in zip(combo_test_labels.tolist(), simple_test_labels.tolist(), strict=False):
        if combo_label != "Other" and combo_label in combo_medians:
            hierarchical_test_predictions.append(float(combo_medians[combo_label]))
            hierarchical_test_usage["Mejor combinación"] += 1
        elif simple_label != "Other" and simple_label in simple_medians:
            hierarchical_test_predictions.append(float(simple_medians[simple_label]))
            hierarchical_test_usage["Mejor segmento simple"] += 1
        else:
            hierarchical_test_predictions.append(float(baseline_prediction))
            hierarchical_test_usage["Mediana global"] += 1

    hierarchical_train_array = np.asarray(hierarchical_train_predictions, dtype=float)
    hierarchical_test_array = np.asarray(hierarchical_test_predictions, dtype=float)
    heuristic_models.append(
        {
            "model_name": "Hierarchical Backoff",
            "family_key": "hierarchical_backoff",
            "family_label": "Hierarchical Backoff",
            "rule_summary": (
                f"Cascada jerárquica: primero intenta predecir con la combinación "
                f"({best_combo['family_label'] if best_combo is not None else 'ninguna combinación representativa'}), "
                f"si el segmento no es representativo baja al segmento simple "
                f"({best_simple['family_label'] if best_simple is not None else 'ningún segmento simple representativo'}), "
                "y si tampoco aplica usa la mediana global."
            ),
            "train_metrics": _metric_summary(train_target, hierarchical_train_array, baseline_train_vector),
            "metrics": _metric_summary(test_target, hierarchical_test_array, baseline_test_vector),
            "predictions": _prediction_rows(test_df, test_target, hierarchical_test_array, week_col),
            "tier_usage": [
                {"source": source, "count": int(count)}
                for source, count in hierarchical_test_usage.items()
            ],
        }
    )

    _step_counter += 1
    _progress(_step_counter, "Heurísticas calculadas")

    heuristic_models = sorted(heuristic_models, key=lambda item: float(item["metrics"]["mae"]))
    best_heuristic = heuristic_models[0] if heuristic_models else None
    best_regression = min(regression_runs, key=lambda item: float(item["metrics"]["mae"]), default=None)

    segment_reports: list[dict[str, Any]] = []
    if best_regression is not None and best_heuristic is not None:
        best_regression_predictions = np.asarray([row["predicted"] for row in best_regression["predictions"]], dtype=float)
        best_heuristic_predictions = np.asarray([row["predicted"] for row in best_heuristic["predictions"]], dtype=float)
        for columns, family_key, family_label, _display_label in SIMPLE_SEGMENT_SPECS + COMBO_SEGMENT_SPECS:
            artifact = segment_artifacts[family_key]
            segment_reports.append(
                _segment_rows(
                    family_key=family_key,
                    family_label=family_label,
                    grouping_type=artifact["grouping_type"],
                    train_labels=artifact["train_labels"],
                    test_labels=artifact["test_labels"],
                    test_target=test_target,
                    regression_predictions=best_regression_predictions,
                    heuristic_predictions=best_heuristic_predictions,
                    baseline_predictions=baseline_test_vector,
                )
            )

    strategy_comparison = None
    if best_regression is not None and best_heuristic is not None:
        mae_gap = float(best_heuristic["metrics"]["mae"] - best_regression["metrics"]["mae"])
        winner = "tie"
        if mae_gap > 1e-9:
            winner = "regression"
        elif mae_gap < -1e-9:
            winner = "heuristic"
        strategy_comparison = {
            "winner": winner,
            "mae_gap": mae_gap,
            "best_regression": _comparison_entry(best_regression, include_strategy=True),
            "best_heuristic": _comparison_entry(best_heuristic, include_strategy=False),
            "narrative": (
                "La mejor regresion supera a la mejor heuristica en MAE."
                if winner == "regression"
                else "La mejor heuristica iguala o supera a la mejor regresion en MAE."
                if winner == "heuristic"
                else "La mejor regresion y la mejor heuristica quedan virtualmente empatadas en MAE."
            ),
        }

    warnings.append(
        {
            "code": "temporal_holdout",
            "severity": "info",
            "column": week_col,
            "message": f"Temporal holdout uses week {holdout_week} as test set.",
            "suggestion": "Keep the last observed week as untouched validation data for comparisons.",
        }
    )

    _progress(_total_steps - 2, "Entrenando clasificación por bandas")

    from app.stats.ml_classification import compute_priority_classification

    priority_classification = compute_priority_classification(df, target_col, week_col)

    _progress(_total_steps, "Análisis completo")

    model_results = _top_level_model_results(regression_runs) + unavailable_model_results
    model_results = sorted(
        model_results,
        key=lambda item: (
            item["metrics"].get("mae") is None,
            float(item["metrics"]["mae"]) if item["metrics"].get("mae") is not None else float("inf"),
            item["model_name"],
        ),
    )

    return {
        "target_present": True,
        "model_built": bool(regression_runs or heuristic_models),
        "target_column": target_col,
        "split": {
            "train_weeks": [str(week) for week in unique_weeks[:-1]],
            "test_weeks": [str(holdout_week)],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "feature_columns": numeric_features + categorical_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "models": model_results,
        "warnings": warnings,
        "preprocessing_benchmarks": preprocessing_benchmarks,
        "segment_reports": segment_reports,
        "heuristic_models": heuristic_models,
        "strategy_comparison": strategy_comparison,
        "target_transformation_diagnostics": target_transformation_diagnostics,
        "priority_classification": priority_classification,
    }
