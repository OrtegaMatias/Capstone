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

SIMPLE_SEGMENT_SPECS: list[tuple[tuple[str, ...], str, str]] = [
    (("Type",), "type", "Type"),
    (("Quality",), "quality", "Quality"),
    (("Owner",), "owner", "Owner"),
    (("Size",), "size", "Size"),
]

COMBO_SEGMENT_SPECS: list[tuple[tuple[str, ...], str, str]] = [
    (("Type", "Quality"), "type_quality", "Type x Quality"),
    (("Owner", "Size"), "owner_size", "Owner x Size"),
    (("Owner", "Type"), "owner_type", "Owner x Type"),
]

STRATEGY_LABELS = {
    "raw": "Raw target",
    "log1p": "Log1p target",
    "log1p_outlier_norm": "Log1p + outlier normalization",
    "winsor_iqr": "Winsor IQR",
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
        "learning_sections": [],
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
    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": r2,
        "medae": float(medae),
        "baseline_mae": float(baseline_mae),
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
        log_norm_values = log_values.copy()
    else:
        log_norm_values = np.clip(log_values, a_min=log_lower, a_max=log_upper)
    restored_values = np.expm1(log_norm_values)

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
            "notes": ["Compresión logarítmica del target antes de cualquier normalización residual."],
        },
        {
            "step_key": "log1p_outlier_norm",
            "step_label": "Log1p + outlier normalization",
            "scale": "log",
            "stats": _boxplot_stats(log_norm_values),
            "notes": [
                (
                    "Se aplica capping IQR sobre la escala logarítmica para normalizar outliers "
                    "que sobreviven después de log1p."
                )
            ],
        },
        {
            "step_key": "log1p_outlier_norm_restored",
            "step_label": "Log1p + outlier normalization (restored days)",
            "scale": "days",
            "stats": _boxplot_stats(restored_values),
            "notes": ["La misma transformación reexpresada en días para comparar antes/después en escala original."],
        },
    ]

    return {
        "scope": "train_only",
        "boxplot_data": [
            {
                "feature": "Escala original (antes vs después)",
                "groups": [
                    {"group": "Raw target", "values": raw_values.tolist()},
                    {"group": "Log1p + norm (restored days)", "values": restored_values.tolist()},
                ],
            },
            {
                "feature": "Espacio logarítmico",
                "groups": [
                    {"group": "Log1p target", "values": log_values.tolist()},
                    {"group": "Log1p + outlier normalization", "values": log_norm_values.tolist()},
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


def _regression_model_specs() -> list[ModelSpec]:
    specs = [
        ModelSpec(
            name="Decision Tree",
            builder=lambda: DecisionTreeRegressor(max_depth=5, random_state=42),
        ),
        ModelSpec(
            name="Random Forest",
            builder=lambda: RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        ModelSpec(
            name="CatBoost",
            builder=(
                None
                if CatBoostRegressor is None
                else lambda: CatBoostRegressor(
                    iterations=100,
                    depth=5,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
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
                else lambda: XGBRegressor(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
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
                else lambda: LGBMRegressor(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=5,
                    random_state=42,
                    verbose=-1,
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


def _prepare_target(strategy_name: str, train_target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {"strategy_name": strategy_name}
    if strategy_name == "raw":
        return train_target.copy(), metadata

    if strategy_name == "log1p":
        return np.log1p(np.clip(train_target, a_min=0.0, a_max=None)), metadata

    if strategy_name == "log1p_outlier_norm":
        logged_target = np.log1p(np.clip(train_target, a_min=0.0, a_max=None))
        lower, upper = _iqr_bounds(logged_target)
        metadata.update({"lower_bound": lower, "upper_bound": upper, "normalized_after": "log1p"})
        if lower is None or upper is None:
            return logged_target, metadata
        return np.clip(logged_target, a_min=lower, a_max=upper), metadata

    if strategy_name == "winsor_iqr":
        q1 = float(np.quantile(train_target, 0.25))
        q3 = float(np.quantile(train_target, 0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        metadata.update({"lower_bound": lower, "upper_bound": upper})
        return np.clip(train_target, a_min=lower, a_max=upper), metadata

    raise ValueError(f"Unknown target strategy '{strategy_name}'")


def _restore_predictions(strategy_name: str, predictions: np.ndarray) -> np.ndarray:
    restored = predictions.copy()
    if strategy_name in {"log1p", "log1p_outlier_norm"}:
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

    payload = {
        "model_name": name,
        "family_key": family_key,
        "family_label": family_label,
        "rule_summary": f"Predice la mediana historica por {family_label} y agrupa segmentos no representativos en Other.",
        "train_metrics": _metric_summary(train_target, train_predictions, np.full_like(train_target, baseline_prediction)),
        "metrics": _metric_summary(test_target, test_predictions, baseline_test_vector),
        "predictions": _prediction_rows(test_df, test_target, test_predictions, WEEK_COL),
        "tier_usage": [{"source": family_label, "count": int(len(test_predictions))}],
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


def _learning_sections(
    *,
    train_target: np.ndarray,
    full_target: pd.Series,
    warnings: list[dict[str, Any]],
    best_regression: dict[str, Any] | None,
    best_heuristic: dict[str, Any] | None,
    target_transformation_diagnostics: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    mean_value = float(np.mean(full_target))
    median_value = float(np.median(full_target))
    skew_value = float(full_target.skew()) if len(full_target) > 2 else 0.0
    q1 = float(full_target.quantile(0.25))
    q3 = float(full_target.quantile(0.75))
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    outlier_ratio = float((full_target > upper).mean()) if not np.isclose(iqr, 0.0) else 0.0

    boosting_status = []
    dependency_warnings = [warning for warning in warnings if warning["code"].endswith("_unavailable")]
    if dependency_warnings:
        boosting_status.extend(
            [
                "Los boosting externos quedaron preparados como dependencias opcionales.",
                "Si faltan librerías, el benchmark sigue respondiendo y reporta la ausencia como warning.",
            ]
        )
    else:
        boosting_status.append("Los boosting externos quedaron disponibles dentro del benchmark de semana 2.")

    comparison_bullets = []
    if best_regression is not None:
        comparison_bullets.append(
            f"Mejor regresion: {best_regression['model_name']} con {best_regression.get('strategy_label', 'estrategia n/a')} (MAE={best_regression['metrics']['mae']:.2f})."
        )
    if best_heuristic is not None:
        comparison_bullets.append(
            f"Mejor heuristica: {best_heuristic['model_name']} (MAE={best_heuristic['metrics']['mae']:.2f})."
        )
    if not comparison_bullets:
        comparison_bullets.append("No hubo suficientes modelos para comparar estrategias en esta ejecucion.")

    diagnostics_by_step = {
        step["step_key"]: step["stats"] for step in (target_transformation_diagnostics or {}).get("steps", [])
    }
    raw_stats = diagnostics_by_step.get("raw", {})
    log_stats = diagnostics_by_step.get("log1p", {})
    normalized_log_stats = diagnostics_by_step.get("log1p_outlier_norm", {})

    return [
        {
            "slug": "asimetria-dwell-time",
            "title": "Asimetria del dwell time",
            "summary": "DaysInDeposit muestra cola larga y una diferencia grande entre media y mediana, por lo que conviene evitar lecturas basadas solo en el promedio.",
            "bullets": [
                f"Media={mean_value:.2f}, mediana={median_value:.2f}, skewness={skew_value:.2f}.",
                "Cuando la cola es larga, RMSE y la media reaccionan fuerte a pocos casos extremos.",
                "En estos escenarios el MedAE y las medianas segmentadas ayudan a leer estabilidad operativa.",
            ],
        },
        {
            "slug": "transformacion-log",
            "title": "Transformacion logaritmica",
            "summary": "La transformacion log1p comprime valores muy altos del target sin perder el orden de magnitud y luego revierte las predicciones a dias reales.",
            "bullets": [
                "Se entrena con log1p(target) y se evalua siempre en escala original para no distorsionar la comparacion.",
                (
                    f"En train, la skewness pasa de {raw_stats.get('skew', 0.0):.2f} a "
                    f"{log_stats.get('skew', 0.0):.2f} después de log1p."
                    if raw_stats and log_stats
                    else "La escala logarítmica reduce la asimetría y compacta la cola alta."
                ),
                "Sirve cuando el costo de la cola domina el ajuste y hace inestable la regresion directa.",
                "No reemplaza la interpretacion del negocio; solo cambia la geometria del aprendizaje.",
            ],
        },
        {
            "slug": "outliers",
            "title": "Outliers y normalizacion residual",
            "summary": "Los outliers no se eliminan por defecto; se comparan contra variantes robustas calculadas solo con train para evitar leakage temporal.",
            "bullets": [
                f"Con la regla IQR, aproximadamente {outlier_ratio:.2%} del target supera el limite superior de {upper:.2f} dias.",
                "La winsorizacion ajusta el target de entrenamiento, pero no toca el holdout.",
                (
                    f"Después de log1p, la proporción residual de outliers pasa de {log_stats.get('outlier_ratio', 0.0):.2%} "
                    f"a {normalized_log_stats.get('outlier_ratio', 0.0):.2%} tras la normalización residual."
                    if log_stats and normalized_log_stats
                    else "La normalización residual opera sobre la escala logarítmica para reducir extremos persistentes."
                ),
                "Esto permite medir si la robustez mejora sin borrar observaciones del fenomeno.",
            ],
        },
        {
            "slug": "boosting",
            "title": "Boosting para benchmark",
            "summary": "CatBoost, XGBoost y LightGBM entran como comparadores de arboles potenciados frente a baselines mas simples.",
            "bullets": boosting_status,
        },
        {
            "slug": "heuristicas-vs-regresiones",
            "title": "Heuristicas vs regresiones",
            "summary": "La comparacion no solo mide error: ayuda a decidir si el problema necesita una regla interpretable o un modelo mas flexible.",
            "bullets": comparison_bullets
            + [
                "Las heuristicas por segmento sirven como benchmark interpretable y como puente hacia decisiones operativas.",
                "Si una heuristica se acerca al mejor regressor, la complejidad adicional puede no justificar el costo.",
            ],
        },
        {
            "slug": "metaheuristicas",
            "title": "Puente hacia heuristicas y metaheuristicas",
            "summary": "Semana 2 no implementa un solver metaheuristico, pero deja insumos concretos para construir reglas, funciones objetivo y vecindarios en semanas posteriores.",
            "bullets": [
                "Los segmentos con mayor error pueden transformarse en penalizaciones o prioridades dentro de una futura funcion objetivo.",
                "Las medianas por segmento ofrecen semillas interpretable para heuristicas constructivas.",
                "Los boosters ayudan a detectar no linealidades que luego se pueden traducir a restricciones o scores de busqueda.",
            ],
        },
    ]


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

    for spec in _regression_model_specs():
        if spec.builder is None:
            warnings.append(_dependency_warning(spec))
            continue

        for strategy_name in ("raw", "log1p", "log1p_outlier_norm", "winsor_iqr"):
            transformed_target, strategy_metadata = _prepare_target(strategy_name, train_target)
            notes: list[str] = []
            if strategy_name == "winsor_iqr":
                notes.append(
                    f"Límites train-only: [{strategy_metadata['lower_bound']:.2f}, {strategy_metadata['upper_bound']:.2f}]"
                )
            if strategy_name == "log1p":
                notes.append("Predicciones revertidas a escala original con expm1.")
            if strategy_name == "log1p_outlier_norm":
                lower_bound = strategy_metadata.get("lower_bound")
                upper_bound = strategy_metadata.get("upper_bound")
                if lower_bound is not None and upper_bound is not None:
                    notes.append(
                        f"Normalización residual en escala log con límites train-only: [{lower_bound:.2f}, {upper_bound:.2f}]"
                    )
                notes.append("Predicciones revertidas a escala original con expm1.")

            try:
                model = Pipeline([("preprocess", preprocessor), ("regressor", spec.builder())])
                model.fit(train_features, transformed_target)

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

    for columns, family_key, family_label in SIMPLE_SEGMENT_SPECS:
        heuristic, grouped_train, grouped_test, medians = _build_segment_heuristic(
            name=f"Heuristic Median - {family_label}",
            family_key=family_key,
            family_label=family_label,
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

    for columns, family_key, family_label in COMBO_SEGMENT_SPECS:
        heuristic, grouped_train, grouped_test, medians = _build_segment_heuristic(
            name=f"Heuristic Median - {family_label}",
            family_key=family_key,
            family_label=family_label,
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
    hierarchical_train_usage = {"Best Combination": 0, "Best Simple Segment": 0, "Global Median": 0}
    for combo_label, simple_label in zip(combo_train_labels.tolist(), simple_train_labels.tolist(), strict=False):
        if combo_label != "Other" and combo_label in combo_medians:
            hierarchical_train_predictions.append(float(combo_medians[combo_label]))
            hierarchical_train_usage["Best Combination"] += 1
        elif simple_label != "Other" and simple_label in simple_medians:
            hierarchical_train_predictions.append(float(simple_medians[simple_label]))
            hierarchical_train_usage["Best Simple Segment"] += 1
        else:
            hierarchical_train_predictions.append(float(baseline_prediction))
            hierarchical_train_usage["Global Median"] += 1

    hierarchical_test_predictions: list[float] = []
    hierarchical_test_usage = {"Best Combination": 0, "Best Simple Segment": 0, "Global Median": 0}
    for combo_label, simple_label in zip(combo_test_labels.tolist(), simple_test_labels.tolist(), strict=False):
        if combo_label != "Other" and combo_label in combo_medians:
            hierarchical_test_predictions.append(float(combo_medians[combo_label]))
            hierarchical_test_usage["Best Combination"] += 1
        elif simple_label != "Other" and simple_label in simple_medians:
            hierarchical_test_predictions.append(float(simple_medians[simple_label]))
            hierarchical_test_usage["Best Simple Segment"] += 1
        else:
            hierarchical_test_predictions.append(float(baseline_prediction))
            hierarchical_test_usage["Global Median"] += 1

    hierarchical_train_array = np.asarray(hierarchical_train_predictions, dtype=float)
    hierarchical_test_array = np.asarray(hierarchical_test_predictions, dtype=float)
    heuristic_models.append(
        {
            "model_name": "Hierarchical Backoff",
            "family_key": "hierarchical_backoff",
            "family_label": "Hierarchical Backoff",
            "rule_summary": (
                f"Usa {best_combo['family_label'] if best_combo is not None else 'ninguna combinacion representativa'}, "
                f"luego {best_simple['family_label'] if best_simple is not None else 'ningun segmento simple representativo'} "
                "y finalmente la mediana global."
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

    heuristic_models = sorted(heuristic_models, key=lambda item: float(item["metrics"]["mae"]))
    best_heuristic = heuristic_models[0] if heuristic_models else None
    best_regression = min(regression_runs, key=lambda item: float(item["metrics"]["mae"]), default=None)

    segment_reports: list[dict[str, Any]] = []
    if best_regression is not None and best_heuristic is not None:
        best_regression_predictions = np.asarray([row["predicted"] for row in best_regression["predictions"]], dtype=float)
        best_heuristic_predictions = np.asarray([row["predicted"] for row in best_heuristic["predictions"]], dtype=float)
        for columns, family_key, family_label in SIMPLE_SEGMENT_SPECS + COMBO_SEGMENT_SPECS:
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

    learning_sections = _learning_sections(
        train_target=train_target,
        full_target=data[target_col].dropna(),
        warnings=warnings,
        best_regression=best_regression,
        best_heuristic=best_heuristic,
        target_transformation_diagnostics=target_transformation_diagnostics,
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
        "models": _top_level_model_results(regression_runs),
        "warnings": warnings,
        "preprocessing_benchmarks": preprocessing_benchmarks,
        "segment_reports": segment_reports,
        "heuristic_models": heuristic_models,
        "strategy_comparison": strategy_comparison,
        "learning_sections": learning_sections,
        "target_transformation_diagnostics": target_transformation_diagnostics,
    }
