from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as sstats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.feature_selection import mutual_info_regression

from app.stats.columns import analytical_columns
from app.stats.warnings import dataframe_quality_warnings


TARGET_COL = "DaysInDeposit"


def _safe_numeric_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _outlier_warning(target: pd.Series) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    clean = target.dropna()
    if clean.empty:
        return warnings
    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    if np.isclose(iqr, 0):
        return warnings
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outlier_ratio = float(((clean < lower) | (clean > upper)).mean())
    if outlier_ratio > 0.05:
        warnings.append(
            {
                "code": "target_outliers",
                "severity": "warning",
                "column": TARGET_COL,
                "message": f"Target has {outlier_ratio:.2%} extreme outliers by IQR rule.",
                "suggestion": "Consider robust models/transformations before causal interpretation.",
            }
        )
    return warnings


def _prepare_mi_features(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, np.ndarray, list[bool], list[str]]:
    encoded = pd.DataFrame(index=df.index)
    discrete_mask: list[bool] = []

    for col in feature_cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            encoded[col] = pd.to_numeric(series, errors="coerce").fillna(series.median() if not series.dropna().empty else 0)
            discrete_mask.append(False)
        else:
            cat = series.astype("string").fillna("<MISSING>")
            codes, _ = pd.factorize(cat)
            encoded[col] = codes
            discrete_mask.append(True)

    target = pd.to_numeric(df[target_col], errors="coerce").fillna(pd.to_numeric(df[target_col], errors="coerce").median())
    return encoded, target.to_numpy(), discrete_mask, feature_cols


def compute_supervised_overview(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, Any]:
    if target_col not in df.columns:
        return {
            "target_present": False,
            "target_stats": {},
            "hist_bins": [],
            "pearson_correlations": [],
            "mutual_information": [],
            "warnings": [
                {
                    "code": "missing_target",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target DaysInDeposit is not available. Supervised analysis disabled.",
                    "suggestion": "Upload out_file or merged dataset with DaysInDeposit.",
                }
            ],
        }

    data = df.copy()
    data[target_col] = _safe_numeric_series(data[target_col])
    data = data.dropna(subset=[target_col])

    if data.empty:
        return {
            "target_present": False,
            "target_stats": {},
            "hist_bins": [],
            "pearson_correlations": [],
            "mutual_information": [],
            "warnings": [
                {
                    "code": "target_all_missing",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target exists but all values are missing or non-numeric.",
                    "suggestion": "Clean target values before supervised analysis.",
                }
            ],
        }

    target = data[target_col]
    feature_cols = [col for col in analytical_columns(data, keep_columns={target_col}) if col != target_col]
    desc = target.describe(percentiles=[0.25, 0.5, 0.75])
    target_stats = {
        "count": int(desc.get("count", 0)),
        "mean": float(desc.get("mean")) if not np.isnan(desc.get("mean", np.nan)) else None,
        "std": float(desc.get("std")) if not np.isnan(desc.get("std", np.nan)) else None,
        "min": float(desc.get("min")) if not np.isnan(desc.get("min", np.nan)) else None,
        "p25": float(desc.get("25%")) if not np.isnan(desc.get("25%", np.nan)) else None,
        "p50": float(desc.get("50%")) if not np.isnan(desc.get("50%", np.nan)) else None,
        "p75": float(desc.get("75%")) if not np.isnan(desc.get("75%", np.nan)) else None,
        "max": float(desc.get("max")) if not np.isnan(desc.get("max", np.nan)) else None,
    }

    hist_counts, hist_edges = np.histogram(target, bins="auto")
    hist_bins = [
        {"left": float(hist_edges[i]), "right": float(hist_edges[i + 1]), "count": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    pearson_rows: list[dict[str, float | str]] = []
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(data[col]):
            clean = data[[col, target_col]].dropna()
            if len(clean) < 3:
                continue
            corr = clean[col].corr(clean[target_col], method="pearson")
            corr_value = _finite_float(corr)
            if corr_value is None:
                continue
            pearson_rows.append({"feature": col, "pearson": corr_value})

    mi_rows: list[dict[str, float | str]] = []
    if len(data) >= 5 and feature_cols:
        x_encoded, y, discrete_mask, feature_cols_ordered = _prepare_mi_features(data, target_col, feature_cols)
        if x_encoded.shape[1] > 0 and x_encoded.shape[0] > 0:
            mi_scores = mutual_info_regression(x_encoded, y, discrete_features=discrete_mask, random_state=42)
        else:
            mi_scores = []
        mi_rows = []
        for idx, score in enumerate(mi_scores):
            mi_value = _finite_float(score)
            if mi_value is None:
                continue
            mi_rows.append({"feature": feature_cols_ordered[idx], "mi": mi_value})
        mi_rows.sort(key=lambda item: item["mi"], reverse=True)

    warnings = dataframe_quality_warnings(data) + _outlier_warning(target)

    return {
        "target_present": True,
        "target_stats": target_stats,
        "hist_bins": hist_bins,
        "pearson_correlations": sorted(pearson_rows, key=lambda x: abs(float(x["pearson"])), reverse=True),
        "mutual_information": mi_rows,
        "warnings": warnings,
    }


def _safe_formula_name(name: str) -> str:
    return name.replace('"', '\\"')


def _empty_multiple_regression_response(
    *,
    warnings: list[dict[str, Any]],
    target_present: bool,
    model_built: bool = False,
    formula: str | None = None,
    source: str = "out",
) -> dict[str, Any]:
    return {
        "source": source,
        "target_present": target_present,
        "model_built": model_built,
        "formula": formula,
        "n_obs": 0,
        "n_features": 0,
        "r_squared": None,
        "adj_r_squared": None,
        "f_statistic": None,
        "f_p_value": None,
        "aic": None,
        "bic": None,
        "coefficients": [],
        "anova_rows": [],
        "conclusions": [],
        "warnings": warnings,
    }


def compute_multiple_regression_out(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, Any]:
    if target_col not in df.columns:
        return _empty_multiple_regression_response(
            target_present=False,
            warnings=[
                {
                    "code": "missing_target",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target DaysInDeposit is not available in OUT source.",
                    "suggestion": "Upload Grupo1_out.csv with DaysInDeposit to enable multiple regression.",
                }
            ],
        )

    data = df.copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col])

    if data.empty:
        return _empty_multiple_regression_response(
            target_present=False,
            warnings=[
                {
                    "code": "target_all_missing",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target exists but all values are missing or non-numeric.",
                    "suggestion": "Clean target values before training multiple regression.",
                }
            ],
        )

    warnings = dataframe_quality_warnings(data) + _outlier_warning(data[target_col])
    feature_cols = [col for col in analytical_columns(data, keep_columns={target_col}) if col != target_col]
    if not feature_cols:
        warnings.append(
            {
                "code": "no_features_available",
                "severity": "warning",
                "column": None,
                "message": "No eligible predictor features available after preprocessing.",
                "suggestion": "Review schema and ensure OUT contains analytical predictors besides target.",
            }
        )
        return _empty_multiple_regression_response(target_present=True, warnings=warnings)

    model_df = pd.DataFrame(index=data.index)
    model_df[target_col] = data[target_col]
    terms: list[str] = []
    term_map: list[tuple[str, str, str]] = []

    for col in feature_cols:
        series = data[col]
        if int(series.nunique(dropna=True)) <= 1:
            warnings.append(
                {
                    "code": "feature_excluded_constant",
                    "severity": "info",
                    "column": col,
                    "message": f"Feature {col} excluded from regression because it is constant.",
                    "suggestion": "Include features with variability for regression signal.",
                }
            )
            continue

        safe_name = _safe_formula_name(col)
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            if numeric.dropna().empty:
                warnings.append(
                    {
                        "code": "feature_excluded_non_numeric",
                        "severity": "info",
                        "column": col,
                        "message": f"Feature {col} excluded because numeric parsing failed.",
                        "suggestion": "Clean or cast feature values before regression.",
                    }
                )
                continue
            fill_value = float(numeric.median()) if _finite_float(numeric.median()) is not None else 0.0
            model_df[col] = numeric.fillna(fill_value)
            term = f'Q("{safe_name}")'
            terms.append(term)
            term_map.append((col, term, "numeric"))
        else:
            categorical = series.astype("string").fillna("<MISSING>")
            if int(categorical.nunique(dropna=True)) <= 1:
                warnings.append(
                    {
                        "code": "feature_excluded_constant",
                        "severity": "info",
                        "column": col,
                        "message": f"Feature {col} excluded from regression because it is constant.",
                        "suggestion": "Include features with variability for regression signal.",
                    }
                )
                continue
            model_df[col] = categorical
            term = f'C(Q("{safe_name}"))'
            terms.append(term)
            term_map.append((col, term, "categorical"))

    if not terms:
        warnings.append(
            {
                "code": "no_features_after_cleaning",
                "severity": "warning",
                "column": None,
                "message": "No predictors remained after removing constant/invalid features.",
                "suggestion": "Check OUT dataset quality and feature variability.",
            }
        )
        return _empty_multiple_regression_response(target_present=True, warnings=warnings)

    formula = f'Q("{_safe_formula_name(target_col)}") ~ ' + " + ".join(terms)
    try:
        model = smf.ols(formula=formula, data=model_df).fit()
    except Exception as exc:
        warnings.append(
            {
                "code": "multiple_regression_fit_failed",
                "severity": "error",
                "column": None,
                "message": f"Multiple regression fit failed: {exc}",
                "suggestion": "Reduce collinearity/high-cardinality features and retry.",
            }
        )
        return _empty_multiple_regression_response(
            target_present=True,
            warnings=warnings,
            formula=formula,
        )

    if int(model.nobs) <= 0:
        warnings.append(
            {
                "code": "multiple_regression_no_observations",
                "severity": "error",
                "column": None,
                "message": "Model fit produced zero effective observations.",
                "suggestion": "Review missing values and feature preprocessing.",
            }
        )
        return _empty_multiple_regression_response(
            target_present=True,
            warnings=warnings,
            formula=formula,
        )

    coefficients: list[dict[str, Any]] = []
    confidence = model.conf_int()
    for term in model.params.index:
        if term == "Intercept":
            continue
        coefficients.append(
            {
                "term": str(term),
                "estimate": _finite_float(model.params.get(term)),
                "std_error": _finite_float(model.bse.get(term)),
                "t_value": _finite_float(model.tvalues.get(term)),
                "p_value": _finite_float(model.pvalues.get(term)),
                "ci_low": _finite_float(confidence.loc[term, 0]) if term in confidence.index else None,
                "ci_high": _finite_float(confidence.loc[term, 1]) if term in confidence.index else None,
            }
        )

    coefficients.sort(
        key=lambda item: (
            item["p_value"] is None,
            item["p_value"] if item["p_value"] is not None else float("inf"),
        )
    )

    anova_rows: list[dict[str, Any]] = []
    try:
        anova_table = sm.stats.anova_lm(model, typ=2)
        residual_ss = _finite_float(anova_table.loc["Residual", "sum_sq"]) if "Residual" in anova_table.index else None
    except Exception:
        anova_table = None
        residual_ss = None
        warnings.append(
            {
                "code": "multiple_regression_anova_unavailable",
                "severity": "warning",
                "column": None,
                "message": "ANOVA for multiple regression could not be computed.",
                "suggestion": "Inspect model assumptions and feature collinearity.",
            }
        )

    if anova_table is not None:
        for feature, term, feature_type in term_map:
            if term not in anova_table.index:
                continue
            row = anova_table.loc[term]
            ss = _finite_float(row.get("sum_sq"))
            df_term = _finite_float(row.get("df"))
            mean_sq = _finite_float(ss / df_term) if ss is not None and df_term not in (None, 0.0) else None
            f_value = _finite_float(row.get("F"))
            p_value = _finite_float(row.get("PR(>F)"))

            partial_eta_squared = None
            if ss is not None and residual_ss is not None and (ss + residual_ss) > 0:
                partial_eta_squared = _finite_float(ss / (ss + residual_ss))

            anova_rows.append(
                {
                    "feature": feature,
                    "feature_type": feature_type,
                    "df": df_term,
                    "sum_sq": ss,
                    "mean_sq": mean_sq,
                    "f_value": f_value,
                    "p_value": p_value,
                    "partial_eta_squared": partial_eta_squared,
                }
            )

    anova_rows.sort(
        key=lambda item: (
            item["p_value"] is None,
            item["p_value"] if item["p_value"] is not None else float("inf"),
        )
    )

    n_obs = int(model.nobs)
    n_features = int(len(term_map))
    r_squared = _finite_float(model.rsquared)
    adj_r_squared = _finite_float(model.rsquared_adj)
    f_statistic = _finite_float(model.fvalue)
    f_p_value = _finite_float(model.f_pvalue)
    aic = _finite_float(model.aic)
    bic = _finite_float(model.bic)

    conclusions: list[str] = []
    if f_p_value is not None and f_statistic is not None:
        if f_p_value < 0.05:
            conclusions.append(
                f"El modelo global es significativo (F={f_statistic:.3f}, p={f_p_value:.3g})."
            )
        else:
            conclusions.append(
                f"El modelo global no alcanza significancia estadística (F={f_statistic:.3f}, p={f_p_value:.3g})."
            )

    if r_squared is not None:
        if adj_r_squared is not None:
            conclusions.append(f"Capacidad explicativa: R²={r_squared:.3f}, R² ajustado={adj_r_squared:.3f}.")
        else:
            conclusions.append(f"Capacidad explicativa: R²={r_squared:.3f}.")

    significant_features = [row for row in anova_rows if row["p_value"] is not None and row["p_value"] < 0.05]
    if significant_features:
        feature_names = ", ".join(row["feature"] for row in significant_features[:3])
        conclusions.append(f"Variables más representativas por ANOVA de términos: {feature_names}.")
    else:
        conclusions.append("No se detectaron términos con p<0.05 en ANOVA del modelo múltiple.")

    if n_obs < max(20, n_features * 3):
        warnings.append(
            {
                "code": "low_sample_for_multivariate_model",
                "severity": "warning",
                "column": None,
                "message": f"Muestra efectiva baja para modelo multivariado (n={n_obs}, features={n_features}).",
                "suggestion": "Interpretar coeficientes con cautela o simplificar el modelo.",
            }
        )

    return {
        "source": "out",
        "target_present": True,
        "model_built": True,
        "formula": formula,
        "n_obs": n_obs,
        "n_features": n_features,
        "r_squared": r_squared,
        "adj_r_squared": adj_r_squared,
        "f_statistic": f_statistic,
        "f_p_value": f_p_value,
        "aic": aic,
        "bic": bic,
        "coefficients": coefficients,
        "anova_rows": anova_rows,
        "conclusions": conclusions,
        "warnings": warnings,
    }


def compute_anova(df: pd.DataFrame, target_col: str = TARGET_COL) -> dict[str, Any]:
    if target_col not in df.columns:
        return {
            "rows": [],
            "warnings": [
                {
                    "code": "missing_target",
                    "severity": "warning",
                    "column": target_col,
                    "message": "Target DaysInDeposit is not available. ANOVA disabled.",
                    "suggestion": "Upload out_file or merged dataset with DaysInDeposit.",
                }
            ],
            "boxplot_data": [],
        }

    data = df.copy()
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    data = data.dropna(subset=[target_col])

    if data.empty:
        return {"rows": [], "warnings": [], "boxplot_data": []}

    global_warnings = _outlier_warning(data[target_col])
    rows: list[dict[str, Any]] = []
    feature_cols = [col for col in analytical_columns(data, keep_columns={target_col}) if col != target_col]

    for feature in feature_cols:
        feature_warnings: list[dict[str, Any]] = []
        feature_series = data[feature]
        non_na = data[[feature, target_col]].dropna()
        if non_na.empty:
            continue

        if pd.api.types.is_numeric_dtype(feature_series):
            if len(non_na) < 3:
                feature_warnings.append(
                    {
                        "code": "too_few_rows",
                        "severity": "warning",
                        "column": feature,
                        "message": "Too few rows for OLS numeric test.",
                        "suggestion": "Collect more data for reliable regression.",
                    }
                )
                rows.append(
                    {
                        "feature": feature,
                        "feature_type": "numeric",
                        "test_used": "ols_numeric",
                        "statistic": None,
                        "p_value": None,
                        "effect_size": None,
                        "n_groups": None,
                        "kruskal_statistic": None,
                        "kruskal_p_value": None,
                        "warnings": feature_warnings,
                    }
                )
                continue

            formula = f'Q("{_safe_formula_name(target_col)}") ~ Q("{_safe_formula_name(feature)}")'
            model = smf.ols(formula=formula, data=non_na).fit()

            param_key = [k for k in model.params.index if k != "Intercept"]
            p_value = float(model.pvalues[param_key[0]]) if param_key else None
            stat_val = float(model.tvalues[param_key[0]]) if param_key else None
            effect = float(model.rsquared)
            p_value = _finite_float(p_value)
            stat_val = _finite_float(stat_val)
            effect = _finite_float(effect)

            rows.append(
                {
                    "feature": feature,
                    "feature_type": "numeric",
                    "test_used": "ols_numeric",
                    "statistic": stat_val,
                    "p_value": p_value,
                    "effect_size": effect,
                    "n_groups": None,
                    "kruskal_statistic": None,
                    "kruskal_p_value": None,
                    "warnings": feature_warnings,
                }
            )
            continue

        grouped = non_na.groupby(feature)[target_col]
        groups = [vals.values for _, vals in grouped]
        n_groups = len(groups)

        if n_groups < 2:
            feature_warnings.append(
                {
                    "code": "too_few_groups",
                    "severity": "warning",
                    "column": feature,
                    "message": "Need at least 2 groups for ANOVA/Kruskal.",
                    "suggestion": "Rebin categories or collect more class diversity.",
                }
            )
            rows.append(
                {
                    "feature": feature,
                    "feature_type": "categorical",
                    "test_used": "anova_kruskal",
                    "statistic": None,
                    "p_value": None,
                    "effect_size": None,
                    "n_groups": n_groups,
                    "kruskal_statistic": None,
                    "kruskal_p_value": None,
                    "warnings": feature_warnings,
                }
            )
            continue

        small_groups = sum(1 for arr in groups if len(arr) < 5)
        if small_groups > 0:
            feature_warnings.append(
                {
                    "code": "small_groups",
                    "severity": "warning",
                    "column": feature,
                    "message": f"{small_groups} groups with n<5 detected.",
                    "suggestion": "Interpret p-values carefully; consider category grouping.",
                }
            )

        formula = f'Q("{_safe_formula_name(target_col)}") ~ C(Q("{_safe_formula_name(feature)}"))'
        model = smf.ols(formula=formula, data=non_na).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        effect_row = anova_table.iloc[0]
        ss_effect = float(effect_row["sum_sq"])
        ss_total = float(anova_table["sum_sq"].sum()) if float(anova_table["sum_sq"].sum()) else np.nan
        eta_sq = float(ss_effect / ss_total) if ss_total and not np.isnan(ss_total) else None
        eta_sq = _finite_float(eta_sq)

        kr_stat, kr_p = sstats.kruskal(*groups)

        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "test_used": "anova_kruskal",
                "statistic": _finite_float(effect_row["F"]) if "F" in effect_row else None,
                "p_value": _finite_float(effect_row["PR(>F)"]) if "PR(>F)" in effect_row else None,
                "effect_size": eta_sq,
                "n_groups": n_groups,
                "kruskal_statistic": _finite_float(kr_stat),
                "kruskal_p_value": _finite_float(kr_p),
                "warnings": feature_warnings,
            }
        )

    rows = sorted(
        rows,
        key=lambda r: (
            r["p_value"] is None,
            r["p_value"] if r["p_value"] is not None else float("inf"),
            -(r["effect_size"] if r["effect_size"] is not None else -1),
        ),
    )

    top_features = [r["feature"] for r in rows[:3] if r.get("feature_type") == "categorical"]
    boxplot_data: list[dict[str, Any]] = []
    for feature in top_features:
        grouped = data[[feature, target_col]].dropna().groupby(feature)[target_col]
        groups = []
        for grp_name, values in grouped:
            groups.append({"group": str(grp_name), "values": [float(v) for v in values.tolist()]})
        boxplot_data.append({"feature": feature, "groups": groups})

    return {
        "rows": rows,
        "warnings": global_warnings,
        "boxplot_data": boxplot_data,
    }
