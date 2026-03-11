from __future__ import annotations

import os
import warnings as pywarnings
from collections import defaultdict
from itertools import combinations
import math
from typing import Any

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
import scipy.stats as sstats
from sklearn.cluster import OPTICS
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from app.stats.columns import analytical_columns, is_technical_id_column
from app.stats.eda import compute_eda
from app.stats.supervised import compute_anova
from app.stats.warnings import dataframe_quality_warnings


def _safe_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


def _dedupe_warnings(warnings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for warning in warnings:
        key = (warning.get("code"), warning.get("column"), warning.get("message"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(warning)
    return deduped


def _series_mode(series: pd.Series) -> Any:
    mode = series.dropna().mode()
    if mode.empty:
        return "<FALTANTE>"
    if pd.api.types.is_numeric_dtype(mode):
        return mode.sort_values().iloc[0]
    normalized = mode.astype("string").sort_values()
    return normalized.iloc[0]


def _semantic_type_counts(df: pd.DataFrame) -> dict[str, int]:
    numeric_count = 0
    categorical_count = 0
    for column in df.columns:
        if is_technical_id_column(str(column)):
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric_count += 1
        else:
            categorical_count += 1
    return {
        "numeric": numeric_count,
        "categorical": categorical_count,
        "technical_excluded": int(sum(1 for column in df.columns if is_technical_id_column(str(column)))),
    }


def _translate_warning(
    warning: dict[str, Any],
    *,
    df: pd.DataFrame | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    code = str(warning.get("code") or "")
    column = warning.get("column")
    column_label = str(column) if column is not None else "dataset"

    translated = {
        "code": code,
        "severity": warning.get("severity", "warning"),
        "column": column,
        "message": str(warning.get("message") or ""),
        "suggestion": warning.get("suggestion"),
    }

    if code == "size_casted_to_categorical_discrete":
        translated["message"] = f"La variable Size en {source_label or 'la fuente'} se analizo como categoria discreta."
        translated["suggestion"] = "Interpreta Size como clase operacional y no como magnitud continua."
    elif code == "size_casted_to_categorical":
        translated["message"] = f"La variable Size en {source_label or 'la fuente'} no pudo validarse como numerica y se trato como categoria."
        translated["suggestion"] = "Estandariza los valores de Size antes de usarla como magnitud."
    elif code.startswith("missing_expected_columns_"):
        missing_part = str(warning.get("message") or "").split(": ", 1)[-1]
        translated["message"] = f"Faltan columnas esperadas en {source_label or 'la fuente'}: {missing_part}."
        translated["suggestion"] = "Verifica el esquema del seed antes de continuar con el EDA."
    elif code == "overlap_column_conflict":
        translated["message"] = "IN y OUT presentan columnas compartidas con valores distintos; el canónico las conserva separadas con sufijos."
        translated["suggestion"] = "Analiza las fuentes por separado y usa el canónico solo como apoyo comparativo."
    elif code == "constant_column":
        translated["message"] = f"La columna {column_label} es constante."
        translated["suggestion"] = "Mantenla solo como contexto o excluyela de comparaciones inferenciales."
    elif code == "near_constant_column":
        translated["message"] = f"La columna {column_label} es casi constante."
        translated["suggestion"] = "Evalua si aporta senal real antes de incluirla en comparaciones."
    elif code == "high_cardinality":
        nunique = int(df[column].astype("string").nunique(dropna=True)) if df is not None and column in df.columns else None
        ratio = (
            float(df[column].astype("string").nunique(dropna=True) / max(len(df), 1))
            if df is not None and column in df.columns
            else None
        )
        translated["message"] = (
            f"La columna {column_label} presenta alta cardinalidad"
            + (f" (n={nunique}, ratio={ratio:.2%})." if nunique is not None and ratio is not None else ".")
        )
        translated["suggestion"] = "Agrupa categorias raras o trata la variable como identificador en etapas posteriores."
    elif code == "week_constant":
        translated["message"] = "La columna week es constante en Week 1."
        translated["suggestion"] = "No la interpretes como evidencia temporal; usala solo como contexto."
    elif code == "missing_values_detected":
        translated["message"] = f"La columna {column_label} presenta valores faltantes."
        translated["suggestion"] = "Audita el origen del NA y define una estrategia de imputacion trazable."
    return translated


def _filter_pipeline_warnings(metadata: dict[str, Any], source: str, df: pd.DataFrame) -> list[dict[str, Any]]:
    source_key = f"{source}_file"
    source_label = source.upper()
    selected: list[dict[str, Any]] = []
    for warning in metadata.get("warnings", []):
        code = str(warning.get("code") or "")
        message = str(warning.get("message") or "")
        if code == f"missing_expected_columns_{source}":
            selected.append(_translate_warning(warning, df=df, source_label=source_label))
            continue
        if code in {"size_casted_to_categorical", "size_casted_to_categorical_discrete"} and source_key in message:
            selected.append(_translate_warning(warning, df=df, source_label=source_label))
            continue
    for warning in dataframe_quality_warnings(df):
        selected.append(_translate_warning(warning, df=df, source_label=source_label))
    return _dedupe_warnings(selected)


def _global_warnings(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    for warning in metadata.get("warnings", []):
        if str(warning.get("code")) == "overlap_column_conflict":
            translated.append(_translate_warning(warning))
    return _dedupe_warnings(translated)


def _merge_variable_dictionary(
    df: pd.DataFrame,
    academic_metadata: dict[str, Any],
    missingness: dict[str, dict[str, float]],
    cardinality: dict[str, int],
) -> list[dict[str, Any]]:
    dictionary_by_name = {
        item["name"]: {
            "name": item["name"],
            "logical_type": item["logical_type"],
            "observed_type": str(df[item["name"]].dtype) if item["name"] in df.columns else "missing",
            "business_description": item["business_description"],
            "analytical_role": item["analytical_role"],
            "notes": item.get("notes"),
            "missing_pct": missingness.get(item["name"], {}).get("pct"),
            "unique_values": cardinality.get(item["name"]),
        }
        for item in academic_metadata["variable_dictionary"]
        if item["name"] in df.columns
    }

    for column in df.columns:
        if is_technical_id_column(str(column)):
            continue
        if str(column) not in dictionary_by_name:
            dictionary_by_name[str(column)] = {
                "name": str(column),
                "logical_type": "observada",
                "observed_type": str(df[column].dtype),
                "business_description": "Variable observada sin descripcion academica explicita.",
                "analytical_role": "apoyo",
                "notes": "Agregala a framework/academic/week-1.json si debe documentarse formalmente.",
                "missing_pct": missingness.get(str(column), {}).get("pct"),
                "unique_values": cardinality.get(str(column)),
            }

    return [dictionary_by_name[name] for name in sorted(dictionary_by_name.keys())]


def _quality_findings(
    df: pd.DataFrame,
    duplicate_count: int,
    missingness: dict[str, dict[str, float]],
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []

    if duplicate_count > 0:
        findings.append(
            {
                "kind": "duplicados_exactos",
                "severity": "warning",
                "hallazgo": f"Se detectaron {duplicate_count} filas duplicadas exactas.",
                "riesgo": "Los duplicados pueden sesgar frecuencias, proporciones y pruebas entre grupos.",
                "decision": "Verificar si corresponden a replicas operacionales legitimas o si deben consolidarse.",
                "affected_columns": [],
            }
        )

    for column, item in missingness.items():
        if item["count"] <= 0:
            continue
        findings.append(
            {
                "kind": "valores_faltantes",
                "severity": "warning",
                "hallazgo": f"La variable {column} presenta {item['count']} NA ({item['pct']:.2f}%).",
                "riesgo": "Los faltantes pueden sesgar estadisticos descriptivos y comparaciones posteriores.",
                "decision": "Mantener el raw para auditoria y usar una capa imputada derivada cuando se necesite completitud.",
                "affected_columns": [column],
            }
        )

    rare_columns: dict[str, list[str]] = defaultdict(list)
    for column in df.columns:
        if is_technical_id_column(str(column)) or pd.api.types.is_numeric_dtype(df[column]):
            continue
        frequencies = df[column].astype("string").value_counts(dropna=True)
        rare_labels = [str(label) for label, count in frequencies.items() if 0 < count / max(len(df), 1) < 0.01]
        if rare_labels:
            rare_columns[str(column)] = rare_labels[:5]

    for column, labels in rare_columns.items():
        findings.append(
            {
                "kind": "categorias_raras",
                "severity": "info",
                "hallazgo": f"La variable {column} contiene categorias marginales como {', '.join(labels)}.",
                "riesgo": "Las categorias muy pequenas vuelven inestables tablas de contingencia y comparaciones entre grupos.",
                "decision": "Usar top-k + otras en comparativos y documentar el sesgo potencial.",
                "affected_columns": [column],
            }
        )

    for warning in warnings:
        code = str(warning.get("code"))
        column = warning.get("column")
        affected_columns = [str(column)] if isinstance(column, str) else []
        if code == "high_cardinality":
            findings.append(
                {
                    "kind": "alta_cardinalidad",
                    "severity": warning.get("severity", "warning"),
                    "hallazgo": warning.get("message"),
                    "riesgo": "La alta cardinalidad dificulta comparativos, tablas de contingencia y clustering interpretable.",
                    "decision": "Mantener la variable, pero leerla con cautela y agrupar categorias en visualizaciones cuando corresponda.",
                    "affected_columns": affected_columns,
                }
            )
        elif code in {"constant_column", "near_constant_column", "week_constant"}:
            findings.append(
                {
                    "kind": "variacion_limitada",
                    "severity": warning.get("severity", "warning"),
                    "hallazgo": warning.get("message"),
                    "riesgo": "Una variable sin variacion real no aporta evidencia analitica fuerte.",
                    "decision": "Conservarla solo como contexto y explicitar que no permite contraste estadistico.",
                    "affected_columns": affected_columns,
                }
            )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for finding in findings:
        key = (finding["kind"], finding["hallazgo"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _numeric_profiles(df: pd.DataFrame, base_eda: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    numeric_cols = [
        column
        for column in base_eda["columns"]
        if column in base_eda["numeric_stats"]
        and int(pd.to_numeric(df[column], errors="coerce").nunique(dropna=True)) > 1
        and not is_technical_id_column(str(column))
    ]
    for column in numeric_cols:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        stats = {
            "mean": _safe_float(series.mean()),
            "median": _safe_float(series.median()),
            "std": _safe_float(series.std(ddof=1)),
            "variance": _safe_float(series.var(ddof=1)),
            "min": _safe_float(series.min()),
            "max": _safe_float(series.max()),
            "p25": _safe_float(series.quantile(0.25)),
            "p75": _safe_float(series.quantile(0.75)),
            "iqr": _safe_float(series.quantile(0.75) - series.quantile(0.25)),
            "skewness": _safe_float(series.skew()),
            "kurtosis": _safe_float(series.kurtosis()),
            "coefficient_variation": _safe_float(series.std(ddof=1) / series.mean())
            if _safe_float(series.mean()) not in (None, 0.0)
            else None,
        }
        profiles.append(
            {
                "column": column,
                "stats": stats,
                "histogram_bins": base_eda["numeric_histograms"].get(column, []),
                "values": [float(value) for value in series.tolist()],
            }
        )
    return profiles


def _categorical_profiles(df: pd.DataFrame, base_eda: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    numeric_cols = set(base_eda["numeric_stats"].keys())
    for column in base_eda["columns"]:
        if column in numeric_cols or is_technical_id_column(str(column)):
            continue
        top_values = base_eda["top_values"].get(column, [])
        total_categories = int(base_eda["cardinality"].get(column, 0))
        top_count = sum(int(item["count"]) for item in top_values)
        total_non_null = int(df[column].notna().sum())
        other_count = max(total_non_null - top_count, 0)
        other_pct = round(float(other_count / max(len(df), 1) * 100), 4)
        profiles.append(
            {
                "column": column,
                "total_categories": total_categories,
                "top_categories": top_values,
                "other_count": other_count,
                "other_pct": other_pct,
            }
        )
    return profiles


def _matrix_to_list(matrix: pd.DataFrame, labels: list[str]) -> list[list[float | None]]:
    payload: list[list[float | None]] = []
    for row_label in labels:
        row_values: list[float | None] = []
        for col_label in labels:
            row_values.append(_safe_float(matrix.loc[row_label, col_label]))
        payload.append(row_values)
    return payload


def _numeric_numeric_section(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    numeric_candidates = [
        str(column)
        for column in analytical_columns(df)
        if str(column) in df.columns
        and pd.api.types.is_numeric_dtype(df[str(column)])
        and int(pd.to_numeric(df[str(column)], errors="coerce").nunique(dropna=True)) > 1
    ]

    if len(numeric_candidates) < 2:
        return {
            "status": "not_applicable",
            "message": "No hay suficientes variables numericas con variacion para correlaciones entre pares.",
            "labels": [],
            "pearson_matrix": [],
            "spearman_matrix": [],
            "target_rankings": [],
        }

    numeric_df = df[numeric_candidates].apply(pd.to_numeric, errors="coerce")
    pearson = numeric_df.corr(method="pearson")
    spearman = numeric_df.corr(method="spearman")
    labels = numeric_candidates

    target_rankings: list[dict[str, Any]] = []
    if target_column and target_column in numeric_df.columns:
        for column in numeric_candidates:
            if column == target_column:
                continue
            subset = numeric_df[[target_column, column]].dropna()
            if len(subset) < 3:
                continue
            target_rankings.append(
                {
                    "feature": column,
                    "pearson": _safe_float(subset[target_column].corr(subset[column], method="pearson")),
                    "spearman": _safe_float(subset[target_column].corr(subset[column], method="spearman")),
                }
            )
        target_rankings.sort(
            key=lambda item: abs(item["spearman"] if item["spearman"] is not None else 0.0),
            reverse=True,
        )

    return {
        "status": "available",
        "message": None,
        "labels": labels,
        "pearson_matrix": _matrix_to_list(pearson, labels),
        "spearman_matrix": _matrix_to_list(spearman, labels),
        "target_rankings": target_rankings,
    }


def _categorical_numeric_section(df: pd.DataFrame, target_column: str | None) -> dict[str, Any]:
    if target_column is None or target_column not in df.columns:
        return {
            "status": "not_applicable",
            "message": "Esta fuente no tiene una variable objetivo numerica para comparaciones cat-num.",
            "rows": [],
            "boxplot_data": [],
        }

    payload = compute_anova(df, target_col=target_column)
    categorical_rows = [row for row in payload["rows"] if row.get("feature_type") == "categorical"]
    if not categorical_rows:
        return {
            "status": "not_applicable",
            "message": "No hay variables categoricas elegibles para ANOVA o Kruskal.",
            "rows": [],
            "boxplot_data": [],
        }
    return {
        "status": "available",
        "message": None,
        "rows": categorical_rows,
        "boxplot_data": payload["boxplot_data"],
    }


def _cramers_v(table: pd.DataFrame) -> float | None:
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return None
    chi2, _, _, _ = sstats.chi2_contingency(table)
    n = table.to_numpy().sum()
    if n <= 0:
        return None
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / max(n - 1, 1))
    rcorr = r - ((r - 1) ** 2) / max(n - 1, 1)
    kcorr = k - ((k - 1) ** 2) / max(n - 1, 1)
    denom = min((kcorr - 1), (rcorr - 1))
    if denom <= 0:
        return None
    return _safe_float(np.sqrt(phi2corr / denom))


def _categorical_categorical_section(df: pd.DataFrame) -> dict[str, Any]:
    eligible_columns: list[str] = []
    for column in analytical_columns(df):
        if column not in df.columns or pd.api.types.is_numeric_dtype(df[column]):
            continue
        nunique = int(df[column].astype("string").nunique(dropna=True))
        if 2 <= nunique <= 25:
            eligible_columns.append(column)

    if len(eligible_columns) < 2:
        return {
            "status": "not_applicable",
            "message": "No hay suficientes variables categoricas con cardinalidad manejable para asociacion cat-cat.",
            "rows": [],
            "contingency_preview": None,
        }

    rows: list[dict[str, Any]] = []
    preview: dict[str, Any] | None = None
    strongest_v = -1.0

    for feature_x, feature_y in combinations(eligible_columns, 2):
        subset = df[[feature_x, feature_y]].dropna()
        if subset.empty:
            continue
        table = pd.crosstab(subset[feature_x].astype("string"), subset[feature_y].astype("string"))
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue
        chi2, p_value, _, _ = sstats.chi2_contingency(table)
        cramers_v = _cramers_v(table)
        rows.append(
            {
                "feature_x": feature_x,
                "feature_y": feature_y,
                "chi2": _safe_float(chi2),
                "p_value": _safe_float(p_value),
                "cramers_v": cramers_v,
                "sample_size": int(subset.shape[0]),
            }
        )
        current_strength = abs(cramers_v) if cramers_v is not None else -1.0
        if current_strength > strongest_v:
            strongest_v = current_strength
            preview_columns = [str(item) for item in table.columns.tolist()[:8]]
            preview_rows = []
            for index_name, row in table.iloc[:8, :8].iterrows():
                row_payload: dict[str, int | str] = {"row_key": str(index_name)}
                for column_name in preview_columns:
                    row_payload[column_name] = int(row[column_name])
                preview_rows.append(row_payload)
            preview = {
                "row_label": feature_x,
                "column_label": feature_y,
                "columns": preview_columns,
                "rows": preview_rows,
            }

    if not rows:
        return {
            "status": "not_applicable",
            "message": "No fue posible construir tablas de contingencia validas para las variables elegibles.",
            "rows": [],
            "contingency_preview": None,
        }

    rows.sort(key=lambda item: abs(item["cramers_v"] if item["cramers_v"] is not None else 0.0), reverse=True)
    return {
        "status": "available",
        "message": None,
        "rows": rows[:10],
        "contingency_preview": preview,
    }


def _temporal_section(df: pd.DataFrame, week_column: str = "week") -> dict[str, Any]:
    if week_column not in df.columns:
        return {
            "status": "not_applicable",
            "message": "La columna week no existe en esta fuente.",
            "unique_periods": [],
            "counts": [],
        }

    clean = df[week_column].dropna()
    unique_periods = sorted({str(value) for value in clean.tolist()})
    if len(unique_periods) <= 1:
        return {
            "status": "not_applicable",
            "message": "Week 1 usa un unico corte temporal; no hay variacion suficiente para diagnostico temporal.",
            "unique_periods": unique_periods,
            "counts": [{"period": period, "count": int((clean.astype('string') == period).sum())} for period in unique_periods],
        }

    counts = clean.astype("string").value_counts().sort_index()
    return {
        "status": "available",
        "message": "La columna week presenta variacion y permite un diagnostico temporal basico.",
        "unique_periods": unique_periods,
        "counts": [{"period": str(index), "count": int(value)} for index, value in counts.items()],
    }


def _impute_source(df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    analysis_df = df.copy()
    missing_summary: dict[str, dict[str, float]] = {}
    strategy_by_column: dict[str, str] = {}
    imputed_counts: dict[str, int] = {}
    notes: list[str] = []
    any_imputation = False

    for column in analysis_df.columns:
        missing_count = int(analysis_df[column].isna().sum())
        missing_pct = round(float(missing_count / max(len(analysis_df), 1) * 100), 4)
        missing_summary[str(column)] = {"count": missing_count, "pct": missing_pct}
        strategy_by_column[str(column)] = "sin imputacion"
        imputed_counts[str(column)] = 0

        if missing_count <= 0 or is_technical_id_column(str(column)):
            continue

        any_imputation = True
        if str(column) == "week":
            fill_value = _series_mode(analysis_df[column])
            strategy = "moda_deterministica"
            notes.append(f"Se imputo week en {source.upper()} usando la moda por tratarse de contexto temporal discreto.")
        elif pd.api.types.is_numeric_dtype(analysis_df[column]):
            clean = pd.to_numeric(analysis_df[column], errors="coerce").dropna()
            fill_value = float(clean.median()) if not clean.empty else 0.0
            strategy = "mediana"
        else:
            fill_value = _series_mode(analysis_df[column])
            strategy = "moda_deterministica"

        analysis_df[column] = analysis_df[column].fillna(fill_value)
        strategy_by_column[str(column)] = strategy
        imputed_counts[str(column)] = missing_count

    if not any_imputation:
        notes.append(f"No se detectaron NA en {source.upper()}; la capa analitica coincide con el dataset raw.")

    return analysis_df, {
        "missing_summary": missing_summary,
        "strategy_by_column": strategy_by_column,
        "imputed_counts": imputed_counts,
        "notes": notes,
        "imputation_applied": any_imputation,
    }


def _outlier_source_section(df: pd.DataFrame) -> dict[str, Any]:
    methods = ["IQR", "robust_z_score"]
    numeric_columns = [
        str(column)
        for column in df.columns
        if not is_technical_id_column(str(column))
        and pd.api.types.is_numeric_dtype(df[column])
        and int(pd.to_numeric(df[column], errors="coerce").nunique(dropna=True)) > 1
    ]
    if not numeric_columns:
        return {
            "status": "not_applicable",
            "methods": methods,
            "columns": [],
            "flagged_counts": 0,
            "flagged_ratio": 0.0,
            "interpretation": "No hay variables numericas con variacion suficiente para diagnosticar outliers.",
        }

    column_results: list[dict[str, Any]] = []
    union_flags = np.zeros(len(df), dtype=bool)

    for column in numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        clean = series.dropna()
        if clean.empty:
            continue

        q1 = float(clean.quantile(0.25))
        q3 = float(clean.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr if not np.isclose(iqr, 0.0) else None
        upper = q3 + 1.5 * iqr if not np.isclose(iqr, 0.0) else None
        flag_iqr = pd.Series(False, index=df.index)
        if lower is not None and upper is not None:
            flag_iqr = (series < lower) | (series > upper)

        median = float(clean.median())
        mad = float(np.median(np.abs(clean - median)))
        max_abs_robust_z = None
        flag_rz = pd.Series(False, index=df.index)
        if not np.isclose(mad, 0.0):
            robust_z = 0.6745 * (series - median) / mad
            flag_rz = robust_z.abs() > 3.5
            max_abs_robust_z = _safe_float(robust_z.abs().max())

        combined = flag_iqr.fillna(False) | flag_rz.fillna(False)
        union_flags |= combined.to_numpy(dtype=bool)
        flagged_count = int(combined.sum())
        column_results.append(
            {
                "column": column,
                "flagged_count": flagged_count,
                "flagged_ratio": round(float(flagged_count / max(len(df), 1) * 100), 4),
                "lower_bound": _safe_float(lower),
                "upper_bound": _safe_float(upper),
                "max_abs_robust_z": max_abs_robust_z,
            }
        )

    column_results.sort(key=lambda item: item["flagged_count"], reverse=True)
    total_flagged = int(union_flags.sum())
    return {
        "status": "available",
        "methods": methods,
        "columns": column_results,
        "flagged_counts": total_flagged,
        "flagged_ratio": round(float(total_flagged / max(len(df), 1) * 100), 4),
        "interpretation": (
            "Los outliers se diagnostican para lectura robusta del fenomeno, pero no se eliminan automaticamente en esta iteracion."
        ),
    }


def _build_source_section(
    raw_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    source: str,
    academic_metadata: dict[str, Any],
    source_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    base_eda = compute_eda(analysis_df)
    duplicate_count = int(raw_df.duplicated().sum())
    total_cells = int(raw_df.shape[0] * raw_df.shape[1]) if raw_df.shape[0] and raw_df.shape[1] else 0
    total_missing = int(raw_df.isna().sum().sum())
    completeness_pct = round(float((1 - (total_missing / total_cells)) * 100), 4) if total_cells else 100.0
    unique_identifier_columns = [
        str(column)
        for column in raw_df.columns
        if int(raw_df[column].astype("string").nunique(dropna=True)) == int(len(raw_df))
    ]
    raw_missingness = {
        str(column): {
            "count": int(raw_df[column].isna().sum()),
            "pct": round(float(raw_df[column].isna().sum() / max(len(raw_df), 1) * 100), 4),
        }
        for column in raw_df.columns
    }
    raw_cardinality = {
        str(column): int(raw_df[column].astype("string").nunique(dropna=True))
        for column in raw_df.columns
    }
    variable_dictionary = _merge_variable_dictionary(
        raw_df,
        academic_metadata=academic_metadata,
        missingness=raw_missingness,
        cardinality=raw_cardinality,
    )
    quality_findings = _quality_findings(
        raw_df,
        duplicate_count=duplicate_count,
        missingness=raw_missingness,
        warnings=source_warnings,
    )
    numeric_profiles = _numeric_profiles(analysis_df, base_eda)
    categorical_profiles = _categorical_profiles(analysis_df, base_eda)
    target_column = academic_metadata.get("target_variable") if academic_metadata.get("target_variable") in analysis_df.columns else None

    return {
        "source": source,
        "title": f"Dataset {source.upper()}",
        "target_variable": target_column,
        "overview_metrics": {
            "row_count": int(raw_df.shape[0]),
            "column_count": int(raw_df.shape[1]),
            "missing_cells": total_missing,
            "completeness_pct": completeness_pct,
            "duplicate_rows_exact": duplicate_count,
            "numeric_variables": _semantic_type_counts(analysis_df)["numeric"],
            "categorical_variables": _semantic_type_counts(analysis_df)["categorical"],
        },
        "dataset_audit": {
            "shape": (int(raw_df.shape[0]), int(raw_df.shape[1])),
            "variable_type_counts": _semantic_type_counts(raw_df),
            "duplicate_rows_exact": duplicate_count,
            "unique_identifier_columns": unique_identifier_columns,
            "completeness_pct": completeness_pct,
            "missingness": raw_missingness,
            "cardinality": raw_cardinality,
            "variable_dictionary": variable_dictionary,
        },
        "data_quality": {"findings": quality_findings},
        "univariate_numeric": {
            "status": "available" if numeric_profiles else "not_applicable",
            "message": None if numeric_profiles else "No hay variables numericas con variacion para analisis univariado.",
            "variables": numeric_profiles,
        },
        "univariate_categorical": {
            "status": "available" if categorical_profiles else "not_applicable",
            "message": None if categorical_profiles else "No hay variables categoricas elegibles para analisis univariado.",
            "variables": categorical_profiles,
        },
        "bivariate_numeric_numeric": _numeric_numeric_section(analysis_df, target_column=target_column),
        "bivariate_categorical_numeric": _categorical_numeric_section(analysis_df, target_column=target_column),
        "bivariate_categorical_categorical": _categorical_categorical_section(analysis_df),
        "temporal_diagnostics": _temporal_section(raw_df),
        "sample_rows": raw_df.head(8).where(pd.notna(raw_df.head(8)), None).to_dict(orient="records"),
        "columns": [str(column) for column in raw_df.columns],
        "warnings": source_warnings,
    }


def _build_comparison(in_df: pd.DataFrame, out_df: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    shared_columns = [
        column
        for column in in_df.columns
        if column in out_df.columns and not is_technical_id_column(str(column))
    ]
    categorical_comparisons: list[dict[str, Any]] = []
    numeric_comparisons: list[dict[str, Any]] = []

    for column in shared_columns:
        in_series = in_df[column]
        out_series = out_df[column]
        is_numeric = pd.api.types.is_numeric_dtype(in_series) and pd.api.types.is_numeric_dtype(out_series)
        in_nunique = int(in_series.astype("string").nunique(dropna=True))
        out_nunique = int(out_series.astype("string").nunique(dropna=True))

        if is_numeric and max(in_nunique, out_nunique) > 12:
            in_clean = pd.to_numeric(in_series, errors="coerce").dropna()
            out_clean = pd.to_numeric(out_series, errors="coerce").dropna()
            numeric_comparisons.append(
                {
                    "column": column,
                    "in_stats": {
                        "mean": _safe_float(in_clean.mean()),
                        "median": _safe_float(in_clean.median()),
                        "std": _safe_float(in_clean.std(ddof=1)),
                        "min": _safe_float(in_clean.min()),
                        "max": _safe_float(in_clean.max()),
                    },
                    "out_stats": {
                        "mean": _safe_float(out_clean.mean()),
                        "median": _safe_float(out_clean.median()),
                        "std": _safe_float(out_clean.std(ddof=1)),
                        "min": _safe_float(out_clean.min()),
                        "max": _safe_float(out_clean.max()),
                    },
                }
            )
            continue

        combined = pd.concat(
            [
                in_series.astype("string").value_counts(dropna=True).rename("in"),
                out_series.astype("string").value_counts(dropna=True).rename("out"),
            ],
            axis=1,
        ).fillna(0)
        combined["total"] = combined["in"] + combined["out"]
        combined = combined.sort_values("total", ascending=False).head(10)
        categories = []
        for category, row in combined.iterrows():
            categories.append(
                {
                    "category": str(category),
                    "in_count": int(row["in"]),
                    "in_pct": round(float(row["in"] / max(len(in_df), 1) * 100), 4),
                    "out_count": int(row["out"]),
                    "out_pct": round(float(row["out"] / max(len(out_df), 1) * 100), 4),
                }
            )
        note = None
        if column == "week" and len(categories) == 1:
            note = "La variable week es constante en ambas fuentes y solo aporta contexto."
        categorical_comparisons.append({"column": column, "categories": categories, "note": note})

    merge_meta = (metadata.get("step_metadata") or {}).get("step_merge", {})
    notes = [
        f"IN contiene {int(len(in_df))} filas y OUT contiene {int(len(out_df))} filas.",
        (
            f"El merge canonico conserva {merge_meta.get('merged_rows', len(in_df))} filas emparejadas "
            f"con {merge_meta.get('unmatched_in_rows', 0)} no emparejadas en IN y "
            f"{merge_meta.get('unmatched_out_rows', 0)} no emparejadas en OUT."
        ),
        "Las comparaciones se concentran en columnas compartidas; Condition se analiza solo en IN y DaysInDeposit solo en OUT.",
    ]

    return {
        "shared_columns": shared_columns,
        "categorical_comparisons": categorical_comparisons,
        "numeric_comparisons": numeric_comparisons,
        "notes": notes,
    }


def _cluster_metadata(label: int) -> tuple[int | None, str, bool]:
    if int(label) == -1:
        return None, "ruido", True
    cluster_id = int(label)
    return cluster_id, f"cluster_{cluster_id}", False


def _run_optics(
    matrix: np.ndarray,
    *,
    min_samples: int,
    min_cluster_size: int,
    xi: float,
) -> tuple[OPTICS, np.ndarray]:
    model = OPTICS(
        min_samples=min_samples,
        min_cluster_size=min_cluster_size,
        xi=xi,
        metric="euclidean",
        n_jobs=1,
    )
    with pywarnings.catch_warnings():
        pywarnings.filterwarnings(
            "ignore",
            message="divide by zero encountered in divide",
            category=RuntimeWarning,
        )
        labels = model.fit_predict(matrix)
    return model, labels


def _candidate_optics_params(n_rows: int) -> list[tuple[int, int, float]]:
    candidates: list[tuple[int, int, float]] = []
    for min_samples_ratio in (0.015, 0.02, 0.025):
        for min_cluster_size_ratio in (0.025, 0.03, 0.04):
            for xi in (0.03, 0.05):
                min_samples = min(max(5, round(min_samples_ratio * n_rows)), n_rows)
                min_cluster_size = min(max(10, round(min_cluster_size_ratio * n_rows)), n_rows)
                min_cluster_size = max(min_cluster_size, min_samples)
                candidate = (int(min_samples), int(min_cluster_size), float(xi))
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates


def _optics_candidate_summary(
    *,
    matrix: np.ndarray,
    labels: np.ndarray,
    min_samples: int,
    min_cluster_size: int,
    xi: float,
) -> dict[str, Any]:
    non_noise_mask = labels != -1
    unique_clusters = sorted(label for label in set(labels.tolist()) if label != -1)
    non_noise_coverage = round(float(non_noise_mask.mean() * 100), 4)
    noise_ratio = round(float((labels == -1).mean() * 100), 4)
    rejected = False
    rejection_reason = None
    silhouette = None

    if len(unique_clusters) < 2:
        rejected = True
        rejection_reason = "menos_de_dos_clusters"
    elif non_noise_coverage < 35:
        rejected = True
        rejection_reason = "cobertura_no_ruido_baja"
    elif non_noise_mask.sum() >= 10:
        silhouette = float(silhouette_score(matrix[non_noise_mask], labels[non_noise_mask]))

    return {
        "min_samples": int(min_samples),
        "min_cluster_size": int(min_cluster_size),
        "xi": float(xi),
        "cluster_count": int(len(unique_clusters)),
        "noise_ratio": noise_ratio,
        "non_noise_coverage": non_noise_coverage,
        "silhouette_non_noise": None if silhouette is None else round(float(silhouette), 4),
        "rejected": rejected,
        "rejection_reason": rejection_reason,
        "selected": False,
    }


def _select_optics_candidate(
    matrix: np.ndarray,
    n_rows: int,
) -> tuple[OPTICS, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    ranked: list[tuple[tuple[float, float, float, float], OPTICS, np.ndarray, dict[str, Any]]] = []

    for min_samples, min_cluster_size, xi in _candidate_optics_params(n_rows):
        model, labels = _run_optics(
            matrix,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            xi=xi,
        )
        summary = _optics_candidate_summary(
            matrix=matrix,
            labels=labels,
            min_samples=min_samples,
            min_cluster_size=min_cluster_size,
            xi=xi,
        )
        summaries.append(summary)
        if summary["rejected"]:
            continue
        rank = (
            float(summary["silhouette_non_noise"] or -1.0),
            -float(summary["noise_ratio"]),
            -float(max(summary["cluster_count"] - 12, 0)),
            float(summary["non_noise_coverage"]),
        )
        ranked.append((rank, model, labels, summary))

    if ranked:
        _, selected_model, selected_labels, selected_summary = max(ranked, key=lambda item: item[0])
        selected_summary["selected"] = True
        return selected_model, selected_labels, selected_summary, summaries

    fallback_min_samples = min(max(20, round(0.02 * n_rows)), n_rows)
    fallback_min_cluster_size = min(max(25, round(0.03 * n_rows)), n_rows)
    fallback_min_cluster_size = max(fallback_min_cluster_size, fallback_min_samples)
    fallback_model, fallback_labels = _run_optics(
        matrix,
        min_samples=int(fallback_min_samples),
        min_cluster_size=int(fallback_min_cluster_size),
        xi=0.05,
    )
    fallback_summary = _optics_candidate_summary(
        matrix=matrix,
        labels=fallback_labels,
        min_samples=int(fallback_min_samples),
        min_cluster_size=int(fallback_min_cluster_size),
        xi=0.05,
    )
    fallback_summary["selected"] = True
    summaries.append(fallback_summary)
    return fallback_model, fallback_labels, fallback_summary, summaries


def _build_umap_embedding(matrix: np.ndarray, n_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    from umap import UMAP

    n_neighbors = int(min(30, max(12, round(math.sqrt(max(n_rows, 1))))))
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.15,
        metric="euclidean",
        random_state=42,
        transform_seed=42,
    )
    with pywarnings.catch_warnings():
        pywarnings.filterwarnings(
            "ignore",
            message="n_jobs value 1 overridden to 1 by setting random_state",
            category=UserWarning,
        )
        pywarnings.filterwarnings(
            "ignore",
            message="'force_all_finite' was renamed to 'ensure_all_finite'",
            category=FutureWarning,
        )
        embedding = reducer.fit_transform(matrix)
    return embedding, {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "min_dist": 0.15,
        "metric": "euclidean",
        "random_state": 42,
    }


def _apply_display_jitter(points: np.ndarray, *, source: str) -> tuple[np.ndarray, bool]:
    if source != "in" or len(points) == 0:
        return points.copy(), False

    span_x = float(points[:, 0].max() - points[:, 0].min()) if len(points) else 0.0
    span_y = float(points[:, 1].max() - points[:, 1].min()) if len(points) else 0.0
    bucket_size = max(max(span_x, span_y, 1.0) / 55, 0.02)
    coordinate_groups: dict[tuple[float, float], list[int]] = defaultdict(list)
    for index, point in enumerate(points):
        coordinate_groups[
            (
                round(float(point[0]) / bucket_size),
                round(float(point[1]) / bucket_size),
            )
        ].append(index)

    duplicates = [indices for indices in coordinate_groups.values() if len(indices) > 1]
    if not duplicates:
        return points.copy(), False

    display = points.copy()
    base_radius = max(span_x, span_y, 1.0) * 0.045
    golden_angle = math.pi * (3 - math.sqrt(5))

    for indices in duplicates:
        count = len(indices)
        for rank, point_index in enumerate(indices):
            radius = base_radius * math.sqrt((rank + 0.5) / count)
            angle = rank * golden_angle
            display[point_index, 0] += math.cos(angle) * radius
            display[point_index, 1] += math.sin(angle) * radius

    return display, True


def _overlap_stats(raw_points: np.ndarray, display_points: np.ndarray, *, jitter_applied: bool, source: str) -> dict[str, Any]:
    if len(raw_points) == 0:
        return {
            "overlap_pct": 0.0,
            "unique_coordinates_raw": 0,
            "unique_coordinates_display": 0,
            "jitter_applied": jitter_applied,
        }

    if source == "in":
        span_x = float(raw_points[:, 0].max() - raw_points[:, 0].min()) if len(raw_points) else 0.0
        span_y = float(raw_points[:, 1].max() - raw_points[:, 1].min()) if len(raw_points) else 0.0
        bucket_size = max(max(span_x, span_y, 1.0) / 55, 0.02)
        raw_unique = int(
            np.unique(
                np.column_stack(
                    [
                        np.round(raw_points[:, 0] / bucket_size),
                        np.round(raw_points[:, 1] / bucket_size),
                    ]
                ),
                axis=0,
            ).shape[0]
        )
    else:
        raw_unique = int(np.unique(np.round(raw_points, 8), axis=0).shape[0])
    display_unique = int(np.unique(np.round(display_points, 8), axis=0).shape[0])
    overlap_pct = round(float((1 - (raw_unique / max(len(raw_points), 1))) * 100), 4)
    return {
        "overlap_pct": overlap_pct,
        "unique_coordinates_raw": raw_unique,
        "unique_coordinates_display": display_unique,
        "jitter_applied": jitter_applied,
    }


def _build_cluster_ranges(ordered_labels: np.ndarray) -> list[dict[str, Any]]:
    ranges: list[dict[str, Any]] = []
    if len(ordered_labels) == 0:
        return ranges

    start = 0
    current = int(ordered_labels[0])
    for index in range(1, len(ordered_labels)):
        label = int(ordered_labels[index])
        if label == current:
            continue
        cluster_id, cluster_label, is_noise = _cluster_metadata(current)
        if not is_noise:
            ranges.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_label": cluster_label,
                    "start_order": int(start),
                    "end_order": int(index - 1),
                }
            )
        current = label
        start = index

    cluster_id, cluster_label, is_noise = _cluster_metadata(current)
    if not is_noise:
        ranges.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "start_order": int(start),
                "end_order": int(len(ordered_labels) - 1),
            }
        )
    return ranges


def _build_cluster_description(cluster_label: str, top_categories: dict[str, str], numeric_means: dict[str, float]) -> str:
    if cluster_label == "ruido":
        return "Puntos dispersos que no alcanzan la densidad mínima para formar un grupo estable."
    
    parts = []
    if top_categories:
        cat_desc = ", ".join(f"{k}='{v}'" for k, v in top_categories.items() if v and str(v).lower() != "nan" and str(v).lower() != "none")
        if cat_desc:
            parts.append(f"Predomina {cat_desc}")
    
    if numeric_means:
        num_desc = ", ".join(f"{k}≈{v:.1f}" for k, v in numeric_means.items() if v is not None)
        if num_desc:
            parts.append(f"promedios: {num_desc}")
            
    if not parts:
        return "Cluster denso sin una separación semántica clara en las variables principales."
        
    return " | ".join(parts).capitalize() + "."


def _build_optics_source_result(source: str, df: pd.DataFrame, week_id: str) -> dict[str, Any]:
    feature_columns = [
        str(column)
        for column in df.columns
        if not is_technical_id_column(str(column))
        and not (str(column) == "week" and int(df[column].nunique(dropna=True)) <= 1)
        and int(df[column].astype("string").nunique(dropna=True)) > 1
    ]
    artifacts = {
        "json_path": f"workspace/{week_id}/optics_{source}.json",
        "analysis_dataset_path": f"workspace/{week_id}/analysis_{source}_imputed.csv",
    }
    if not feature_columns:
        return {
            "source": source,
            "status": "not_applicable",
            "feature_columns": [],
            "preprocessing": [],
            "parameters": {},
            "selected_optics_parameters": {},
            "candidate_search_summary": [],
            "embedding_method": "umap",
            "embedding_parameters": {},
            "embedding_quality": {
                "trustworthiness": None,
                "overlap_stats": {
                    "overlap_pct": 0.0,
                    "unique_coordinates_raw": 0,
                    "unique_coordinates_display": 0,
                    "jitter_applied": False,
                },
                "pca_explained_variance_2d": None,
            },
            "cluster_count": 0,
            "noise_ratio": 0.0,
            "cluster_summary": [],
            "embedding_points": [],
            "pca_points": [],
            "reachability": [],
            "overlap_stats": {
                "overlap_pct": 0.0,
                "unique_coordinates_raw": 0,
                "unique_coordinates_display": 0,
                "jitter_applied": False,
            },
            "cluster_ranges": [],
            "artifacts": artifacts,
            "warnings": [
                {
                    "code": "optics_sin_features",
                    "severity": "warning",
                    "column": None,
                    "message": f"No hay columnas con variacion suficiente para ejecutar OPTICS en {source.upper()}.",
                    "suggestion": "Revisa si la fuente contiene solo variables constantes o tecnicas.",
                }
            ],
            "interpretation": "La fuente no tiene estructura suficiente para clustering automatico.",
        }

    numeric_features = [column for column in feature_columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical_features = [column for column in feature_columns if column not in numeric_features]
    preprocess = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )
    matrix = preprocess.fit_transform(df[feature_columns])
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.shape[1] == 0:
        return {
            "source": source,
            "status": "not_applicable",
            "feature_columns": feature_columns,
            "preprocessing": [],
            "parameters": {},
            "selected_optics_parameters": {},
            "candidate_search_summary": [],
            "embedding_method": "umap",
            "embedding_parameters": {},
            "embedding_quality": {
                "trustworthiness": None,
                "overlap_stats": {
                    "overlap_pct": 0.0,
                    "unique_coordinates_raw": 0,
                    "unique_coordinates_display": 0,
                    "jitter_applied": False,
                },
                "pca_explained_variance_2d": None,
            },
            "cluster_count": 0,
            "noise_ratio": 0.0,
            "cluster_summary": [],
            "embedding_points": [],
            "pca_points": [],
            "reachability": [],
            "overlap_stats": {
                "overlap_pct": 0.0,
                "unique_coordinates_raw": 0,
                "unique_coordinates_display": 0,
                "jitter_applied": False,
            },
            "cluster_ranges": [],
            "artifacts": artifacts,
            "warnings": [
                {
                    "code": "optics_matriz_vacia",
                    "severity": "warning",
                    "column": None,
                    "message": f"La matriz transformada de {source.upper()} quedo vacia tras el preprocesamiento.",
                    "suggestion": "Ajusta las features disponibles antes de clustering.",
                }
            ],
            "interpretation": "No fue posible construir una representacion valida para clustering.",
        }

    n_rows = int(len(df))
    model, labels, selected_candidate, candidate_summaries = _select_optics_candidate(matrix, n_rows)
    unique_clusters = sorted(label for label in set(labels.tolist()) if label != -1)
    cluster_count = len(unique_clusters)
    noise_ratio = round(float((labels == -1).mean() * 100), 4)

    pca_model = PCA(n_components=2 if matrix.shape[1] >= 2 else 1, random_state=42)
    pca_projection = pca_model.fit_transform(matrix)
    if pca_projection.shape[1] == 1:
        pca_projection = np.column_stack([pca_projection[:, 0], np.zeros(len(pca_projection))])
    pca_explained_variance_2d = round(float(np.sum(pca_model.explained_variance_ratio_[:2])), 4)

    umap_projection, embedding_parameters = _build_umap_embedding(matrix, n_rows)
    display_projection, jitter_applied = _apply_display_jitter(umap_projection, source=source)
    overlap_stats = _overlap_stats(umap_projection, display_projection, jitter_applied=jitter_applied, source=source)
    trustworthiness_neighbors = int(min(10, max(2, len(matrix) - 1)))
    embedding_trustworthiness = (
        round(float(trustworthiness(matrix, umap_projection, n_neighbors=trustworthiness_neighbors)), 4)
        if len(matrix) > 2
        else None
    )
    raw_group_sizes = pd.Series(list(map(tuple, np.round(umap_projection, 8)))).value_counts()

    embedding_points = []
    pca_points = []
    for point, pca_point, label, raw_point in zip(display_projection, pca_projection, labels, umap_projection, strict=True):
        cluster_id, cluster_label, is_noise = _cluster_metadata(int(label))
        display_weight = int(raw_group_sizes.get(tuple(np.round(raw_point, 8)), 1))
        embedding_points.append(
            {
                "x": float(point[0]),
                "y": float(point[1]),
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "is_noise": is_noise,
                "display_weight": display_weight,
            }
        )
        pca_points.append(
            {
                "x": float(pca_point[0]),
                "y": float(pca_point[1]),
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "is_noise": is_noise,
                "display_weight": display_weight,
            }
        )

    ordered_reachability = model.reachability_[model.ordering_]
    ordered_labels = labels[model.ordering_]
    reachability = [
        {
            "order": int(index),
            "reachability": _safe_float(value),
            "cluster_id": _cluster_metadata(int(label))[0],
            "cluster_label": _cluster_metadata(int(label))[1],
            "is_noise": _cluster_metadata(int(label))[2],
        }
        for index, (value, label) in enumerate(zip(ordered_reachability, ordered_labels, strict=True))
    ]
    cluster_ranges = _build_cluster_ranges(ordered_labels)

    summary_rows: list[dict[str, Any]] = []
    categorical_summary_columns = categorical_features[:3]
    numeric_summary_columns = numeric_features[:3]
    for label in sorted(set(labels.tolist())):
        cluster_df = df[labels == label]
        if cluster_df.empty:
            continue
        cluster_id, cluster_label, _ = _cluster_metadata(int(label))
        top_categories = {}
        for column in categorical_summary_columns:
            top_categories[column] = str(_series_mode(cluster_df[column].astype("string")))
        numeric_means = {}
        for column in numeric_summary_columns:
            numeric_means[column] = float(pd.to_numeric(cluster_df[column], errors="coerce").mean())
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_label": cluster_label,
                "description": _build_cluster_description(cluster_label, top_categories, numeric_means),
                "size": int(len(cluster_df)),
                "pct": round(float(len(cluster_df) / max(len(df), 1) * 100), 4),
                "top_categories": top_categories,
                "numeric_means": numeric_means,
            }
        )

    optics_warnings: list[dict[str, Any]] = []
    if cluster_count == 0:
        optics_warnings.append(
            {
                "code": "optics_sin_clusters",
                "severity": "warning",
                "column": None,
                "message": f"OPTICS no detecto clusters densos en {source.upper()}; la fuente quedo dominada por ruido.",
                "suggestion": "Interpreta el resultado como baja densidad o alta heterogeneidad en la fuente.",
            }
        )
    if noise_ratio >= 60:
        optics_warnings.append(
            {
                "code": "optics_ruido_alto",
                "severity": "warning",
                "column": None,
                "message": f"{source.upper()} presenta {noise_ratio:.2f}% de ruido bajo OPTICS.",
                "suggestion": "Usa el clustering como apoyo descriptivo y no como segmentacion cerrada.",
            }
        )
    if overlap_stats["overlap_pct"] >= 20:
        optics_warnings.append(
            {
                "code": "embedding_overlap_alto",
                "severity": "warning",
                "column": None,
                "message": f"La proyeccion visual de {source.upper()} concentra {overlap_stats['overlap_pct']:.2f}% de overlap antes del jitter.",
                "suggestion": "Interpretar la nube junto con densidades, centroides y tabla resumen; no solo por dispersion visual.",
            }
        )

    interpretation = (
        f"OPTICS identifica {cluster_count} clusters densos en {source.upper()} con {noise_ratio:.2f}% de ruido."
        if cluster_count > 0
        else f"OPTICS no encontro clusters densos estables en {source.upper()}."
    )

    return {
        "source": source,
        "status": "available",
        "feature_columns": feature_columns,
        "preprocessing": [
            "RobustScaler en variables numericas",
            "OneHotEncoder(handle_unknown='ignore') en variables categoricas",
            "OPTICS sobre el espacio transformado",
            "UMAP 2D para visualizacion",
        ],
        "parameters": {
            "min_samples": int(selected_candidate["min_samples"]),
            "min_cluster_size": int(selected_candidate["min_cluster_size"]),
            "xi": float(selected_candidate["xi"]),
            "metric": "euclidean",
        },
        "selected_optics_parameters": {
            "min_samples": int(selected_candidate["min_samples"]),
            "min_cluster_size": int(selected_candidate["min_cluster_size"]),
            "xi": float(selected_candidate["xi"]),
            "metric": "euclidean",
        },
        "candidate_search_summary": candidate_summaries,
        "embedding_method": "umap",
        "embedding_parameters": embedding_parameters,
        "embedding_quality": {
            "trustworthiness": embedding_trustworthiness,
            "overlap_stats": overlap_stats,
            "pca_explained_variance_2d": pca_explained_variance_2d,
        },
        "cluster_count": cluster_count,
        "noise_ratio": noise_ratio,
        "cluster_summary": summary_rows,
        "embedding_points": embedding_points,
        "pca_points": pca_points,
        "reachability": reachability,
        "overlap_stats": overlap_stats,
        "cluster_ranges": cluster_ranges,
        "artifacts": artifacts,
        "warnings": optics_warnings,
        "interpretation": interpretation,
    }


def _build_insights(
    sources: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    imputation: dict[str, Any],
    outliers: dict[str, Any],
    clustering: dict[str, Any],
) -> list[dict[str, str]]:
    insights: list[dict[str, str]] = []

    if not imputation["imputation_applied"]:
        insights.append(
            {
                "title": "No se detectaron NA en los seeds actuales",
                "evidence": "Las capas raw de IN y OUT llegan completas en Week 1.",
                "implication": "Las metricas descriptivas no dependen de imputacion en este seed concreto.",
                "next_step": "Mantener la capacidad de imputacion preparada para semanas futuras o datasets alternativos.",
            }
        )

    out_source = sources["out"]
    cat_num_rows = out_source["bivariate_categorical_numeric"]["rows"]
    significant = [row for row in cat_num_rows if row.get("p_value") is not None and float(row["p_value"]) < 0.05]
    if significant:
        strongest = significant[0]
        insights.append(
            {
                "title": "OUT muestra diferencias entre grupos sobre DaysInDeposit",
                "evidence": f"{strongest['feature']} presenta evidencia estadistica con {strongest['test_used']} (p={float(strongest['p_value']):.3g}).",
                "implication": "La variable puede ser una explicativa fuerte para la etapa de modelado.",
                "next_step": "Priorizar esta variable en Week 2 y en el modelamiento matematico posterior.",
            }
        )

    condition_profile = next(
        (item for item in sources["in"]["univariate_categorical"]["variables"] if item["column"] == "Condition"),
        None,
    )
    if condition_profile and condition_profile["top_categories"]:
        top = condition_profile["top_categories"][0]
        insights.append(
            {
                "title": "IN esta dominado por una condicion operacional principal",
                "evidence": f"En IN, la categoria dominante de Condition es {top['value']} con {top['pct']:.2f}% de participacion.",
                "implication": "La composicion de entradas no es uniforme y puede condicionar la lectura del resto de variables.",
                "next_step": "Usar Condition como eje descriptivo propio de IN en el portafolio academico.",
            }
        )

    owner_comparison = next((item for item in comparison["categorical_comparisons"] if item["column"] == "Owner"), None)
    if owner_comparison and owner_comparison["categories"]:
        top_owner = owner_comparison["categories"][0]
        insights.append(
            {
                "title": "La mezcla de propietarios difiere entre IN y OUT",
                "evidence": f"La categoria dominante compartida es {top_owner['category']} con {top_owner['in_pct']:.2f}% en IN y {top_owner['out_pct']:.2f}% en OUT.",
                "implication": "La comparacion entre fuentes debe hacerse por distribucion y no solo por volumen agregado.",
                "next_step": "Mantener graficos comparativos por variable compartida en las entregas academicas.",
            }
        )

    out_outliers = outliers["sources"]["out"]
    if out_outliers["status"] == "available" and out_outliers["flagged_counts"] > 0:
        insights.append(
            {
                "title": "OUT presenta valores extremos relevantes",
                "evidence": f"Se marcaron {out_outliers['flagged_counts']} registros extremos ({out_outliers['flagged_ratio']:.2f}%) en la fuente OUT.",
                "implication": "El target puede requerir medidas robustas antes de inferir efectos causales o ajustar modelos.",
                "next_step": "Comparar media y mediana del target y documentar sensibilidad en semanas posteriores.",
            }
        )

    for source in ("in", "out"):
        optics_source = clustering["sources"][source]
        if optics_source["status"] == "available":
            insights.append(
                {
                    "title": f"OPTICS sobre {source.upper()} detecta estructura {'densa' if optics_source['cluster_count'] > 0 else 'difusa'}",
                    "evidence": optics_source["interpretation"],
                    "implication": "El clustering funciona como una capa exploratoria adicional para entender segmentos operacionales.",
                    "next_step": "Usar los clusters solo como apoyo descriptivo y no como verdad cerrada del sistema.",
                }
            )
            if optics_source["cluster_count"] > 0:
                break

    return insights[:6]


def build_week1_academic_eda_bundle(
    *,
    week_id: str,
    source_frames: dict[str, pd.DataFrame],
    canonical_df: pd.DataFrame,
    metadata: dict[str, Any],
    academic_metadata: dict[str, Any],
    clustering_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_in = source_frames["in"].copy()
    raw_out = source_frames["out"].copy()
    analysis_in, imputation_in = _impute_source(raw_in, "in")
    analysis_out, imputation_out = _impute_source(raw_out, "out")

    source_sections = {
        "in": _build_source_section(
            raw_in,
            analysis_in,
            "in",
            academic_metadata=academic_metadata,
            source_warnings=_filter_pipeline_warnings(metadata, "in", raw_in),
        ),
        "out": _build_source_section(
            raw_out,
            analysis_out,
            "out",
            academic_metadata=academic_metadata,
            source_warnings=_filter_pipeline_warnings(metadata, "out", raw_out),
        ),
    }

    comparison = _build_comparison(raw_in, raw_out, metadata)
    imputation = {
        "raw_missing_summary": {
            "in": imputation_in["missing_summary"],
            "out": imputation_out["missing_summary"],
        },
        "imputation_applied": bool(imputation_in["imputation_applied"] or imputation_out["imputation_applied"]),
        "strategy_by_column": {
            "in": imputation_in["strategy_by_column"],
            "out": imputation_out["strategy_by_column"],
        },
        "imputed_counts": {
            "in": imputation_in["imputed_counts"],
            "out": imputation_out["imputed_counts"],
        },
        "analysis_dataset_paths": {
            "in": f"workspace/{week_id}/analysis_in_imputed.csv",
            "out": f"workspace/{week_id}/analysis_out_imputed.csv",
        },
        "notes": imputation_in["notes"] + imputation_out["notes"],
    }
    outliers = {
        "policy": "diagnosticar_sin_filtrar",
        "sources": {
            "in": _outlier_source_section(raw_in),
            "out": _outlier_source_section(raw_out),
        },
    }

    resolved_clustering_payload = clustering_payload
    if resolved_clustering_payload is None:
        resolved_clustering_payload = {
            "week_id": week_id,
            "stage_name": "EDA",
            "sources": {
                "in": _build_optics_source_result("in", analysis_in, week_id),
                "out": _build_optics_source_result("out", analysis_out, week_id),
            },
        }
        resolved_clustering_payload["warnings"] = _dedupe_warnings(
            resolved_clustering_payload["sources"]["in"]["warnings"] + resolved_clustering_payload["sources"]["out"]["warnings"]
        )
    else:
        resolved_clustering_payload = {
            "week_id": resolved_clustering_payload.get("week_id", week_id),
            "stage_name": resolved_clustering_payload.get("stage_name", "EDA"),
            "sources": resolved_clustering_payload["sources"],
            "warnings": _dedupe_warnings(resolved_clustering_payload.get("warnings", [])),
        }

    optics_summary = {
        "endpoint": f"/api/v1/weeks/{week_id}/clustering",
        "sources": {
            source: {
                "status": payload["status"],
                "cluster_count": payload["cluster_count"],
                "noise_ratio": payload["noise_ratio"],
                "artifact_path": payload["artifacts"]["json_path"],
            }
            for source, payload in resolved_clustering_payload["sources"].items()
        },
    }

    top_level_warnings = _dedupe_warnings(
        _global_warnings(metadata)
        + source_sections["in"]["warnings"]
        + source_sections["out"]["warnings"]
        + resolved_clustering_payload["warnings"]
    )

    eda_payload = {
        "problem_definition": academic_metadata,
        "sources": source_sections,
        "comparison": comparison,
        "imputation": imputation,
        "outliers": outliers,
        "optics_summary": optics_summary,
        "insights": _build_insights(source_sections, comparison, imputation, outliers, resolved_clustering_payload),
        "warnings": top_level_warnings,
    }

    return {
        "eda_payload": eda_payload,
        "clustering_payload": resolved_clustering_payload,
        "imputed_frames": {
            "in": analysis_in,
            "out": analysis_out,
        },
        "canonical_df": canonical_df,
    }
