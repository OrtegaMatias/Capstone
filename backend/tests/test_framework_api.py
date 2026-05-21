from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.framework_service import framework_service


def test_framework_summary_bootstraps_workspace(framework_client, framework_repo_root: Path) -> None:
    response = framework_client.get("/api/v1/framework")
    assert response.status_code == 200
    payload = response.json()

    assert payload["framework_name"]
    assert len(payload["weeks"]) == 6
    assert [week["week_id"] for week in payload["weeks"][:2]] == ["week-1", "week-2"]
    assert payload["weeks"][0]["status"] == "active"
    assert payload["weeks"][1]["status"] == "active"
    assert payload["weeks"][2]["status"] == "active"

    assert (framework_repo_root / "workspace/week-1/canonical.csv").exists()
    assert (framework_repo_root / "workspace/week-1/analysis_in_imputed.csv").exists()
    assert (framework_repo_root / "workspace/week-1/analysis_out_imputed.csv").exists()
    assert (framework_repo_root / "workspace/week-1/optics_in.json").exists()
    assert (framework_repo_root / "workspace/week-1/optics_out.json").exists()
    assert (framework_repo_root / "workspace/week-2/canonical.csv").exists()
    assert (framework_repo_root / "workspace/week-3/canonical.csv").exists()
    assert (framework_repo_root / "workspace/week-3/optimization_model.json").exists()
    assert (framework_repo_root / "workspace/week-3/weekly_balance.csv").exists()
    assert (framework_repo_root / "workspace/week-3/official_model.mod").exists()
    assert (framework_repo_root / "workspace/week-3/official_model.dat").exists()
    assert (framework_repo_root / "workspace/week-3/baseline_validation.mod").exists()
    assert (framework_repo_root / "workspace/week-3/baseline_validation.dat").exists()
    assert (framework_repo_root / "workspace/week-3/cplex_segregations.csv").exists()
    assert (framework_repo_root / "workspace/week-3/baseline_segregations.csv").exists()
    assert (framework_repo_root / "workspace/week-3/cplex_ml_signals.csv").exists()
    assert (framework_repo_root / "workspace/week-3/baseline_inventory_seed.csv").exists()
    assert (framework_repo_root / "workspace/week-3/report.md").exists()


def test_week_1_eda_and_report_are_available(framework_client, framework_repo_root: Path) -> None:
    preview = framework_client.get("/api/v1/weeks/week-1/preview")
    assert preview.status_code == 200
    assert preview.json()["total_rows"] > 1000

    eda = framework_client.get("/api/v1/weeks/week-1/eda")
    assert eda.status_code == 200
    eda_payload = eda.json()
    assert eda_payload["problem_definition"]["objective"]
    assert eda_payload["sources"]["in"]["dataset_audit"]["shape"][0] > 1000
    assert eda_payload["sources"]["out"]["dataset_audit"]["shape"][0] > 1000
    assert eda_payload["comparison"]["categorical_comparisons"]
    assert eda_payload["imputation"]["imputation_applied"] is False
    assert eda_payload["outliers"]["sources"]["in"]["status"] in {"available", "not_applicable"}
    assert eda_payload["outliers"]["sources"]["out"]["status"] in {"available", "not_applicable"}
    assert eda_payload["sources"]["in"]["temporal_diagnostics"]["status"] == "not_applicable"
    assert eda_payload["sources"]["out"]["temporal_diagnostics"]["status"] == "not_applicable"

    clustering = framework_client.get("/api/v1/weeks/week-1/clustering")
    assert clustering.status_code == 200
    clustering_payload = clustering.json()
    assert set(clustering_payload["sources"].keys()) == {"in", "out"}
    assert clustering_payload["sources"]["in"]["embedding_method"] == "umap"
    assert clustering_payload["sources"]["out"]["embedding_method"] == "umap"
    assert "Unnamed: 0" not in clustering_payload["sources"]["in"]["feature_columns"]
    assert "week" not in clustering_payload["sources"]["in"]["feature_columns"]
    assert "Unnamed: 0" not in clustering_payload["sources"]["out"]["feature_columns"]
    assert "week" not in clustering_payload["sources"]["out"]["feature_columns"]
    assert clustering_payload["sources"]["in"]["embedding_points"]
    assert clustering_payload["sources"]["out"]["embedding_points"]
    assert clustering_payload["sources"]["in"]["candidate_search_summary"]
    assert clustering_payload["sources"]["out"]["candidate_search_summary"]
    assert any(item["selected"] for item in clustering_payload["sources"]["in"]["candidate_search_summary"])
    assert any(item["selected"] for item in clustering_payload["sources"]["out"]["candidate_search_summary"])
    assert clustering_payload["sources"]["in"]["embedding_quality"]["overlap_stats"]["jitter_applied"] is True
    assert clustering_payload["sources"]["in"]["overlap_stats"]["overlap_pct"] > 0
    assert clustering_payload["sources"]["out"]["selected_optics_parameters"]["min_samples"] >= 5
    assert clustering_payload["sources"]["out"]["cluster_ranges"]
    assert "pca_points" in clustering_payload["sources"]["in"]

    report = framework_client.get("/api/v1/weeks/week-1/report")
    assert report.status_code == 200
    report_payload = report.json()
    assert "Semana 1 - EDA" in report_payload["markdown_content"]
    assert "## Introduccion" in report_payload["markdown_content"]
    assert "## Metodologia" in report_payload["markdown_content"]
    assert "## Analisis del dataset IN" in report_payload["markdown_content"]
    assert "## Analisis del dataset OUT" in report_payload["markdown_content"]
    assert "## Comparacion IN vs OUT" in report_payload["markdown_content"]
    assert "## Calidad de datos e imputacion" in report_payload["markdown_content"]
    assert "## Outliers y anomalias" in report_payload["markdown_content"]
    assert "## Clustering OPTICS" in report_payload["markdown_content"]
    assert "## Conclusiones e hipotesis" in report_payload["markdown_content"]
    assert "UMAP 2D se usa solo para visualizacion" in report_payload["markdown_content"]
    assert "<h1>Week 1 - EDA</h1>" not in report_payload["html_content"]
    assert "<h1>Semana 1 - EDA</h1>" in report_payload["html_content"]
    assert (framework_repo_root / "workspace/week-1/report.html").exists()


def test_week_1_imputation_creates_derived_dataset_when_missing_values_exist(
    framework_repo_root: Path,
) -> None:
    in_seed = framework_repo_root / "seed/Week1/Grupo1_in.csv"
    out_seed = framework_repo_root / "seed/Week1/Grupo1_out.csv"

    seed_in_df = pd.read_csv(in_seed, sep=None, engine="python")
    seed_out_df = pd.read_csv(out_seed, sep=None, engine="python")
    seed_in_df.loc[:9, "Owner"] = np.nan
    seed_out_df.loc[:11, "DaysInDeposit"] = np.nan
    seed_in_df.to_csv(in_seed, index=False, na_rep="")
    seed_out_df.to_csv(out_seed, index=False, na_rep="")

    framework_service.reconfigure(
        repo_root=framework_repo_root,
        seed_dir=framework_repo_root / "seed",
        workspace_dir=framework_repo_root / "workspace",
        manifest_path=framework_repo_root / "framework/manifest.json",
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/weeks/week-1/eda")
        assert response.status_code == 200
        payload = response.json()
        assert payload["imputation"]["raw_missing_summary"]["in"]["Owner"]["count"] == 10
        assert payload["imputation"]["raw_missing_summary"]["out"]["DaysInDeposit"]["count"] == 12
        assert payload["imputation"]["imputation_applied"] is True
        assert payload["imputation"]["imputed_counts"]["in"]["Owner"] == 10
        assert payload["imputation"]["imputed_counts"]["out"]["DaysInDeposit"] == 12
        assert (framework_repo_root / "workspace/week-1/analysis_in_imputed.csv").exists()
        assert (framework_repo_root / "workspace/week-1/analysis_out_imputed.csv").exists()

    framework_service.reconfigure()


def test_week_2_ml_overview_uses_temporal_holdout(framework_client) -> None:
    response = framework_client.get("/api/v1/weeks/week-2/ml/overview")
    assert response.status_code == 200
    payload = response.json()

    assert payload["model_built"] is True
    assert payload["split"]["test_weeks"] == ["5"]
    assert payload["split"]["train_rows"] > 0
    assert payload["models"]
    assert payload["preprocessing_benchmarks"]
    assert payload["heuristic_models"]
    assert payload["segment_reports"]
    assert payload["target_transformation_diagnostics"]["scope"] == "train_only"
    assert payload["target_transformation_diagnostics"]["boxplot_data"]
    assert payload["strategy_comparison"]["best_regression"]["metrics"]["mae"] is not None
    assert payload["strategy_comparison"]["best_heuristic"]["metrics"]["mae"] is not None
    assert any(row["strategy_name"] == "log1p" for row in payload["preprocessing_benchmarks"])
    assert any(row["strategy_name"] == "log1p_drop_outliers" for row in payload["preprocessing_benchmarks"])
    assert any(model["model_name"] == "Hierarchical Backoff" for model in payload["heuristic_models"])
    assert any(warning["code"] == "temporal_holdout" for warning in payload["warnings"])

    report = framework_client.get("/api/v1/weeks/week-2/report")
    assert report.status_code == 200
    report_payload = report.json()
    assert "## Evaluacion ML" in report_payload["markdown_content"]
    assert "## Transformacion del target" in report_payload["markdown_content"]
    assert "## Benchmark de preprocesamiento" in report_payload["markdown_content"]
    assert "## Segmentacion representativa" in report_payload["markdown_content"]


def test_week_2_invalidates_legacy_ml_cache(framework_client, framework_repo_root: Path) -> None:
    cache_path = framework_repo_root / "workspace/week-2/ml_overview.json"
    cache_path.write_text(
        json.dumps(
            {
                "week_id": "week-2",
                "split": {"train_weeks": ["1"], "test_weeks": ["2"], "train_rows": 1, "test_rows": 1},
            }
        ),
        encoding="utf-8",
    )

    cached = framework_client.get("/api/v1/weeks/week-2/ml/cached")
    assert cached.status_code == 204
    assert not cache_path.exists()

    overview = framework_client.get("/api/v1/weeks/week-2/ml/overview")
    assert overview.status_code == 200
    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "fingerprint" in cache_payload
    assert cache_payload["payload"]["split"]["test_weeks"] == ["5"]


def test_week_3_optimization_payload_is_available(framework_client, framework_repo_root: Path) -> None:
    response = framework_client.get("/api/v1/weeks/week-3/optimization-model")
    assert response.status_code == 200
    payload = response.json()

    assert payload["summary"]["bay_count"] == 77
    assert payload["summary"]["planning_weeks"] == 33
    assert payload["summary"]["total_capacity_teu"] > 5000
    assert payload["summary"]["peak_active_teu"] > payload["summary"]["total_capacity_teu"]
    assert payload["ml_integration"]["urgency_rule"] == "u_c = P(Alta) + 0.5 * P(Media)"
    assert payload["owner_selections"]
    assert any(row["decision"] == "owner" for row in payload["owner_selections"])
    assert payload["academic_formulation"]["dimensions"]
    assert payload["academic_formulation"]["given_data"]
    assert payload["academic_formulation"]["parameters"]
    assert payload["academic_formulation"]["decision_variables"]
    assert payload["academic_formulation"]["objective"]["equation_latex"]
    assert payload["academic_formulation"]["constraints"]
    assert payload["academic_formulation"]["week4_extensions"]
    assert payload["notation_bridge"]["equations"]
    assert payload["notation_bridge"]["aggregation_rules"]
    assert payload["notation_bridge"]["segregation_samples"]
    assert payload["cplex_export"]["version"] == "week3_cplex_official_v2"
    assert payload["cplex_export"]["j_count"] == 88
    assert payload["cplex_export"]["jd_count"] == 44
    assert payload["cplex_export"]["jo_count"] == 44
    assert payload["cplex_export"]["k_count"] == 44
    assert payload["cplex_export"]["owners"] == [str(index) for index in range(1, 12)]
    assert payload["cplex_export"]["lambda_ml"] == 1.0
    assert payload["cplex_export"]["w_urg"] == 2.0
    assert payload["cplex_export"]["w_slow"] == 1.0
    assert "uBar" in payload["cplex_export"]["ml_signal_rule"]
    assert payload["cplex_export"]["sample_pairs"]
    assert payload["baseline_export"]["version"] == "week3_baseline_validation_v6"
    assert payload["baseline_export"]["j_count"] == 44
    assert payload["baseline_export"]["b_count"] == 77
    assert payload["baseline_export"]["c_labels"] == ["D", "O"]
    assert payload["baseline_export"]["validated_week"] == 1
    assert payload["baseline_export"]["inspection_mapping"] == "I -> D"
    assert payload["baseline_export"]["sample_segregations"]
    assert payload["container_samples"]
    first_dimension = payload["academic_formulation"]["dimensions"][0]
    assert "symbol_latex" in first_dimension
    assert "description" in first_dimension
    objective = payload["academic_formulation"]["objective"]
    assert objective["components"]
    first_sample = payload["container_samples"][0]
    assert 0.0 <= first_sample["u_c"] <= 1.0
    assert first_sample["penalty_examples"]
    first_penalty = first_sample["penalty_examples"][0]
    assert 0.0 <= first_penalty["q_b"] <= 1.0
    assert first_penalty["p_cb"] >= 0.0
    first_segregation = payload["notation_bridge"]["segregation_samples"][0]
    assert 0.0 <= first_segregation["avg_u_jt"] <= 1.0
    assert first_segregation["alpha_l"] == 1
    assert any(warning["code"] == "yard_capacity_exceeded" for warning in payload["warnings"])
    assert "Conjuntos" not in payload["formulation"]  # schema should stay structured

    report = framework_client.get("/api/v1/weeks/week-3/report")
    assert report.status_code == 200
    report_payload = report.json()
    assert "## Dimensiones e indices" in report_payload["markdown_content"]
    assert "## Datos observados" in report_payload["markdown_content"]
    assert "## Parametros" in report_payload["markdown_content"]
    assert "## Variables" in report_payload["markdown_content"]
    assert "## Funcion objetivo" in report_payload["markdown_content"]
    assert "## Restricciones" in report_payload["markdown_content"]
    assert "## Puente con la capa operacional" in report_payload["markdown_content"]
    assert "## Export CPLEX oficial" in report_payload["markdown_content"]
    assert "## Baseline de validacion del primer .mod" in report_payload["markdown_content"]
    assert "## Supuestos y extensiones" in report_payload["markdown_content"]
    assert "$$\\min Z =" in report_payload["markdown_content"]
    assert "$$u_c = P_c(\\mathrm{Alta}) + 0.5\\,P_c(\\mathrm{Media})$$" in report_payload["markdown_content"]
    assert "<h1>Semana 3 - Modelamiento matematico</h1>" in report_payload["html_content"]

    dat_path = framework_repo_root / "workspace/week-3/official_model.dat"
    dat_text = dat_path.read_text(encoding="utf-8")
    mod_path = framework_repo_root / "workspace/week-3/official_model.mod"
    mod_text = mod_path.read_text(encoding="utf-8")
    assert "J = 1..88;" in dat_text
    assert "B = 1..77;" in dat_text
    assert "T = 1..33;" in dat_text
    assert "JD = {1, 3, 5" in dat_text
    assert "JO = {2, 4, 6" in dat_text
    assert "K = {" in dat_text
    assert "Gamma =" in dat_text
    assert "tau =" in dat_text
    assert "q =" in dat_text
    assert "uBar =" in dat_text
    assert "lambdaML = 1.000000;" in dat_text
    assert "wUrg = 2.000000;" in dat_text
    assert "wSlow = 1.000000;" in dat_text
    assert "CFbase =" in dat_text
    assert "range J = ...;" in mod_text
    assert "tuple Transition" in mod_text
    assert "float tau[J] = ...;" in mod_text
    assert "float q[B] = ...;" in mod_text
    assert "float uBar[J][T] = ...;" in mod_text
    assert "float CFbase[J][B] = ...;" in mod_text
    assert "dvar int+ r[K][T];" in mod_text
    assert "forall(j in JD, t in T)" in mod_text
    assert "CFbase[j][b] + lambdaML" in mod_text

    baseline_dat_path = framework_repo_root / "workspace/week-3/baseline_validation.dat"
    baseline_dat_text = baseline_dat_path.read_text(encoding="utf-8")
    baseline_mod_path = framework_repo_root / "workspace/week-3/baseline_validation.mod"
    baseline_mod_text = baseline_mod_path.read_text(encoding="utf-8")
    assert "J = 1..44;" in baseline_dat_text
    assert "B = 1..77;" in baseline_dat_text
    assert 'C = {"D", "O"};' in baseline_dat_text
    assert "dem =" in baseline_dat_text
    assert "Inv0 =" in baseline_dat_text
    assert "F = 999999;" in baseline_dat_text
    assert '{string} C = {"D", "O"};' in baseline_mod_text
    assert "int dem[J][C] = ...;" in baseline_mod_text
    assert "int Inv0[J][C] = ...;" in baseline_mod_text
    assert "dvar int+ s0[J][B][C];" in baseline_mod_text
    assert "sum(b in B) s0[j][b][c] == Inv0[j][c];" in baseline_mod_text
    assert 'I_pos[j]["D"][b]' in baseline_mod_text
    assert 'I_pos[j]["O"][b]' in baseline_mod_text

    segregation_df = pd.read_csv(framework_repo_root / "workspace/week-3/cplex_segregations.csv")
    assert segregation_df.shape[0] == 88
    assert set(segregation_df["subset"]) == {"JD", "JO"}
    assert segregation_df["owner"].astype(str).nunique() == 11
    assert "tau" in segregation_df.columns
    ml_signal_df = pd.read_csv(framework_repo_root / "workspace/week-3/cplex_ml_signals.csv")
    assert ml_signal_df.shape[0] == 88 * 33
    assert ml_signal_df["j"].nunique() == 88
    assert ml_signal_df["week"].nunique() == 33
    assert ml_signal_df["uBar"].between(0.0, 1.0).all()
    assert set(ml_signal_df["signal_origin"].unique()) <= {"weekly", "group_mean_fallback"}

    baseline_segregation_df = pd.read_csv(framework_repo_root / "workspace/week-3/baseline_segregations.csv")
    assert baseline_segregation_df.shape[0] == 44
    assert baseline_segregation_df["owner"].astype(str).nunique() == 11
    assert baseline_segregation_df["RF"].isin([0, 1]).all()
    baseline_inventory_df = pd.read_csv(framework_repo_root / "workspace/week-3/baseline_inventory_seed.csv")
    positive_inventory = baseline_inventory_df[baseline_inventory_df["assigned_units"].fillna(0) > 0]
    assert positive_inventory["bay_id"].notna().all()


def test_week_notes_persist_inside_workspace(framework_client, framework_repo_root: Path) -> None:
    content = "Hallazgo clave: week 1 tiene calidad suficiente para iniciar EDA."
    save = framework_client.put("/api/v1/weeks/week-1/notes", json={"content": content})
    assert save.status_code == 200
    assert save.json()["ok"] is True

    get_notes = framework_client.get("/api/v1/weeks/week-1/notes")
    assert get_notes.status_code == 200
    assert get_notes.json()["content"] == content
    assert (framework_repo_root / "workspace/week-1/notes.md").read_text(encoding="utf-8") == content


def test_week_1_summary_exposes_academic_context(framework_client) -> None:
    response = framework_client.get("/api/v1/weeks/week-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["academic_context"]["objective"]
    assert payload["academic_context"]["unit_of_observation"]
    assert payload["academic_context"]["initial_hypotheses"]


def test_framework_bootstrap_fails_if_week_1_academic_metadata_is_missing(framework_repo_root: Path) -> None:
    academic_path = framework_repo_root / "framework/academic/week-1.json"
    academic_path.unlink()

    framework_service.reconfigure(
        repo_root=framework_repo_root,
        seed_dir=framework_repo_root / "seed",
        workspace_dir=framework_repo_root / "workspace",
        manifest_path=framework_repo_root / "framework/manifest.json",
    )

    with pytest.raises(FileNotFoundError, match="Academic metadata for week-1 not found"):
        framework_service.bootstrap()

    framework_service.reconfigure()
