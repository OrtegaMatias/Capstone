from __future__ import annotations

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
    assert payload["weeks"][2]["status"] == "scaffolded"

    assert (framework_repo_root / "workspace/week-1/canonical.csv").exists()
    assert (framework_repo_root / "workspace/week-1/analysis_in_imputed.csv").exists()
    assert (framework_repo_root / "workspace/week-1/analysis_out_imputed.csv").exists()
    assert (framework_repo_root / "workspace/week-1/optics_in.json").exists()
    assert (framework_repo_root / "workspace/week-1/optics_out.json").exists()
    assert (framework_repo_root / "workspace/week-2/canonical.csv").exists()
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
    assert payload["metrics"]["mae"] is not None
    assert payload["metrics"]["baseline_mae"] is not None
    assert payload["predictions"]
    assert any(warning["code"] == "temporal_holdout" for warning in payload["warnings"])


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
