from __future__ import annotations

import json
import hashlib
import threading
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.core.config import Settings, get_settings
from app.etl.pipeline import ETLPipeline
from app.etl.step_cast_types import _cast_single_dataframe
from app.etl.step_clean_columns import _clean_dataframe
from app.etl.step_read_csv import robust_read_csv
from app.etl.types import DatasetInput, PipelineContext
from app.schemas.framework import WeekStatus
from app.stats.academic_eda import build_week1_academic_eda_bundle
from app.stats.eda import compute_eda
from app.stats.ml import compute_temporal_ml_overview
from app.stats.supervised import compute_anova, compute_multiple_regression_out, compute_supervised_overview
from app.stats.week3_optimization import build_week3_optimization_bundle
from app.storage.week_workspace_store import WeekWorkspaceStore


EXPECTED_IN_COLUMNS = ["Unnamed: 0", "Condition", "Owner", "Size", "Type", "Quality", "week"]
EXPECTED_OUT_COLUMNS = ["Unnamed: 0", "Owner", "Size", "Type", "Quality", "DaysInDeposit", "week"]


class FrameworkService:
    def __init__(self, settings: Settings | None = None):
        self.pipeline = ETLPipeline()
        self.reconfigure(settings=settings or get_settings())

    def reconfigure(
        self,
        settings: Settings | None = None,
        *,
        repo_root: Path | None = None,
        seed_dir: Path | None = None,
        workspace_dir: Path | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        resolved_settings = settings or get_settings()
        self.settings = resolved_settings
        self.repo_root = Path(repo_root or resolved_settings.repo_root).resolve()
        self.seed_dir = Path(seed_dir or resolved_settings.seed_dir).resolve()
        self.workspace_dir = Path(workspace_dir or resolved_settings.workspace_dir).resolve()
        self.manifest_path = Path(manifest_path or resolved_settings.framework_manifest_path).resolve()
        self.workspace_store = WeekWorkspaceStore(self.workspace_dir)
        self._week1_bundle_cache: dict[str, dict[str, Any]] = {}
        self._bootstrap_lock = threading.RLock()
        self._week1_bundle_lock = threading.RLock()

    def bootstrap(self) -> None:
        with self._bootstrap_lock:
            manifest = self._load_manifest()
            for week in manifest["weeks"]:
                self.workspace_store.ensure_week_dir(week["week_id"])
                self._ensure_academic_assets(week)
                if week["status"] == "active":
                    if self._active_week_needs_refresh(week):
                        self._build_active_week(week)
                    if week["week_id"] == "week-1" and not self._week1_derived_assets_exist(week["week_id"]):
                        self.refresh_week_report(week["week_id"])
                    if not self._report_exists(week["week_id"]):
                        self.refresh_week_report(week["week_id"])
                else:
                    notes_path = self.workspace_store.notes_path(week["week_id"])
                    if not notes_path.exists():
                        notes_path.write_text("", encoding="utf-8")
                    if not self._report_exists(week["week_id"]):
                        self.refresh_week_report(week["week_id"])

    def get_framework_summary(self) -> dict[str, Any]:
        self.bootstrap()
        manifest = self._load_manifest()
        return {
            "framework_name": manifest["framework_name"],
            "summary": manifest["summary"],
            "workspace_root": str(self.workspace_dir.relative_to(self.repo_root)),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "weeks": [self._build_week_summary(week) for week in manifest["weeks"]],
        }

    def get_week(self, week_id: str) -> dict[str, Any]:
        self.bootstrap()
        week = self._get_week_definition(week_id)
        summary = self._build_week_summary(week)
        summary.update(
            {
                "description": week["description"],
                "expected_inputs": week["expected_inputs"],
                "checklist": week["checklist"],
                "deliverables": week["deliverables"],
                "academic_context": self._week_academic_context_summary(week),
            }
        )
        return summary

    def get_week_preview(self, week_id: str, limit: int = 20) -> dict[str, Any]:
        week = self._require_analysis(week_id, expected="preview")
        df, _ = self.workspace_store.load_dataset(week["week_id"])
        preview_df = df.head(limit).copy()
        preview_df = preview_df.where(pd.notna(preview_df), None)
        return {
            "columns": [str(col) for col in df.columns],
            "rows": preview_df.to_dict(orient="records"),
            "total_rows": int(df.shape[0]),
        }

    def get_week_eda(self, week_id: str) -> dict[str, Any]:
        week = self._require_analysis(week_id, expected="eda")
        df, metadata = self.workspace_store.load_dataset(week["week_id"])
        if week["week_id"] == "week-1":
            return self._build_week1_eda_payload(week["week_id"], metadata)

        payload = compute_eda(df)
        payload["warnings"] = self._dedupe_warnings(metadata.get("warnings", []) + payload.get("warnings", []))
        return payload

    def get_week_clustering(self, week_id: str) -> dict[str, Any]:
        week = self._require_analysis(week_id, expected="clustering")
        if week["week_id"] != "week-1":
            raise ValueError(f"Clustering is not available for week '{week_id}'")
        return self._build_week1_bundle(week_id)["clustering_payload"]

    def get_week_ml_overview(self, week_id: str) -> dict[str, Any]:
        cached = self.get_week_ml_cached(week_id)
        if cached is not None:
            return cached
        week = self._require_analysis(week_id, expected="ml_overview")
        payload = self._build_week_ml_payload(week)
        payload["week_id"] = week["week_id"]
        payload["stage_name"] = week["stage_name"]
        payload["artifacts"] = self._artifact_payload(week)
        self._save_ml_cache(week_id, payload)
        return payload

    def get_week_optimization_model(self, week_id: str) -> dict[str, Any]:
        week = self._require_analysis(week_id, expected="optimization_model")
        if week["week_id"] != "week-3":
            raise ValueError(f"Optimization model is not available for week '{week_id}'")
        return self._build_week3_optimization_payload(week)

    def get_week_ml_cached(self, week_id: str) -> dict[str, Any] | None:
        """Return cached ML results or None if no cache exists."""
        cache_path = self.workspace_dir / week_id / "ml_overview.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
            except Exception:
                cache_path.unlink(missing_ok=True)
                return None

            if not isinstance(cached, dict):
                cache_path.unlink(missing_ok=True)
                return None

            expected_fingerprint = self._ml_cache_fingerprint(week_id)
            if cached.get("fingerprint") != expected_fingerprint or not isinstance(cached.get("payload"), dict):
                cache_path.unlink(missing_ok=True)
                return None

            return cached["payload"]
        return None

    def run_week_ml_with_progress(
        self,
        week_id: str,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> dict[str, Any]:
        """Run ML analysis with progress callback and cache the result."""
        from app.stats.ml import compute_temporal_ml_overview as _compute_ml
        from app.stats.supervised import compute_anova, compute_multiple_regression_out, compute_supervised_overview

        week = self._require_analysis(week_id, expected="ml_overview")

        try:
            out_df = self._load_source_dataframe(week["week_id"], "out")
            analysis_df = out_df
        except ValueError:
            out_df = None
            analysis_df, _ = self.workspace_store.load_dataset(week["week_id"])

        temporal_payload = _compute_ml(analysis_df, on_progress=on_progress)
        supervised_payload = compute_supervised_overview(analysis_df)
        anova_payload = compute_anova(analysis_df)

        if out_df is not None:
            multiple_regression_payload = compute_multiple_regression_out(out_df)
        else:
            multiple_regression_payload = {
                "source": "out",
                "target_present": False,
                "model_built": False,
                "formula": None,
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
                "warnings": [
                    {
                        "code": "missing_out_source",
                        "severity": "warning",
                        "column": None,
                        "message": "OUT source is not available for this week.",
                        "suggestion": "Provide an OUT dataset to enable multiple regression context.",
                    }
                ],
            }

        temporal_payload["warnings"] = self._dedupe_warnings(
            temporal_payload.get("warnings", [])
            + supervised_payload.get("warnings", [])
            + anova_payload.get("warnings", [])
            + multiple_regression_payload.get("warnings", [])
        )
        temporal_payload["supervised_overview"] = supervised_payload
        temporal_payload["anova"] = anova_payload
        temporal_payload["multiple_regression"] = multiple_regression_payload

        temporal_payload["week_id"] = week["week_id"]
        temporal_payload["stage_name"] = week["stage_name"]
        temporal_payload["artifacts"] = self._artifact_payload(week)

        self._save_ml_cache(week_id, temporal_payload)
        return temporal_payload

    def _save_ml_cache(self, week_id: str, payload: dict[str, Any]) -> None:
        cache_dir = self.workspace_dir / week_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / "ml_overview.json"
        cache_payload = {
            "fingerprint": self._ml_cache_fingerprint(week_id),
            "payload": payload,
        }
        cache_path.write_text(json.dumps(cache_payload, default=str, ensure_ascii=False), encoding="utf-8")

    def get_week_notes(self, week_id: str) -> dict[str, Any]:
        self._get_week_definition(week_id)
        self.bootstrap()
        content, updated_at = self.workspace_store.read_notes(week_id)
        return {"content": content, "updated_at": updated_at}

    def save_week_notes(self, week_id: str, content: str) -> dict[str, Any]:
        self._get_week_definition(week_id)
        self.bootstrap()
        updated_at = self.workspace_store.write_notes(week_id, content)
        return {"ok": True, "updated_at": updated_at}

    def get_week_report(self, week_id: str) -> dict[str, Any]:
        week = self._get_week_definition(week_id)
        self.bootstrap()
        if not self._report_exists(week_id):
            return self.refresh_week_report(week_id)
        markdown_content, html_content, updated_at = self.workspace_store.read_report(week_id)
        return {
            "week_id": week_id,
            "stage_name": week["stage_name"],
            "markdown_content": markdown_content,
            "html_content": html_content,
            "updated_at": updated_at,
            "artifacts": self._artifact_payload(week),
        }

    def get_week_artifact_text(self, week_id: str, filename: str) -> dict[str, Any]:
        week = self._get_week_definition(week_id)
        self.bootstrap()

        requested_name = Path(filename).name
        if requested_name != filename:
            raise ValueError("Artifact filename must not include directories")

        allowed_filenames = {
            Path(str(item.get("relative_path", ""))).name
            for item in week.get("artifact_templates") or []
            if item.get("relative_path")
        }
        if requested_name not in allowed_filenames:
            raise ValueError(f"Artifact '{requested_name}' is not declared for week '{week_id}'")

        allowed_suffixes = {".mod", ".dat", ".md", ".html", ".json", ".csv", ".txt"}
        if Path(requested_name).suffix.lower() not in allowed_suffixes:
            raise ValueError(f"Artifact '{requested_name}' is not exposed as text")

        content = self.workspace_store.read_text_artifact(week_id, requested_name)
        return {
            "week_id": week_id,
            "filename": requested_name,
            "content": content,
        }

    def refresh_week_report(self, week_id: str) -> dict[str, Any]:
        week = self._get_week_definition(week_id)
        markdown_content = self._render_markdown_report(week)
        html_content = self._render_html_report(week)
        updated_at = self.workspace_store.write_report(week_id, markdown_content, html_content)
        return {
            "week_id": week_id,
            "stage_name": week["stage_name"],
            "markdown_content": markdown_content,
            "html_content": html_content,
            "updated_at": updated_at,
            "artifacts": self._artifact_payload(week),
        }

    def _load_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _academic_metadata_path(self, week: dict[str, Any]) -> Path | None:
        relative_path = week.get("academic_metadata_path")
        if not relative_path:
            return None
        return self.repo_root / relative_path

    def _ensure_academic_assets(self, week: dict[str, Any]) -> None:
        metadata_path = self._academic_metadata_path(week)
        if metadata_path is None:
            return
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Academic metadata for {week['week_id']} not found: {metadata_path.relative_to(self.repo_root)}"
            )

    def _load_week_academic_metadata(self, week_id: str) -> dict[str, Any]:
        week = self._get_week_definition(week_id)
        metadata_path = self._academic_metadata_path(week)
        if metadata_path is None or not metadata_path.exists():
            raise FileNotFoundError(f"Academic metadata for {week_id} not found")
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def _week_academic_context_summary(self, week: dict[str, Any]) -> dict[str, Any] | None:
        metadata_path = self._academic_metadata_path(week)
        if metadata_path is None or not metadata_path.exists():
            return None
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return {
            "objective": metadata["objective"],
            "analytical_goal": metadata["analytical_goal"],
            "unit_of_observation": metadata["unit_of_observation"],
            "target_variable": metadata.get("target_variable"),
            "initial_hypotheses": metadata["initial_hypotheses"],
        }

    def _week1_derived_assets_exist(self, week_id: str) -> bool:
        return (
            self.workspace_store.analysis_imputed_path(week_id, "in").exists()
            and self.workspace_store.analysis_imputed_path(week_id, "out").exists()
            and self.workspace_store.optics_path(week_id, "in").exists()
            and self.workspace_store.optics_path(week_id, "out").exists()
        )

    def _persist_week1_bundle_artifacts(self, week_id: str, bundle: dict[str, Any]) -> None:
        for source, df in bundle["imputed_frames"].items():
            self.workspace_store.write_analysis_imputed(week_id, source, df)
        for source, payload in bundle["clustering_payload"]["sources"].items():
            self.workspace_store.write_optics_payload(week_id, source, payload)

    def _week1_bundle_fingerprint(self, week_id: str) -> tuple[str]:
        week = self._get_week_definition(week_id)
        academic_path = self.repo_root / week["academic_metadata_path"]
        paths = [
            self.repo_root / week["seed_paths"]["in_file"],
            self.repo_root / week["seed_paths"]["out_file"],
            academic_path,
        ]
        digest = hashlib.sha256()
        for path in paths:
            digest.update(path.read_bytes())
        return (digest.hexdigest(),)

    def _load_week1_persisted_clustering_payload(self, week_id: str) -> dict[str, Any] | None:
        try:
            in_payload = self.workspace_store.read_optics_payload(week_id, "in")
            out_payload = self.workspace_store.read_optics_payload(week_id, "out")
        except FileNotFoundError:
            return None
        warnings = self._dedupe_warnings(in_payload.get("warnings", []) + out_payload.get("warnings", []))
        return {
            "week_id": week_id,
            "stage_name": "EDA",
            "sources": {"in": in_payload, "out": out_payload},
            "warnings": warnings,
        }

    def _build_week1_eda_payload(self, week_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        in_df = self._load_source_dataframe(week_id, "in")
        out_df = self._load_source_dataframe(week_id, "out")
        canonical_df, _ = self.workspace_store.load_dataset(week_id)
        bundle = build_week1_academic_eda_bundle(
            week_id=week_id,
            source_frames={"in": in_df, "out": out_df},
            canonical_df=canonical_df,
            metadata=metadata,
            academic_metadata=self._load_week_academic_metadata(week_id),
            clustering_payload=self._load_week1_persisted_clustering_payload(week_id),
        )
        for source, df in bundle["imputed_frames"].items():
            self.workspace_store.write_analysis_imputed(week_id, source, df)
        return bundle["eda_payload"]

    def _build_week1_bundle(self, week_id: str) -> dict[str, Any]:
        with self._week1_bundle_lock:
            fingerprint = self._week1_bundle_fingerprint(week_id)
            cached = self._week1_bundle_cache.get(week_id)
            if cached and cached["fingerprint"] == fingerprint:
                return cached["bundle"]

            canonical_df, metadata = self.workspace_store.load_dataset(week_id)
            in_df = self._load_source_dataframe(week_id, "in")
            out_df = self._load_source_dataframe(week_id, "out")
            bundle = build_week1_academic_eda_bundle(
                week_id=week_id,
                source_frames={"in": in_df, "out": out_df},
                canonical_df=canonical_df,
                metadata=metadata,
                academic_metadata=self._load_week_academic_metadata(week_id),
            )
            self._persist_week1_bundle_artifacts(week_id, bundle)
            self._week1_bundle_cache[week_id] = {"fingerprint": fingerprint, "bundle": bundle}
            return bundle

    def _get_week_definition(self, week_id: str) -> dict[str, Any]:
        manifest = self._load_manifest()
        for week in manifest["weeks"]:
            if week["week_id"] == week_id:
                return week
        raise ValueError(f"Unknown week_id '{week_id}'")

    def _active_week_needs_refresh(self, week: dict[str, Any]) -> bool:
        if week["week_id"] == "week-3":
            payload_path = self.workspace_store.artifact_path(week["week_id"], "optimization_model.json")
            if not payload_path.exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "official_model.mod").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "official_model.dat").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "baseline_validation.mod").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "baseline_validation.dat").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "cplex_segregations.csv").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "cplex_ml_signals.csv").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "baseline_segregations.csv").exists():
                return True
            if not self.workspace_store.artifact_path(week["week_id"], "baseline_inventory_seed.csv").exists():
                return True
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except Exception:
                return True
            if not self._week3_payload_is_current(payload):
                return True

        metadata_path = self.workspace_store.metadata_path(week["week_id"])
        canonical_path = self.workspace_store.canonical_path(week["week_id"])
        if not metadata_path.exists() or not canonical_path.exists():
            return True

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return True

        saved_snapshot = metadata.get("seed_snapshot", {})
        current_snapshot = self._seed_snapshot(week)
        return saved_snapshot != current_snapshot

    def _seed_snapshot(self, week: dict[str, Any]) -> dict[str, dict[str, int | str]]:
        snapshot: dict[str, dict[str, int | str]] = {}
        for key, relative_path in (week.get("seed_paths") or {}).items():
            if not relative_path:
                continue
            path = self.repo_root / relative_path
            if not path.exists():
                continue
            stat = path.stat()
            snapshot[key] = {
                "path": relative_path,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        return snapshot

    def _build_active_week(self, week: dict[str, Any]) -> None:
        if week["week_id"] == "week-3":
            self._build_week3_active_week(week)
            return

        seed_paths = week.get("seed_paths") or {}
        in_path = self.repo_root / seed_paths["in_file"] if seed_paths.get("in_file") else None
        out_path = self.repo_root / seed_paths["out_file"] if seed_paths.get("out_file") else None

        in_bytes = in_path.read_bytes() if in_path and in_path.exists() else None
        out_bytes = out_path.read_bytes() if out_path and out_path.exists() else None

        context = PipelineContext(
            in_input=DatasetInput(filename=in_path.name, content=in_bytes) if in_path and in_bytes is not None else None,
            out_input=DatasetInput(filename=out_path.name, content=out_bytes) if out_path and out_bytes is not None else None,
        )
        context = self.pipeline.run(context)
        if context.merged_df is None:
            raise ValueError(f"Week {week['week_id']} could not produce a canonical dataset")

        df = context.merged_df
        schema_detected = {
            "in_file": [str(col) for col in context.in_df.columns] if context.in_df is not None else [],
            "out_file": [str(col) for col in context.out_df.columns] if context.out_df is not None else [],
            "canonical": [str(col) for col in df.columns],
        }

        warnings = context.warnings + self._schema_warnings(schema_detected)
        metadata = {
            "week_id": week["week_id"],
            "stage_name": week["stage_name"],
            "status": week["status"],
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": [str(col) for col in df.columns],
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "has_target": "DaysInDeposit" in df.columns,
            "schema": schema_detected,
            "warnings": warnings,
            "step_metadata": context.step_metadata,
            "seed_snapshot": self._seed_snapshot(week),
        }

        self.workspace_store.save_dataset(
            week_id=week["week_id"],
            canonical_df=df,
            metadata=metadata,
            in_bytes=in_bytes,
            out_bytes=out_bytes,
        )
        self._invalidate_week_derivatives(week["week_id"])

    def _build_week3_active_week(self, week: dict[str, Any]) -> None:
        self._invalidate_week_derivatives(week["week_id"])
        seed_paths = week.get("seed_paths") or {}
        instance_relative_path = seed_paths.get("instance_file")
        if not instance_relative_path:
            raise ValueError("Week 3 requires an 'instance_file' in manifest seed_paths")
        history_relative_paths = [
            value for key, value in sorted(seed_paths.items()) if str(key).startswith("history_file_") and value
        ]
        if not history_relative_paths:
            raise ValueError("Week 3 requires historical Week2 OUT files in manifest seed_paths")

        bundle = build_week3_optimization_bundle(
            self.repo_root / str(instance_relative_path),
            [self.repo_root / str(path) for path in history_relative_paths],
        )
        canonical_df = bundle["canonical_df"]
        metadata = {
            "week_id": week["week_id"],
            "stage_name": week["stage_name"],
            "status": week["status"],
            "shape": [int(canonical_df.shape[0]), int(canonical_df.shape[1])],
            "columns": [str(col) for col in canonical_df.columns],
            "dtypes": {col: str(dtype) for col, dtype in canonical_df.dtypes.items()},
            "has_target": False,
            "schema": {
                "canonical": [str(col) for col in canonical_df.columns],
                "instance_file": str(instance_relative_path),
                "history_files": [str(path) for path in history_relative_paths],
            },
            "warnings": bundle["warnings"],
            "step_metadata": {
                "builder": "week3_optimization",
                "weekly_balance_rows": int(bundle["weekly_balance_df"].shape[0]),
                "scored_containers": int(canonical_df.shape[0]),
            },
            "seed_snapshot": self._seed_snapshot(week),
        }
        self.workspace_store.save_dataset(
            week_id=week["week_id"],
            canonical_df=canonical_df,
            metadata=metadata,
        )
        weekly_balance_path = self.workspace_store.artifact_path(week["week_id"], "weekly_balance.csv")
        bundle["weekly_balance_df"].to_csv(weekly_balance_path, index=False)
        cplex_mapping_path = self.workspace_store.artifact_path(week["week_id"], "cplex_segregations.csv")
        bundle["cplex_segregation_df"].to_csv(cplex_mapping_path, index=False)
        cplex_ubar_path = self.workspace_store.artifact_path(week["week_id"], "cplex_ml_signals.csv")
        bundle["cplex_ubar_df"].to_csv(cplex_ubar_path, index=False)
        cplex_mod_path = self.workspace_store.artifact_path(week["week_id"], "official_model.mod")
        cplex_mod_path.write_text(bundle["cplex_mod_text"], encoding="utf-8")
        cplex_dat_path = self.workspace_store.artifact_path(week["week_id"], "official_model.dat")
        cplex_dat_path.write_text(bundle["cplex_dat_text"], encoding="utf-8")
        baseline_mapping_path = self.workspace_store.artifact_path(week["week_id"], "baseline_segregations.csv")
        bundle["baseline_segregation_df"].to_csv(baseline_mapping_path, index=False)
        baseline_inventory_seed_path = self.workspace_store.artifact_path(week["week_id"], "baseline_inventory_seed.csv")
        bundle["baseline_inventory_seed_df"].to_csv(baseline_inventory_seed_path, index=False)
        baseline_mod_path = self.workspace_store.artifact_path(week["week_id"], "baseline_validation.mod")
        baseline_mod_path.write_text(bundle["baseline_mod_text"], encoding="utf-8")
        baseline_dat_path = self.workspace_store.artifact_path(week["week_id"], "baseline_validation.dat")
        baseline_dat_path.write_text(bundle["baseline_dat_text"], encoding="utf-8")
        self.workspace_store.write_json_artifact(week["week_id"], "optimization_model.json", bundle["payload"])

    def _ml_cache_fingerprint(self, week_id: str) -> str:
        metadata_path = self.workspace_store.metadata_path(week_id)
        if not metadata_path.exists():
            return ""
        digest = hashlib.sha256()
        digest.update(metadata_path.read_bytes())
        return digest.hexdigest()

    def _invalidate_week_derivatives(self, week_id: str) -> None:
        paths = [
            self.workspace_store.report_markdown_path(week_id),
            self.workspace_store.report_html_path(week_id),
            self.workspace_dir / week_id / "ml_overview.json",
        ]
        if week_id == "week-3":
            paths.extend(
                [
                    self.workspace_store.artifact_path(week_id, "optimization_model.json"),
                    self.workspace_store.artifact_path(week_id, "weekly_balance.csv"),
                    self.workspace_store.artifact_path(week_id, "official_model.mod"),
                    self.workspace_store.artifact_path(week_id, "cplex_segregations.csv"),
                    self.workspace_store.artifact_path(week_id, "cplex_ml_signals.csv"),
                    self.workspace_store.artifact_path(week_id, "official_model.dat"),
                    self.workspace_store.artifact_path(week_id, "baseline_validation.mod"),
                    self.workspace_store.artifact_path(week_id, "baseline_validation.dat"),
                    self.workspace_store.artifact_path(week_id, "baseline_segregations.csv"),
                    self.workspace_store.artifact_path(week_id, "baseline_inventory_seed.csv"),
                ]
            )
        for path in paths:
            path.unlink(missing_ok=True)

    def _build_week_summary(self, week: dict[str, Any]) -> dict[str, Any]:
        self.workspace_store.ensure_week_dir(week["week_id"])
        _, notes_updated_at = self.workspace_store.read_notes(week["week_id"])
        return {
            "week_id": week["week_id"],
            "week_number": int(week["week_number"]),
            "title": week["title"],
            "stage_name": week["stage_name"],
            "status": week["status"],
            "summary": week["summary"],
            "seed_paths": {
                "in_file": (week.get("seed_paths") or {}).get("in_file"),
                "out_file": (week.get("seed_paths") or {}).get("out_file"),
            },
            "analysis_available": list(week.get("analysis_available") or []),
            "artifacts": self._artifact_payload(week),
            "notes_updated_at": notes_updated_at,
            "report_available": self._report_exists(week["week_id"]),
        }

    def _artifact_payload(self, week: dict[str, Any]) -> list[dict[str, Any]]:
        artifacts = []
        for item in week.get("artifact_templates") or []:
            path = str(item["relative_path"])
            artifacts.append(
                {
                    "kind": item["kind"],
                    "label": item["label"],
                    "path": path,
                    "available": (self.repo_root / path).exists(),
                }
            )
        return artifacts

    def _report_exists(self, week_id: str) -> bool:
        return self.workspace_store.report_markdown_path(week_id).exists() and self.workspace_store.report_html_path(
            week_id
        ).exists()

    def _require_analysis(self, week_id: str, expected: str) -> dict[str, Any]:
        self.bootstrap()
        week = self._get_week_definition(week_id)
        if expected not in set(week.get("analysis_available") or []):
            raise ValueError(f"Analysis '{expected}' is not available for week '{week_id}'")
        return week

    def _load_source_dataframe(self, week_id: str, source: str) -> pd.DataFrame:
        if not self.workspace_store.source_exists(week_id, source):
            raise ValueError(f"Source '{source}' is not available for {week_id}")
        raw_bytes = self.workspace_store.read_source_bytes(week_id, source)
        raw_df, _ = robust_read_csv(raw_bytes, filename=f"raw_{source}.csv")
        clean_df, _, _ = _clean_dataframe(raw_df, source=f"{source}_file")
        cast_df, _, _ = _cast_single_dataframe(clean_df, source=f"{source}_file")
        return cast_df

    def _build_week3_optimization_payload(self, week: dict[str, Any]) -> dict[str, Any]:
        payload_path = self.workspace_store.artifact_path(week["week_id"], "optimization_model.json")
        if not payload_path.exists():
            self._build_active_week(week)
        payload = self.workspace_store.read_json_artifact(week["week_id"], "optimization_model.json")
        if not self._week3_payload_is_current(payload):
            self._build_active_week(week)
            payload = self.workspace_store.read_json_artifact(week["week_id"], "optimization_model.json")
        payload["artifacts"] = self._artifact_payload(week)
        return payload

    def _week3_payload_is_current(self, payload: dict[str, Any]) -> bool:
        academic = payload.get("academic_formulation")
        bridge = payload.get("notation_bridge")
        cplex_export = payload.get("cplex_export")
        baseline_export = payload.get("baseline_export")
        if not isinstance(academic, dict) or not isinstance(bridge, dict) or not isinstance(cplex_export, dict):
            return False
        if not isinstance(baseline_export, dict):
            return False
        required_sections = [
            "dimensions",
            "given_data",
            "parameters",
            "decision_variables",
            "objective",
            "constraints",
            "assumptions",
            "week4_extensions",
        ]
        if any(section not in academic for section in required_sections):
            return False
        objective = academic.get("objective")
        if not isinstance(objective, dict) or "equation_latex" not in objective or "components" not in objective:
            return False
        if not isinstance(bridge.get("equations"), list):
            return False
        if cplex_export.get("version") != "week3_cplex_official_v2":
            return False
        if baseline_export.get("version") != "week3_baseline_validation_v6":
            return False
        baseline_samples = baseline_export.get("sample_segregations")
        if not isinstance(baseline_samples, list):
            return False
        if baseline_samples:
            required_baseline_keys = {
                "j",
                "base_code",
                "type",
                "size_teu",
                "owner",
                "dem_d",
                "dem_o",
                "out_s",
                "inv0_d",
                "inv0_o",
            }
            if not isinstance(baseline_samples[0], dict) or not required_baseline_keys.issubset(baseline_samples[0].keys()):
                return False
        return True

    def _build_week_ml_payload(self, week: dict[str, Any]) -> dict[str, Any]:
        try:
            out_df = self._load_source_dataframe(week["week_id"], "out")
            analysis_df = out_df
        except ValueError:
            out_df = None
            analysis_df, _ = self.workspace_store.load_dataset(week["week_id"])

        temporal_payload = compute_temporal_ml_overview(analysis_df)
        supervised_payload = compute_supervised_overview(analysis_df)
        anova_payload = compute_anova(analysis_df)

        if out_df is not None:
            multiple_regression_payload = compute_multiple_regression_out(out_df)
        else:
            multiple_regression_payload = {
                "source": "out",
                "target_present": False,
                "model_built": False,
                "formula": None,
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
                "warnings": [
                    {
                        "code": "missing_out_source",
                        "severity": "warning",
                        "column": None,
                        "message": "OUT source is not available for this week.",
                        "suggestion": "Provide an OUT dataset to enable multiple regression context.",
                    }
                ],
            }

        temporal_payload["warnings"] = self._dedupe_warnings(
            temporal_payload.get("warnings", [])
            + supervised_payload.get("warnings", [])
            + anova_payload.get("warnings", [])
            + multiple_regression_payload.get("warnings", [])
        )
        temporal_payload["supervised_overview"] = supervised_payload
        temporal_payload["anova"] = anova_payload
        temporal_payload["multiple_regression"] = multiple_regression_payload
        return temporal_payload

    def _week_ml_markdown_sections(self, ml: dict[str, Any]) -> list[str]:
        lines = [
            "",
            "## Evaluacion ML",
            f"- Semanas train: {', '.join(ml['split']['train_weeks']) or 'n/a'}",
            f"- Semanas test: {', '.join(ml['split']['test_weeks']) or 'n/a'}",
        ]

        comparison = ml.get("strategy_comparison")
        best_regression = comparison.get("best_regression") if comparison else None
        best_heuristic = comparison.get("best_heuristic") if comparison else None
        if best_regression:
            lines.extend(
                [
                    f"- Mejor regresion: {best_regression['model_name']} ({best_regression.get('strategy_label') or 'n/a'})",
                    f"- MAE regresion: {best_regression['metrics'].get('mae')}",
                    f"- RMSE regresion: {best_regression['metrics'].get('rmse')}",
                    f"- MedAE regresion: {best_regression['metrics'].get('medae')}",
                ]
            )
        if best_heuristic:
            lines.extend(
                [
                    f"- Mejor heuristica: {best_heuristic['model_name']}",
                    f"- MAE heuristica: {best_heuristic['metrics'].get('mae')}",
                    f"- RMSE heuristica: {best_heuristic['metrics'].get('rmse')}",
                    f"- MedAE heuristica: {best_heuristic['metrics'].get('medae')}",
                ]
            )
        if comparison:
            lines.append(f"- Comparacion: {comparison.get('narrative')}")

        diagnostics = ml.get("target_transformation_diagnostics")
        if diagnostics and diagnostics.get("steps"):
            lines.extend(["", "## Transformacion del target"])
            lines.append(
                f"- Alcance del diagnostico: {diagnostics.get('scope')}. Los boxplots se exponen en frontend y las estadisticas resumen el before/after."
            )
            for step in diagnostics.get("steps", [])[:4]:
                stats = step.get("stats", {})
                lines.append(
                    f"- {step['step_label']} ({step.get('scale')}): media={stats.get('mean')}, mediana={stats.get('p50')}, skew={stats.get('skew')}, outliers={stats.get('outlier_count')} ({stats.get('outlier_ratio')})."
                )

        lines.extend(["", "## Benchmark de preprocesamiento"])
        benchmarks = sorted(
            ml.get("preprocessing_benchmarks", []),
            key=lambda item: float(item.get("metrics", {}).get("mae") or 10**9),
        )
        for row in benchmarks[:8]:
            lines.append(
                f"- {row['model_name']} / {row['strategy_label']}: MAE={row['metrics'].get('mae')}, RMSE={row['metrics'].get('rmse')}, MedAE={row['metrics'].get('medae')}."
            )

        segment_reports = ml.get("segment_reports", [])
        if segment_reports:
            lines.extend(["", "## Segmentacion representativa"])
            for report in segment_reports[:4]:
                lines.append(f"- {report['family_label']}:")
                for row in report.get("rows", [])[:3]:
                    lines.append(
                        f"  - {row['segment']}: test={row['test_count']}, mae_reg={row.get('regression_mae')}, mae_heur={row.get('heuristic_mae')}."
                    )

        warning_messages = [warning["message"] for warning in ml.get("warnings", [])[:6]]
        if warning_messages:
            lines.extend(["", "## Advertencias principales"])
            lines.extend(f"- {message}" for message in warning_messages)

        return lines

    def _week_ml_html_sections(self, ml: dict[str, Any]) -> list[str]:
        body_parts = [
            "<h2>Evaluacion ML</h2>",
            "<ul>",
            f"<li>Semanas train: {escape(', '.join(ml['split']['train_weeks']) or 'n/a')}</li>",
            f"<li>Semanas test: {escape(', '.join(ml['split']['test_weeks']) or 'n/a')}</li>",
        ]

        comparison = ml.get("strategy_comparison")
        best_regression = comparison.get("best_regression") if comparison else None
        best_heuristic = comparison.get("best_heuristic") if comparison else None
        if best_regression:
            body_parts.extend(
                [
                    f"<li>Mejor regresion: {escape(str(best_regression['model_name']))} ({escape(str(best_regression.get('strategy_label') or 'n/a'))})</li>",
                    f"<li>MAE regresion: {escape(str(best_regression['metrics'].get('mae')))}</li>",
                    f"<li>RMSE regresion: {escape(str(best_regression['metrics'].get('rmse')))}</li>",
                    f"<li>MedAE regresion: {escape(str(best_regression['metrics'].get('medae')))}</li>",
                ]
            )
        if best_heuristic:
            body_parts.extend(
                [
                    f"<li>Mejor heuristica: {escape(str(best_heuristic['model_name']))}</li>",
                    f"<li>MAE heuristica: {escape(str(best_heuristic['metrics'].get('mae')))}</li>",
                    f"<li>RMSE heuristica: {escape(str(best_heuristic['metrics'].get('rmse')))}</li>",
                    f"<li>MedAE heuristica: {escape(str(best_heuristic['metrics'].get('medae')))}</li>",
                ]
            )
        if comparison:
            body_parts.append(f"<li>Comparacion: {escape(str(comparison.get('narrative')))}</li>")
        body_parts.append("</ul>")

        diagnostics = ml.get("target_transformation_diagnostics")
        if diagnostics and diagnostics.get("steps"):
            diagnostics_html = "".join(
                (
                    f"<li><strong>{escape(str(step['step_label']))}</strong> "
                    f"({escape(str(step.get('scale')))}): "
                    f"media={escape(str(step.get('stats', {}).get('mean')))}, "
                    f"mediana={escape(str(step.get('stats', {}).get('p50')))}, "
                    f"skew={escape(str(step.get('stats', {}).get('skew')))}, "
                    f"outliers={escape(str(step.get('stats', {}).get('outlier_count')))} "
                    f"({escape(str(step.get('stats', {}).get('outlier_ratio')))}).</li>"
                )
                for step in diagnostics.get("steps", [])[:4]
            )
            body_parts.extend(
                [
                    "<h2>Transformacion del target</h2>",
                    f"<p>Alcance del diagnostico: {escape(str(diagnostics.get('scope')))}. Los boxplots se visualizan en frontend.</p>",
                    f"<ul>{diagnostics_html}</ul>",
                ]
            )

        benchmarks = sorted(
            ml.get("preprocessing_benchmarks", []),
            key=lambda item: float(item.get("metrics", {}).get("mae") or 10**9),
        )
        benchmarks_html = "".join(
            f"<li>{escape(str(row['model_name']))} / {escape(str(row['strategy_label']))}: "
            f"MAE={escape(str(row['metrics'].get('mae')))}, "
            f"RMSE={escape(str(row['metrics'].get('rmse')))}, "
            f"MedAE={escape(str(row['metrics'].get('medae')))}.</li>"
            for row in benchmarks[:8]
        )
        body_parts.extend(
            [
                "<h2>Benchmark de preprocesamiento</h2>",
                f"<ul>{benchmarks_html or '<li>Sin benchmarks disponibles</li>'}</ul>",
            ]
        )

        segment_reports = ml.get("segment_reports", [])
        if segment_reports:
            segment_parts = []
            for report in segment_reports[:4]:
                rows_html = "".join(
                    f"<li>{escape(str(row['segment']))}: test={escape(str(row['test_count']))}, "
                    f"mae_reg={escape(str(row.get('regression_mae')))}, "
                    f"mae_heur={escape(str(row.get('heuristic_mae')))}.</li>"
                    for row in report.get("rows", [])[:3]
                )
                segment_parts.append(
                    f"<li><strong>{escape(str(report['family_label']))}</strong><ul>{rows_html or '<li>Sin segmentos representativos</li>'}</ul></li>"
                )
            body_parts.extend(
                [
                    "<h2>Segmentacion representativa</h2>",
                    f"<ul>{''.join(segment_parts)}</ul>",
                ]
            )

        warnings_html = "".join(
            f"<li>{escape(str(warning['message']))}</li>" for warning in ml.get("warnings", [])[:6]
        )
        if warnings_html:
            body_parts.extend(["<h2>Advertencias principales</h2>", f"<ul>{warnings_html}</ul>"])

        return body_parts

    def _week3_markdown_math_entries(self, entries: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for entry in entries:
            lines.append(f"- ${entry['symbol_latex']}$")
            if entry.get("domain_latex"):
                lines.append(f"  Dominio: ${entry['domain_latex']}$")
            lines.append(f"  Descripción: {entry['description']}")
            if entry.get("notes"):
                lines.append(f"  Nota: {entry['notes']}")
        return lines

    def _week3_markdown_constraints(self, constraints: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for constraint in constraints:
            lines.extend(
                [
                    f"### {constraint['name']}",
                    f"$${constraint['equation_latex']}$$",
                    f"Descripción: {constraint['description']}",
                ]
            )
            if constraint.get("notes"):
                lines.append(f"Nota: {constraint['notes']}")
            lines.append("")
        return lines

    def _week3_markdown_equations(self, equations: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for equation in equations:
            lines.extend(
                [
                    f"### {equation['name']}",
                    f"$${equation['equation_latex']}$$",
                    f"Descripción: {equation['description']}",
                    "",
                ]
            )
        return lines

    def _week3_html_math_entries(self, entries: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for entry in entries:
            item = ["<li>", f"<div class='math-inline'>${escape(str(entry['symbol_latex']))}$</div>"]
            if entry.get("domain_latex"):
                item.append(
                    f"<div><strong>Dominio:</strong> <span class='math-inline'>${escape(str(entry['domain_latex']))}$</span></div>"
                )
            item.append(f"<div><strong>Descripción:</strong> {escape(str(entry['description']))}</div>")
            if entry.get("notes"):
                item.append(f"<div><strong>Nota:</strong> {escape(str(entry['notes']))}</div>")
            item.append("</li>")
            parts.append("".join(item))
        return "".join(parts)

    def _week3_html_constraints(self, constraints: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for constraint in constraints:
            block = [
                "<li>",
                f"<strong>{escape(str(constraint['name']))}</strong>",
                f"<div class='math-block'>$$ {escape(str(constraint['equation_latex']))} $$</div>",
                f"<div>{escape(str(constraint['description']))}</div>",
            ]
            if constraint.get("notes"):
                block.append(f"<div><strong>Nota:</strong> {escape(str(constraint['notes']))}</div>")
            block.append("</li>")
            blocks.append("".join(block))
        return "".join(blocks)

    def _week3_html_equations(self, equations: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for equation in equations:
            blocks.append(
                "".join(
                    [
                        "<li>",
                        f"<strong>{escape(str(equation['name']))}</strong>",
                        f"<div class='math-block'>$$ {escape(str(equation['equation_latex']))} $$</div>",
                        f"<div>{escape(str(equation['description']))}</div>",
                        "</li>",
                    ]
                )
            )
        return "".join(blocks)

    def _render_week3_markdown_report(self, week: dict[str, Any]) -> str:
        payload = self._build_week3_optimization_payload(week)
        summary = payload["summary"]
        ml = payload["ml_integration"]
        academic = payload["academic_formulation"]
        bridge = payload["notation_bridge"]
        cplex_export = payload["cplex_export"]
        baseline_export = payload["baseline_export"]
        objective = academic["objective"]
        lines = [
            "# Semana 3 - Modelamiento matematico",
            "",
            week["description"],
            "",
            "La formulación académica incorpora del avance del compañero la notación por segregación, la lógica de procesos y la estructura formal de restricciones, mientras que la implementación operacional mantiene el parámetro `u_c` por contenedor y la penalización `P_{c,b}`.",
            "",
            "## Resumen de instancia",
            f"- Bahias: {summary['bay_count']}",
            f"- Semanas del horizonte: {summary['planning_weeks']}",
            f"- Capacidad total del yard (TEU): {summary['total_capacity_teu']}",
            f"- Stock inicial: {summary['initial_stock_units']} contenedores / {summary['initial_stock_teu']} TEU",
            f"- Pico de contenedores activos: {summary['peak_active_units']} contenedores / {summary['peak_active_teu']} TEU",
            f"- Utilizacion peak: {summary['peak_utilization_pct']}%",
            f"- Pares de bahias compatibles: {summary['compatibility_pairs']}",
            "",
            "## Dimensiones e indices",
        ]
        lines.extend(self._week3_markdown_math_entries(academic["dimensions"]))
        lines.extend(["", "## Datos observados"])
        lines.extend(self._week3_markdown_math_entries(academic["given_data"]))
        lines.extend(["", "## Parametros"])
        lines.extend(self._week3_markdown_math_entries(academic["parameters"]))
        lines.extend(["", "## Variables"])
        lines.extend(self._week3_markdown_math_entries(academic["decision_variables"]))
        lines.extend(
            [
                "",
                "## Funcion objetivo",
                f"$${objective['equation_latex']}$$",
                objective["description"],
                "",
                "### Componentes de la funcion objetivo",
            ]
        )
        for component in objective["components"]:
            lines.append(f"- ${component['term_latex']}$")
            lines.append(f"  Descripción: {component['description']}")
            if component.get("notes"):
                lines.append(f"  Nota: {component['notes']}")

        lines.extend(["", "## Restricciones"])
        lines.extend(self._week3_markdown_constraints(academic["constraints"]))
        lines.extend(
            [
                "## Puente con la capa operacional",
                f"- Unidad academica: {bridge['academic_unit']}",
                f"- Unidad operacional: {bridge['operational_unit']}",
                f"- Modelo global ganador: {ml.get('global_best_model')}",
                f"- Adjacent accuracy global: {ml.get('global_adjacent_accuracy')}",
                f"- Correlacion priority score global: {ml.get('global_priority_corr')}",
                f"- Regla de urgencia: `{ml.get('urgency_rule')}`",
                f"- Parametros objetivo: alpha={ml.get('alpha')}, beta={ml.get('beta')}, lambda_init={ml.get('lambda_init')}, lambda_tour={ml.get('lambda_tour')}, lambda_dwell={ml.get('lambda_dwell')}",
                "",
                "### Ecuaciones del puente",
            ]
        )
        lines.extend(self._week3_markdown_equations(bridge["equations"]))
        lines.append("### Reglas de agregacion")
        lines.extend(f"- {item}" for item in bridge["aggregation_rules"])
        lines.append("")
        lines.append("### Construccion de costos")
        lines.extend(f"- {item}" for item in bridge["cost_bridge"])
        lines.append("")
        lines.append("### Logica de procesos")
        lines.extend(f"- {item}" for item in bridge["process_logic"])
        lines.append("")
        lines.append("### Compatibilidad")
        lines.extend(f"- {item}" for item in bridge["compatibility_rule"])
        lines.extend(["", "### Selector owner/global"])
        for owner_row in payload["owner_selections"]:
            lines.append(
                f"- Owner {owner_row['owner']}: decision={owner_row['decision']}, modelo={owner_row.get('best_model') or 'global'}, test={owner_row['test_rows']}, adj={owner_row.get('adjacent_accuracy')}, corr={owner_row.get('priority_corr')}. {owner_row['reason']}"
            )
        lines.extend(["", "### Muestras de segregacion agregada"])
        for sample in bridge["segregation_samples"][:6]:
            lines.append(
                f"- {sample['segregation_id']} | week={sample['week']} | count={sample['container_count']} | $\\bar u_{{jt}}$={sample['avg_u_jt']} | $T_j$={sample.get('dwell_mean_tj')} | $\\alpha_j^L$={sample['alpha_l']} | $\\alpha_j^P$={sample['alpha_p']} | $\\alpha_j^M$={sample['alpha_m']}"
            )

        lines.extend(["", "## Export CPLEX oficial"])
        lines.extend(
            [
                f"- Version: {cplex_export['version']}",
                f"- Dimensiones exportadas: |J|={cplex_export['j_count']}, |JD|={cplex_export['jd_count']}, |JO|={cplex_export['jo_count']}, |K|={cplex_export['k_count']}, |B|={cplex_export['b_count']}, |T|={cplex_export['t_count']}",
                f"- Owners incluidos: {', '.join(cplex_export['owners'])}",
                f"- Capacidad relajada de procesos en baseline: {cplex_export['relaxed_process_capacity']}",
                f"- Penalizacion ML en objetivo: `{cplex_export['ml_signal_rule']}`",
                f"- Pesos ML: lambdaML={cplex_export['lambda_ml']}, wUrg={cplex_export['w_urg']}, wSlow={cplex_export['w_slow']}",
                "- Artefactos generados: `official_model.mod`, `official_model.dat`, `cplex_segregations.csv`, `cplex_ml_signals.csv`",
                "",
                "### Supuestos del export .dat",
            ]
        )
        lines.extend(f"- {item}" for item in cplex_export["assumptions"])
        lines.extend(["", "### Muestras de pares K"])
        for pair in cplex_export["sample_pairs"]:
            lines.append(
                f"- <{pair['damaged_j']},{pair['operational_j']}> -> {pair['type']} | size={pair['size_teu']} | owner={pair['owner']}"
            )

        lines.extend(["", "## Baseline de validacion del primer .mod"])
        lines.extend(
            [
                f"- Version: {baseline_export['version']}",
                f"- Dimensiones exportadas: |J|={baseline_export['j_count']}, |B|={baseline_export['b_count']}, |C|={len(baseline_export['c_labels'])}",
                f"- Semana validada: {baseline_export['validated_week']}",
                f"- Owners incluidos: {', '.join(baseline_export['owners'])}",
                f"- Mapeo de inspeccion: {baseline_export['inspection_mapping']}",
                f"- Inventario inicial agregado: {baseline_export['total_inv0_units']} contenedores / {baseline_export['total_inv0_teu']} TEU",
                f"- Inventario inicial dañado: {baseline_export['total_inv0_d_units']} contenedores",
                f"- Inventario inicial operativo: {baseline_export['total_inv0_o_units']} contenedores",
                f"- Regla de modelado de inventario: {baseline_export['initial_inventory_modeling_rule']}",
                f"- Capacidad de taller usada en validacion: {baseline_export['relaxed_workshop_capacity']}",
                "- Artefactos generados: `baseline_validation.mod`, `baseline_validation.dat`, `baseline_segregations.csv`, `baseline_inventory_seed.csv`",
                "",
                "### Supuestos del baseline",
            ]
        )
        lines.extend(f"- {item}" for item in baseline_export["assumptions"])
        lines.extend(["", "### Muestras de segregacion base"])
        for sample in baseline_export["sample_segregations"]:
            lines.append(
                f"- j={sample['j']} | {sample['base_code']} | dem_D={sample['dem_d']} | dem_O={sample['dem_o']} | out_s={sample['out_s']} | Inv0_D={sample['inv0_d']} | Inv0_O={sample['inv0_o']}"
            )

        lines.extend(["", "## Supuestos y extensiones"])
        lines.append("### Supuestos activos en semana 3")
        lines.extend(f"- {item}" for item in academic["assumptions"])
        lines.append("")
        lines.append("### Extensiones naturales para semana 4")
        lines.extend(self._week3_markdown_constraints(academic["week4_extensions"]))

        warning_messages = [warning["message"] for warning in payload.get("warnings", [])[:6]]
        if warning_messages:
            lines.extend(["## Advertencias de la instancia"])
            lines.extend(f"- {message}" for message in warning_messages)

        return "\n".join(lines).strip() + "\n"

    def _render_week3_html_report(self, week: dict[str, Any]) -> str:
        payload = self._build_week3_optimization_payload(week)
        summary = payload["summary"]
        ml = payload["ml_integration"]
        academic = payload["academic_formulation"]
        bridge = payload["notation_bridge"]
        cplex_export = payload["cplex_export"]
        baseline_export = payload["baseline_export"]
        objective = academic["objective"]
        owner_rows = "".join(
            (
                f"<li><strong>Owner {escape(str(row['owner']))}</strong>: decision={escape(str(row['decision']))}, "
                f"modelo={escape(str(row.get('best_model') or 'global'))}, test={escape(str(row['test_rows']))}, "
                f"adj={escape(str(row.get('adjacent_accuracy')))}, corr={escape(str(row.get('priority_corr')))}. "
                f"{escape(str(row['reason']))}</li>"
            )
            for row in payload["owner_selections"]
        )
        objective_components = "".join(
            (
                "<li>"
                f"<div class='math-inline'>${escape(str(component['term_latex']))}$</div>"
                f"<div>{escape(str(component['description']))}</div>"
                + (
                    f"<div><strong>Nota:</strong> {escape(str(component['notes']))}</div>"
                    if component.get("notes")
                    else ""
                )
                + "</li>"
            )
            for component in objective["components"]
        )
        bridge_aggregation = "".join(f"<li>{escape(str(item))}</li>" for item in bridge["aggregation_rules"])
        bridge_costs = "".join(f"<li>{escape(str(item))}</li>" for item in bridge["cost_bridge"])
        bridge_processes = "".join(f"<li>{escape(str(item))}</li>" for item in bridge["process_logic"])
        bridge_compatibility = "".join(f"<li>{escape(str(item))}</li>" for item in bridge["compatibility_rule"])
        segregation_rows = "".join(
            (
                f"<li><strong>{escape(str(sample['segregation_id']))}</strong>: week={escape(str(sample['week']))}, "
                f"count={escape(str(sample['container_count']))}, ū_jt={escape(str(sample['avg_u_jt']))}, "
                f"T_j={escape(str(sample.get('dwell_mean_tj')))}, α_j^L={escape(str(sample['alpha_l']))}, "
                f"α_j^P={escape(str(sample['alpha_p']))}, α_j^M={escape(str(sample['alpha_m']))}</li>"
            )
            for sample in bridge["segregation_samples"][:6]
        )
        cplex_pairs = "".join(
            (
                f"<li>&lt;{escape(str(pair['damaged_j']))},{escape(str(pair['operational_j']))}&gt; -> "
                f"{escape(str(pair['type']))}, size={escape(str(pair['size_teu']))}, owner={escape(str(pair['owner']))}</li>"
            )
            for pair in cplex_export["sample_pairs"]
        )
        baseline_assumptions_html = "".join(
            f"<li>{escape(str(item))}</li>" for item in baseline_export["assumptions"]
        )
        baseline_samples_html = "".join(
            (
                f"<li>j={escape(str(sample['j']))} | {escape(str(sample['base_code']))} | "
                f"dem_D={escape(str(sample['dem_d']))} | dem_O={escape(str(sample['dem_o']))} | "
                f"out_s={escape(str(sample['out_s']))} | Inv0_D={escape(str(sample['inv0_d']))} | "
                f"Inv0_O={escape(str(sample['inv0_o']))}</li>"
            )
            for sample in baseline_export["sample_segregations"]
        )
        warnings_html = "".join(
            f"<li>{escape(str(warning['message']))}</li>" for warning in payload.get("warnings", [])[:6]
        )
        body = "\n".join(
            [
                "<h1>Semana 3 - Modelamiento matematico</h1>",
                f"<p>{escape(week['description'])}</p>",
                "<p>La formulación académica incorpora del avance del compañero la notación por segregación, la lógica de procesos y la estructura formal de restricciones, mientras que la implementación operacional mantiene el parámetro <code>u_c</code> por contenedor y la penalización <code>P_{c,b}</code>.</p>",
                "<h2>Resumen de instancia</h2>",
                "<ul>",
                f"<li>Bahias: {summary['bay_count']}</li>",
                f"<li>Semanas del horizonte: {summary['planning_weeks']}</li>",
                f"<li>Capacidad total del yard (TEU): {summary['total_capacity_teu']}</li>",
                f"<li>Stock inicial: {summary['initial_stock_units']} contenedores / {summary['initial_stock_teu']} TEU</li>",
                f"<li>Pico activo: {summary['peak_active_units']} contenedores / {summary['peak_active_teu']} TEU</li>",
                f"<li>Utilizacion peak: {summary['peak_utilization_pct']}%</li>",
                f"<li>Pares de bahias compatibles: {summary['compatibility_pairs']}</li>",
                "</ul>",
                "<h2>Dimensiones e indices</h2>",
                f"<ul>{self._week3_html_math_entries(academic['dimensions'])}</ul>",
                "<h2>Datos observados</h2>",
                f"<ul>{self._week3_html_math_entries(academic['given_data'])}</ul>",
                "<h2>Parametros</h2>",
                f"<ul>{self._week3_html_math_entries(academic['parameters'])}</ul>",
                "<h2>Variables</h2>",
                f"<ul>{self._week3_html_math_entries(academic['decision_variables'])}</ul>",
                "<h2>Funcion objetivo</h2>",
                f"<div class='math-block'>$$ {escape(str(objective['equation_latex']))} $$</div>",
                f"<p>{escape(str(objective['description']))}</p>",
                "<h3>Componentes de la funcion objetivo</h3>",
                f"<ul>{objective_components}</ul>",
                "<h2>Restricciones</h2>",
                f"<ul>{self._week3_html_constraints(academic['constraints'])}</ul>",
                "<h2>Puente con la capa operacional</h2>",
                "<ul>",
                f"<li>Unidad academica: {escape(str(bridge['academic_unit']))}</li>",
                f"<li>Unidad operacional: {escape(str(bridge['operational_unit']))}</li>",
                f"<li>Modelo global ganador: {escape(str(ml.get('global_best_model')))}</li>",
                f"<li>Adjacent accuracy global: {escape(str(ml.get('global_adjacent_accuracy')))}</li>",
                f"<li>Correlacion priority score global: {escape(str(ml.get('global_priority_corr')))}</li>",
                f"<li>Regla de urgencia: <code>{escape(str(ml.get('urgency_rule')))}</code></li>",
                f"<li>Parametros objetivo: alpha={escape(str(ml.get('alpha')))}, beta={escape(str(ml.get('beta')))}, lambda_init={escape(str(ml.get('lambda_init')))}, lambda_tour={escape(str(ml.get('lambda_tour')))}, lambda_dwell={escape(str(ml.get('lambda_dwell')))}</li>",
                "</ul>",
                "<h3>Ecuaciones del puente</h3>",
                f"<ul>{self._week3_html_equations(bridge['equations'])}</ul>",
                f"<h3>Reglas de agregacion</h3><ul>{bridge_aggregation}</ul>",
                f"<h3>Construccion de costos</h3><ul>{bridge_costs}</ul>",
                f"<h3>Logica de procesos</h3><ul>{bridge_processes}</ul>",
                f"<h3>Compatibilidad</h3><ul>{bridge_compatibility}</ul>",
                "<h3>Selector owner/global</h3>",
                f"<ul>{owner_rows}</ul>",
                "<h3>Muestras de segregacion agregada</h3>",
                f"<ul>{segregation_rows or '<li>Sin muestras agregadas</li>'}</ul>",
                "<h2>Export CPLEX oficial</h2>",
                "<ul>",
                f"<li>Version: {escape(str(cplex_export['version']))}</li>",
                f"<li>Dimensiones exportadas: |J|={escape(str(cplex_export['j_count']))}, |JD|={escape(str(cplex_export['jd_count']))}, |JO|={escape(str(cplex_export['jo_count']))}, |K|={escape(str(cplex_export['k_count']))}, |B|={escape(str(cplex_export['b_count']))}, |T|={escape(str(cplex_export['t_count']))}</li>",
                f"<li>Owners incluidos: {escape(', '.join(cplex_export['owners']))}</li>",
                f"<li>Capacidad relajada de procesos en baseline: {escape(str(cplex_export['relaxed_process_capacity']))}</li>",
                f"<li>Penalizacion ML en objetivo: <code>{escape(str(cplex_export['ml_signal_rule']))}</code></li>",
                f"<li>Pesos ML: lambdaML={escape(str(cplex_export['lambda_ml']))}, wUrg={escape(str(cplex_export['w_urg']))}, wSlow={escape(str(cplex_export['w_slow']))}</li>",
                "<li>Artefactos generados: <code>official_model.mod</code>, <code>official_model.dat</code>, <code>cplex_segregations.csv</code>, <code>cplex_ml_signals.csv</code></li>",
                "</ul>",
                f"<h3>Supuestos del export .dat</h3><ul>{''.join(f'<li>{escape(str(item))}</li>' for item in cplex_export['assumptions'])}</ul>",
                f"<h3>Muestras de pares K</h3><ul>{cplex_pairs or '<li>Sin pares de ejemplo</li>'}</ul>",
                "<h2>Baseline de validacion del primer .mod</h2>",
                "<ul>",
                f"<li>Version: {escape(str(baseline_export['version']))}</li>",
                f"<li>Dimensiones exportadas: |J|={escape(str(baseline_export['j_count']))}, |B|={escape(str(baseline_export['b_count']))}, |C|={escape(str(len(baseline_export['c_labels'])))}</li>",
                f"<li>Semana validada: {escape(str(baseline_export['validated_week']))}</li>",
                f"<li>Owners incluidos: {escape(', '.join(baseline_export['owners']))}</li>",
                f"<li>Mapeo de inspeccion: {escape(str(baseline_export['inspection_mapping']))}</li>",
                f"<li>Inventario inicial agregado: {escape(str(baseline_export['total_inv0_units']))} contenedores / {escape(str(baseline_export['total_inv0_teu']))} TEU</li>",
                f"<li>Inventario inicial dañado: {escape(str(baseline_export['total_inv0_d_units']))} contenedores</li>",
                f"<li>Inventario inicial operativo: {escape(str(baseline_export['total_inv0_o_units']))} contenedores</li>",
                f"<li>Regla de modelado de inventario: {escape(str(baseline_export['initial_inventory_modeling_rule']))}</li>",
                f"<li>Capacidad de taller usada en validacion: {escape(str(baseline_export['relaxed_workshop_capacity']))}</li>",
                "<li>Artefactos generados: <code>baseline_validation.mod</code>, <code>baseline_validation.dat</code>, <code>baseline_segregations.csv</code>, <code>baseline_inventory_seed.csv</code></li>",
                "</ul>",
                f"<h3>Supuestos del baseline</h3><ul>{baseline_assumptions_html}</ul>",
                f"<h3>Muestras de segregacion base</h3><ul>{baseline_samples_html or '<li>Sin muestras base</li>'}</ul>",
                "<h2>Supuestos y extensiones</h2>",
                f"<h3>Supuestos activos en semana 3</h3><ul>{''.join(f'<li>{escape(str(item))}</li>' for item in academic['assumptions'])}</ul>",
                f"<h3>Extensiones naturales para semana 4</h3><ul>{self._week3_html_constraints(academic['week4_extensions'])}</ul>",
                "<h2>Advertencias de la instancia</h2>",
                f"<ul>{warnings_html or '<li>Sin advertencias principales</li>'}</ul>",
            ]
        )
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<title>Semana 3 - Modelamiento matematico</title>"
            "<style>body{font-family:Manrope,Arial,sans-serif;margin:32px;line-height:1.5;color:#173043;}"
            "h1,h2,h3{margin-bottom:8px;} ul{padding-left:20px;} code{background:#eef4f6;padding:2px 4px;border-radius:4px;}"
            ".math-block,.math-inline{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;background:#f4f8fb;border:1px solid #d7e2ea;border-radius:8px;padding:8px 10px;margin:8px 0;}"
            ".math-inline{display:inline-block;padding:2px 6px;margin:0 4px 0 0;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )

    def _render_markdown_report(self, week: dict[str, Any]) -> str:
        if week["week_id"] == "week-1":
            return self._render_week1_academic_markdown_report(week)
        if week["week_id"] == "week-3":
            return self._render_week3_markdown_report(week)
        lines = [
            f"# {week['title']} - {week['stage_name']}",
            "",
            week["description"],
            "",
            "## Estado",
            f"- Estado: {week['status']}",
            "- Reporte disponible: si",
            "",
            "## Seeds",
            f"- IN: {(week.get('seed_paths') or {}).get('in_file', 'n/a')}",
            f"- OUT: {(week.get('seed_paths') or {}).get('out_file', 'n/a')}",
            "",
            "## Checklist",
        ]
        lines.extend(f"- {item}" for item in week["checklist"])
        lines.extend(["", "## Entregables"])
        lines.extend(f"- {item}" for item in week["deliverables"])

        if week["status"] == "active":
            df, metadata = self.workspace_store.load_dataset(week["week_id"])
            lines.extend(
                [
                    "",
                    "## Dataset",
                    f"- Filas: {int(df.shape[0])}",
                    f"- Columnas: {int(df.shape[1])}",
                    f"- Target disponible: {'si' if metadata.get('has_target') else 'no'}",
                ]
            )
            if "eda" in week.get("analysis_available", []):
                eda = compute_eda(df)
                lines.extend(["", "## Resumen EDA"])
                for key, value in eda["global_metrics"].items():
                    lines.append(f"- {key}: {value}")
                warning_messages = [warning["message"] for warning in eda["warnings"][:6]]
                if warning_messages:
                    lines.extend(["", "## Advertencias principales"])
                    lines.extend(f"- {message}" for message in warning_messages)
            if "ml_overview" in week.get("analysis_available", []):
                ml = self._build_week_ml_payload(week)
                lines.extend(self._week_ml_markdown_sections(ml))
        else:
            lines.extend(
                [
                    "",
                    "## Estado del scaffold",
                    "Esta semana aun no ejecuta algoritmos automaticos. Sirve como guia de implementacion y portafolio.",
                ]
            )

        return "\n".join(lines).strip() + "\n"

    def _render_html_report(self, week: dict[str, Any]) -> str:
        if week["week_id"] == "week-1":
            return self._render_week1_academic_html_report(week)
        if week["week_id"] == "week-3":
            return self._render_week3_html_report(week)
        checklist_html = "".join(f"<li>{escape(item)}</li>" for item in week["checklist"])
        deliverables_html = "".join(f"<li>{escape(item)}</li>" for item in week["deliverables"])
        seeds = week.get("seed_paths") or {}
        body_parts = [
            f"<h1>{escape(week['title'])} - {escape(week['stage_name'])}</h1>",
            f"<p>{escape(week['description'])}</p>",
            "<h2>Estado</h2>",
            "<ul>",
            f"<li>Estado: {escape(str(week['status']))}</li>",
            "<li>Reporte disponible: si</li>",
            "</ul>",
            "<h2>Seeds</h2>",
            "<ul>",
            f"<li>IN: {escape(str(seeds.get('in_file') or 'n/a'))}</li>",
            f"<li>OUT: {escape(str(seeds.get('out_file') or 'n/a'))}</li>",
            "</ul>",
            "<h2>Checklist</h2>",
            f"<ul>{checklist_html}</ul>",
            "<h2>Entregables</h2>",
            f"<ul>{deliverables_html}</ul>",
        ]

        if week["status"] == "active":
            df, metadata = self.workspace_store.load_dataset(week["week_id"])
            body_parts.extend(
                [
                    "<h2>Dataset</h2>",
                    "<ul>",
                    f"<li>Filas: {int(df.shape[0])}</li>",
                    f"<li>Columnas: {int(df.shape[1])}</li>",
                    f"<li>Target disponible: {'si' if metadata.get('has_target') else 'no'}</li>",
                    "</ul>",
                ]
            )
            if "eda" in week.get("analysis_available", []):
                eda = compute_eda(df)
                metrics_html = "".join(
                    f"<li>{escape(str(key))}: {escape(str(value))}</li>" for key, value in eda["global_metrics"].items()
                )
                warnings_html = "".join(
                    f"<li>{escape(str(warning['message']))}</li>" for warning in eda["warnings"][:6]
                )
                body_parts.extend(
                    [
                        "<h2>Resumen EDA</h2>",
                        f"<ul>{metrics_html}</ul>",
                        "<h2>Advertencias principales</h2>",
                        f"<ul>{warnings_html or '<li>Sin advertencias principales</li>'}</ul>",
                    ]
                )
            if "ml_overview" in week.get("analysis_available", []):
                ml = self._build_week_ml_payload(week)
                body_parts.extend(self._week_ml_html_sections(ml))
        else:
            body_parts.extend(
                [
                    "<h2>Estado del scaffold</h2>",
                    "<p>Esta semana aun no ejecuta algoritmos automaticos. Sirve como guia de implementacion y portafolio.</p>",
                ]
            )

        body = "\n".join(body_parts)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{escape(week['title'])}</title>"
            "<style>body{font-family:Manrope,Arial,sans-serif;margin:32px;line-height:1.5;color:#173043;}"
            "h1,h2{margin-bottom:8px;} ul{padding-left:20px;} code{background:#eef4f6;padding:2px 4px;border-radius:4px;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )

    def _render_week1_academic_markdown_report(self, week: dict[str, Any]) -> str:
        bundle = self._build_week1_bundle(week["week_id"])
        academic_payload = bundle["eda_payload"]
        clustering_payload = bundle["clustering_payload"]
        problem = academic_payload["problem_definition"]
        in_source = academic_payload["sources"]["in"]
        out_source = academic_payload["sources"]["out"]

        lines = [
            "# Semana 1 - EDA",
            "",
            "## Introduccion",
            problem["domain_context"],
            "",
            f"- Objetivo: {problem['objective']}",
            f"- Unidad de observacion: {problem['unit_of_observation']}",
            f"- Variable objetivo principal: {problem.get('target_variable') or 'no aplica'}",
            "",
            "## Metodologia",
            "- Se analizaron por separado los datasets IN y OUT.",
            "- El dataset canonico se mantuvo solo como apoyo comparativo y de consistencia.",
            "- Los NA se auditan siempre y, si existen, se trabaja con una capa imputada derivada sin alterar el raw.",
            "- Los outliers se diagnostican, pero no se eliminan automaticamente en esta iteracion.",
            "- Se ejecuto OPTICS por separado sobre las capas analiticas de IN y OUT.",
            "- OPTICS se calcula en el espacio transformado y UMAP 2D se usa solo para visualizacion.",
            "- PCA queda como diagnostico secundario y ya no es la vista oficial del clustering.",
            "",
            "## Analisis del dataset IN",
            f"- Filas: {in_source['overview_metrics']['row_count']}",
            f"- Columnas: {in_source['overview_metrics']['column_count']}",
            f"- Completitud: {in_source['overview_metrics']['completeness_pct']}%",
            f"- Duplicados exactos: {in_source['overview_metrics']['duplicate_rows_exact']}",
            f"- Diagnostico temporal: {in_source['temporal_diagnostics']['message']}",
        ]
        for finding in in_source["data_quality"]["findings"][:6]:
            lines.append(f"- Calidad IN: {finding['hallazgo']}")
        for item in in_source["univariate_categorical"]["variables"][:4]:
            top = item["top_categories"][0] if item["top_categories"] else {"value": "n/a", "pct": 0}
            lines.append(
                f"- Distribucion IN {item['column']}: categoria dominante {top['value']} ({top['pct']}%)."
            )

        lines.extend(
            [
                "",
                "## Analisis del dataset OUT",
                f"- Filas: {out_source['overview_metrics']['row_count']}",
                f"- Columnas: {out_source['overview_metrics']['column_count']}",
                f"- Completitud: {out_source['overview_metrics']['completeness_pct']}%",
                f"- Duplicados exactos: {out_source['overview_metrics']['duplicate_rows_exact']}",
                f"- Diagnostico temporal: {out_source['temporal_diagnostics']['message']}",
            ]
        )
        for item in out_source["univariate_numeric"]["variables"][:3]:
            lines.append(
                f"- Variable numerica OUT {item['column']}: media={item['stats'].get('mean')}, mediana={item['stats'].get('median')}, std={item['stats'].get('std')}."
            )
        for row in out_source["bivariate_categorical_numeric"]["rows"][:5]:
            lines.append(
                f"- Relacion OUT {row['feature']}: {row['test_used']} con p={row.get('p_value')} y efecto={row.get('effect_size')}."
            )

        lines.extend(["", "## Comparacion IN vs OUT"])
        for note in academic_payload["comparison"]["notes"]:
            lines.append(f"- {note}")
        for comparison in academic_payload["comparison"]["categorical_comparisons"][:4]:
            if comparison["categories"]:
                top = comparison["categories"][0]
                lines.append(
                    f"- {comparison['column']}: categoria principal {top['category']} con {top['in_pct']}% en IN y {top['out_pct']}% en OUT."
                )

        lines.extend(
            [
                "",
                "## Calidad de datos e imputacion",
                (
                    "- Se aplico imputacion."
                    if academic_payload["imputation"]["imputation_applied"]
                    else "- No hubo NA en los seeds actuales; no fue necesario imputar."
                ),
            ]
        )
        for source in ("in", "out"):
            lines.append(f"- Estrategias {source.upper()}:")
            for column, strategy in academic_payload["imputation"]["strategy_by_column"][source].items():
                imputed = academic_payload["imputation"]["imputed_counts"][source][column]
                lines.append(f"  - {column}: {strategy}, imputados={imputed}")

        lines.extend(["", "## Outliers y anomalias"])
        for source in ("in", "out"):
            outlier_payload = academic_payload["outliers"]["sources"][source]
            lines.append(
                f"- {source.upper()}: {outlier_payload['interpretation']} Filas marcadas={outlier_payload['flagged_counts']} ({outlier_payload['flagged_ratio']}%)."
            )
            for column in outlier_payload["columns"][:3]:
                lines.append(
                    f"  - {column['column']}: {column['flagged_count']} registros, rango IQR=({column.get('lower_bound')}, {column.get('upper_bound')})"
                )

        lines.extend(["", "## Clustering OPTICS"])
        for source in ("in", "out"):
            optics_payload = clustering_payload["sources"][source]
            lines.append(
                f"- {source.upper()}: {optics_payload['interpretation']} Variables={', '.join(optics_payload['feature_columns']) or 'n/a'}."
            )
            lines.append(
                f"  - Embedding visual: {optics_payload['embedding_method'].upper()} con trustworthiness={optics_payload['embedding_quality'].get('trustworthiness')} y overlap raw={optics_payload['overlap_stats']['overlap_pct']}%."
            )
            lines.append(
                f"  - Parametros seleccionados: min_samples={optics_payload['selected_optics_parameters']['min_samples']}, min_cluster_size={optics_payload['selected_optics_parameters']['min_cluster_size']}, xi={optics_payload['selected_optics_parameters']['xi']}."
            )

        lines.extend(["", "## Conclusiones e hipotesis"])
        lines.extend(f"- {insight['title']}: {insight['implication']}" for insight in academic_payload["insights"])
        return "\n".join(lines).strip() + "\n"

    def _render_week1_academic_html_report(self, week: dict[str, Any]) -> str:
        markdown_content = self._render_week1_academic_markdown_report(week)
        sections = []
        current_heading = ""
        current_items: list[str] = []
        main_title = f"Semana {week['week_number']} - {week['stage_name']}"
        for line in markdown_content.splitlines():
            if line.startswith("## "):
                if current_heading:
                    sections.append((current_heading, current_items))
                current_heading = line[3:]
                current_items = []
                continue
            if line.startswith("# "):
                main_title = line[2:]
                current_heading = ""
                continue
            current_items.append(line)
        if current_heading:
            sections.append((current_heading, current_items))

        body_parts = [f"<h1>{escape(main_title)}</h1>"]
        for heading, items in sections:
            body_parts.append(f"<h2>{escape(heading)}</h2>")
            bullet_items = [item for item in items if item.startswith("- ")]
            paragraphs = [item for item in items if item and not item.startswith("- ")]
            for paragraph in paragraphs:
                body_parts.append(f"<p>{escape(paragraph)}</p>")
            if bullet_items:
                body_parts.append("<ul>")
                for bullet in bullet_items:
                    body_parts.append(f"<li>{escape(bullet[2:])}</li>")
                body_parts.append("</ul>")

        body = "\n".join(body_parts)
        return (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{escape(week['title'])}</title>"
            "<style>body{font-family:Manrope,Arial,sans-serif;margin:32px;line-height:1.55;color:#173043;}"
            "h1,h2{margin-bottom:10px;} ul{padding-left:20px;} p{max-width:900px;}"
            "</style></head><body>"
            f"{body}</body></html>"
        )

    @staticmethod
    def _schema_warnings(schema: dict[str, list[str]]) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []

        in_columns = set(schema.get("in_file", []))
        out_columns = set(schema.get("out_file", []))

        if in_columns:
            missing_in = [col for col in EXPECTED_IN_COLUMNS if col not in in_columns]
            if missing_in:
                warnings.append(
                    {
                        "code": "missing_expected_columns_in",
                        "severity": "warning",
                        "column": None,
                        "message": f"in_file missing expected columns: {', '.join(missing_in)}",
                        "suggestion": "Verify the seed schema before extending the weekly analysis.",
                    }
                )

        if out_columns:
            missing_out = [col for col in EXPECTED_OUT_COLUMNS if col not in out_columns]
            if missing_out:
                warnings.append(
                    {
                        "code": "missing_expected_columns_out",
                        "severity": "warning",
                        "column": None,
                        "message": f"out_file missing expected columns: {', '.join(missing_out)}",
                        "suggestion": "Verify the seed schema before extending the weekly analysis.",
                    }
                )

        return warnings

    @staticmethod
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


framework_service = FrameworkService()
