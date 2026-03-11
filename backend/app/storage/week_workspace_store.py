from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class WeekWorkspaceStore:
    def __init__(self, workspace_root: Path):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def week_dir(self, week_id: str) -> Path:
        return self.workspace_root / week_id

    def ensure_week_dir(self, week_id: str) -> Path:
        week_dir = self.week_dir(week_id)
        week_dir.mkdir(parents=True, exist_ok=True)
        (week_dir / "exports").mkdir(parents=True, exist_ok=True)
        return week_dir

    def source_path(self, week_id: str, source: str) -> Path:
        if source not in {"in", "out"}:
            raise ValueError("source must be either 'in' or 'out'")
        return self.week_dir(week_id) / f"raw_{source}.csv"

    def canonical_path(self, week_id: str) -> Path:
        return self.week_dir(week_id) / "canonical.csv"

    def metadata_path(self, week_id: str) -> Path:
        return self.week_dir(week_id) / "metadata.json"

    def analysis_imputed_path(self, week_id: str, source: str) -> Path:
        if source not in {"in", "out"}:
            raise ValueError("source must be either 'in' or 'out'")
        return self.week_dir(week_id) / f"analysis_{source}_imputed.csv"

    def optics_path(self, week_id: str, source: str) -> Path:
        if source not in {"in", "out"}:
            raise ValueError("source must be either 'in' or 'out'")
        return self.week_dir(week_id) / f"optics_{source}.json"

    def notes_path(self, week_id: str) -> Path:
        return self.week_dir(week_id) / "notes.md"

    def report_markdown_path(self, week_id: str) -> Path:
        return self.week_dir(week_id) / "report.md"

    def report_html_path(self, week_id: str) -> Path:
        return self.week_dir(week_id) / "report.html"

    def save_dataset(
        self,
        week_id: str,
        canonical_df: pd.DataFrame,
        metadata: dict[str, Any],
        in_bytes: bytes | None = None,
        out_bytes: bytes | None = None,
    ) -> None:
        self.ensure_week_dir(week_id)

        if in_bytes is not None:
            self.source_path(week_id, "in").write_bytes(in_bytes)
        if out_bytes is not None:
            self.source_path(week_id, "out").write_bytes(out_bytes)

        canonical_df.to_csv(self.canonical_path(week_id), index=False)
        self.metadata_path(week_id).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        notes_path = self.notes_path(week_id)
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")

    def load_dataset(self, week_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        canonical_path = self.canonical_path(week_id)
        metadata_path = self.metadata_path(week_id)

        if not canonical_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Week dataset {week_id} not found")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        df = pd.read_csv(canonical_path, dtype=str)
        dtype_map = metadata.get("dtypes", {})

        for col in df.columns:
            dtype_name = str(dtype_map.get(col, "object"))
            if "int" in dtype_name or "float" in dtype_name:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                df[col] = df[col].astype("string")

        return df, metadata

    def read_source_bytes(self, week_id: str, source: str) -> bytes:
        path = self.source_path(week_id, source)
        if not path.exists():
            raise FileNotFoundError(f"Raw source '{source}' for week {week_id} not found")
        return path.read_bytes()

    def write_analysis_imputed(self, week_id: str, source: str, df: pd.DataFrame) -> None:
        self.ensure_week_dir(week_id)
        df.to_csv(self.analysis_imputed_path(week_id, source), index=False)

    def write_optics_payload(self, week_id: str, source: str, payload: dict[str, Any]) -> None:
        self.ensure_week_dir(week_id)
        self.optics_path(week_id, source).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def read_optics_payload(self, week_id: str, source: str) -> dict[str, Any]:
        path = self.optics_path(week_id, source)
        if not path.exists():
            raise FileNotFoundError(f"OPTICS payload for source '{source}' and week {week_id} not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def source_exists(self, week_id: str, source: str) -> bool:
        return self.source_path(week_id, source).exists()

    def read_notes(self, week_id: str) -> tuple[str, str | None]:
        path = self.notes_path(week_id)
        if not path.exists():
            return "", None
        content = path.read_text(encoding="utf-8")
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return content, updated_at

    def write_notes(self, week_id: str, content: str) -> str:
        self.ensure_week_dir(week_id)
        path = self.notes_path(week_id)
        path.write_text(content, encoding="utf-8")
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()

    def read_report(self, week_id: str) -> tuple[str, str, str | None]:
        md_path = self.report_markdown_path(week_id)
        html_path = self.report_html_path(week_id)
        if not md_path.exists() or not html_path.exists():
            raise FileNotFoundError(f"Report for week {week_id} not found")
        updated_at = datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).isoformat()
        return md_path.read_text(encoding="utf-8"), html_path.read_text(encoding="utf-8"), updated_at

    def write_report(self, week_id: str, markdown_content: str, html_content: str) -> str:
        self.ensure_week_dir(week_id)
        md_path = self.report_markdown_path(week_id)
        html_path = self.report_html_path(week_id)
        md_path.write_text(markdown_content, encoding="utf-8")
        html_path.write_text(html_content, encoding="utf-8")
        return datetime.fromtimestamp(md_path.stat().st_mtime, tz=timezone.utc).isoformat()
