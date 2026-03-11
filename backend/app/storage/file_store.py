from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


class DatasetFileStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def dataset_dir(self, dataset_id: str) -> Path:
        return self.base_dir / dataset_id

    def source_path(self, dataset_id: str, source: str) -> Path:
        if source not in {"in", "out"}:
            raise ValueError("source must be either 'in' or 'out'")
        return self.dataset_dir(dataset_id) / f"raw_{source}.csv"

    def source_exists(self, dataset_id: str, source: str) -> bool:
        return self.source_path(dataset_id, source).exists()

    def read_source_bytes(self, dataset_id: str, source: str) -> bytes:
        path = self.source_path(dataset_id, source)
        if not path.exists():
            raise FileNotFoundError(f"Raw source '{source}' for dataset {dataset_id} not found")
        return path.read_bytes()

    def save_dataset(
        self,
        dataset_id: str,
        canonical_df: pd.DataFrame,
        metadata: dict[str, Any],
        in_bytes: bytes | None = None,
        out_bytes: bytes | None = None,
    ) -> None:
        ds_dir = self.dataset_dir(dataset_id)
        ds_dir.mkdir(parents=True, exist_ok=True)

        if in_bytes is not None:
            (ds_dir / "raw_in.csv").write_bytes(in_bytes)
        if out_bytes is not None:
            (ds_dir / "raw_out.csv").write_bytes(out_bytes)

        canonical_df.to_csv(ds_dir / "canonical.csv", index=False)
        (ds_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        notes_path = ds_dir / "notes.md"
        if not notes_path.exists():
            notes_path.write_text("", encoding="utf-8")

    def load_dataset(self, dataset_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        ds_dir = self.dataset_dir(dataset_id)
        canonical_path = ds_dir / "canonical.csv"
        metadata_path = ds_dir / "metadata.json"

        if not canonical_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")

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

    def read_notes(self, dataset_id: str) -> tuple[str, str | None]:
        ds_dir = self.dataset_dir(dataset_id)
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        notes_path = ds_dir / "notes.md"
        if not notes_path.exists():
            return "", None
        content = notes_path.read_text(encoding="utf-8")
        updated_at = datetime.fromtimestamp(notes_path.stat().st_mtime, tz=timezone.utc).isoformat()
        return content, updated_at

    def write_notes(self, dataset_id: str, content: str) -> str:
        ds_dir = self.dataset_dir(dataset_id)
        if not ds_dir.exists():
            raise FileNotFoundError(f"Dataset {dataset_id} not found")
        notes_path = ds_dir / "notes.md"
        notes_path.write_text(content, encoding="utf-8")
        return datetime.fromtimestamp(notes_path.stat().st_mtime, tz=timezone.utc).isoformat()
