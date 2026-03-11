from __future__ import annotations

from app.storage.file_store import DatasetFileStore


class NotesService:
    def __init__(self, file_store: DatasetFileStore):
        self.file_store = file_store

    def get_notes(self, dataset_id: str) -> dict[str, str | None]:
        content, updated_at = self.file_store.read_notes(dataset_id)
        return {"content": content, "updated_at": updated_at}

    def save_notes(self, dataset_id: str, content: str) -> dict[str, str | bool]:
        updated_at = self.file_store.write_notes(dataset_id, content)
        return {"ok": True, "updated_at": updated_at}
