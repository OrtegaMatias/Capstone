from fastapi import APIRouter, HTTPException

from app.schemas.notes import NotesPayload, NotesResponse, NotesSaveResponse
from app.services.dataset_service import DatasetService
from app.services.notes_service import NotesService

router = APIRouter(tags=["notes"])
dataset_service = DatasetService()
notes_service = NotesService(dataset_service.file_store)


@router.get("/datasets/{dataset_id}/notes", response_model=NotesResponse)
def get_notes(dataset_id: str) -> NotesResponse:
    try:
        payload = notes_service.get_notes(dataset_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NotesResponse(**payload)


@router.put("/datasets/{dataset_id}/notes", response_model=NotesSaveResponse)
def save_notes(dataset_id: str, notes: NotesPayload) -> NotesSaveResponse:
    try:
        payload = notes_service.save_notes(dataset_id, notes.content)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NotesSaveResponse(**payload)
