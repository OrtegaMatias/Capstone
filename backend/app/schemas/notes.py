from pydantic import BaseModel


class NotesPayload(BaseModel):
    content: str


class NotesResponse(BaseModel):
    content: str
    updated_at: str | None = None


class NotesSaveResponse(BaseModel):
    ok: bool
    updated_at: str
