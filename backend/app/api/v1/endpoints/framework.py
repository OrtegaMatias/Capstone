from __future__ import annotations

import json
from queue import Queue

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse

from app.schemas.academic_eda import WeekAcademicEDAResponse, WeekClusteringResponse
from app.schemas.dataset import PreviewResponse
from app.schemas.framework import FrameworkSummary, MlEvaluationSummary, WeekConfig, WeekReportSummary
from app.schemas.notes import NotesPayload, NotesResponse, NotesSaveResponse
from app.services.framework_service import framework_service

router = APIRouter(tags=["framework"])


@router.get("/framework", response_model=FrameworkSummary)
def get_framework_summary() -> FrameworkSummary:
    payload = framework_service.get_framework_summary()
    return FrameworkSummary(**payload)


@router.get("/weeks/{week_id}", response_model=WeekConfig)
def get_week(week_id: str) -> WeekConfig:
    try:
        payload = framework_service.get_week(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WeekConfig(**payload)


@router.get("/weeks/{week_id}/preview", response_model=PreviewResponse)
def get_week_preview(week_id: str, limit: int = Query(20, ge=1, le=200)) -> PreviewResponse:
    try:
        payload = framework_service.get_week_preview(week_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PreviewResponse(**payload)


@router.get("/weeks/{week_id}/eda", response_model=WeekAcademicEDAResponse)
def get_week_eda(week_id: str) -> WeekAcademicEDAResponse:
    try:
        payload = framework_service.get_week_eda(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WeekAcademicEDAResponse(**payload)


@router.get("/weeks/{week_id}/clustering", response_model=WeekClusteringResponse)
def get_week_clustering(week_id: str) -> WeekClusteringResponse:
    try:
        payload = framework_service.get_week_clustering(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return WeekClusteringResponse(**payload)


@router.get("/weeks/{week_id}/ml/cached")
def get_week_ml_cached(week_id: str) -> Response:
    """Return cached ML results or 204 if no cache exists."""
    cached = framework_service.get_week_ml_cached(week_id)
    if cached is None:
        return Response(status_code=204)
    return Response(
        content=json.dumps(cached, default=str, ensure_ascii=False),
        media_type="application/json",
    )


@router.post("/weeks/{week_id}/ml/run")
def run_week_ml_sse(week_id: str) -> StreamingResponse:
    """Run ML analysis with SSE progress events."""
    progress_queue: Queue[str | None] = Queue()

    def on_progress(current: int, total: int, message: str) -> None:
        data = json.dumps({"current": current, "total": total, "message": message})
        progress_queue.put(f"data: {data}\n\n")

    def event_stream():  # type: ignore[no-untyped-def]
        import threading

        def _run() -> None:
            try:
                payload = framework_service.run_week_ml_with_progress(week_id, on_progress)
                done_data = json.dumps(payload, default=str, ensure_ascii=False)
                progress_queue.put(f"event: done\ndata: {done_data}\n\n")
            except Exception as exc:
                err_data = json.dumps({"detail": str(exc)})
                progress_queue.put(f"event: error\ndata: {err_data}\n\n")
            finally:
                progress_queue.put(None)  # sentinel

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()

        while True:
            item = progress_queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/weeks/{week_id}/ml/overview", response_model=MlEvaluationSummary)
def get_week_ml_overview(week_id: str) -> MlEvaluationSummary:
    try:
        payload = framework_service.get_week_ml_overview(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return MlEvaluationSummary(**payload)


@router.get("/weeks/{week_id}/notes", response_model=NotesResponse)
def get_week_notes(week_id: str) -> NotesResponse:
    try:
        payload = framework_service.get_week_notes(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NotesResponse(**payload)


@router.put("/weeks/{week_id}/notes", response_model=NotesSaveResponse)
def save_week_notes(week_id: str, notes: NotesPayload) -> NotesSaveResponse:
    try:
        payload = framework_service.save_week_notes(week_id, notes.content)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return NotesSaveResponse(**payload)


@router.post("/weeks/{week_id}/report/refresh", response_model=WeekReportSummary)
def refresh_week_report(week_id: str) -> WeekReportSummary:
    try:
        payload = framework_service.refresh_week_report(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WeekReportSummary(**payload)


@router.get("/weeks/{week_id}/report", response_model=WeekReportSummary)
def get_week_report(week_id: str) -> WeekReportSummary:
    try:
        payload = framework_service.get_week_report(week_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return WeekReportSummary(**payload)
