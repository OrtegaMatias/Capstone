from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

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
