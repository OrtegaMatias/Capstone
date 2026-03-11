from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from app.schemas.pivot import (
    PivotMetadataResponse,
    PivotQueryRequest,
    PivotQueryResponse,
    PivotSourcesResponse,
)
from app.services.dataset_service import DatasetService

router = APIRouter(tags=["pivot"])
service = DatasetService()


@router.get("/datasets/{dataset_id}/pivot/sources", response_model=PivotSourcesResponse)
def get_pivot_sources(dataset_id: str) -> PivotSourcesResponse:
    payload = service.get_pivot_sources(dataset_id)
    return PivotSourcesResponse(**payload)


@router.get("/datasets/{dataset_id}/pivot/metadata", response_model=PivotMetadataResponse)
def get_pivot_metadata(dataset_id: str, source: str = Query(..., pattern="^(in|out)$")) -> PivotMetadataResponse:
    try:
        payload = service.get_pivot_metadata(dataset_id, source)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PivotMetadataResponse(**payload)


@router.post("/datasets/{dataset_id}/pivot/query", response_model=PivotQueryResponse)
def run_pivot_query(dataset_id: str, request: PivotQueryRequest) -> PivotQueryResponse:
    try:
        payload = service.run_pivot_query(dataset_id, request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PivotQueryResponse(**payload)
