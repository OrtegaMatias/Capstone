from fastapi import APIRouter, Query

from app.schemas.dataset import PreviewResponse
from app.schemas.eda import EDAResponse
from app.schemas.variability import VariabilityResponse
from app.services.dataset_service import DatasetService

router = APIRouter(tags=["datasets"])
service = DatasetService()


@router.get("/datasets/{dataset_id}/preview", response_model=PreviewResponse)
def get_preview(dataset_id: str, limit: int = Query(20, ge=1, le=200)) -> PreviewResponse:
    payload = service.get_preview(dataset_id, limit=limit)
    return PreviewResponse(**payload)


@router.get("/datasets/{dataset_id}/eda", response_model=EDAResponse)
def get_eda(dataset_id: str) -> EDAResponse:
    payload = service.get_eda(dataset_id)
    return EDAResponse(**payload)


@router.get("/datasets/{dataset_id}/variability", response_model=VariabilityResponse)
def get_variability(
    dataset_id: str,
    custom_mode: str = Query("freq_only", pattern="^(freq_only|ordinal_map)$"),
    ordinal_strategy: str = Query("frequency", pattern="^(frequency|alphabetical)$"),
) -> VariabilityResponse:
    payload = service.get_variability(dataset_id, custom_mode=custom_mode, ordinal_strategy=ordinal_strategy)
    return VariabilityResponse(**payload)
