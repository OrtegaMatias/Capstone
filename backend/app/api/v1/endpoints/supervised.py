from fastapi import APIRouter

from app.schemas.supervised import AnovaResponse, MultipleRegressionResponse, SupervisedOverviewResponse
from app.services.dataset_service import DatasetService

router = APIRouter(tags=["supervised"])
service = DatasetService()


@router.get("/datasets/{dataset_id}/supervised/overview", response_model=SupervisedOverviewResponse)
def get_supervised_overview(dataset_id: str) -> SupervisedOverviewResponse:
    payload = service.get_supervised_overview(dataset_id)
    return SupervisedOverviewResponse(**payload)


@router.get("/datasets/{dataset_id}/anova", response_model=AnovaResponse)
def get_anova(dataset_id: str) -> AnovaResponse:
    payload = service.get_anova(dataset_id)
    return AnovaResponse(**payload)


@router.get("/datasets/{dataset_id}/supervised/multiple-regression", response_model=MultipleRegressionResponse)
def get_multiple_regression(dataset_id: str) -> MultipleRegressionResponse:
    payload = service.get_multiple_regression(dataset_id)
    return MultipleRegressionResponse(**payload)
