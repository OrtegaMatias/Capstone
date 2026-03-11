from fastapi import APIRouter, UploadFile

from app.schemas.upload import UploadResponse
from app.services.dataset_service import DatasetService

router = APIRouter(tags=["upload"])
service = DatasetService()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(in_file: UploadFile | None = None, out_file: UploadFile | None = None) -> UploadResponse:
    payload = await service.upload_dataset(in_file=in_file, out_file=out_file)
    return UploadResponse(**payload)
