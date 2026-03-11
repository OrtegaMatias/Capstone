from fastapi import APIRouter

from app.api.v1.endpoints.datasets import router as datasets_router
from app.api.v1.endpoints.framework import router as framework_router
from app.api.v1.endpoints.notes import router as notes_router
from app.api.v1.endpoints.pivot import router as pivot_router
from app.api.v1.endpoints.supervised import router as supervised_router
from app.api.v1.endpoints.upload import router as upload_router

api_router = APIRouter()
api_router.include_router(framework_router)
api_router.include_router(upload_router)
api_router.include_router(datasets_router)
api_router.include_router(supervised_router)
api_router.include_router(notes_router)
api_router.include_router(pivot_router)
