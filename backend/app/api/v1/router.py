from fastapi import APIRouter

from app.api.v1.endpoints.framework import router as framework_router

api_router = APIRouter()
api_router.include_router(framework_router)
