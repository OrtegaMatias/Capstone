from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: str
    message: str
    details: dict | None = None


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
        payload = ErrorResponse(
            code="http_error",
            message=str(exc.detail),
            details={"status_code": exc.status_code},
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        payload = ErrorResponse(
            code="internal_error",
            message="Unexpected error",
            details={"error": str(exc)},
        )
        return JSONResponse(status_code=500, content=payload.model_dump())
