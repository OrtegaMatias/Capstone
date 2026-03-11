import sys
import time
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from loguru import logger


LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "request_id={extra[request_id]} dataset_id={extra[dataset_id]} "
    "path={extra[path]} method={extra[method]} status={extra[status]} duration_ms={extra[duration_ms]} | "
    "<level>{message}</level>"
)


def configure_logging() -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format=LOG_FORMAT)


def _bound_logger(
    request_id: str,
    dataset_id: str | None = None,
    path: str = "",
    method: str = "",
    status: int = 0,
    duration_ms: float = 0.0,
):
    return logger.bind(
        request_id=request_id,
        dataset_id=dataset_id or "-",
        path=path,
        method=method,
        status=status,
        duration_ms=f"{duration_ms:.2f}",
    )


def install_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_logging_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.perf_counter()
        dataset_id = request.path_params.get("dataset_id") if hasattr(request, "path_params") else None

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start) * 1000
            _bound_logger(
                request_id=request_id,
                dataset_id=dataset_id,
                path=request.url.path,
                method=request.method,
                status=response.status_code,
                duration_ms=duration_ms,
            ).info("request completed")
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000
            _bound_logger(
                request_id=request_id,
                dataset_id=dataset_id,
                path=request.url.path,
                method=request.method,
                status=500,
                duration_ms=duration_ms,
            ).exception("request failed")
            raise
