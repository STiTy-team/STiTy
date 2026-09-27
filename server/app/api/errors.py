from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from core.utils import logging

from app.api.schema import ProblemDetail
from app.utils.api import describe_validation_errors, status_code_name
from app.utils.errors import INTERNAL_MESSAGE, STiTyApiError

log = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(STiTyApiError, on_api_error)
    app.add_exception_handler(HTTPException, on_http_error)
    app.add_exception_handler(RequestValidationError, on_validation_error)
    app.add_exception_handler(Exception, on_unexpected_error)


async def on_api_error(request: Request, error: STiTyApiError) -> JSONResponse:
    if error.status_code >= 500:
        log.error("Request failed (%s): %s", error.code, error, exc_info=error)
    else:
        log.warning("Request refused (%s): %s", error.code, error)

    return ProblemDetail.for_request(
        request,
        error.status_code,
        error.code,
        error.message,
    ).to_response()


async def on_http_error(request: Request, error: HTTPException) -> JSONResponse:
    code = status_code_name(error.status_code)
    return ProblemDetail.for_request(
        request, error.status_code, code, str(error.detail)
    ).to_response()


async def on_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    problems = describe_validation_errors(error.errors())
    return ProblemDetail.for_request(request, 422, "invalid_request", problems).to_response()


async def on_unexpected_error(request: Request, error: Exception) -> JSONResponse:
    log.error("Request failed unexpectedly: %s", error, exc_info=error)
    return ProblemDetail.for_request(request, 500, "internal_error", INTERNAL_MESSAGE).to_response()
