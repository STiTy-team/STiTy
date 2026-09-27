from http import HTTPStatus
from typing import Any, Self

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

from app.utils.errors import STiTyApiError

PROBLEM_JSON = "application/problem+json"


class InvalidMessage(STiTyApiError):
    code = "invalid_message"
    status_code = 422


class ProblemDetail(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = "about:blank"
    title: str
    status: int
    detail: str | None = None
    instance: str | None = None

    @classmethod
    def for_status(
        cls,
        status: int,
        detail: str | None = None,
        **extensions: Any,
    ) -> Self:
        return cls(
            title=HTTPStatus(status).phrase,
            status=status,
            detail=detail,
            **extensions,
        )

    @classmethod
    def for_request(
        cls,
        request: Request,
        status: int,
        code: str,
        detail: str,
    ) -> Self:
        return cls.for_status(
            status,
            detail,
            instance=request.url.path,
            code=code,
        )

    def to_response(self) -> JSONResponse:
        return JSONResponse(
            self.model_dump(exclude_none=True),
            status_code=self.status,
            media_type=PROBLEM_JSON,
        )
