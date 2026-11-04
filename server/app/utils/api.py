from collections.abc import Iterable
from http import HTTPStatus
from typing import Any


def status_code_name(status: int) -> str:
    return HTTPStatus(status).name.lower()


def describe_validation_errors(errors: Iterable[dict[str, Any]]) -> str:
    return "; ".join(f"{'.'.join(map(str, error['loc']))}: {error['msg']}" for error in errors)
