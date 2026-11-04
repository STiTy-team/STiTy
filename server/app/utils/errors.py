from abc import ABC, abstractmethod

from core.errors import STiTyError

INTERNAL_MESSAGE = "Something went wrong on the server"
BASE_CLOSE_CODE = 4000


def close_code_for(status_code: int) -> int:
    return BASE_CLOSE_CODE + status_code


class STiTyApiError(STiTyError, ABC):
    exit_code = 1

    @property
    @abstractmethod
    def code(self) -> str: ...

    @property
    @abstractmethod
    def status_code(self) -> int: ...

    @property
    def message(self) -> str:
        return INTERNAL_MESSAGE if self.status_code >= 500 else str(self)

    @property
    def close_code(self) -> int:
        return close_code_for(self.status_code)
