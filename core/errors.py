from abc import ABC, abstractmethod


class STiTyError(Exception, ABC):
    @property
    @abstractmethod
    def exit_code(self) -> int: ...


class ConfigError(STiTyError):
    exit_code = 2


class DataError(STiTyError):
    exit_code = 1


class AudioError(DataError):
    pass


class ProcessError(STiTyError):
    exit_code = 1
