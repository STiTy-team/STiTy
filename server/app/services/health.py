from core.pipeline import describe

from app.config import ServerConfig


class HealthService:
    def __init__(self, config: ServerConfig):
        self._config = config

    def check(self) -> str:
        return "ok"

    def pipeline(self) -> dict:
        return describe(self._config)
