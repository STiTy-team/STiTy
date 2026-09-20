from core.errors import ConfigError
from core.utils import logging

log = logging.getLogger(__name__)


class Pipeline:
    REQUIRED: tuple = ()
    OPTIONAL: tuple = ()

    def __init__(self, settings: dict, *, parts: dict, cfg):
        self.settings = settings
        self.parts = parts
        self.cfg = cfg

    @classmethod
    def validate(cls, leftover_options: dict) -> dict:
        options = leftover_options
        if options:
            takes = sorted(r.kind for r in cls.REQUIRED + cls.OPTIONAL)
            raise ConfigError(
                f"pipeline: {cls.NAME!r} does not take {sorted(options)} "
                f"(its parts are {takes}; it has no other settings)"
            )
        return {}

    def part(self, kind: str):
        return self.parts.get(kind)

    async def load(self) -> None:
        for part in self.parts.values():
            await part.load()

    async def close(self) -> None:
        for part in self.parts.values():
            try:
                await part.close()
            except Exception as e:  # noqa: BLE001
                log.warning("[CLOSE-FAILED] %s: %s", type(part).__name__, e)

    def start(self, *, src_lang: str | None, target_lang: str) -> None:
        detector = self.part("vad")
        for part in self.parts.values():
            part.start(language=src_lang, target_lang=target_lang,
                       vad=None if part is detector else detector)

    async def listen(self, audio: bytes) -> list:
        raise NotImplementedError

    async def finish(self) -> list:
        raise NotImplementedError
