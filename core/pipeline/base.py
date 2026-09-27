from dataclasses import dataclass
from enum import StrEnum

from core.components.registry import Component
from core.errors import ConfigError
from core.utils import logging

from . import stages

log = logging.getLogger(__name__)


class RunsPer(StrEnum):
    ROOM = "room"
    TARGET_LANG = "target_lang"


class Processor:
    """One stage, run over a live stream.

    A ROOM stage is fed the room's audio; a TARGET_LANG stage is fed what the ROOM
    stage produced, once per target language. `process` returns what the stage made
    of one item, `finish` whatever it was still holding when the stream ended.
    `skip` is an item the caller chose not to process, which a stage that keeps
    context may still want to see.
    """

    def __init__(self, parts: dict[str, Component]):
        self.parts = parts

    def start(self, *, languages: list[str], target_lang: str = "") -> None:
        stages.start(self.parts, languages=languages, target_lang=target_lang)

    def accepts(self, item) -> bool:
        return True

    async def process(self, item) -> list:
        raise NotImplementedError

    def skip(self, item) -> None:
        pass

    async def finish(self) -> list:
        return []

    async def close(self) -> None:
        await stages.close(self.parts)


@dataclass(frozen=True)
class Stage:
    parts: tuple[str, ...]
    runs_per: RunsPer
    processor: type[Processor]


class Pipeline:
    REQUIRED: tuple = ()
    OPTIONAL: tuple = ()
    STAGES: tuple[Stage, ...] = ()

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
        await stages.close(self.parts)

    def start(self, *, languages: list[str], target_lang: str,
              own_lang: str | None = None) -> None:
        stages.start(self.parts, languages=languages, target_lang=target_lang)

    async def listen(self, audio: bytes) -> list:
        raise NotImplementedError

    async def finish(self) -> list:
        raise NotImplementedError
