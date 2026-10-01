import dataclasses

from core.components.correction import correctors
from core.components.langid import langids
from core.components.mixer import mixers
from core.components.registry import Transcribed
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors
from core.utils import langs, logging

from . import pipelines, stages
from .base import Pipeline

log = logging.getLogger(__name__)


async def detect_language(parts: dict, item: Transcribed) -> Transcribed:
    detector = parts.get("langid")
    if detector is None:
        return item
    try:
        detected = await detector.detect(item.original)
    except Exception as e:  # noqa: BLE001
        log.warning("[LANGID-FAILED] %s", e)
        return item
    if detected is None or detected == item.language:
        return item
    log.debug("[LANG-FIX] %r -> %r: %r", item.language, detected, item.original)
    return dataclasses.replace(item, language=detected)


@pipelines.register("cascade:v2")
class CascadePipelineV2(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (mixers, detectors, correctors, translators, langids)

    def start(
        self, *, languages: list[str], target_lang: str, own_lang: str | None = None
    ) -> None:
        self.target_lang = target_lang
        self.own_lang = own_lang
        self.said_so_far: list[str] = []
        stages.start(self.parts, languages=languages, target_lang=target_lang)

    async def listen(self, audio: bytes) -> list:
        return await self._translated(await stages.hear(self.parts, audio))

    async def finish(self) -> list:
        return await self._translated(await stages.finish(self.parts))

    async def _translated(self, produced: list) -> list:
        out = []
        for item in produced:
            out.append(item)
            if not isinstance(item, Transcribed):
                continue
            item = await detect_language(self.parts, item)
            target = langs.pick_target(
                item.language, lang=self.own_lang or "", target_lang=self.target_lang
            )
            translated = await stages.translate(
                self.parts, item, target_lang=target, context=list(self.said_so_far)
            )
            if translated.original:
                self.said_so_far.append(translated.original)
            out.append(translated)
        return out
