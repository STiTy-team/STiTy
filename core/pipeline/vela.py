import dataclasses

from core.components.correction import correctors
from core.components.langid import langids
from core.components.mixer import mixers
from core.components.registry import Transcribed
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors
from core.utils import langs, logging
from core.utils.audio import MultiChannelAudio

from . import pipelines, stages
from .base import Pipeline, Processor, RunsPer, Stage

log = logging.getLogger(__name__)


def hear_language(parts: dict, audio: bytes) -> None:
    detector = parts.get("langid")
    if detector is not None:
        detector.hear(audio)


async def detect_language(parts: dict, item: Transcribed) -> Transcribed:
    detector = parts.get("langid")
    if detector is None:
        return item
    try:
        detected = await detector.detect(item)
    except Exception as e:  # noqa: BLE001
        log.warning("[LANGID-FAILED] %s", e)
        return item
    if detected is None or detected == item.language:
        return item
    log.debug("[LANG-FIX] %r -> %r: %r", item.language, detected, item.original)
    return dataclasses.replace(item, language=detected)


async def hear(parts: dict, audio: bytes) -> list:
    hear_language(parts, audio)
    return await identified(parts, await stages.hear(parts, audio))


async def identified(parts: dict, produced: list) -> list:
    return [await detect_language(parts, item) if isinstance(item, Transcribed) else item
            for item in produced]


class SpeechProcessor(Processor):
    async def process(self, audio: MultiChannelAudio) -> list:
        return await hear(self.parts, self.parts["mixer"].mix(audio))

    async def finish(self) -> list:
        return await identified(self.parts, await stages.finish(self.parts))


class TranslationProcessor(Processor):
    def start(self, *, languages: list[str], target_lang: str = "") -> None:
        super().start(languages=languages, target_lang=target_lang)
        self.target_lang = target_lang
        self.languages = set(languages)
        self.said_so_far: list[str] = []

    def accepts(self, item) -> bool:
        return isinstance(item, Transcribed)

    async def process(self, item: Transcribed) -> list:
        allowed = not self.languages or item.language in self.languages
        try:
            translated = await stages.translate(
                self.parts, item, target_lang=self.target_lang,
                context=list(self.said_so_far), allowed=allowed)
        except Exception:
            log.exception("[TRANSLATE-FAILED] to %s", self.target_lang)
            translated = stages.untranslated(item, self.target_lang)
        self._remember(translated.original)
        return [translated]

    def skip(self, item: Transcribed) -> None:
        self._remember(item.original)

    def _remember(self, original: str) -> None:
        if original:
            self.said_so_far.append(original)


@pipelines.register("vela")
class VelaPipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (mixers, detectors, langids, correctors, translators)
    STAGES = (
        Stage(parts=("mixer", "vad", "transcription", "langid"), runs_per=RunsPer.ROOM,
              processor=SpeechProcessor),
        Stage(parts=("correction", "translation"), runs_per=RunsPer.TARGET_LANG,
              processor=TranslationProcessor),
    )

    def start(self, *, languages: list[str], target_lang: str,
              own_lang: str | None = None) -> None:
        self.target_lang = target_lang
        self.own_lang = own_lang
        self.said_so_far: list[str] = []
        super().start(languages=languages, target_lang=target_lang)

    async def listen(self, audio: bytes) -> list:
        return await self._translated(await hear(self.parts, audio))

    async def finish(self) -> list:
        return await self._translated(
            await identified(self.parts, await stages.finish(self.parts)))

    async def _translated(self, produced: list) -> list:
        out = []
        for item in produced:
            out.append(item)
            if not isinstance(item, Transcribed):
                continue
            target = langs.pick_target(
                item.language, lang=self.own_lang or "", target_lang=self.target_lang)
            translated = await stages.translate(
                self.parts, item, target_lang=target, context=list(self.said_so_far))
            if translated.original:
                self.said_so_far.append(translated.original)
            out.append(translated)
        return out
