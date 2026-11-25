"""zero: the bare-bones speech translation pipeline.

Speech goes in, the transcriber commits a line, and that line is translated on its own.
Nothing sits between the two steps: no language re-detection, no correction, no context
from earlier lines, no fallback when a translation fails.
"""

from core.components.mixer import mixers
from core.components.registry import Transcribed
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors
from core.utils import langs

from . import pipelines, stages
from .base import Pipeline, Processor, RunsPer, Stage
from .cascade import SpeechProcessor


class TranslationProcessor(Processor):
    """Server stage: translates every committed line into one target language."""

    def start(self, *, languages: list[str], target_lang: str = "") -> None:
        super().start(languages=languages, target_lang=target_lang)
        self.target_lang = target_lang

    def accepts(self, item) -> bool:
        return isinstance(item, Transcribed)

    async def process(self, item: Transcribed) -> list:
        return [await stages.translate(self.parts, item, target_lang=self.target_lang, context=[])]


@pipelines.register("zero")
class ZeroPipeline(Pipeline):
    """Transcription and translation are required; the mixer and VAD are optional.

    The server runs STAGES: speech once per room, translation once per target language.
    Bench calls listen/finish directly and gets the same output in one list.
    """

    REQUIRED = (transcribers, translators)
    OPTIONAL = (mixers, detectors)
    STAGES = (
        Stage(
            parts=("mixer", "vad", "transcription"),
            runs_per=RunsPer.ROOM,
            processor=SpeechProcessor,
        ),
        Stage(
            parts=("translation",),
            runs_per=RunsPer.TARGET_LANG,
            processor=TranslationProcessor,
        ),
    )

    def start(self, *, languages: list[str], target_lang: str, own_lang: str | None = None) -> None:
        self.target_lang = target_lang
        self.own_lang = own_lang
        super().start(languages=languages, target_lang=target_lang)

    async def listen(self, audio: bytes) -> list:
        return await self._with_translations(await stages.hear(self.parts, audio))

    async def finish(self) -> list:
        return await self._with_translations(await stages.finish(self.parts))

    async def _with_translations(self, produced: list) -> list:
        """Returns what the transcriber produced, each committed line followed by its translation."""
        out = []
        for item in produced:
            out.append(item)
            if isinstance(item, Transcribed):
                target = langs.pick_target(
                    item.language, lang=self.own_lang or "", target_lang=self.target_lang
                )
                out.append(await stages.translate(self.parts, item, target_lang=target, context=[]))
        return out
