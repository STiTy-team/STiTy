from core.components.correction import correctors
from core.components.mixer import mixers
from core.components.registry import Transcribed, Translated
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors
from core.utils import timing

from . import pipelines
from .base import Pipeline, RunsPer, Stage


@pipelines.register("cascade")
class CascadePipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (mixers, detectors, correctors, translators)
    STAGES = (
        Stage(parts=("mixer", "vad", "transcription"), runs_per=RunsPer.ROOM),
        Stage(parts=("correction", "translation"), runs_per=RunsPer.TARGET_LANG),
    )

    def start(self, *, languages: list[str], target_lang: str) -> None:
        self.target_lang = target_lang
        self.said_so_far: list[str] = []
        super().start(languages=languages, target_lang=target_lang)

    async def listen(self, audio: bytes) -> list:
        transcriber = self.parts["transcription"]
        detector = self.part("vad")
        speech = None if detector is None else detector.detect(audio)
        produced = [] if speech is None else [speech]
        produced.extend(await transcriber.transcribe(audio))
        if speech is not None:
            produced.extend(await transcriber.flush("vad", speech))
        return await self._translated(produced)

    async def finish(self) -> list:
        return await self._translated(await self.parts["transcription"].finish())

    async def _translated(self, produced: list) -> list:
        corrector, translator = self.part("correction"), self.part("translation")
        out = []
        for item in produced:
            if not isinstance(item, Transcribed):
                out.append(item)
                continue

            out.append(item)
            original = item.original
            language = item.language or ""
            if corrector is not None:
                original = await corrector.correct(original, language)

            translation, detected = "", ""
            target = self.target_lang
            if language == target:
                translation = original
            elif translator is not None:
                translation, detected = await translator.translate(
                    original, target, language or None, context=list(self.said_so_far))
            if original:
                self.said_so_far.append(original)

            out.append(Translated(
                original=original,
                translation=translation,
                language=item.language or detected or "",
                target_lang=target,
                commit_reason=item.commit_reason,
                decision_audio_sec=item.decision_audio_sec,
                committed_elapsed_sec=item.committed_elapsed_sec,
                translated_elapsed_sec=timing.elapsed(),
            ))
        return out
