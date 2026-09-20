from core.components.correction import correctors
from core.components.registry import Final, Transcribed
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors

from . import pipelines
from .base import Pipeline


@pipelines.register("cascade")
class CascadePipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (detectors, correctors, translators)

    def start(self, *, src_lang: str | None, target_lang: str) -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.said_so_far: list[str] = []
        super().start(src_lang=src_lang, target_lang=target_lang)

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
            language = item.language or self.src_lang or ""
            if corrector is not None:
                original = await corrector.correct(original, language)

            translation, detected = "", ""
            if translator is not None:
                translation, detected = await translator.translate(
                    original, self.target_lang, language,
                    context=list(self.said_so_far))
            if original:
                self.said_so_far.append(original)

            out.append(Final(
                original=original,
                translation=translation,
                language=item.language or detected or "",
                target_lang=self.target_lang if translator is not None else "",
                commit_reason=item.commit_reason,
                decision_audio_sec=item.decision_audio_sec,
                recv_elapsed_sec=item.recv_elapsed_sec,
            ))
        return out
