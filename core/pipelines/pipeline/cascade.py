from core.utils import logging

from ..correction import correctors
from ..transcription import transcribers
from ..translation import translators
from ..translation.dialogue import Dialogue
from ..vad import detectors
from . import pipelines
from .base import Pipeline

logger = logging.getLogger("bench")


@pipelines.register("cascade")
class CascadePipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (detectors, correctors, translators)

    def __init__(self, settings: dict, *, parts: dict, cfg):
        super().__init__(settings, parts=parts, cfg=cfg)
        self.dialogue = Dialogue()

    def start(self, *, src_lang: str | None, target_lang: str, group: str = "") -> None:
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.dialogue.open(group=group)
        super().start(src_lang=src_lang, target_lang=target_lang, group=group)

    async def listen(self, audio: bytes) -> None:
        transcriber = self.parts["transcription"]
        detector = self.part("vad")
        with logging.collect("transcribed") as transcribed:
            await transcriber.transcribe(audio)
            speech = None if detector is None else detector.detect(audio)
            if speech is not None:
                await transcriber.flush("vad", speech)
        await self._emit_finals(transcribed)

    async def finish(self) -> None:
        with logging.collect("transcribed") as transcribed:
            await self.parts["transcription"].finish()
        await self._emit_finals(transcribed)

    async def _emit_finals(self, transcribed: list[dict]) -> None:
        corrector, translator = self.part("correction"), self.part("translation")
        for event in transcribed:
            original = event.get("original") or ""
            language = event.get("language") or self.src_lang
            if corrector is not None:
                original = await corrector.correct(original, language)

            speaker = event.get("speaker") or ""
            translation, detected = ("", "")
            if translator is not None:
                translation, detected = await translator.translate(
                    original, self.target_lang, language,
                    context=self.dialogue.context(self.target_lang),
                    speaker=self.dialogue.label(speaker))
            self.dialogue.said(original, translation, lang=language or "",
                               target=self.target_lang, speaker=speaker)

            payload = {k: v for k, v in event.items()
                       if k not in ("t", "type", "session", "item", "audio")}
            payload["original"] = original
            payload["language"] = event.get("language") or detected or ""
            payload["translation"] = translation
            if translator is not None:
                payload["target_lang"] = self.target_lang
            logging.emit("final", **payload)
