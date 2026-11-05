from dataclasses import replace

from core.utils import dedup, timing

from . import transcribers
from ..registry import Folded, Speech, Transcribed, Translated
from .base import Transcriber


@transcribers.register("brig")
class BrigTranscription(Transcriber):
    SETTINGS = {
        "asr": ("asr", dict),
        "adapter": ("adapter", dict),
        "keep_recent_sec": ("keep_recent_sec", float),
    }

    @classmethod
    def validate(cls, options: dict, *, kind: str = "transcription") -> dict:
        out = super().validate(options, kind=kind)
        for key in ("asr", "adapter"):
            sub = dict(out.get(key) or {})
            name = sub.pop("name", None)
            if not name:
                raise ValueError(f"transcription brig needs `{key}` with a `name`")
            out[key] = {
                "name": name,
                "kwargs": transcribers.get(name).validate(sub, kind=f"{kind}.{key}"),
            }
        return out

    def _child(self, key: str):
        spec = self.settings[key]
        return transcribers.get(spec["name"])(spec["kwargs"], cfg=self.cfg)

    async def load(self) -> None:
        self._asr = self._child("asr")
        self._adapter = self._child("adapter")
        await self._asr.load()
        await self._adapter.load()

    async def close(self) -> None:
        await self._asr.close()
        await self._adapter.close()

    def start(
        self, languages: list[str] | None = None, *, target_lang: str = "", vad=None, **_
    ) -> None:
        self._transcript: list[tuple[float, str]] = []
        self._recent: list[Translated] = []
        self._asr.start(languages, target_lang=target_lang, vad=vad)
        self._adapter.start(languages, target_lang=target_lang, vad=vad)

    async def transcribe(self, audio: bytes) -> list:
        heard = await self._asr.transcribe(audio)
        folds = self._absorb(heard)
        produced = self._route(await self._adapter.transcribe(audio))
        return heard + folds + produced

    async def flush(self, reason: str, speech: Speech | None = None) -> list:
        heard = await self._asr.flush(reason, speech)
        folds = self._absorb(heard)
        produced = self._route(await self._adapter.flush(reason, speech))
        return heard + folds + produced

    async def finish(self, reason: str = "finish", speech: Speech | None = None) -> list:
        heard = await self._asr.finish(reason, speech)
        folds = self._absorb(heard)
        produced = self._route(await self._adapter.finish(reason, speech))
        return heard + folds + produced

    def _absorb(self, heard: list) -> list:
        arrived = [item for item in heard if isinstance(item, Transcribed)]
        if not arrived:
            return []
        for item in arrived:
            self._transcript.append((item.decision_audio_sec, dedup.norm(item.original)))
        horizon = arrived[-1].decision_audio_sec - dedup.MATCH_WINDOW_BACK_SEC * 2
        self._transcript = [row for row in self._transcript if row[0] >= horizon]
        return self._refold()

    def _route(self, produced: list) -> list:
        keep_sec = float(self.settings.get("keep_recent_sec", dedup.MATCH_WINDOW_BACK_SEC))
        out = []
        for item in produced:
            if not isinstance(item, Translated):
                out.append(item)
                continue
            window = dedup.window_text(self._transcript, item.decision_audio_sec)
            text = dedup.region_fold(item.translation, window)
            if not dedup.norm(text):
                continue
            if text != item.translation:
                item = replace(item, original=text, translation=text)
            out.append(item)
            self._recent.append(item)
        if self._recent:
            horizon = self._recent[-1].decision_audio_sec - keep_sec
            self._recent = [r for r in self._recent if r.decision_audio_sec >= horizon]
        return out

    def _refold(self) -> list:
        folds, kept = [], []
        for item in self._recent:
            window = dedup.window_text(self._transcript, item.decision_audio_sec)
            text = dedup.region_fold(item.translation, window)
            if text == item.translation:
                kept.append(item)
                continue
            if not dedup.norm(text):
                text = ""
            folds.append(
                Folded(
                    original=item.translation,
                    translation=text,
                    language=item.language,
                    target_lang=item.target_lang,
                    commit_reason="dedup",
                    decision_audio_sec=item.decision_audio_sec,
                    committed_elapsed_sec=round(timing.elapsed() or 0.0, 4),
                )
            )
            if text:
                kept.append(replace(item, original=text, translation=text))
        self._recent = kept
        return folds
