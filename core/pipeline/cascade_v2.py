from core.components.correction import correctors
from core.components.enhancement import enhancers
from core.components.enhancement.base import Routed
from core.components.filter import filters
from core.components.labeler import labelers
from core.components.mixer import mixers
from core.components.transcription import transcribers
from core.components.translation import translators
from core.components.vad import detectors
from core.utils import audio as audio_mod
from core.utils.audio import MultiChannelAudio

from . import pipelines, stages
from .base import Pipeline, Processor, RunsPer, Stage
from .cascade import TranslationProcessor

ROOM_PARTS = ("mixer", "enhancement", "vad", "transcription", "labeler", "filter")
TRANSLATION_PARTS = ("correction", "translation")


def pipeline_settings(parts: dict) -> dict:
    for part in parts.values():
        cfg = getattr(part, "cfg", None)
        if cfg is not None:
            return cfg.stity.resolved.get("pipeline", {})
    return {}


class Room:
    def __init__(self, parts: dict, *, split_at_vad: bool):
        self.parts = parts
        self.split_at_vad = split_at_vad
        self.heard = 0

    async def hear(self, pcm: bytes) -> list:
        enhancer = self.parts.get("enhancement")
        routed = enhancer.enhance(pcm) if enhancer is not None else Routed(vad=pcm, asr=pcm)
        detector = self.parts.get("vad")
        speech = None if detector is None else detector.detect(routed.vad)
        transcriber = self.parts["transcription"]
        produced = [] if speech is None else [speech]
        samples = len(routed.asr) // 2
        if speech is not None and self.split_at_vad:
            cut = min(max(int(round(speech.ended_at * audio_mod.SAMPLING_RATE)) - self.heard, 0),
                      samples)
            produced.extend(await transcriber.transcribe(routed.asr[:cut * 2]))
            produced.extend(await transcriber.flush("vad", speech))
            if cut < samples:
                produced.extend(await transcriber.transcribe(routed.asr[cut * 2:]))
        else:
            produced.extend(await transcriber.transcribe(routed.asr))
            if speech is not None:
                produced.extend(await transcriber.flush("vad", speech))
        self.heard += samples
        return self.post(produced)

    async def finish(self) -> list:
        return self.post(await self.parts["transcription"].finish())

    def post(self, produced: list) -> list:
        labeler, keeper = self.parts.get("labeler"), self.parts.get("filter")
        if labeler is not None:
            produced = labeler.label(produced)
        if keeper is not None:
            produced = keeper.filter(produced)
        return produced


class SpeechProcessorV2(Processor):
    def start(self, *, languages: list[str], target_lang: str = "") -> None:
        super().start(languages=languages, target_lang=target_lang)
        self.room = Room(self.parts,
                         split_at_vad=pipeline_settings(self.parts).get("split_at_vad", False))

    async def process(self, audio: MultiChannelAudio) -> list:
        return await self.room.hear(self.parts["mixer"].mix(audio))

    async def finish(self) -> list:
        return await self.room.finish()


@pipelines.register("cascade:v2")
class CascadeV2Pipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (mixers, enhancers, detectors, labelers, filters, correctors, translators)
    STAGES = (
        Stage(parts=ROOM_PARTS, runs_per=RunsPer.ROOM, processor=SpeechProcessorV2),
        Stage(parts=TRANSLATION_PARTS, runs_per=RunsPer.TARGET_LANG,
              processor=TranslationProcessor),
    )

    @classmethod
    def validate(cls, leftover_options: dict) -> dict:
        options = dict(leftover_options)
        split = options.pop("split_at_vad", None)
        super().validate(options)
        return {} if split is None else {"split_at_vad": bool(split)}

    def start(self, *, languages: list[str], target_lang: str,
              own_lang: str | None = None) -> None:
        stages.start(self.parts, languages=languages, target_lang=target_lang)
        self.room = Room(self.parts, split_at_vad=self.settings.get("split_at_vad", False))
        self.translation = TranslationProcessor(
            {kind: self.parts[kind] for kind in TRANSLATION_PARTS if kind in self.parts})
        self.translation.target_lang = target_lang
        self.translation.languages = set()
        self.translation.said_so_far = []

    async def listen(self, audio: bytes) -> list:
        return await self._translated(await self.room.hear(audio))

    async def finish(self) -> list:
        return await self._translated(await self.room.finish())

    async def _translated(self, produced: list) -> list:
        out = []
        for item in produced:
            out.append(item)
            if self.translation.accepts(item):
                out.extend(await self.translation.process(item))
        return out
