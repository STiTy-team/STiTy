from core.components.mixer import mixers
from core.components.transcription import transcribers
from core.components.vad import detectors

from . import pipelines, stages
from .base import Pipeline, RunsPer, Stage
from .cascade import SpeechProcessor


@pipelines.register("brig")
class BrigPipeline(Pipeline):
    REQUIRED = (transcribers,)
    OPTIONAL = (mixers, detectors)
    STAGES = (
        Stage(parts=("mixer", "vad", "transcription"), runs_per=RunsPer.ROOM,
              processor=SpeechProcessor),
    )

    async def listen(self, audio: bytes) -> list:
        return await stages.hear(self.parts, audio)

    async def finish(self) -> list:
        return await stages.finish(self.parts)
