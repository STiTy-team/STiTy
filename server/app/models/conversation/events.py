from dataclasses import dataclass
from typing import Self

from core.components.registry import Folded, Partial, Transcribed, Translated


@dataclass(frozen=True)
class SegmentId:
    conversation: str
    number: int

    def __str__(self) -> str:
        return f"{self.conversation}:{self.number}"


@dataclass(frozen=True)
class ConversationEvent:
    segment: SegmentId
    record: Partial | Transcribed | Translated | Folded
    started_at: float
    ended_at: float


class SegmentTimeline:
    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id
        self._next_number = 1
        self._started_at: float | None = None
        self._last_ended_at = 0.0
        self._translated_segments: dict[tuple[str, float, str], int] = {}
        self._translated_ends: dict[str, float] = {}

    def stamp(self, record: object, heard_from: float, heard_to: float) -> ConversationEvent | None:
        if isinstance(record, Translated) and not record.original:
            return None
        if isinstance(record, (Translated, Folded)):
            return self._stamp_stream(record, heard_to)
        if isinstance(record, Partial):
            if self._started_at is None and record.text:
                self._started_at = heard_from
            number = self._next_number
        elif isinstance(record, Transcribed):
            number = self._next_number
            self._next_number += 1
        else:
            return None
        started_at = self._last_ended_at if self._started_at is None else self._started_at
        if isinstance(record, Transcribed):
            self._started_at, self._last_ended_at = None, heard_to
        return ConversationEvent(
            SegmentId(self._conversation_id, number), record, started_at, heard_to
        )

    def _stamp_stream(
        self, record: Translated | Folded, heard_to: float
    ) -> ConversationEvent | None:
        key = (record.target_lang, record.decision_audio_sec, record.original)
        if isinstance(record, Folded):
            number = self._translated_segments.pop(key, None)
            if number is None:
                return None
            if record.translation:
                self._translated_segments[
                    (record.target_lang, record.decision_audio_sec, record.translation)
                ] = number
            started_at = record.decision_audio_sec
        else:
            number = self._next_number
            self._next_number += 1
            horizon = heard_to - 60.0
            self._translated_segments = {
                k: v for k, v in self._translated_segments.items() if k[1] >= horizon
            }
            self._translated_segments[key] = number
            started_at = self._translated_ends.get(record.target_lang, 0.0)
            self._translated_ends[record.target_lang] = heard_to
        return ConversationEvent(
            SegmentId(self._conversation_id, number), record, started_at, heard_to
        )


@dataclass(frozen=True)
class PartialTranscript:
    text: str
    language: str
    seq: int

    @classmethod
    def of(cls, event: ConversationEvent) -> Self:
        record = event.record
        return cls(text=record.text, language=record.language, seq=record.seq)


@dataclass(frozen=True)
class FinalOutput:
    segment: int
    started_at: float
    ended_at: float
    original: str
    translation: str
    language: str
    target_lang: str
    commit_reason: str

    @classmethod
    def of(cls, event: ConversationEvent) -> Self:
        record = event.record
        return cls(
            segment=event.segment.number,
            started_at=event.started_at,
            ended_at=event.ended_at,
            original=record.original,
            translation=getattr(record, "translation", ""),
            language=record.language,
            target_lang=getattr(record, "target_lang", ""),
            commit_reason=record.commit_reason,
        )


@dataclass(frozen=True)
class FoldOutput:
    segment: int
    target_lang: str
    original: str
    replacement: str

    @classmethod
    def of(cls, event: ConversationEvent) -> Self:
        record = event.record
        return cls(
            segment=event.segment.number,
            target_lang=record.target_lang,
            original=record.original,
            replacement=record.translation,
        )


@dataclass(frozen=True)
class JoinedConversation:
    conversation_id: str


@dataclass(frozen=True)
class ConversationSettingsChanged:
    pass


@dataclass(frozen=True)
class Closed:
    error: Exception


def is_partial(item: object) -> bool:
    return isinstance(item, ConversationEvent) and isinstance(item.record, Partial)
