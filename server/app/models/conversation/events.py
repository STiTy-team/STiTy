from dataclasses import dataclass
from typing import Self

from core.components.registry import Partial, Transcribed, Translated


@dataclass(frozen=True)
class SegmentId:
    conversation: str
    number: int

    def __str__(self) -> str:
        return f"{self.conversation}:{self.number}"


@dataclass(frozen=True)
class ConversationEvent:
    segment: SegmentId
    record: Partial | Transcribed | Translated
    started_at: float
    ended_at: float


class SegmentTimeline:
    def __init__(self, conversation_id: str):
        self._conversation_id = conversation_id
        self._next_number = 1
        self._started_at: float | None = None
        self._last_ended_at = 0.0

    def stamp(self, record: object, heard_from: float, heard_to: float) -> ConversationEvent | None:
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
    started_at: float
    ended_at: float
    original: str
    translation: str
    language: str
    commit_reason: str

    @classmethod
    def of(cls, event: ConversationEvent) -> Self:
        record = event.record
        return cls(
            started_at=event.started_at,
            ended_at=event.ended_at,
            original=record.original,
            translation=getattr(record, "translation", ""),
            language=record.language,
            commit_reason=record.commit_reason,
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
