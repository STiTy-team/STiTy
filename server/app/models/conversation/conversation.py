import asyncio
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

from app.models.audio import ConversationAudio
from app.models.conversation.events import (
    Closed,
    ConversationEvent,
    ConversationSettingsChanged,
    JoinedConversation,
    SegmentTimeline,
)
from app.models.conversation.feed import Feed, Inbox
from app.models.conversation.participant import Participant


class ConversationState(StrEnum):
    OPEN = "open"
    CLOSING = "closing"


class Worker(Protocol):
    async def close(self) -> None: ...


class SettlingWorker(Worker, Protocol):
    async def settle(self) -> None: ...


@dataclass
class Conversation:
    id: str
    state: ConversationState = ConversationState.OPEN
    participants: dict[str, Participant] = field(default_factory=dict)
    audio: ConversationAudio | None = None
    transcription_worker: Worker | None = None
    translation_workers: dict[str, SettlingWorker] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _timeline: SegmentTimeline = field(init=False, repr=False)
    _feed: Feed = field(default_factory=Feed, init=False, repr=False)
    _translation_feeds: dict[str, Feed] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._timeline = SegmentTimeline(self.id)

    @property
    def is_open(self) -> bool:
        return self.state == ConversationState.OPEN

    @property
    def speaks(self) -> list[str]:
        spoken = [code for p in self.participants.values() for code in p.speaks]
        return list(dict.fromkeys(spoken))

    @property
    def reads(self) -> set[str]:
        return set().union(*(p.reads for p in self.participants.values()))

    @property
    def channels(self) -> list[str]:
        return [p.channel for p in self.participants.values() if p.channel is not None]

    @property
    def missing_translation_langs(self) -> list[str]:
        return sorted(self.reads - set(self.translation_workers))

    def wants_translation(self, spoken: str, into: str) -> bool:
        return any(p.target_for(spoken) == into for p in self.participants.values())

    def add(self, participant: Participant) -> None:
        self.participants[participant.id] = participant
        self.subscribe(participant.inbox, participant.reads)
        participant.inbox.put(JoinedConversation(self.id))

    def remove(self, participant: Participant) -> list[Worker | None]:
        del self.participants[participant.id]
        worker = self.close_channel(participant)
        self.unsubscribe(participant.inbox)
        participant.inbox.end()
        return [worker, *self.detach_unused_translation_workers()]

    def change_languages(
        self,
        participant: Participant,
        lang: str | None,
        target_lang: str | None,
        lang_map: object | None,
    ) -> None:
        participant.set_languages(lang, target_lang, lang_map)
        self.unsubscribe(participant.inbox)
        participant.delivery.reset()
        self.subscribe(participant.inbox, participant.reads)
        participant.inbox.put(ConversationSettingsChanged())

    def open_channel(self, participant: Participant, channel: str) -> None:
        participant.channel = channel
        self.audio.open_channel(channel)

    def push_audio(self, participant: Participant, pcm: bytes) -> None:
        if participant.channel is not None and self.audio is not None:
            self.audio.push(participant.channel, pcm)

    def close_channel(self, participant: Participant) -> Worker | None:
        if participant.channel is None:
            return None
        if self.audio is not None:
            self.audio.close_channel(participant.channel)
        participant.channel = None
        if self.channels:
            return None
        worker, self.transcription_worker = self.transcription_worker, None
        return worker

    def stamp(self, record: object, heard_from: float, heard_to: float) -> ConversationEvent | None:
        return self._timeline.stamp(record, heard_from, heard_to)

    def publish(self, event: ConversationEvent, lang: str | None = None) -> None:
        feed = self._feed if lang is None else self._translation_feeds.get(lang)
        if feed is not None:
            feed.publish(event)

    def subscribe(self, inbox: Inbox, langs: Iterable[str] = ()) -> Inbox:
        self._feed.subscribe(inbox)
        for lang in langs:
            self._translation_feeds.setdefault(lang, Feed()).subscribe(inbox)
        return inbox

    def unsubscribe(self, inbox: Inbox) -> None:
        self._feed.unsubscribe(inbox)
        for feed in self._translation_feeds.values():
            feed.unsubscribe(inbox)

    def detach_unused_translation_workers(self) -> list[Worker]:
        unused = sorted(set(self.translation_workers) - self.reads)
        for lang in unused:
            self._translation_feeds.pop(lang, None)
        return [self.translation_workers.pop(lang) for lang in unused]

    async def settle_translations(self) -> None:
        for worker in list(self.translation_workers.values()):
            await worker.settle()

    def start_closing(self) -> list[Worker | None]:
        self.state = ConversationState.CLOSING
        for participant in self.participants.values():
            participant.channel = None
        workers = [self.transcription_worker, *self.translation_workers.values()]
        self.transcription_worker = None
        self.translation_workers.clear()
        return workers

    def dismiss_participants(self, error: Exception | None) -> None:
        for participant in self.participants.values():
            if error is not None:
                participant.inbox.put(Closed(error))
            participant.inbox.end()
        self.participants.clear()
