from collections import deque
from dataclasses import dataclass, field

from core.components.registry import Folded, Partial, Transcribed, Translated
from core.utils import langs

from app.models.conversation.events import (
    ConversationEvent,
    FinalOutput,
    FoldOutput,
    PartialTranscript,
)
from app.models.conversation.feed import Inbox


@dataclass
class Participant:
    id: str
    inbox: Inbox
    expects_translations: bool = True
    lang: str = ""
    target_lang: str = ""
    lang_map: dict[str, tuple[str, ...]] = field(default_factory=dict)
    channel: str | None = None
    delivery: "Delivery" = field(init=False)

    def __post_init__(self) -> None:
        self.delivery = Delivery(self)

    def set_languages(
        self,
        lang: str | None = None,
        target_lang: str | None = None,
        lang_map: object | None = None,
    ) -> None:
        if lang:
            self.lang = langs.norm_code(lang)
        if target_lang:
            self.target_lang = langs.norm_code(target_lang)
        if lang_map is not None:
            self.lang_map = langs.parse_map(lang_map)

    @property
    def speaks(self) -> list[str]:
        if self.lang_map:
            return list(self.lang_map)
        if not self.lang:
            return []
        return [code for code in (self.lang, self.target_lang) if code]

    @property
    def reads(self) -> set[str]:
        mapped = (code for targets in self.lang_map.values() for code in targets)
        return {code for code in (self.lang, self.target_lang, *mapped) if code}

    def targets_for(self, language: str) -> tuple[str, ...]:
        return langs.pick_targets(
            language, lang=self.lang, target_lang=self.target_lang, lang_map=self.lang_map
        )


class Delivery:
    def __init__(self, participant: Participant):
        self._participant = participant
        self._queues: dict[str, deque[int]] = {}
        self._ready: dict[str, dict[int, ConversationEvent]] = {}

    def reset(self) -> None:
        self._queues.clear()
        self._ready.clear()

    def accept(self, item: object) -> list:
        if not isinstance(item, ConversationEvent):
            return [item]
        record, number = item.record, item.segment.number
        if isinstance(record, Partial):
            return [PartialTranscript.of(item)]
        if isinstance(record, Transcribed):
            targets = (
                self._participant.targets_for(record.language)
                if self._participant.expects_translations
                else ()
            )
            if targets:
                for target in targets:
                    self._enqueue(target, number)
            else:
                self._enqueue("", number)
                self._ready[""][number] = item
            return self._flush()
        if isinstance(record, Folded):
            if record.target_lang in self._participant.reads:
                return [FoldOutput.of(item)]
            return []
        if isinstance(record, Translated):
            if number in self._queues.get(record.target_lang, ()):
                self._ready[record.target_lang][number] = item
                return self._flush()
            if not self._participant.expects_translations \
                    and record.target_lang in self._participant.reads:
                self._enqueue(record.target_lang, number)
                self._ready[record.target_lang][number] = item
                return self._flush()
        return []

    def _enqueue(self, target: str, number: int) -> None:
        self._queues.setdefault(target, deque()).append(number)
        self._ready.setdefault(target, {})

    def _flush(self) -> list:
        out = []
        for target, queue in self._queues.items():
            ready = self._ready[target]
            while queue and queue[0] in ready:
                event = ready.pop(queue.popleft())
                if event.record.original.strip():
                    out.append(FinalOutput.of(event))
        return out
