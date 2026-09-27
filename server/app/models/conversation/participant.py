from collections import deque
from dataclasses import dataclass, field

from core.components.registry import Partial, Transcribed, Translated
from core.utils import langs

from app.models.conversation.events import ConversationEvent, FinalOutput, PartialTranscript
from app.models.conversation.feed import Inbox


@dataclass
class Participant:
    id: str
    inbox: Inbox
    expects_translations: bool = True
    lang: str = ""
    target_lang: str = ""
    lang_map: dict[str, str] = field(default_factory=dict)
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
        return {code for code in (self.lang, self.target_lang, *self.lang_map.values()) if code}

    def target_for(self, language: str) -> str:
        return langs.pick_target(
            language, lang=self.lang, target_lang=self.target_lang, lang_map=self.lang_map
        )


class Delivery:
    def __init__(self, participant: Participant):
        self._participant = participant
        self._order: deque[int] = deque()
        self._awaited_translations: dict[int, str] = {}
        self._ready: dict[int, ConversationEvent] = {}

    def reset(self) -> None:
        self._order.clear()
        self._awaited_translations.clear()
        self._ready.clear()

    def accept(self, item: object) -> list:
        if not isinstance(item, ConversationEvent):
            return [item]
        record, number = item.record, item.segment.number
        if isinstance(record, Partial):
            return [PartialTranscript.of(item)]
        if isinstance(record, Transcribed):
            self._order.append(number)
            target = self._participant.target_for(record.language)
            if self._participant.expects_translations and target:
                self._awaited_translations[number] = target
            else:
                self._ready[number] = item
            return self._flush()
        if isinstance(record, Translated) and self._awaited_translations.get(number) == (
            record.target_lang
        ):
            del self._awaited_translations[number]
            self._ready[number] = item
            return self._flush()
        return []

    def _flush(self) -> list:
        out = []
        while self._order and self._order[0] in self._ready:
            event = self._ready.pop(self._order.popleft())
            if event.record.original.strip():
                out.append(FinalOutput.of(event))
        return out
