from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, TypeAdapter
from pydantic.alias_generators import to_camel

from core.utils import clock

from app.models.conversation import FinalOutput, Participant, PartialTranscript


ConversationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$", max_length=64)]


class Message(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")

    def encode(self) -> dict:
        return self.model_dump(by_alias=True)


class LanguageMessage(Message):
    def languages(self) -> dict:
        return self.model_dump(include={"lang", "target_lang", "lang_map"})


class StartMessage(LanguageMessage):
    type: Literal["start"]
    conversation_id: ConversationId | None = None
    lang: str = "auto"
    target_lang: str = ""
    lang_map: dict | None = None


class ConfigMessage(LanguageMessage):
    type: Literal["config"]
    lang: str | None = None
    target_lang: str | None = None
    lang_map: dict | None = None


class FinishMessage(Message):
    type: Literal["finish"]


class StopMessage(Message):
    type: Literal["stop"]


class PingMessage(Message):
    type: Literal["ping"]


ClientMessage = Annotated[
    StartMessage | ConfigMessage | FinishMessage | StopMessage | PingMessage,
    Field(discriminator="type"),
]

client_message = TypeAdapter(ClientMessage)


class HelloMessage(Message):
    type: Literal["hello"] = "hello"
    message: str = "STiTy Server"
    server_config: dict


class ReadyMessage(Message):
    type: Literal["ready"] = "ready"
    message: str = "Ready to receive audio"
    conversation_id: ConversationId


class ConversationConfigMessage(Message):
    type: Literal["config_ok"] = "config_ok"
    lang: str
    target_lang: str
    lang_map: dict[str, str]

    @classmethod
    def of(cls, participant: Participant) -> Self:
        return cls(
            lang=participant.lang or "auto",
            target_lang=participant.target_lang,
            lang_map=participant.lang_map,
        )


class PartialMessage(Message):
    type: Literal["partial"] = "partial"
    text: str
    language: str
    seq: int

    @classmethod
    def of(cls, transcript: PartialTranscript) -> Self:
        return cls(
            text=transcript.text,
            language=transcript.language,
            seq=transcript.seq,
        )


class FinalMessage(Message):
    type: Literal["final"] = "final"
    start: str
    end: str
    original: str
    translation: str
    language: str
    commit_reason: str

    @classmethod
    def of(cls, transcript: FinalOutput) -> Self:
        return cls(
            start=clock.format_duration(transcript.started_at),
            end=clock.format_duration(transcript.ended_at),
            original=transcript.original,
            translation=transcript.translation,
            language=transcript.language,
            commit_reason=transcript.commit_reason,
        )
