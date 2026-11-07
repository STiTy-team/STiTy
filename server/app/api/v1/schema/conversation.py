from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)
from pydantic.alias_generators import to_camel

from core.utils import clock

from app.models.conversation import (
    FinalOutput,
    FoldOutput,
    Participant,
    PartialTranscript,
    RoomChanged,
)


ConversationId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]+$", max_length=64)]


DEVICE_FIELDS = ("platform", "version", "model", "headset")
DEVICE_VALUE_LENGTH = 64


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
    join: bool = False
    device: dict | None = None
    lang: str = "auto"
    target_lang: str = ""
    lang_map: dict | None = None
    recording: bool = False

    @field_validator("device", mode="before")
    @classmethod
    def _keep_known_device_fields(cls, device: object) -> dict | None:
        if not isinstance(device, dict):
            return None
        return {
            key: value if isinstance(value, bool) else str(value)[:DEVICE_VALUE_LENGTH]
            for key, value in device.items()
            if key in DEVICE_FIELDS and value is not None
        }


class ConfigMessage(LanguageMessage):
    type: Literal["config"]
    lang: str | None = None
    target_lang: str | None = None
    lang_map: dict | None = None
    recording: bool | None = None


class FinishMessage(Message):
    type: Literal["finish"]


class StopMessage(Message):
    type: Literal["stop"]


class PingMessage(Message):
    type: Literal["ping"]


class SetMicMessage(Message):
    type: Literal["set_mic"]
    participant_id: str
    can_speak: bool


class SetAllMicsMessage(Message):
    type: Literal["set_all_mics"]
    can_speak: bool


class RemoveParticipantMessage(Message):
    type: Literal["remove_participant"]
    participant_id: str


ClientMessage = Annotated[
    StartMessage
    | ConfigMessage
    | FinishMessage
    | StopMessage
    | PingMessage
    | SetMicMessage
    | SetAllMicsMessage
    | RemoveParticipantMessage,
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
    participant_id: str
    recording: bool


class RoomParticipantMessage(Message):
    id: str
    number: int
    can_speak: bool


class RoomMessage(Message):
    type: Literal["room"] = "room"
    host_id: str | None
    participants: list[RoomParticipantMessage]

    @classmethod
    def of(cls, event: RoomChanged) -> Self:
        return cls(
            host_id=event.host_id,
            participants=[
                RoomParticipantMessage(id=p.id, number=p.number, can_speak=p.can_speak)
                for p in event.participants
            ],
        )


class ConversationConfigMessage(Message):
    type: Literal["config_ok"] = "config_ok"
    lang: str
    target_lang: str
    lang_map: dict[str, list[str]]
    recording: bool

    @classmethod
    def of(cls, participant: Participant, recording: bool) -> Self:
        return cls(
            lang=participant.lang or "auto",
            target_lang=participant.target_lang,
            lang_map={source: list(targets) for source, targets in participant.lang_map.items()},
            recording=recording,
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


class SourceMessage(Message):
    participant_id: str
    share: float


class FinalMessage(Message):
    type: Literal["final"] = "final"
    segment_id: int
    start: str
    end: str
    original: str
    translation: str
    language: str
    target_lang: str
    commit_reason: str
    sources: list[SourceMessage] = []

    @classmethod
    def of(cls, transcript: FinalOutput) -> Self:
        return cls(
            segment_id=transcript.segment,
            start=clock.format_duration(transcript.started_at),
            end=clock.format_duration(transcript.ended_at),
            original=transcript.original,
            translation=transcript.translation,
            language=transcript.language,
            target_lang=transcript.target_lang,
            commit_reason=transcript.commit_reason,
            sources=[
                SourceMessage(participant_id=s.participant_id, share=s.share)
                for s in transcript.sources
            ],
        )


class FoldMessage(Message):
    type: Literal["fold"] = "fold"
    segment_id: int
    target_lang: str
    original: str
    replacement: str

    @classmethod
    def of(cls, fold: "FoldOutput") -> Self:
        return cls(
            segment_id=fold.segment,
            target_lang=fold.target_lang,
            original=fold.original,
            replacement=fold.replacement,
        )
