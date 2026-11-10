from collections.abc import Awaitable
from typing import AsyncIterator

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, WebSocket

from core.pipeline import describe
from core.utils import logging

from app.api.v1.schema.conversation import (
    ClientMessage,
    ConfigMessage,
    ConversationConfigMessage,
    ConversationId,
    FinalMessage,
    FinishMessage,
    FoldMessage,
    HelloMessage,
    Message,
    PartialMessage,
    PingMessage,
    ReadyMessage,
    RemoveParticipantMessage,
    RoomMessage,
    SetAllMicsMessage,
    SetMicMessage,
    StartMessage,
    StopMessage,
    client_message,
)
from app.api.websocket import WebSocketConnectionHandler
from app.containers import Container
from app.models.conversation import (
    ConversationSettingsChanged,
    FinalOutput,
    FoldOutput,
    JoinedConversation,
    PartialTranscript,
    RoomChanged,
)
from app.services.conversation import (
    ConversationError,
    ConversationService,
    ConversationSession,
)
from app.services.recording import RecordingService, SessionRecording
from app.config import ServerConfig

log = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])


@router.websocket("/ws")
@inject
async def conversation(
    websocket: WebSocket,
    service: ConversationService = Depends(Provide[Container.conversation_service]),
    config: ServerConfig = Depends(Provide[Container.server_config]),
    recorder: RecordingService = Depends(Provide[Container.recording_service]),
):
    await ConversationWebSocketConnectionHandler(
        websocket,
        service,
        config,
        recorder,
    ).run()


@router.get("/rooms/{room_code}", status_code=204)
@inject
async def room(
    room_code: ConversationId,
    service: ConversationService = Depends(Provide[Container.conversation_service]),
) -> None:
    service.ensure_open(room_code)


class ConversationWebSocketConnectionHandler(WebSocketConnectionHandler[ClientMessage]):
    def __init__(
        self,
        websocket: WebSocket,
        service: ConversationService,
        config: ServerConfig,
        recorder: RecordingService,
    ):
        super().__init__(websocket, client_message)
        self.service = service
        self.config = config
        self.recorder = recorder
        self.session: ConversationSession
        self.recording: SessionRecording | None = None

    async def run(self) -> None:
        try:
            await super().run()
        finally:
            if self.recording:
                self.recording.close()

    async def handle_connect(self) -> None:
        self.session = self.service.open_session()
        client = self.websocket.client
        log.info(
            "Client %s connected as participant %s",
            f"{client.host}:{client.port}" if client else "unknown",
            self.session.participant.id,
        )

    async def produce_message(self) -> AsyncIterator[Message]:
        async for message in self._messages():
            if self.recording:
                self.recording.message("out", message.encode())
            yield message

    async def _messages(self) -> AsyncIterator[Message]:
        yield HelloMessage(server_config=describe(self.config))

        async for event in self.session.events():
            match event:
                case JoinedConversation():
                    yield ReadyMessage(
                        conversation_id=event.conversation_id,
                        participant_id=self.session.participant.id,
                        recording=self.recording is not None,
                    )
                case RoomChanged():
                    yield RoomMessage.of(event)
                case ConversationSettingsChanged():
                    yield ConversationConfigMessage.of(
                        self.session.participant,
                        recording=self.recording is not None,
                    )
                case PartialTranscript():
                    yield PartialMessage.of(event)
                case FinalOutput():
                    yield FinalMessage.of(event)
                case FoldOutput():
                    yield FoldMessage.of(event)

    async def handle_message(self, message: ClientMessage | bytes) -> None:
        if isinstance(message, StartMessage):
            self.session.set_device(message.device)
        if isinstance(message, StartMessage | ConfigMessage) and message.recording is not None:
            self._set_recording(message.recording)
        self._record(message)
        match message:
            case PingMessage():
                pass
            case StartMessage():
                await self.session.start(
                    message.conversation_id, must_exist=message.join, **message.languages()
                )
            case ConfigMessage():
                await self._refusable(
                    self.session.configure(**message.languages()), "config change"
                )
            case bytes():
                await self.session.audio(message)
            case SetMicMessage():
                await self._refusable(
                    self.session.set_can_speak(message.participant_id, message.can_speak),
                    "host action",
                )
            case SetAllMicsMessage():
                await self._refusable(
                    self.session.set_everyone_can_speak(message.can_speak), "host action"
                )
            case RemoveParticipantMessage():
                await self._refusable(
                    self.session.remove_participant(message.participant_id), "host action"
                )
            case FinishMessage():
                await self.session.finish()
            case StopMessage():
                await self.session.stop()
                self.stop()

    async def handle_disconnect(self) -> None:
        await self.session.leave()

    def _set_recording(self, on: bool) -> None:
        if on and self.recording is None:
            participant = self.session.participant
            self.recording = self.recorder.open(participant.id, participant.device)
        elif not on and self.recording is not None:
            self.recording.close()
            self.recording = None
        self.session.set_recording(self.recording is not None)

    def _record(self, message: ClientMessage | bytes) -> None:
        if self.recording is None or isinstance(message, PingMessage):
            return
        if isinstance(message, bytes):
            self.recording.audio(message)
        else:
            self.recording.message("in", message.encode())

    async def _refusable(self, action: Awaitable[None], what: str) -> None:
        try:
            await action
        except ConversationError as e:
            log.warning("Refused %s (%s): %s", what, e.code, e)
