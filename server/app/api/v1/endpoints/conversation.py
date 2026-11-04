from typing import AsyncIterator

from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, WebSocket

from core.pipeline import describe
from core.utils import logging

from app.api.v1.schema.conversation import (
    ClientMessage,
    ConfigMessage,
    ConversationConfigMessage,
    FinalMessage,
    FinishMessage,
    HelloMessage,
    Message,
    PartialMessage,
    PingMessage,
    ReadyMessage,
    StartMessage,
    StopMessage,
    client_message,
)
from app.api.websocket import WebSocketConnectionHandler
from app.containers import Container
from app.models.conversation import (
    ConversationSettingsChanged,
    FinalOutput,
    JoinedConversation,
    PartialTranscript,
)
from app.services.conversation import (
    ConversationError,
    ConversationService,
    ConversationSession,
)
from app.config import ServerConfig

log = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])


@router.websocket("/ws")
@inject
async def conversation(
    websocket: WebSocket,
    service: ConversationService = Depends(Provide[Container.conversation_service]),
    config: ServerConfig = Depends(Provide[Container.server_config]),
):
    await ConversationWebSocketConnectionHandler(
        websocket,
        service,
        config,
    ).run()


class ConversationWebSocketConnectionHandler(WebSocketConnectionHandler[ClientMessage]):
    def __init__(
        self,
        websocket: WebSocket,
        service: ConversationService,
        config: ServerConfig,
    ):
        super().__init__(websocket, client_message)
        self.service = service
        self.config = config
        self.session: ConversationSession

    async def handle_connect(self) -> None:
        self.session = self.service.open_session()
        client = self.websocket.client
        log.info(
            "Client %s connected as participant %s",
            f"{client.host}:{client.port}" if client else "unknown",
            self.session.participant.id,
        )

    async def produce_message(self) -> AsyncIterator[Message]:
        yield HelloMessage(server_config=describe(self.config))

        async for event in self.session.events():
            match event:
                case JoinedConversation():
                    yield ReadyMessage(conversation_id=event.conversation_id)
                case ConversationSettingsChanged():
                    yield ConversationConfigMessage.of(self.session.participant)
                case PartialTranscript():
                    yield PartialMessage.of(event)
                case FinalOutput():
                    yield FinalMessage.of(event)

    async def handle_message(self, message: ClientMessage | bytes) -> None:
        match message:
            case PingMessage():
                pass
            case StartMessage():
                await self.session.start(message.conversation_id, **message.languages())
            case ConfigMessage():
                await self._configure(message)
            case bytes():
                await self.session.audio(message)
            case FinishMessage():
                await self.session.finish()
            case StopMessage():
                await self.session.stop()
                self.stop()

    async def handle_disconnect(self) -> None:
        await self.session.leave()

    async def _configure(self, message: ConfigMessage) -> None:
        try:
            await self.session.configure(**message.languages())
        except ConversationError as e:
            log.warning("Refused config change (%s): %s", e.code, e)
