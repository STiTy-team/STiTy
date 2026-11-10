from collections.abc import AsyncIterator

from app.config import ServerConfig
from app.services.conversation.conversation import ConversationService
from app.services.conversation.errors import (
    AudioFailed,
    ConversationClosed,
    ConversationError,
    ConversationNotFound,
    ConversationNotStarted,
    NotRoomHost,
    ParticipantTooSlow,
    RemovedFromRoom,
    RoomEnded,
    ServerShutdown,
)
from app.services.conversation.session import ConversationSession
from app.services.recording import RecordingService

__all__ = [
    "AudioFailed",
    "ConversationClosed",
    "ConversationError",
    "ConversationNotFound",
    "ConversationNotStarted",
    "ConversationService",
    "ConversationSession",
    "NotRoomHost",
    "ParticipantTooSlow",
    "RemovedFromRoom",
    "RoomEnded",
    "ServerShutdown",
    "init_conversation_service",
]


async def init_conversation_service(
    config: ServerConfig, recorder: RecordingService
) -> AsyncIterator[ConversationService]:
    service = ConversationService(config, recorder)
    await service.load_models()
    yield service
    await service.close_all()
