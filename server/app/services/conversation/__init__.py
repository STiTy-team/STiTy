from collections.abc import AsyncIterator

from app.config import ServerConfig
from app.services.conversation.conversation import ConversationService
from app.services.conversation.errors import (
    AudioFailed,
    ConversationClosed,
    ConversationError,
    ConversationNotStarted,
    ParticipantTooSlow,
    ServerShutdown,
)
from app.services.conversation.session import ConversationSession

__all__ = [
    "AudioFailed",
    "ConversationClosed",
    "ConversationError",
    "ConversationNotStarted",
    "ConversationService",
    "ConversationSession",
    "ParticipantTooSlow",
    "ServerShutdown",
    "init_conversation_service",
]


async def init_conversation_service(config: ServerConfig) -> AsyncIterator[ConversationService]:
    service = ConversationService(config)
    await service.load_models()
    yield service
    await service.close_all()
