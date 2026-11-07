from app.utils.errors import STiTyApiError


class ConversationError(STiTyApiError):
    pass


class ConversationNotStarted(ConversationError):
    code = "conversation_not_started"
    status_code = 409


class ConversationNotFound(ConversationError):
    code = "conversation_not_found"
    status_code = 404


class ConversationClosed(ConversationError):
    code = "conversation_closed"
    status_code = 410


class NotRoomHost(ConversationError):
    code = "not_room_host"
    status_code = 403


class RemovedFromRoom(ConversationError):
    code = "removed_from_room"
    status_code = 403


class RoomEnded(ConversationError):
    code = "room_ended"
    status_code = 410


class AudioFailed(ConversationError):
    code = "audio_failed"
    status_code = 500


class ServerShutdown(ConversationError):
    code = "shutdown"
    status_code = 503


class ParticipantTooSlow(ConversationError):
    code = "too_slow"
    status_code = 422
