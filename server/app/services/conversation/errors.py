from app.utils.errors import STiTyApiError


class ConversationError(STiTyApiError):
    pass


class ConversationNotStarted(ConversationError):
    code = "conversation_not_started"
    status_code = 409


class ConversationClosed(ConversationError):
    code = "conversation_closed"
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
