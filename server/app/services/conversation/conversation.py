import asyncio
import uuid
from collections.abc import Awaitable

from core.pipeline import RunsPer, build_processor, stages_of
from core.utils import clock, logging

from app.config import ServerConfig
from app.models.audio import ConversationAudio
from app.models.conversation import Conversation, Inbox, Participant, Worker, is_partial
from app.services.conversation.errors import (
    AudioFailed,
    ConversationClosed,
    ConversationError,
    ServerShutdown,
)
from app.services.conversation.session import ConversationSession
from app.services.conversation.workers import (
    TranscriptionWorker,
    TranslationWorker,
    close_workers,
)

log = logging.getLogger(__name__)


class ConversationService:
    def __init__(self, config: ServerConfig):
        self.config = config
        stages = stages_of(config)
        self._transcription_stage = next(s for s in stages if s.runs_per == RunsPer.ROOM)
        self._translation_stage = next(
            (s for s in stages if s.runs_per == RunsPer.TARGET_LANG), None
        )
        self.conversations: dict[str, Conversation] = {}
        self._tasks: set[asyncio.Task] = set()

    async def load_models(self) -> None:
        log.info("Loading models for pipeline [%s]", self.config.stity.pipeline.name)
        started = clock.monotonic()
        for stage in (self._transcription_stage, self._translation_stage):
            if stage is not None:
                await (await build_processor(self.config, stage)).close()
        log.info("Loaded models in %.1fs", clock.elapsed_since(started))

    def open_session(self) -> ConversationSession:
        participant = Participant(id=str(uuid.uuid4()), inbox=Inbox(droppable=is_partial))
        participant.delivery.has_target_stage = self._translation_stage is not None
        return ConversationSession(self, participant)

    async def join(self, participant: Participant, conversation_id: str | None) -> Conversation:
        conversation = self._get_conversation(conversation_id)
        try:
            async with conversation.lock:
                if not conversation.is_open:
                    raise ConversationClosed(f"{conversation.id} is closed")
                conversation.add(participant)
                unused = await self.sync_translation_workers(conversation)
        except Exception:
            await self.leave(conversation, participant)
            raise
        await close_workers(unused)

        log.info(
            "Participant %s joined conversation %s (lang: %s, target: %s, map: %s)",
            participant.id,
            conversation.id,
            participant.lang or "auto",
            participant.target_lang or "-",
            participant.lang_map or "-",
        )
        return conversation

    async def leave(self, conversation: Conversation, participant: Participant) -> None:
        async with conversation.lock:
            if participant.id not in conversation.participants:
                return
            workers = conversation.remove(participant)
        await close_workers(workers)
        log.info("Participant %s left conversation %s", participant.id, conversation.id)
        if not conversation.participants:
            await self.close(conversation)

    async def close(
        self, conversation: Conversation, error: ConversationError | None = None
    ) -> None:
        async with conversation.lock:
            if not conversation.is_open:
                return
            workers = conversation.start_closing()
        await close_workers(workers)
        conversation.dismiss_participants(error)
        if self.conversations.get(conversation.id) is conversation:
            del self.conversations[conversation.id]
        reason = error.code if error is not None else "empty"
        log.info("Closed conversation %s (reason: %s)", conversation.id, reason)

    async def close_all(self) -> None:
        log.info("Closing %d open conversation(s)", len(self.conversations))
        for conversation in list(self.conversations.values()):
            await self.close(conversation, ServerShutdown("server is shutting down"))
        log.info("Closed conversation service")

    async def open_channel(self, conversation: Conversation, participant: Participant) -> None:
        async with conversation.lock:
            if participant.channel is not None or not conversation.is_open:
                return
            if conversation.transcription_worker is None:
                conversation.audio = ConversationAudio()
                conversation.transcription_worker = TranscriptionWorker(
                    conversation,
                    await build_processor(self.config, self._transcription_stage),
                    on_fail=lambda: self._spawn(
                        self.close(conversation, AudioFailed("audio pipeline failed"))
                    ),
                )
            conversation.open_channel(participant, str(uuid.uuid4()))

    async def sync_translation_workers(self, conversation: Conversation) -> list[Worker]:
        if self._translation_stage is not None:
            for lang in conversation.missing_translation_langs:
                conversation.translation_workers[lang] = TranslationWorker(
                    conversation,
                    lang,
                    await build_processor(self.config, self._translation_stage),
                )
        return conversation.detach_unused_translation_workers()

    def _get_conversation(self, conversation_id: str | None) -> Conversation:
        existing = self.conversations.get(conversation_id) if conversation_id else None
        if existing is not None and existing.is_open:
            return existing
        conversation = Conversation(id=conversation_id or str(uuid.uuid4()))
        self.conversations[conversation.id] = conversation
        log.info("Opened conversation %s", conversation.id)
        return conversation

    def _spawn(self, work: Awaitable[None]) -> None:
        task = asyncio.ensure_future(work)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
