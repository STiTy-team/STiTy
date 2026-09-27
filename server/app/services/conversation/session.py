from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from app.models.conversation import Closed, Conversation, Overflow, Participant
from app.services.conversation.errors import ConversationNotStarted, ParticipantTooSlow
from app.services.conversation.workers import close_workers

if TYPE_CHECKING:
    from app.services.conversation.conversation import ConversationService


class ConversationSession:
    def __init__(self, service: "ConversationService", participant: Participant):
        self._service = service
        self.participant = participant
        self.conversation: Conversation | None = None

    async def start(
        self,
        conversation_id: str | None,
        *,
        lang: str,
        target_lang: str,
        lang_map: object | None,
    ) -> None:
        if self.conversation is not None:
            await self.configure(lang=lang, target_lang=target_lang, lang_map=lang_map)
            return
        self.participant.set_languages(lang, target_lang, lang_map)
        self.conversation = await self._service.join(self.participant, conversation_id)

    async def configure(
        self,
        *,
        lang: str | None,
        target_lang: str | None,
        lang_map: object | None,
    ) -> None:
        conversation = self._started()
        async with conversation.lock:
            conversation.change_languages(self.participant, lang, target_lang, lang_map)
            unused = await self._service.sync_translation_workers(conversation)
        await close_workers(unused)

    async def audio(self, pcm: bytes) -> None:
        conversation = self._started()
        if self.participant.channel is None:
            await self._service.open_channel(conversation, self.participant)
        conversation.push_audio(self.participant, pcm)

    async def finish(self) -> None:
        conversation = self._started()
        async with conversation.lock:
            worker = conversation.close_channel(self.participant)
        await close_workers([worker])
        await conversation.settle_translations()

    async def stop(self) -> None:
        if self.conversation is not None:
            await self.finish()
        await self.leave()

    async def leave(self) -> None:
        if self.conversation is None:
            self.participant.inbox.end()
            return
        await self._service.leave(self.conversation, self.participant)

    async def events(self) -> AsyncIterator[object]:
        try:
            while (item := await self.participant.inbox.get()) is not None:
                if isinstance(item, Closed):
                    raise item.error
                for event in self.participant.delivery.accept(item):
                    yield event
        except Overflow as e:
            raise ParticipantTooSlow(f"participant {self.participant.id} is too slow") from e

    def _started(self) -> Conversation:
        if self.conversation is None:
            raise ConversationNotStarted("send start first")
        return self.conversation
