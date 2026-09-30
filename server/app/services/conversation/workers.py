import asyncio
from collections.abc import Callable, Iterable
from dataclasses import replace

from core.pipeline import Processor
from core.utils import logging, stream

from app.models.conversation import Conversation, ConversationEvent, Inbox, Worker, is_partial

log = logging.getLogger(__name__)

SETTLE_POLL_SEC = 0.01


class TranscriptionWorker:
    def __init__(
        self, conversation: Conversation, processor: Processor, on_fail: Callable[[], None]
    ):
        self._conversation = conversation
        self._audio = conversation.audio
        self._processor = processor
        self._on_fail = on_fail
        self._task = asyncio.create_task(self._run(), name=f"transcription:{conversation.id}")

    async def close(self) -> None:
        self._audio.end()
        await self._task

    async def _run(self) -> None:
        conversation_id = self._conversation.id
        stream.bind(conversation=conversation_id, worker="transcription")
        stream.start_clock()
        languages = self._conversation.speaks
        self._processor.start(languages=languages)
        log.info(
            "Started transcription worker for conversation %s (languages: %s)",
            conversation_id,
            languages or "any",
        )
        try:
            async for chunk in self._audio.chunks():
                stream.audio_position(chunk.start)
                self._publish(await self._processor.process(chunk), chunk.start)
            self._publish(await self._processor.finish(), self._audio.seconds)
        except Exception:
            log.exception("Transcription worker failed for conversation %s", conversation_id)
            self._audio.end()
            self._on_fail()
        finally:
            await self._processor.close()
            log.info("Stopped transcription worker for conversation %s", conversation_id)

    def _publish(self, records: list, heard_from: float) -> None:
        for record in records:
            event = self._conversation.stamp(record, heard_from, self._audio.seconds)
            if event is not None:
                self._conversation.publish(event)


class TranslationWorker:
    def __init__(self, conversation: Conversation, lang: str, processor: Processor):
        self._conversation = conversation
        self._lang = lang
        self._processor = processor
        self._inbox = conversation.subscribe(Inbox(droppable=is_partial))
        self._idle = asyncio.Event()
        self._task = asyncio.create_task(self._run(), name=f"translation:{conversation.id}:{lang}")

    async def settle(self) -> None:
        while len(self._inbox) and not self._task.done():
            await asyncio.sleep(SETTLE_POLL_SEC)
        await self._idle.wait()

    async def close(self) -> None:
        self._conversation.unsubscribe(self._inbox)
        self._inbox.end()
        await self._task

    async def _run(self) -> None:
        conversation_id, lang = self._conversation.id, self._lang
        stream.bind(conversation=conversation_id, worker="translation", lang=lang)
        stream.start_clock()
        self._processor.start(languages=[], target_lang=lang)
        log.info("Started %s translation worker for conversation %s", lang, conversation_id)
        try:
            while (event := await self._next()) is not None:
                await self._translate(event)
        except Exception:
            log.exception("%s translation worker failed for conversation %s", lang, conversation_id)
            self._conversation.unsubscribe(self._inbox)
        finally:
            self._idle.set()
            await self._processor.close()
            log.info("Stopped %s translation worker for conversation %s", lang, conversation_id)

    async def _next(self) -> ConversationEvent | None:
        if not len(self._inbox):
            self._idle.set()
        event = await self._inbox.get()
        self._idle.clear()
        return event

    async def _translate(self, event: ConversationEvent) -> None:
        record = event.record
        if not self._processor.accepts(record):
            return
        if not self._conversation.wants_translation(record.language, into=self._lang):
            self._processor.skip(record)
            return
        for output in await self._processor.process(record):
            self._conversation.publish(replace(event, record=output), self._lang)


async def close_workers(workers: Iterable[Worker | None]) -> None:
    for worker in workers:
        if worker is not None:
            await worker.close()
