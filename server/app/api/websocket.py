import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Protocol

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError
from starlette.types import Message as Frame
from starlette.websockets import WebSocketState

from core.utils import logging
from core.utils.json import dumps

from app.api.schema import InvalidMessage
from app.utils.api import describe_validation_errors
from app.utils.errors import STiTyApiError, close_code_for

log = logging.getLogger(__name__)

NORMAL_CLOSE = 1000


class OutgoingMessage(Protocol):
    def encode(self) -> dict: ...


class WebSocketConnectionHandler[IncomingMessage](ABC):
    def __init__(self, websocket: WebSocket, parser: TypeAdapter[IncomingMessage]):
        self.websocket = websocket
        self._parser = parser
        self._stopped = False

    @abstractmethod
    async def handle_connect(self) -> None: ...

    @abstractmethod
    async def handle_message(self, message: IncomingMessage | bytes) -> None: ...

    @abstractmethod
    def produce_message(self) -> AsyncIterator[OutgoingMessage]: ...

    @abstractmethod
    async def handle_disconnect(self) -> None: ...

    async def run(self) -> None:
        await self.websocket.accept()
        try:
            await self.handle_connect()
        except Exception as e:
            await self._fail(e)
            return

        sender = asyncio.create_task(self._send_all())
        try:
            async for message in self._receive_all():
                await self.handle_message(message)
        except Exception as e:
            await self._fail(e)
        finally:
            await self.handle_disconnect()
            await sender
        await self._close(NORMAL_CLOSE)

    def stop(self) -> None:
        self._stopped = True

    async def _receive_all(self) -> AsyncIterator[IncomingMessage | bytes]:
        while not self._stopped and (frame := await self._receive_frame()) is not None:
            if frame.get("bytes") is not None:
                yield frame["bytes"]
            elif (message := self._parse(frame.get("text") or "")) is not None:
                yield message

    async def _receive_frame(self) -> Frame | None:
        try:
            frame = await self.websocket.receive()
        except (WebSocketDisconnect, RuntimeError):
            return None
        return None if frame["type"] == "websocket.disconnect" else frame

    def _parse(self, text: str) -> IncomingMessage | None:
        try:
            return self._parser.validate_json(text)
        except ValidationError as e:
            field_errors = [error for error in e.errors() if error["loc"]]
            if not field_errors:
                log.warning("Ignored malformed or unknown message: %s", text[:100])
                return None
            raise InvalidMessage(describe_validation_errors(field_errors)) from e

    async def _send_all(self) -> None:
        try:
            async for message in self.produce_message():
                await self.websocket.send_text(dumps(message.encode()))
        except (WebSocketDisconnect, RuntimeError):
            log.debug("Stopped sending: client disconnected")
        except Exception as e:
            await self._fail(e)

    async def _fail(self, error: Exception) -> None:
        if isinstance(error, STiTyApiError):
            code, reason = error.close_code, error.code
        else:
            code, reason = close_code_for(500), "internal_error"
        log.error("Connection failed (%d %s): %s", code, reason, error, exc_info=error)
        await self._close(code, reason)

    async def _close(self, code: int, reason: str = "") -> None:
        if self.websocket.client_state != WebSocketState.CONNECTED:
            return
        try:
            await self.websocket.close(code, reason)
        except RuntimeError:
            pass
