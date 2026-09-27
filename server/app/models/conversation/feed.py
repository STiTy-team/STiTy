import asyncio
from collections import deque
from collections.abc import Callable

INBOX_SIZE = 256


class Overflow(Exception):
    pass


class Inbox[T]:
    def __init__(
        self, size: int | None = INBOX_SIZE, droppable: Callable[[T], bool] = lambda _: False
    ):
        self._size = size
        self._droppable = droppable
        self._items: deque[T] = deque()
        self._overflowed = False
        self._ended = False
        self._ready = asyncio.Event()

    def __len__(self) -> int:
        return len(self._items)

    def put(self, item: T) -> None:
        if self._ended or self._overflowed:
            return
        if self._size is not None and len(self._items) >= self._size:
            victim = next((old for old in self._items if self._droppable(old)), None)
            if victim is None:
                self._overflowed = True
                self._ready.set()
                return
            self._items.remove(victim)
        self._items.append(item)
        self._ready.set()

    def end(self) -> None:
        self._ended = True
        self._ready.set()

    async def get(self) -> T | None:
        while True:
            if self._overflowed:
                raise Overflow(f"inbox holds {self._size} items that cannot be dropped")
            if self._items:
                return self._items.popleft()
            if self._ended:
                return None
            self._ready.clear()
            await self._ready.wait()


class Feed[T]:
    def __init__(self) -> None:
        self._inboxes: set[Inbox[T]] = set()

    def publish(self, item: T) -> None:
        for inbox in list(self._inboxes):
            inbox.put(item)

    def subscribe(self, inbox: Inbox[T]) -> Inbox[T]:
        self._inboxes.add(inbox)
        return inbox

    def unsubscribe(self, inbox: Inbox[T]) -> None:
        self._inboxes.discard(inbox)
