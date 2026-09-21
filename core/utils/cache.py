import asyncio

_entries: dict = {}


async def load(key, build):
    if key not in _entries:
        _entries[key] = asyncio.create_task(build())
    try:
        return await asyncio.shield(_entries[key])
    except asyncio.CancelledError:
        raise
    except Exception:
        evict(key)
        raise


def evict(key) -> None:
    _entries.pop(key, None)


def clear() -> None:
    _entries.clear()
