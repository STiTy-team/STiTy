"""Answers read from S3, kept for one refresh period of the web app."""

import time
from functools import wraps

TTL_SEC = 30

_entries: dict[tuple, tuple[float, object]] = {}


def cached(fn):
    @wraps(fn)
    def wrapper(*args):
        key = (fn.__qualname__, *args)
        hit = _entries.get(key)
        if hit is not None and time.monotonic() - hit[0] < TTL_SEC:
            return hit[1]
        value = fn(*args)
        _entries[key] = (time.monotonic(), value)
        return value

    return wrapper


def clear() -> None:
    _entries.clear()
