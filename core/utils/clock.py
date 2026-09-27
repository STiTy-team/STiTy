import time
from datetime import datetime, timezone

STAMP_FORMAT = "%Y%m%dT%H%M%S"


def now() -> datetime:
    return datetime.now(timezone.utc)


def stamp(at: datetime | None = None) -> str:
    return (at or now()).strftime(STAMP_FORMAT)


def monotonic() -> float:
    return time.perf_counter()


def elapsed_since(start: float) -> float:
    return time.perf_counter() - start
