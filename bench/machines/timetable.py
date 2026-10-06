from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from .machine import MachineSettings, WindowInfo, minutes_of

LOOKAHEAD_DAYS = 8


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime


def merge_touching(spans: list[Window]) -> list[Window]:
    merged: list[Window] = []
    for span in sorted(spans, key=lambda window: window.start):
        if merged and span.start <= merged[-1].end:
            merged[-1] = Window(merged[-1].start, max(merged[-1].end, span.end))
        else:
            merged.append(span)
    return merged


def windows(settings: MachineSettings, at: datetime, days: int = LOOKAHEAD_DAYS) -> list[Window]:
    zone = ZoneInfo(settings.timezone)
    today = at.astimezone(zone).date()
    spans = []
    for offset in range(-1, days + 1):
        date = today + timedelta(days=offset)
        midnight = datetime.combine(date, time(), zone)
        spans.extend(
            Window(
                midnight + timedelta(minutes=minutes_of(block.start)),
                midnight + timedelta(minutes=minutes_of(block.end)),
            )
            for block in settings.blocks
            if block.day == date.weekday()
        )
    return [window for window in merge_touching(spans) if window.end > at]


def current_window(settings: MachineSettings, at: datetime) -> Window | None:
    return next(
        (window for window in windows(settings, at) if window.start <= at < window.end), None
    )


def next_window(settings: MachineSettings, at: datetime) -> Window | None:
    return next((window for window in windows(settings, at) if window.start > at), None)


def window_info(settings: MachineSettings, at: datetime) -> WindowInfo:
    current = current_window(settings, at)
    if current:
        return WindowInfo(open=True, ends_at=current.end)
    upcoming = next_window(settings, at)
    return WindowInfo(open=False, next_opens_at=upcoming.start if upcoming else None)

