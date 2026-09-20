"""How long something took.

One decorator. A measured call writes `{"type": "timing", "tag": ..., "dur": ...}`
when it returns or raises, so a duration cannot be left open -- there is no way to
start one without ending it.

`registry.register` puts this on every component method, which is why backend code
contains no timing line at all. The tag names the work, not the model: `decode`, so
that two backends' decoding compares in one lane.
"""
import functools
import inspect

from core.utils import stream

elapsed = stream.elapsed


def measure(tag: str):
    def deco(call):
        if getattr(call, "__measured__", False):
            return call

        if inspect.iscoroutinefunction(call):
            @functools.wraps(call)
            async def measured(*args, **kwargs):
                started, audio = stream.elapsed(), stream.audio_position()
                try:
                    return await call(*args, **kwargs)
                finally:
                    _write(tag, started, audio)
        else:
            @functools.wraps(call)
            def measured(*args, **kwargs):
                started, audio = stream.elapsed(), stream.audio_position()
                try:
                    return call(*args, **kwargs)
                finally:
                    _write(tag, started, audio)

        measured.__measured__ = True
        return measured

    return deco


def _write(tag: str, started: float | None, audio: float | None) -> None:
    if started is None:
        return
    fields = {"tag": tag, "dur": round((stream.elapsed() or started) - started, 4)}
    if audio is not None:
        fields["audio"] = round(audio, 3)
    stream.record("timing", t=started, **fields)
