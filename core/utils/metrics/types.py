"""The shape every metric reads: one committed segment, one utterance.

The field names are the WebSocket protocol's own (docs/WEBSOCKET_PROTOCOL.md) --
`translation`, `language`, `commitReason`, `fsl_sec` -- plus the timing fields the
eval server adds to `final`. They are not a benchmark invention, which is why the
metrics can own them: any caller holding STiTy `final` payloads can score with
these without inventing a translation layer first.

`from_final` / `from_row` take the snake_case payloads a pipeline emits and
normalize once -- codes lowercased, text stripped, numbers
floated. Every metric downstream reads attributes, so no caller has to remember
that `decision_audio_sec` is the commit-decision clock and `audio_end_sec` is not.
"""
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable


@runtime_checkable
class TargetPolicy(Protocol):
    """Whatever decides which language an utterance should be translated into.

    Narrow on purpose: the routing and BLEU metrics need this one answer, and
    depending on a whole config object would tie them to one caller's settings.
    """

    def expected_target(self, src_lang: str) -> str:
        ...


def _code(value) -> str:
    return value.strip().lower() if isinstance(value, str) else ""


def _text(value) -> str:
    return value.strip() if isinstance(value, str) else ""


def _number(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class Segment:
    """One committed segment -- a `final` message."""

    original: str = ""
    translation: str = ""
    language: str = ""
    target_lang: str = ""
    commit_reason: str = ""
    decision_audio_sec: float | None = None
    recv_elapsed_sec: float | None = None
    direction_refixed: bool = False

    @property
    def fsl_sec(self) -> float | None:
        """How far behind the audio this commit was delivered, in seconds.

        Derived, not recorded: it is the wall clock at delivery minus the audio
        clock at the decision, both of which are already here. A pipeline cannot
        report it inconsistently with them because it does not report it at all.

        Meaningful under real-time pacing, which is what a latency number is for.
        A run that pushed audio faster than real time finishes ahead of the
        recording and reports this negative -- which is the honest reading of
        "delivered before that moment would have arrived", not an error.
        """
        if self.recv_elapsed_sec is None or self.decision_audio_sec is None:
            return None
        return self.recv_elapsed_sec - self.decision_audio_sec

    @classmethod
    def from_final(cls, payload: Mapping) -> "Segment":
        return cls(
            original=_text(payload.get("original")),
            translation=_text(payload.get("translation")),
            language=_code(payload.get("language")),
            target_lang=_code(payload.get("target_lang")),
            commit_reason=_code(payload.get("commit_reason")),
            decision_audio_sec=_number(payload.get("decision_audio_sec")),
            recv_elapsed_sec=_number(payload.get("recv_elapsed_sec")),
            direction_refixed=bool(payload.get("direction_refixed")),
        )


@dataclass(frozen=True)
class Utterance:
    """One scored unit: the references, what came back, and the segments it took.

    `hypothesis` is the concatenated ASR output and `hypothesis_translation` the
    concatenated translation, because WER scores the first and BLEU the second.
    """

    id: str = ""
    src_lang: str = ""
    reference: str = ""
    hypothesis: str = ""
    hypothesis_translation: str = ""
    duration_sec: float = 0.0
    reference_translations: Mapping[str, str] = field(default_factory=dict)
    segments: tuple[Segment, ...] = ()

    @classmethod
    def from_row(cls, row: Mapping) -> "Utterance":
        refs = row.get("reference_translations") or {}
        return cls(
            id=str(row.get("id") or ""),
            src_lang=_code(row.get("src_lang")),
            reference=_text(row.get("reference")),
            hypothesis=_text(row.get("hypothesis")),
            hypothesis_translation=_text(row.get("hypothesis_translation")),
            duration_sec=_number(row.get("duration_sec")) or 0.0,
            reference_translations={_code(k): _text(v) for k, v in refs.items()},
            segments=tuple(Segment.from_final(s) for s in (row.get("segments") or [])),
        )

    def reference_translation(self, target_lang: str) -> str:
        return self.reference_translations.get(_code(target_lang), "")
