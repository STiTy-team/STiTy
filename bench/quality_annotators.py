from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from core.utils.metrics.critical_information import VALUE_TYPES

from .critical_values import Span, extract_critical_values


QUALITY_ANNOTATOR_VERSION = "2026-09-21.2"
CRITICAL_REFERENCE_PROMPT_VERSION = "ko-en-critical-reference-v2"
CRITICAL_CANDIDATE_PROMPT_VERSION = "ko-en-critical-candidate-v2"
FLUENCY_PROMPT_VERSION = "spoken-fluency-mqm-v1"

ENTITY_TYPES = {"person", "location", "organization", "product", "term", "address",
                "email", "url"}
_TYPE_ALIASES = {"domain_term": "term", "terminology": "term", "postal_address": "address",
                 "place": "location", "org": "organization", "company": "organization",
                 "name": "person"}

FLUENCY_SYSTEM_PROMPT = """You are a strict evaluator of spoken-language fluency.
Judge only the candidate text as conversation in the target language. Do not infer or judge source meaning or translation faithfulness.
Use this 1-5 rubric: 5 fully natural spoken language; 4 natural with a small awkwardness; 3 understandable but noticeably awkward; 2 difficult or repeatedly ungrammatical; 1 unusable.
Also annotate MQM fluency/style errors. Allowed categories are grammar, word_order, word_form, spelling, punctuation, register, awkwardness, repetition, untranslated_fragment, and consistency. Severity is minor, major, or critical. Critical is reserved for text that is effectively unusable as target-language conversation.
Every error must identify an exact substring using zero-based start and exclusive end character offsets in the candidate. Do not create an error merely because a different wording would be preferable.
Return one JSON object with keys score, reason, and errors. errors is an array of objects with category, severity, start, end, text, and explanation."""

_CRITICAL_SPAN_RULES = """A span is critical only if it names a specific entity or a domain-specific term whose mistranslation would change who, what or where is meant. Do not extract generic common nouns (for example: cities, coffee, the stage, police headquarters, baked goods), pronouns, or descriptions.
Types are person, location, organization, product, term, address, email and url.
Every span must use an exact substring of its own text with zero-based start and exclusive end character offsets. Each span has type, text, start, end, canonical_value and accepted_values."""

CRITICAL_REFERENCE_SYSTEM_PROMPT = f"""You annotate critical information in the reference translation of one Korean-English conversation turn. The source, when given, is only for disambiguation.
Extract spans from the reference only. Numeric values (numbers, ordinals, dates, times, money, measurements, phone numbers) are already in known_reference_spans; do not repeat them.
{_CRITICAL_SPAN_RULES}
Give each span a language-neutral canonical_value (the lowercase English form of the name) and list valid alternative renderings (transliteration, translation, abbreviation, original script) in accepted_values.
Return one JSON object with key reference_spans."""

CRITICAL_CANDIDATE_SYSTEM_PROMPT = f"""You annotate critical information in a candidate translation against a fixed list of reference_spans. The reference_spans are final: never add, drop or change them.
1. Extract the candidate's critical spans. When a candidate span refers to the same entity as a reference span, copy that reference span's canonical_value exactly; transliteration, translation, abbreviation and original-script retention may be equivalent. When it refers to something else, or to a fact absent from the reference, give it its own canonical_value: invented facts must be extracted.
2. known_candidate_value_spans are the candidate's numeric values found by rule. If a reference span of type number, ordinal, date, time, money or unit has no candidate value span with the same canonical_value, but the candidate states exactly that value in other words (for example "a dozen" for 12), add a candidate span with that type, the reference's canonical_value, and the candidate's words as text. Never add a value span for a different value.
{_CRITICAL_SPAN_RULES}
Return one JSON object with keys candidate_spans and alignments. Each alignment has reference_index (into reference_spans), candidate_index (into your candidate_spans) and relation, which is equivalent or conflicting."""


class JsonJudge(Protocol):
    model: str

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        ...


class OpenAIJsonJudge:
    retry_delay_sec = 1.0
    report_every = 25

    def __init__(self, *, model: str = "gpt-5.4-mini", api_key: str | None = None,
                 max_retries: int = 3, usage_log: str | Path | None = None, client=None):
        from core.meaning_segmentator.autoseg.infra.gateway import Usage, _price_of

        if client is None:
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY is required for automatic quality annotation")
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=key)
        price = _price_of(model)
        if price is None:
            print(f"[judge] warning: no price for {model!r}; cost will be recorded as 0",
                  flush=True)
        self.client = client
        self.model = model
        self.max_retries = max_retries
        self.usage = Usage(price=price)
        self.usage_log = Path(usage_log) if usage_log else None

    def _record(self, purpose: str, payload: dict) -> None:
        before = self.usage.cost
        self.usage.add(payload, purpose)
        usage = payload.get("usage") or {}
        if self.usage_log is not None:
            line = {
                "at": datetime.now(timezone.utc).isoformat(),
                "purpose": purpose,
                "model": self.model,
                "prompt_tokens": usage.get("prompt_tokens", 0) or 0,
                "completion_tokens": usage.get("completion_tokens", 0) or 0,
                "cached_tokens": (usage.get("prompt_tokens_details") or {})
                .get("cached_tokens", 0) or 0,
                "cost": self.usage.cost - before,
                "cumulative_cost": self.usage.cost,
            }
            with open(self.usage_log, "a", encoding="utf-8") as output:
                output.write(json.dumps(line, ensure_ascii=False) + "\n")
        if self.usage.calls % self.report_every == 0:
            print(f"[judge] {self.usage.calls} calls, estimated cost "
                  f"${self.usage.cost:.4f}", flush=True)

    async def ask(self, *, purpose: str, system: str, payload: dict) -> dict:
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                self._record(purpose, response.model_dump())
                value = json.loads(response.choices[0].message.content)
                if not isinstance(value, dict):
                    raise ValueError(f"{purpose} did not return a JSON object")
                return value
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    await asyncio.sleep(min(self.retry_delay_sec * 2 ** attempt, 8))
        raise RuntimeError(f"{purpose} failed after {self.max_retries} attempts: {last_error}")


def _locate(text: str, surface: str, start, end) -> tuple[int, int] | None:
    """Offsets of ``surface`` in ``text``, trusting the judge's offsets only if they hold.

    Judges often miscount characters. When the substring occurs more than once (a
    repeated word is the typical case), the occurrence nearest the claimed start is
    taken; a surface that is not in the text at all cannot be located.
    """
    if (isinstance(start, int) and isinstance(end, int)
            and 0 <= start < end <= len(text) and text[start:end] == surface):
        return start, end
    positions = [match.start() for match in re.finditer(re.escape(surface), text)]
    if not positions:
        return None
    anchor = start if isinstance(start, int) else 0
    best = min(positions, key=lambda position: (abs(position - anchor), position))
    return best, best + len(surface)


def _span_type(raw) -> str:
    kind = re.sub(r"[\s-]+", "_", str(raw or "").strip().lower())
    return _TYPE_ALIASES.get(kind, kind)


def _validated_span(text: str, raw: dict, allowed_types: set[str], origin: str) -> dict | None:
    kind = _span_type(raw.get("type"))
    surface = str(raw.get("text") or "")
    canonical = str(raw.get("canonical_value") or "").strip()
    if kind not in allowed_types or not surface or not canonical:
        return None
    located = _locate(text, surface, raw.get("start"), raw.get("end"))
    if located is None:
        return None
    accepted = [str(value) for value in raw.get("accepted_values") or [] if str(value).strip()]
    return {"type": kind, "text": surface, "start": located[0], "end": located[1],
            "canonical_value": canonical, "accepted_values": accepted, "origin": origin}


def _unlocated(text: str, raw: dict) -> int:
    """1 when a well-formed span was dropped only because its text is not in ``text``."""
    surface = str(raw.get("text") or "")
    return int(bool(surface and str(raw.get("canonical_value") or "").strip())
               and _locate(text, surface, raw.get("start"), raw.get("end")) is None)


def _with_origin(spans, origin: str) -> list[dict]:
    return [{**span, "origin": span.get("origin") or origin} for span in spans]


def gold_reference_spans(block: dict) -> list[dict]:
    """The human-made reference spans of a ``critical_information`` block.

    Spans this annotator writes carry an ``origin``; human spans from a dataset
    manifest carry none (or ``origin: gold``). A block written before origins were
    recorded has ``annotation_source`` but unmarked spans, which cannot be told apart,
    so none of them is taken for human.
    """
    spans = [dict(span) for span in block.get("reference_spans") or [] if isinstance(span, dict)]
    if "annotation_source" in block and not any("origin" in span for span in spans):
        return []
    return [{**span, "origin": "gold"} for span in spans
            if span.get("origin", "gold") == "gold"]


def _merge_spans(primary: list[dict], secondary: list[dict]) -> list[dict]:
    output = [dict(value) for value in primary]
    occupied = [(value["start"], value["end"]) for value in output
                if isinstance(value.get("start"), int) and isinstance(value.get("end"), int)]
    identities = {(str(value.get("type") or "").lower(),
                   str(value.get("canonical_value") or "").casefold()) for value in output}
    for value in secondary:
        identity = (str(value.get("type") or "").lower(),
                    str(value.get("canonical_value") or "").casefold())
        if identity in identities:
            existing = next(item for item in output
                            if (str(item.get("type") or "").lower(),
                                str(item.get("canonical_value") or "").casefold()) == identity)
            for key in ("text", "start", "end"):
                if existing.get(key) is None and value.get(key) is not None:
                    existing[key] = value[key]
            accepted = list(dict.fromkeys((existing.get("accepted_values") or [])
                                          + (value.get("accepted_values") or [])))
            if accepted:
                existing["accepted_values"] = accepted
            continue
        located = (isinstance(value.get("start"), int)
                   and isinstance(value.get("end"), int))
        if located and any(value["start"] < end and start < value["end"]
                           for start, end in occupied):
            continue
        output.append(value)
        identities.add(identity)
        if located:
            occupied.append((value["start"], value["end"]))
    return sorted(output, key=lambda value: (value.get("start", len(output) + 1),
                                             value.get("end", len(output) + 1)))


def _check_pair(source_lang: str, target_lang: str) -> None:
    if {source_lang, target_lang} != {"ko", "en"}:
        raise ValueError("critical-information annotation supports only ko<->en")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def annotate_reference_critical_information(
        *, judge: JsonJudge, source: str, reference: str, source_lang: str,
        target_lang: str, gold_reference_spans: list[dict] | None = None) -> dict:
    """Critical spans of the reference translation, without looking at any candidate.

    This set is the denominator every system is scored against, so it is extracted
    once per reference and shared. ``source`` should be the gold transcript: the ASR
    transcript differs per ASR system and would make the set system-dependent.
    """
    _check_pair(source_lang, target_lang)
    gold = [{**dict(span), "origin": "gold"} for span in gold_reference_spans or []]
    seed = _merge_spans(gold, _with_origin(extract_critical_values(reference, target_lang),
                                           "rule"))
    payload = {"source_language": source_lang, "target_language": target_lang,
               "reference": reference, "known_reference_spans": seed}
    if source:
        payload["source"] = source
    result = await judge.ask(purpose="critical_information_reference",
                             system=CRITICAL_REFERENCE_SYSTEM_PROMPT, payload=payload)
    llm, unlocated = [], 0
    for raw in result.get("reference_spans") or []:
        if not isinstance(raw, dict):
            continue
        span = _validated_span(reference, raw, ENTITY_TYPES, "llm")
        if span:
            llm.append(span)
        else:
            unlocated += _unlocated(reference, raw)
    return {"reference_spans": _merge_spans(seed, llm),
            "reference_unlocated_spans": unlocated,
            "reference_prompt_version": CRITICAL_REFERENCE_PROMPT_VERSION,
            "reference_judge_model": judge.model,
            "reference_sha256": _sha256(reference)}


async def annotate_candidate_critical_information(
        *, judge: JsonJudge, source: str, reference: str, candidate: str,
        reference_spans: list[dict], source_lang: str, target_lang: str) -> dict:
    """Critical spans of one candidate, matched against a fixed reference span list."""
    _check_pair(source_lang, target_lang)
    rule = _with_origin(extract_critical_values(candidate, target_lang), "rule")
    payload = {"source_language": source_lang, "target_language": target_lang,
               "reference": reference, "reference_spans": reference_spans,
               "candidate": candidate, "known_candidate_value_spans": rule}
    if source:
        payload["source"] = source
    result = await judge.ask(purpose="critical_information_candidate",
                             system=CRITICAL_CANDIDATE_SYSTEM_PROMPT, payload=payload)
    raws = [raw for raw in result.get("candidate_spans") or [] if isinstance(raw, dict)]

    # The judge may state a value the rules missed ("a dozen"), but only a reference
    # value that no rule-found candidate value already matches; it can neither
    # override a rule-found value nor introduce a value of its own.
    found = {(span["type"], span["canonical_value"]) for span in rule}
    missing = {(span.get("type"), span.get("canonical_value")) for span in reference_spans
               if span.get("type") in VALUE_TYPES} - found
    rescued, entities, unlocated = [], [], 0
    for raw in raws:
        kind = _span_type(raw.get("type"))
        if kind in VALUE_TYPES:
            span = _validated_span(candidate, raw, VALUE_TYPES, "llm_value")
            if span and (span["type"], span["canonical_value"]) in missing:
                rescued.append(span)
        elif kind in ENTITY_TYPES:
            span = _validated_span(candidate, raw, ENTITY_TYPES, "llm")
            if span:
                entities.append(span)
            else:
                unlocated += _unlocated(candidate, raw)
    alignment = []
    for link in result.get("alignments") or []:
        if not isinstance(link, dict):
            continue
        ref_index, cand_index = link.get("reference_index"), link.get("candidate_index")
        if (isinstance(ref_index, int) and isinstance(cand_index, int)
                and 0 <= ref_index < len(reference_spans) and 0 <= cand_index < len(raws)):
            alignment.append({
                "reference_index": ref_index,
                "reference_canonical_value": reference_spans[ref_index].get("canonical_value"),
                "candidate_text": raws[cand_index].get("text"),
                "candidate_canonical_value": raws[cand_index].get("canonical_value"),
                "relation": str(link.get("relation") or ""),
            })
    return {"candidate_spans": _merge_spans(_merge_spans(rule, rescued), entities),
            "alignment": alignment,
            "candidate_unlocated_spans": unlocated,
            "candidate_prompt_version": CRITICAL_CANDIDATE_PROMPT_VERSION,
            "candidate_sha256": _sha256(candidate)}


class ReferenceSpanCache:
    """Reference spans shared by every system annotated against the same reference.

    An append-only JSONL file. Several annotation processes may run at once; two
    that both miss a key both ask the judge, and the judge need not answer alike. So
    after writing, a process reads the file back and takes the *first* entry for
    the key -- every process then scores against the same spans.
    """

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self._memory: dict[str, dict] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def key(*parts) -> str:
        return _sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str))

    def _read(self, key: str) -> dict | None:
        if self.path is None or not self.path.is_file():
            return None
        with open(self.path, encoding="utf-8") as source:
            for line in source:
                try:
                    entry = json.loads(line)
                except ValueError:  # a line another process is still writing
                    continue
                if isinstance(entry, dict) and entry.get("key") == key:
                    return entry.get("value")
        return None

    async def get(self, key: str, factory) -> dict:
        async with self._locks.setdefault(key, asyncio.Lock()):
            if key in self._memory:
                return self._memory[key]
            value = self._read(key)
            if value is None:
                value = await factory()
                if self.path is not None:
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.path, "a", encoding="utf-8") as output:
                        output.write(json.dumps({"key": key, "value": value},
                                                ensure_ascii=False) + "\n")
                    value = self._read(key)
            self._memory[key] = value
            return value


async def annotate_critical_information(*, judge: JsonJudge, source: str,
                                        reference: str, candidate: str,
                                        source_lang: str, target_lang: str,
                                        gold_reference_spans: list[dict] | None = None,
                                        reference_cache: ReferenceSpanCache | None = None
                                        ) -> dict:
    gold = [dict(span) for span in gold_reference_spans or []]

    async def extract():
        return await annotate_reference_critical_information(
            judge=judge, source=source, reference=reference, source_lang=source_lang,
            target_lang=target_lang, gold_reference_spans=gold)

    if reference_cache is None:
        reference_block = await extract()
    else:
        reference_block = await reference_cache.get(ReferenceSpanCache.key(
            QUALITY_ANNOTATOR_VERSION, CRITICAL_REFERENCE_PROMPT_VERSION, judge.model,
            source_lang, target_lang, source, reference, gold), extract)
    candidate_block = await annotate_candidate_critical_information(
        judge=judge, source=source, reference=reference, candidate=candidate,
        reference_spans=reference_block["reference_spans"], source_lang=source_lang,
        target_lang=target_lang)
    origins = {span.get("origin") for span in reference_block["reference_spans"]}
    return {
        **reference_block,
        **candidate_block,
        "annotation_source": "human_gold+automatic" if "gold" in origins else "automatic",
        "annotator_version": QUALITY_ANNOTATOR_VERSION,
        "judge_model": judge.model,
    }


def target_token_count(text: str, lang: str) -> int:
    if lang == "ko":
        return len(re.findall(r"\S+", text))
    return len(re.findall(r"\b\w+(?:['’-]\w+)*\b", text, re.UNICODE))


async def annotate_fluency(*, judge: JsonJudge, candidate: str,
                           target_lang: str, previous_turns: list[str] | None = None) -> dict:
    result = await judge.ask(
        purpose="spoken_fluency_mqm",
        system=FLUENCY_SYSTEM_PROMPT,
        payload={
            "target_language": target_lang,
            "preceding_target_turns": list(previous_turns or [])[-3:],
            "candidate": candidate,
        },
    )
    score = result.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not 1 <= float(score) <= 5:
        raise ValueError("fluency judge returned a score outside [1, 5]")
    allowed = {"grammar", "word_order", "word_form", "spelling", "punctuation",
               "register", "awkwardness", "repetition", "untranslated_fragment", "consistency"}
    errors = []
    unlocated = 0
    for raw in result.get("errors") or []:
        if not isinstance(raw, dict):
            continue
        kind = str(raw.get("category") or "").lower()
        surface = str(raw.get("text") or "")
        if kind not in allowed or not surface:
            continue
        located = _locate(candidate, surface, raw.get("start"), raw.get("end"))
        if located is None:
            unlocated += 1
            continue
        start, end = located
        severity = str(raw.get("severity") or "minor").lower()
        if severity not in {"minor", "major", "critical"}:
            severity = "minor"
        errors.append({"category": kind, "severity": severity, "start": start, "end": end,
                       "text": surface, "explanation": str(raw.get("explanation") or "")})
    return {
        "judge": {"score": float(score), "reason": str(result.get("reason") or "")},
        "mqm_errors": errors,
        "mqm_unlocated_errors": unlocated,
        "target_token_count": target_token_count(candidate, target_lang),
        "target_lang": target_lang,
        "annotator_version": QUALITY_ANNOTATOR_VERSION,
        "prompt_version": FLUENCY_PROMPT_VERSION,
        "judge_model": judge.model,
        "candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
    }


__all__ = ["OpenAIJsonJudge", "ReferenceSpanCache", "Span",
           "annotate_candidate_critical_information", "annotate_critical_information",
           "annotate_fluency", "annotate_reference_critical_information",
           "extract_critical_values", "gold_reference_spans", "target_token_count"]
