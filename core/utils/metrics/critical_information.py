"""Critical-information preservation: annotating spans and scoring them.

Spans carry ``type`` and a language-neutral ``canonical_value`` (plus optional
``accepted_values``). Human-reviewed reference spans from the dataset come first;
the annotation step adds rule-extracted values (``critical_values``) and LLM-judged
entities, once per reference, and annotates each candidate against that fixed list.
The scoring functions only compare spans.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .critical_values import extract_critical_values
from .judge import QUALITY_ANNOTATOR_VERSION, JsonJudge, locate


VALUE_TYPES = {"number", "ordinal", "date", "time", "money", "unit"}
ENTITY_TYPES = {"person", "location", "organization", "product", "term", "address",
                "email", "url"}
_TYPE_ALIASES = {"domain_term": "term", "terminology": "term", "postal_address": "address",
                 "place": "location", "org": "organization", "company": "organization",
                 "name": "person"}

CRITICAL_REFERENCE_PROMPT_VERSION = "ko-en-critical-reference-v2"
CRITICAL_CANDIDATE_PROMPT_VERSION = "ko-en-critical-candidate-v2"

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


def _norm(value) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).casefold().split())


def _forms(span: dict) -> set[str]:
    values = [span.get("canonical_value"), span.get("text")]
    values.extend(span.get("accepted_values") or [])
    return {_norm(v) for v in values if _norm(v)}


def _matches(expected: dict, predicted: dict) -> bool:
    return (_norm(expected.get("type")) == _norm(predicted.get("type"))
            and bool(_forms(expected) & _forms(predicted)))


def _match_counts(expected, predicted, *, allowed_types=None) -> tuple[int, int, int, dict]:
    refs = [dict(v) for v in expected
            if allowed_types is None or _norm(v.get("type")) in allowed_types]
    hyps = [dict(v) for v in predicted
            if allowed_types is None or _norm(v.get("type")) in allowed_types]
    used = set()
    correct = 0
    by_type = defaultdict(lambda: Counter(total=0, correct=0))
    for ref in refs:
        kind = _norm(ref.get("type")) or "unknown"
        by_type[kind]["total"] += 1
        found = next((i for i, hyp in enumerate(hyps)
                      if i not in used and _matches(ref, hyp)), None)
        if found is not None:
            used.add(found)
            correct += 1
            by_type[kind]["correct"] += 1
    detail = {kind: {"correct": counts["correct"], "total": counts["total"],
                     "accuracy": counts["correct"] / counts["total"]}
              for kind, counts in sorted(by_type.items()) if counts["total"]}
    return correct, len(refs), len(hyps), detail


def normalized_value_accuracy(expected, predicted) -> dict:
    """Accuracy after dataset-provided canonical value normalisation."""
    correct, total, _, by_type = _match_counts(
        expected, predicted, allowed_types=VALUE_TYPES)
    if not total:
        raise ValueError("no normalized value annotations")
    return {"accuracy": correct / total, "correct": correct, "total": total,
            "by_type": by_type}


def critical_span_f1(expected, predicted) -> dict:
    """Micro precision/recall/F1 over reviewed critical spans."""
    true_positive, n_ref, n_hyp, _ = _match_counts(expected, predicted)
    precision = true_positive / n_hyp if n_hyp else (1.0 if not n_ref else 0.0)
    recall = true_positive / n_ref if n_ref else (1.0 if not n_hyp else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": true_positive, "predicted": n_hyp, "reference": n_ref}


def critical_fact_error_rate(utterances) -> dict:
    """Fraction of critical-information-bearing utterances with any span error."""
    eligible = errors = invented_only = 0
    per_item = {}
    for row in utterances:
        expected = row.get("reference_spans") or []
        predicted = row.get("candidate_spans") or []
        if not expected and not predicted:
            continue
        match = critical_span_f1(expected, predicted)
        has_error = not (match["true_positive"] == match["reference"] == match["predicted"])
        eligible += 1
        errors += int(has_error)
        invented_only += int(not expected)
        per_item[str(row.get("id") or eligible)] = has_error
    if not eligible:
        raise ValueError("no utterance contains reviewed critical spans")
    return {"error_rate": errors / eligible, "error_items": errors,
            "eligible_items": eligible, "invented_only_items": invented_only,
            "per_item": per_item}


def corpus(items, **_) -> tuple[dict, dict]:
    rows = []
    reference_items = 0
    for item in items:
        block = item.metric_inputs.get("critical_information") or {}
        if "reference_spans" in block:
            reference_items += 1
        # Missing means "the new candidate has not been annotated"; an explicit
        # empty list means "annotation ran and found no spans". Conflating the two
        # would turn every translation-only replay into a false critical failure.
        if "reference_spans" in block and "candidate_spans" in block:
            rows.append({"id": item.id,
                         "reference_spans": block.get("reference_spans") or [],
                         "candidate_spans": block.get("candidate_spans") or []})
    if not rows:
        reason = ("missing metric_inputs.critical_information.candidate_spans"
                  if reference_items else
                  "missing metric_inputs.critical_information.reference_spans")
        return {}, {name: reason for name in (
            "critical_information.normalized_value_accuracy",
            "critical_information.critical_span_f1",
            "critical_information.critical_fact_error_rate")}

    values, unavailable = {}, {}
    # Match within an utterance. Flattening first would let the same value in a
    # different turn conceal a local omission.
    value_correct = value_total = 0
    value_types = defaultdict(lambda: Counter(total=0, correct=0))
    span_tp = span_ref = span_hyp = 0
    for row in rows:
        correct, total, _, detail = _match_counts(
            row["reference_spans"], row["candidate_spans"], allowed_types=VALUE_TYPES)
        value_correct += correct
        value_total += total
        for kind, cell in detail.items():
            value_types[kind]["correct"] += cell["correct"]
            value_types[kind]["total"] += cell["total"]
        tp, n_ref, n_hyp, _ = _match_counts(
            row["reference_spans"], row["candidate_spans"])
        span_tp += tp
        span_ref += n_ref
        span_hyp += n_hyp
    if value_total:
        values["normalized_value_accuracy"] = {
            "accuracy": value_correct / value_total,
            "correct": value_correct, "total": value_total,
            "n_annotated_items": len(rows),
            "n_missing_candidate_annotations": reference_items - len(rows),
            "by_type": {kind: {"correct": cell["correct"], "total": cell["total"],
                               "accuracy": cell["correct"] / cell["total"]}
                        for kind, cell in sorted(value_types.items())},
        }
    else:
        unavailable["critical_information.normalized_value_accuracy"] = (
            "no normalized value annotations")
    precision = span_tp / span_hyp if span_hyp else (1.0 if not span_ref else 0.0)
    recall = span_tp / span_ref if span_ref else (1.0 if not span_hyp else 0.0)
    values["critical_span_f1"] = {
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "true_positive": span_tp, "predicted": span_hyp, "reference": span_ref,
        "n_annotated_items": len(rows),
        "n_missing_candidate_annotations": reference_items - len(rows),
    }
    try:
        values["critical_fact_error_rate"] = critical_fact_error_rate(rows)
        values["critical_fact_error_rate"]["n_missing_candidate_annotations"] = (
            reference_items - len(rows))
    except ValueError as exc:
        unavailable["critical_information.critical_fact_error_rate"] = str(exc)
    return ({"critical_information": values} if values else {}), unavailable


def _span_type(raw) -> str:
    kind = re.sub(r"[\s-]+", "_", str(raw or "").strip().lower())
    return _TYPE_ALIASES.get(kind, kind)


def _validated_span(text: str, raw: dict, allowed_types: set[str], origin: str) -> dict | None:
    kind = _span_type(raw.get("type"))
    surface = str(raw.get("text") or "")
    canonical = str(raw.get("canonical_value") or "").strip()
    if kind not in allowed_types or not surface or not canonical:
        return None
    located = locate(text, surface, raw.get("start"), raw.get("end"))
    if located is None:
        return None
    accepted = [str(value) for value in raw.get("accepted_values") or [] if str(value).strip()]
    return {"type": kind, "text": surface, "start": located[0], "end": located[1],
            "canonical_value": canonical, "accepted_values": accepted, "origin": origin}


def _unlocated(text: str, raw: dict) -> int:
    surface = str(raw.get("text") or "")
    return int(bool(surface and str(raw.get("canonical_value") or "").strip())
               and locate(text, surface, raw.get("start"), raw.get("end")) is None)


def _with_origin(spans, origin: str) -> list[dict]:
    return [{**span, "origin": span.get("origin") or origin} for span in spans]


def gold_reference_spans(block: dict) -> list[dict]:
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
                except ValueError:
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


__all__ = ["VALUE_TYPES", "ENTITY_TYPES", "normalized_value_accuracy", "critical_span_f1",
           "critical_fact_error_rate", "corpus", "gold_reference_spans", "ReferenceSpanCache",
           "annotate_reference_critical_information",
           "annotate_candidate_critical_information", "annotate_critical_information"]
