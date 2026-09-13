"""판단형 루프의 Critic·PE. `agents_distill` 과 갈라 두는 이유는 **출력 단위가 다르기** 때문이다.

`agents_distill` 의 Critic 은 실패를 토큰 조건(`left_last`/`right_first`)으로 내야 하고,
관문이 그 토큰 매칭으로 규칙을 검증한다. 그 형태가 한계였다 — 규칙 하나가 dev 3,399 경계 중
3~5자리에만 걸리는데 개정 한 번이 남긴 경계의 37%를 흔들어, 직접 효과가 교란에 묻혔다.
손으로 쓴 판단 기준 프롬프트가 그 루프가 6런 동안 못 넘은 v0 를 넘긴 것이 그 증거다
(dev overlap 0.4767 → 0.5485).

여기서는 Critic 이 **판단 기준**을 내고 PE 가 `[Core Principles]`/`[Examples]` 를 고친다.
`[Scoring Rules]` 는 측정 절차·결합식·순위 규칙만 두고, `[Output Rules]` 와 함께 건드리지 않는다.
측정 절차를 서술하는 곳은 `[Scoring Rules]` 하나다.
"""

from __future__ import annotations

import json

from .agents_distill import SECTIONS, check_skeleton, replace_section  # noqa: F401  (재수출)

CRITIC_SYSTEM = """You diagnose a scoring prompt for streaming-translation cut positions.

Setup. A sentence is scored once: every candidate cut position gets a number 0-100. For each
latency budget T a deterministic step keeps the top-scored positions (a minimum distance between
cuts is enforced for you). The kept positions are the segmentation that is actually shipped: each
piece is translated ON ITS OWN and the pieces are read one after another, in order, by a listener
who cannot go back.

How the target was MEASURED, per cut set:
  cohesion   the pieces were translated separately, joined in order, and a reference-free
             quality estimator scored how faithfully that joined text renders the WHOLE source
             sentence. No human translation is involved.
  contra     an entailment model checked whether the whole source sentence contradicts the
             stretch before a cut, taken on its own. The worst cut in the set is what counts.
  H_set = cohesion x (1 - worst contra)

You receive cases. Each case is one sentence at one budget T and contains:
- "policy": the cut set this prompt produced, its H_set, and the pieces with their translations.
- "target": a better cut set found offline, its H_set, and the same detail.
- "diff": positions the policy kept but the target did not ("dropped"), and positions the target
  kept but the policy did not ("added"). Each carries the per-boundary measurements: "H" (the
  same product measured with that cut ALONE), "H_pct" (its rank inside the sentence, 0-100),
  "cohesion", "contra".
- "error_type":
    "boundary"    the policy kept a position whose own H is low. It is a bad cut on its own
                  terms, and the prompt should have seen that from the source alone.
    "interaction" the policy kept a position whose own H is HIGH, yet the better set does not
                  use it. Cutting there is fine alone but wrong in combination — usually two
                  cuts too close in meaning, a piece left too short to stand, or a better cut
                  one or two words away that makes both neighbours whole.

Your job: name the JUDGEMENT the prompt is getting wrong, not the tokens it fires on. A finding
is worth reporting only if it recurs across cases — one sentence is an anecdote. Write it so a
reader scoring an unseen sentence could apply it: a question to ask, or a condition on meaning.
Never write token lists ("if the previous word is 'the'"), never quote more than 40 characters of
source text, never name this sentence.

For "interaction" findings, the fix is usually a rule about the SET, not the position: what to do
when two candidates compete, or how the choice at one position constrains the next.

Return ONLY JSON:
{
  "findings": [
    {"error_type": "boundary" | "interaction",
     "diagnosis": "one sentence: what the prompt mis-judges, in terms of meaning",
     "evidence": "how many cases show it and what they share",
     "edit": {"where": "core_principles" | "examples",
              "action": "add" | "replace",
              "target": "first few words of the line to replace (omit when adding)",
              "text": "the new line, or an Input/Output example pair"}}
  ]
}
At most 3 findings. Order them by how many cases they explain."""

ENGINEER_SYSTEM = """You revise the system prompt of a scoring model, one iteration at a time.

Hard constraints:
1. Keep the section skeleton exactly: [Role], [Core Principles], [Scoring Rules],
   [Decision Procedure], [Output Rules], [Examples]. Same headers, same order.
2. Copy [Scoring Rules] and [Output Rules] verbatim from the current prompt. Both are frozen —
   [Scoring Rules] states how the target was measured, and the measurement did not change.
3. Do not add scoring conditions anywhere. Judgements belong in [Core Principles], worked cases
   in [Examples].
4. SIZE: at most __BUDGET__ characters (current prompt is __CURLEN__). The size is held fixed
   across iterations: if a new line supersedes an old one, REPLACE it rather than append, and cut
   weaker material to make room for what you add. You cannot count characters reliably, so code
   counts them for you: the input field "size" gives every section's length, and "editable_cap"
   is the most [Core Principles] and [Examples] may hold together. Trust those numbers.
   If "size_feedback" is present, "your_previous_attempt" was over the limit by "over_by"
   characters: keep its changes, and cut at least that much from the sections in "cut_from".
5. At most 8 examples. If you add one, remove a weaker one. Examples keep the Input/Output form
   with <SEG:?> at every candidate position in the input.
6. Never write token lists or punctuation rules. A token condition fires on a handful of
   positions while the measurement is taken at every position; a judgement applies everywhere.
7. Never name or depend on a specific target language pair beyond what the current prompt says.
8. Consult the attempt history: entries with "adopted": false were measured and rejected. Do not
   repeat them or minor variants — move in a different direction.
9. Every change must be traceable to a finding in the critique.

What the model is judged on: the cut sets its scores produce are translated piece by piece and
scored against the source as a whole, with the worst contradiction risk in the set applied as a
penalty. Only the ORDER of the numbers inside one sentence is ever read.

Return ONLY JSON:
{
  "sections_changed": ["[Core Principles]", "..."],
  "changelog": ["one line per change, stating what and why"],
  "prompt": "the complete revised prompt text"
}"""

WRITER_SYSTEM = """You write the system prompt for a scoring model used in streaming speech translation.

The model receives one source sentence in which EVERY possible cut position is already marked
with <SEG:?>. It replaces each ? with a score 0-100. Only the ORDER of those numbers inside one
sentence is ever read: a deterministic step keeps the highest-scored positions that fit the
current latency budget, and the kept positions are the segmentation that ships. Each piece is
translated ON ITS OWN and read in order by a listener who cannot go back.

The scores are judged against a MEASURED target, and your prompt must let the model reproduce it
from the SOURCE TEXT ALONE:
  cohesion   the pieces were translated separately into __TARGETS__, joined in order, and a
             reference-free quality estimator scored how faithfully that joined text renders the
             WHOLE source sentence.
  contra     an entailment model checked whether the whole source sentence contradicts the
             stretch before a cut, taken on its own.
  target = cohesion x (1 - contra)

Hard requirements:
- Section headers, verbatim and in this order:
  [Role], [Core Principles], [Scoring Rules], [Decision Procedure], [Output Rules], [Examples]
- [Output Rules] MUST be copied verbatim from the block given to you.
- [Core Principles] is the substance. Write JUDGEMENTS — questions the model asks about MEANING at
  each position — not surface-form rules. Two judgements carry the measurement: whether what
  follows overturns the stretch already heard, and whether translating the two sides apart still
  adds up to the source. Say what damage looks like in THIS source language, using what the
  profile tells you about how it builds clauses, where it puts negation and heads, and what it
  leaves implicit. Name the language's own devices; do not name individual tokens as triggers.
- **Do NOT write surface-form rules** — no lists of function words, no punctuation rules, no
  "if the previous token is X". A token condition fires on a handful of positions and says
  nothing about the rest, while the measurement is taken at every position; a judgement applies
  everywhere. Grammar labels are not the criterion either: a cut between two complete clauses
  can measure badly, and a cut inside a phrase can measure well.
- [Scoring Rules] is the ONLY place that states how the target was measured — [Output Rules]
  points to it and says nothing about the measurement. State cohesion, contra and the product
  exactly as given above, then how to combine the judgements into one number and the ranking
  rules (distinct integers, use the full range). contra is a graded probability and enters only
  through the product: do not turn it into a yes/no flag or a separate tier that outranks
  cohesion. No scoring conditions there.
- [Examples]: 3-4 pairs, in the SOURCE language, each Input/Output with <SEG:?> at every
  candidate position of the input and integers in the output. Build them from the sample
  sentences you are given, not from invented text.
- __SPACING__
- Keep the whole prompt under 8000 characters.

Return ONLY the prompt text. No commentary, no code fences."""


def writer_system(spaced: bool, targets: list[str]) -> str:
    unit = "words" if spaced else "characters"
    return (WRITER_SYSTEM
            .replace("__TARGETS__", ", ".join(targets))
            .replace("__SPACING__",
                     f"The source is written in {unit}; a cut position sits between two {unit}."))


ALLOWED_WHERE = {"core_principles": "[Core Principles]", "examples": "[Examples]"}
FROZEN = ("[Output Rules]", "[Scoring Rules]")


def critic_system() -> str:
    return CRITIC_SYSTEM


def engineer_system(budget: int, cur_len: int) -> str:
    return (ENGINEER_SYSTEM.replace("__BUDGET__", str(budget))
            .replace("__CURLEN__", str(cur_len)))


def clean_findings(blob: dict) -> list[dict]:
    """Critic 출력에서 쓸 수 있는 finding 만 남긴다. 토큰 조건은 여기서 잘라낸다."""
    out = []
    for f in (blob or {}).get("findings", []) or []:
        if not isinstance(f, dict):
            continue
        edit = f.get("edit") or {}
        if edit.get("where") not in ALLOWED_WHERE:
            continue
        if not (edit.get("text") or "").strip():
            continue
        if edit.get("action") == "replace" and not (edit.get("target") or "").strip():
            continue
        out.append({"error_type": f.get("error_type", "boundary"),
                    "diagnosis": (f.get("diagnosis") or "").strip(),
                    "evidence": (f.get("evidence") or "").strip(),
                    "edit": {k: edit.get(k) for k in ("where", "action", "target", "text")}})
    return out[:3]


def frozen_intact(before: str, after: str) -> list[str]:
    """PE 가 건드리면 안 되는 섹션이 그대로인지. 다르면 그 섹션 이름을 돌려준다."""
    bad = []
    for header in FROZEN:
        if section_of(before, header) != section_of(after, header):
            bad.append(header)
    return bad


def section_of(prompt: str, header: str) -> str:
    i = prompt.find(header)
    if i < 0:
        return ""
    nxt = [prompt.find(s, i + len(header)) for s in SECTIONS if s != header]
    nxt = [k for k in nxt if k > i]
    return prompt[i:min(nxt)].strip() if nxt else prompt[i:].strip()


def parse_prompt(blob: dict, current: str, budget: int) -> tuple[str | None, list[str]]:
    """PE 출력 검증 — 통과하면 (프롬프트, 변경목록), 아니면 (None, 사유)."""
    errs: list[str] = []
    pr = (blob or {}).get("prompt") or ""
    if not pr.strip():
        return None, ["prompt 가 비었다"]
    errs += check_skeleton(pr)
    errs += [f"동결 섹션이 바뀌었다: {h}" for h in frozen_intact(current, pr)]
    if len(pr) > budget:
        errs.append(f"길이 초과: {len(pr)} > {budget}")
    if errs:
        return None, errs
    return pr, [str(x) for x in (blob.get("changelog") or [])]


# ── 길이 — 모델은 글자 수를 못 센다. 코드가 세서 넘긴다 ─────────────────────
# judge01 에서 Writer 는 "8000자 이하" 지시에 10,947자를, PE 는 상한 9,000 에 9,895자를 냈다.
# 상한 숫자만 주면 쓰는 도중에 길이를 가늠하지 못하므로, 섹션별 실측과 초과량을 준다.

EDITABLE = tuple(ALLOWED_WHERE.values())


def section_sizes(prompt: str) -> dict[str, int]:
    return {h: len(section_of(prompt, h)) for h in SECTIONS}


def size_brief(prompt: str, budget: int) -> dict:
    """PE 입력용 — 섹션별 길이와, 고칠 수 있는 두 섹션이 함께 가질 수 있는 최대 길이."""
    sizes = section_sizes(prompt)
    fixed = len(prompt) - sum(sizes[h] for h in EDITABLE)
    return {"counted_by": "code", "budget": budget, "current_total": len(prompt),
            "sections": sizes, "editable_sections": list(EDITABLE),
            "editable_cap": budget - fixed}


def size_feedback(draft: str, current: str, budget: int) -> dict:
    """길이 초과로 반려된 초안을 한 번 되돌려 보낼 때 붙이는 실측."""
    now, was = section_sizes(draft), section_sizes(current)
    return {"counted_by": "code", "attempt_total": len(draft), "budget": budget,
            "over_by": len(draft) - budget,
            "sections": {h: {"attempt": now[h], "current": was[h], "change": now[h] - was[h]}
                         for h in SECTIONS},
            "cut_from": sorted(EDITABLE, key=lambda h: now[h] - was[h], reverse=True)}


def only_too_long(errs: list[str]) -> bool:
    """반려 사유가 길이 초과뿐인가 — 그때만 되돌려 보낼 값어치가 있다."""
    return bool(errs) and all(e.startswith("길이 초과") for e in errs)
