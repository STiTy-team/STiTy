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
import re

from .agents_distill import SECTIONS, replace_section  # noqa: F401  (재수출)
from .agents_distill import check_skeleton as _check_skeleton

# judge 루프의 골격 — distill 의 여섯 섹션에서 [Decision Procedure] 를 뺐다. 그 절은 [Core
# Principles] 의 두 판단을 순서로 다시 쓴 것에 영어 힌트를 보탠 것이었고(judge09 v0), PE 가 원칙만
# 고치므로 절차의 옛 힌트가 개정과 어긋난 채 남았다. 순위 규칙(다른 정수·넓은 범위)은 코드가 만드는
# [Output Rules] 에 있다.
JUDGE_SECTIONS = ["[Role]", "[Core Principles]", "[Scoring Rules]", "[Output Rules]", "[Examples]"]


def check_skeleton(prompt: str) -> list[str]:
    return _check_skeleton(prompt, JUDGE_SECTIONS)

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
- "latency_bin": the average piece length of this cut set ("≤3" = pieces of about 3 words).
  Cases are spread across bins on purpose. In the short bins good cut sets are NECESSARILY dense;
  a finding that makes the prompt avoid cutting near other cuts in general will hurt those bins.
- "contra_kill": true when a single policy cut with a high contradiction ("policy_worst_contra")
  is what zeroed the set. Such a case says "this one position is overturned later" — not that
  short pieces or dense cuts are bad. Do not turn it into a general caution about cutting.
- "last_revision" (only after the first measured revision): what the previous revision did and
  how it landed — its "edits", the measured "delta", "by_bin" (mean H_set change per latency bin),
  "n_worse"/"n_better", and a post-mortem ("why", "blamed" units, "lesson"). If it was rejected,
  do not ask for the same change again in other words; the bins say where it hurt.

Your job: name the JUDGEMENT the prompt is getting wrong, not the tokens it fires on. A finding
is worth reporting only if it recurs across cases — one sentence is an anecdote. Write it so a
reader scoring an unseen sentence could apply it: a question to ask, or a condition on meaning.
Never write token lists ("if the previous word is 'the'"), never quote more than 40 characters of
source text, never name this sentence.

Return ONLY JSON:
{
  "findings": [
    {"diagnosis": "one sentence: what the prompt mis-judges, in terms of meaning",
     "evidence": "how many cases show it and what they share",
     "edit": {"where": "core_principles" | "examples",
              "action": "add" | "replace",
              "target": "first few words of the line to replace (omit when adding)",
              "text": "the new line, or an Input/Output example pair"}}
  ]
}
At most 3 findings. Order them by how many cases they explain."""

POSTMORTEM_SYSTEM = """You explain why one revision of a scoring prompt did or did not work.

The prompt scores every possible cut position in a sentence; a deterministic step keeps the top
scores for each number of cuts k, and the kept set is measured as H_set = cohesion x (1 - worst
contradiction). You are given, for the revision that was just measured:
- "verdict" and "delta": the paired result on dev-A (mean and 95% CI of H_set change).
- "edits": which unit changed, "was" its text before, "now" after.
- "by_bin": the mean H_set change per latency bin ("≤3" = pieces of about 3 words).
- "n_worse" / "n_better": how many (sentence, k) pairs moved each way.
- "worst" / "best": the five pairs that moved most in each direction, each with the cut set
  before and after ("‖" marks a cut).

Read the cut sets: say what the edits made the model DO differently (cut earlier, avoid cutting
near heads, split enumerations, ...), and which edit is responsible. Be concrete about the bins —
a revision that helps long pieces and hurts short ones is a different fact from one that hurts
everywhere. Never propose a new rule here; that is the critic's job.

Return ONLY JSON:
{
  "why": "one or two sentences: what changed in behaviour and why it moved the measurement",
  "blamed": ["C3", "..."],
  "lesson": "one sentence a later revision should respect"
}"""

ENGINEER_SYSTEM = """You revise the system prompt of a scoring model, one iteration at a time, by EDITING
numbered units of it. You do not rewrite the prompt — code applies your edits.

What you can edit:
- "units" lists every unit of [Core Principles] (ids C1, C2, ...; one line each) and of
  [Examples] (ids E1, E2, ...; one Input/Output pair each), with its exact text, its length
  ("chars") and its evidence:
    "origin"         "v0", or the iteration whose ADOPTED revision introduced this text
    "adopted_delta"  the measured gain of that adoption, "adopted_ci_lo" its lower bound
                     (null for v0 units — nothing was measured about them one by one)
- Every other section — [Role], [Scoring Rules], [Output Rules] — stays as
  it is. [Scoring Rules] states how the target was measured, and the measurement did not change.

Hard constraints:
1. Judgements belong in [Core Principles], worked cases in [Examples]. Do not add scoring
   conditions anywhere.
2. SIZE: the edited prompt must be at most __BUDGET__ characters (it is __CURLEN__ now) — a small
   growth over the last adopted prompt. You cannot count characters reliably, so code counts
   them and "size.headroom" tells you how much you may add NET, in characters and in WORDS
   ("words" is what you can estimate: a typical [Core Principles] unit is
   "typical_principle_words" words, and "headroom.principles" says how many such units fit).
   Every insertion or lengthening must be paid for within that headroom, or by a deletion or
   paraphrase in the same edit list. When what you want to add does not fit, make room by
   EVIDENCE, not by length:
   - replace or delete the units with the weakest evidence first: "v0" units with no measured
     gain, and units the current critique faults;
   - keep units whose "adopted_ci_lo" is positive — their gain was measured — unless you
     paraphrase them shorter with the same meaning;
   - paraphrasing any unit shorter without changing what it asks is allowed. Mark such an edit
     "kind": "paraphrase"; an edit that adds or changes a judgement is "kind": "change".
   If "size_feedback" is present, "your_previous_edits" produced a prompt "over_by" characters
   ("over_by_words" words) too long, and "your_edits" shows the exact change each of those edits
   made. Return a complete new edit list that removes at least that many words more than it
   adds.
3. At most 8 examples. An example unit is exactly "Input: ..." then a newline and "Output: ...",
   with <SEG:?> at every candidate position in the input and integers in the output.
4. Never write token lists or punctuation rules. A token condition fires on a handful of
   positions while the measurement is taken at every position; a judgement applies everywhere.
5. Never name or depend on a specific target language pair beyond what the current prompt says.
6. Consult the attempt history: entries with "adopted": false were measured and rejected. Do not
   repeat them or minor variants — move in a different direction. Each entry lists its "edits":
   which unit it changed ("was", the unit's text at the time) and into what ("now"). Do not make
   the same change to the same units again, even reworded.
7. Every change must be traceable to a finding in the critique.
8. If "sibling_candidates" is present, those revisions were already proposed in this iteration
   and will be measured alongside yours. Propose a DIFFERENT revision: address other findings,
   edit other units, or take a different direction on the same finding. Do not restate a sibling.
9. If "labeled_examples" is present, each entry is a real sentence from the measured set with
   its MEASURED scores already written as an Input/Output pair ("unit"), and the case it came
   from ("id", its "latency_bin", and "gap" = how much the current prompt lost there). To put one
   into [Examples], write an edit with "labeled_example": "<id>" INSTEAD of "text" — code pastes
   the exact pair, e.g. {"op": "replace", "id": "E2", "labeled_example": "en_us_1591",
   "kind": "example"} or {"op": "insert_after", "id": "E_end", "labeled_example": "..."}.
   A measured example teaches the ranking directly where a principle only describes it; prefer
   replacing a hand-written example with a measured one over adding another principle.
10. If "constraint" is present, obey it — code enforces it by dropping edits that violate it:
   "examples_only": edit only [Examples] units (replace or insert measured examples, delete a
   weak hand-written one); no [Core Principles] edit at all.
   "single_small": exactly ONE edit, and its new text is at most 60 words — a small, precise
   change whose effect can be attributed.

What the model is judged on: the cut sets its scores produce are translated piece by piece and
scored against the source as a whole, with the worst contradiction risk in the set applied as a
penalty. Only the ORDER of the numbers inside one sentence is ever read.

Return ONLY JSON:
{
  "changelog": ["one line per change, stating what and why"],
  "edits": [
    {"op": "replace", "id": "C3", "kind": "change", "text": "the complete new text of that unit"},
    {"op": "replace", "id": "C1", "kind": "paraphrase", "text": "the same judgement, shorter"},
    {"op": "delete", "id": "E4"},
    {"op": "insert_after", "id": "C5", "kind": "change", "text": "a new unit"}
  ]
}
"insert_after" with id "C0" or "E0" inserts at the TOP of that section, and "C_end" or "E_end"
appends at the BOTTOM — use those instead of inventing an id past the last one. Ids are renumbered
every iteration, so name only ids from the "units" list you were given. Each existing id may be
replaced or deleted at most once."""

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
  [Role], [Core Principles], [Scoring Rules], [Output Rules], [Examples]
  No other section — in particular no [Decision Procedure]: the procedure IS the judgements in
  [Core Principles], applied at every marker; do not restate them as steps.
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


def postmortem_system() -> str:
    return POSTMORTEM_SYSTEM


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
        out.append({"diagnosis": (f.get("diagnosis") or "").strip(),
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
    return {h: len(section_of(prompt, h)) for h in JUDGE_SECTIONS if h in prompt}


def words(text: str) -> int:
    return len(text.split())


def chars_per_word(prompt: str) -> float:
    """이 프롬프트의 단어당 글자 수 — 글자 예산을 단어로 환산할 때 쓴다."""
    return len(prompt) / max(1, words(prompt))


def headroom(prompt: str, budget: int) -> dict:
    """더 넣을 수 있는 순증가량. LLM 은 글자보다 단어 수를 훨씬 잘 어림하므로 둘 다 준다.

    judge05~08 에서 상한 초과 6건 중 5건이 '추가만 하고 줄이지 않은' 편집이었다. 5% 여유(약
    500자)가 원칙 하나(400~650자) 크기라는 것을 글자 수만으로는 가늠하지 못한 탓이다."""
    cpw = chars_per_word(prompt)
    free = budget - len(prompt)
    principles = [u["chars"] for u in edit_units(prompt) if u["id"].startswith("C")]
    typical = int(sum(principles) / len(principles)) if principles else 0
    return {"chars": free, "words": int(free / cpw),
            "typical_principle_words": int(typical / cpw),
            "principles": round(free / typical, 1) if typical else None}


def size_brief(prompt: str, budget: int) -> dict:
    """PE 입력용 — 섹션별 길이, 고칠 수 있는 두 섹션의 최대 길이, 남은 여유(글자·단어)."""
    sizes = section_sizes(prompt)
    fixed = len(prompt) - sum(sizes[h] for h in EDITABLE)
    return {"counted_by": "code", "budget": budget, "current_total": len(prompt),
            "sections": sizes, "editable_sections": list(EDITABLE),
            "editable_cap": budget - fixed, "headroom": headroom(prompt, budget)}


def only_too_long(errs: list[str]) -> bool:
    """반려 사유가 길이 초과뿐인가 — 그때만 되돌려 보낼 값어치가 있다."""
    return bool(errs) and all(e.startswith("길이 초과") for e in errs)


# ── 편집 단위 — PE 는 프롬프트를 다시 쓰지 않고 번호 붙은 단위를 고친다 ─────────
# judge03 에서 초과량을 알려 되돌려도 다섯 번 중 네 번이 상한을 못 맞췄다(두 번은 더 길어졌다).
# 1만 자를 통째로 다시 쓰는 한 길이는 모델 손을 떠난다. 편집만 받고 적용과 길이는 코드가 한다.

UNIT_TAGS = {"[Core Principles]": "C", "[Examples]": "E"}
MAX_EXAMPLES = 8


def _items(prompt: str, header: str) -> list[str | None]:
    """섹션 본문을 단위 목록으로. 빈 줄은 None 으로 남겨 재조립 때 모양을 지킨다.
    [Core Principles] 는 한 줄이 한 단위, [Examples] 는 "Input:" 줄부터 다음 "Input:" 이나
    빈 줄 전까지가 한 단위다."""
    body = section_of(prompt, header)[len(header):].strip("\n")
    items: list[str | None] = []
    for line in body.split("\n"):
        if not line.strip():
            items.append(None)
        elif (header == "[Examples]" and items and items[-1] is not None
              and not line.lstrip().startswith("Input:")):
            items[-1] += "\n" + line
        else:
            items.append(line)
    return items


def edit_units(prompt: str) -> list[dict]:
    """PE 입력용 — 고칠 수 있는 단위마다 id·섹션·실제 글자 수·본문."""
    out = []
    for header, tag in UNIT_TAGS.items():
        k = 0
        for it in _items(prompt, header):
            if it is None:
                continue
            k += 1
            out.append({"id": f"{tag}{k}", "section": header, "chars": len(it), "text": it})
    return out


def apply_edits(prompt: str, edits) -> tuple[str | None, list[dict], list[dict]]:
    """편집 목록을 적용한다 — (새 프롬프트 또는 None, 건너뛴 편집, 적용한 편집의 글자 증감).

    잘못된 편집은 **그것만 건너뛰고** 나머지를 적용한다. 묶음을 통째로 반려하면 멀쩡한 편집까지
    버려지고 이터 하나가 날아간다 (judge07 iter 1: 편집 넷 중 하나가 없는 id 를 가리켜 전부 반려).
    남는 편집이 하나도 없을 때만 None 이다."""
    if not isinstance(edits, list) or not edits:
        return None, [{"edit": None, "id": None, "reason": "edits 가 비었다"}], []
    known = {u["id"]: u["text"] for u in edit_units(prompt)}
    last = {t: 0 for t in UNIT_TAGS.values()}
    for uid in known:
        last[uid[0]] = max(last[uid[0]], int(uid[1:]))

    def anchor(uid: str) -> str | None:
        """`insert_after` 가 가리키는 자리. "C0"/"E0" 은 섹션 맨 앞, "C_end"/"E_end" 는 맨 끝이고,
        **마지막 id 바로 다음 번호**(C6 까지 있을 때의 C7)도 맨 끝으로 읽는다 — 끝에 붙이는 자리가
        없어서 PE 가 그 번호를 만들어 썼고, judge07 은 그 때문에 다섯 이터 중 넷을 날렸다."""
        if uid in known:
            return uid
        m = re.fullmatch(r"([CE])(?:_(?:end|last)|(\d+))", uid or "")
        if not m or m.group(1) not in last:
            return None
        tag, num = m.group(1), m.group(2)
        if num is None:
            return f"{tag}_end"
        n = int(num)
        if n == 0:
            return f"{tag}0"
        return f"{tag}_end" if n == last[tag] + 1 else None
    plan: dict[str, str | None] = {}
    inserts: dict[str, list[str]] = {}
    errs: list[dict] = []
    deltas: list[dict] = []
    for n, e in enumerate(edits):
        if not isinstance(e, dict):
            errs.append({"edit": n, "id": None, "reason": "객체가 아니다"})
            continue
        op, uid = e.get("op"), str(e.get("id") or "")
        text = e["text"].strip("\n") if isinstance(e.get("text"), str) else ""
        # PE 가 단위 id 를 본문 머리에 써 넣는다("C11: At every marker …", judge05 iter 4). 그 글자는
        # 분절기 프롬프트에 그대로 들어가므로 뗀다.
        text = re.sub(r"^\s*[CE]\d+\s*[:.)\-–]\s*", "", text)
        if op not in ("replace", "delete", "insert_after"):
            errs.append({"edit": n, "id": uid, "reason": f"모르는 op {op!r}"})
            continue
        at = anchor(uid) if op == "insert_after" else (uid if uid in known else None)
        if at is None:
            errs.append({"edit": n, "id": uid, "reason": f"없는 id {uid!r}"})
            continue
        if op != "delete":
            if not text.strip():
                errs.append({"edit": n, "id": uid, "reason": "text 가 비었다"})
                continue
            if any(h in text for h in SECTIONS):
                errs.append({"edit": n, "id": uid, "reason": "text 에 섹션 헤더가 들어 있다"})
                continue
            if uid.startswith("E") and not (text.lstrip().startswith("Input:")
                                            and "\nOutput:" in text):
                errs.append({"edit": n, "id": uid,
                             "reason": "예시는 'Input: …' 다음 줄 'Output: …' 한 쌍이어야 한다"})
                continue
        kind = "delete" if op == "delete" else str(e.get("kind") or "change")
        if op == "insert_after":
            inserts.setdefault(at, []).append(text)
            deltas.append({"edit": n, "op": op, "id": at, "kind": kind,
                           "chars_change": len(text) + 1})
            continue
        if uid in plan:
            errs.append({"edit": n, "id": uid, "reason": "같은 단위를 두 번 고친다"})
            continue
        plan[uid] = None if op == "delete" else text
        change = -(len(known[uid]) + 1) if op == "delete" else len(text) - len(known[uid])
        deltas.append({"edit": n, "op": op, "id": uid, "kind": kind, "chars_change": change})
    if not deltas:
        return None, errs, deltas
    out = prompt
    for header, tag in UNIT_TAGS.items():
        gap = [""] if header == "[Examples]" else []        # 예시 쌍 사이는 빈 줄
        lines: list[str] = []
        for t in inserts.get(f"{tag}0", []):
            lines += [t] + gap
        k = 0
        for it in _items(prompt, header):
            if it is None:
                lines.append("")
                continue
            k += 1
            uid = f"{tag}{k}"
            kept = plan.get(uid, it)
            if kept is not None:
                lines.append(kept)
            for t in inserts.get(uid, []):
                lines += gap + [t]
        for t in inserts.get(f"{tag}_end", []):
            lines += gap + [t]
        body = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
        out = replace_section(out, header, f"{header}\n{body}")
    # replace_section 은 섹션 뒤에 빈 줄을 붙인다. 마지막 섹션이면 파일 끝 모양이 바뀌어, 아무것도
    # 안 바꾼 편집도 길이가 늘어 상한에 걸린다(judge03 v0: 8,664 → 8,666). 원래 끝을 되살린다.
    tail = prompt[len(prompt.rstrip("\n")):]
    return out.rstrip("\n") + tail, errs, deltas


ROLES = ("free", "examples_only", "single_small", "rewrite")


def enforce_role(edits, role: str) -> tuple[list, list[dict]]:
    """후보 역할별 제약을 코드로 건다. 어기는 편집은 건너뛴다.

    같은 이터에 보폭이 다른 후보를 섞기 위해서다 — 원칙을 통째로 갈아끼운 후보(judge05~08 의
    기본형)는 매번 전 구간을 흔들었고, 어느 편집이 무엇을 바꿨는지도 남지 않았다."""
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    if role == "examples_only":
        keep = [e for e in edits if str(e.get("id") or "").startswith("E")]
        bad = [{"edit": n, "id": e.get("id"), "reason": "examples_only 인데 예시 단위가 아니다"}
               for n, e in enumerate(edits) if not str(e.get("id") or "").startswith("E")]
        return keep, bad
    if role == "single_small":
        bad = [{"edit": n, "id": e.get("id"), "reason": "single_small 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        if keep and len(str(keep[0].get("text") or "").split()) > 60 and "labeled_example" not in keep[0]:
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": f"single_small 인데 {len(str(keep[0]['text']).split())}단어 > 60"})
            keep = []
        return keep, bad
    return edits, []


def resolve_labeled_examples(edits, examples: dict[str, str]) -> tuple[list, list[dict]]:
    """`labeled_example` 로 실측 예시를 가리킨 편집에 그 본문을 넣는다. 없는 id 는 건너뛴다."""
    out, skipped = [], []
    for n, e in enumerate(edits or []):
        if isinstance(e, dict) and "labeled_example" in e and not e.get("text"):
            ex = examples.get(str(e["labeled_example"]))
            if ex is None:
                skipped.append({"edit": n, "id": e.get("id"),
                                "reason": f"없는 labeled_example {e['labeled_example']!r}"})
                continue
            e = {**e, "text": ex, "kind": e.get("kind") or "example"}
        out.append(e)
    return out, skipped


def parse_edits(blob: dict, current: str, budget: int) -> tuple[
        str | None, list[str], str | None, list[dict], list[dict]]:
    """PE 편집 출력을 적용하고 검증한다 — (프롬프트 또는 None, 변경목록 또는 사유, 적용 결과,
    편집별 증감, 건너뛴 편집). 적용 결과는 검증에 떨어져도 돌려준다 — 길이 피드백을 만들 때 쓴다."""
    blob = blob or {}
    draft, skipped, deltas = apply_edits(current, blob.get("edits"))
    if draft is None:
        return (None, [s["reason"] for s in skipped] or ["적용할 편집이 없다"], None, deltas,
                skipped)
    n_ex = sum(1 for u in edit_units(draft)
               if u["section"] == "[Examples]" and u["text"].lstrip().startswith("Input:"))
    if n_ex > MAX_EXAMPLES:
        return None, [f"예시 {n_ex}개 > {MAX_EXAMPLES}"], draft, deltas, skipped
    pr, note = parse_prompt({"prompt": draft, "changelog": blob.get("changelog")}, current, budget)
    return pr, note, draft, deltas, skipped


def init_provenance(prompt: str) -> dict[str, dict]:
    """단위 본문 → 출처 기록. 시작 프롬프트의 단위는 전부 v0 이고 따로 잰 이득이 없다."""
    return {u["text"]: {"origin": "v0", "adopted_delta": None, "adopted_ci_lo": None}
            for u in edit_units(prompt)}


def adopt_provenance(prov: dict, new_prompt: str, it: int, gain: dict) -> dict:
    """채택된 개정본의 출처표. 그대로 남은 단위는 기록을 잇고, 새로 들어오거나 바뀐 단위는 이번
    채택의 Δ 를 받는다(압축한 단위도 새 문구로 다시 잰 것이다). 사라진 단위의 기록은 버린다."""
    out = {}
    for u in edit_units(new_prompt):
        t = u["text"]
        out[t] = prov[t] if t in prov else {
            "origin": f"iter {it}", "adopted_delta": round(gain["mean"], 4),
            "adopted_ci_lo": round(gain["lo"], 4)}
    return out


def history_brief(history: list[dict]) -> list[dict]:
    """PE 에게 주는 이력 — 방향을 되풀이하지 않게 하는 데 필요한 것만. 전체 이력은 후보마다
    부트스트랩 전체·소견·부검을 다 담아 judge09 4이터에 이미 20KB 였다."""
    out = []
    for h in history:
        d = h.get("gain") or h.get("delta") or {}
        b = {"iter": h["iter"], "adopted": bool(h.get("adopted")),
             "delta_mean": round(d["mean"], 4) if "mean" in d else None,
             "delta_lo": round(d["lo"], 4) if "lo" in d else None,
             "edits": h.get("edits") or []}
        if h.get("candidate") is not None:
            b["candidate"] = h["candidate"]
        if h.get("screened_out"):
            b["measured_on"] = "screen subset, lost to a sibling"
        elif h.get("screened_only"):
            b["measured_on"] = "screen subset only"
        if h.get("reason"):
            b["rejected_before_measuring"] = h["reason"]
        lesson = (h.get("diagnosis") or {}).get("lesson")
        if lesson:
            b["lesson"] = lesson
        out.append(b)
    return out


def edit_summary(prompt: str, edits) -> list[dict]:
    """이력용 — 어느 단위를 무엇으로 바꿨는지 짧게. 이력에 사유·Δ 만 있으면 PE 가 기각된 편집을
    알아보지 못하고 되풀이한다(judge05 iter 2~4). id 는 이터마다 다시 매겨지므로 원문 앞부분을 싣는다."""
    known = {u["id"]: u["text"] for u in edit_units(prompt)}
    out = []
    for e in edits or []:
        if not isinstance(e, dict):
            continue
        op = e.get("op")
        out.append({"op": op, "id": e.get("id"),
                    "kind": "delete" if op == "delete" else (e.get("kind") or "change"),
                    "was": known.get(str(e.get("id")), "")[:80],
                    "now": (e.get("text") or "")[:80]})
    return out


def units_with_provenance(prompt: str, prov: dict) -> list[dict]:
    """PE 입력용 — 편집 단위에 출처 기록을 붙인다."""
    blank = {"origin": "v0", "adopted_delta": None, "adopted_ci_lo": None}
    return [{**u, **prov.get(u["text"], blank)} for u in edit_units(prompt)]


def edit_feedback(draft: str, current: str, budget: int, deltas: list[dict]) -> dict:
    """길이만 넘은 편집을 한 번 되돌려 보낼 때 붙이는 실측."""
    units = sorted(edit_units(current), key=lambda u: -u["chars"])
    over = len(draft) - budget
    return {"counted_by": "code", "result_total": len(draft), "budget": budget,
            "over_by": over, "over_by_words": -(-over // int(chars_per_word(current))),
            "your_edits": deltas,
            "largest_units": [{"id": u["id"], "chars": u["chars"]} for u in units[:6]]}
