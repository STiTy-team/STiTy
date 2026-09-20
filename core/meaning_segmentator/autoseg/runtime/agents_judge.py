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

from .agents_distill import SECTIONS  # noqa: F401  (재수출)
from .agents_distill import check_skeleton as _check_skeleton

# judge 루프의 골격 — distill 의 여섯 섹션에서 [Decision Procedure] 를 뺐다. 그 절은 [Core
# Principles] 의 두 판단을 순서로 다시 쓴 것에 영어 힌트를 보탠 것이었고(judge09 v0), PE 가 원칙만
# 고치므로 절차의 옛 힌트가 개정과 어긋난 채 남았다. 순위 규칙(다른 정수·넓은 범위)은 코드가 만드는
# [Output Rules] 에 있다.
# `[Order Principles]` 는 **같은 등급 안에서 두 자리 중 어느 쪽을 먼저 택할지**만 담는 칸이다.
# `[Core Principles]` 와 갈라 두는 이유는 `[Scoring Rules]` 의 절차가 두 단계이기 때문이다 —
# 등급을 고르는 단계와 등급 안에서 하나씩 뽑아 비교하는 단계. 원칙 여덟은 전부 단항 질문이라
# 앞 단계용이고, 뒤 단계에 쓸 근거를 주지 않는다. 판별한 것(judge31): 이항 비교문을 원칙 칸에
# 넣은 후보 둘이 −0.0060·−0.0077 이고 깊이 8·13위가 안 움직였다. 칸을 갈라야 골격이 그 문장을
# 뒤 단계에서 읽는다.
# 순서: 골격이 판단 절보다 **먼저** 온다. [Scoring Rules] 가 목표와, 점수를 만드는 절차와, 두
# 판단 절이 각각 무엇을 정하는지를 말하므로 그것을 읽고 나서 판단 절을 읽는 것이 맞다.
JUDGE_SECTIONS = ["[Role]", "[Scoring Rules]", "[Core Principles]", "[Order Principles]",
                  "[Output Rules]", "[Examples]"]


OPTIONAL_SECTIONS = ("[Scoring Rules]", "[Order Principles]")
# 이 칸은 **사람이 골격에 넣을 때만** 생긴다. Writer 는 만들지 않고 PE 도 새로 만들 수 없으므로
# 없는 프롬프트를 틀렸다고 하지 않는다 — 대신 있으면 위치·중복을 검사한다.
ALWAYS_OPTIONAL = ("[Order Principles]",)


# `section_of`·`replace_section` 이 섹션의 끝을 찾을 때 쓰는 경계 목록. **distill 의 `SECTIONS`
# 를 직접 늘리면 안 된다** — 그 루프의 `check_skeleton` 기본값이 곧 그 목록이어서 judge 전용 칸을
# 필수 섹션으로 요구하게 되고 distill 프롬프트가 깨진다. 여기에 없는 헤더는 경계로 인식되지 않아
# **앞 섹션이 그것을 삼킨다** — `[Core Principles]` 를 편집하는 순간 뒤따르는 칸이 통째로 사라진다.
BOUNDARIES = tuple(dict.fromkeys([*SECTIONS, *JUDGE_SECTIONS]))


def replace_section(prompt: str, header: str, body: str) -> str:
    """`header` 섹션을 `body` 로 통째로 바꾼다 (다음 섹션 헤더 직전까지).

    distill 판과 같은 일을 하지만 경계 목록이 `BOUNDARIES` 다."""
    i = prompt.find(header)
    if i < 0:
        return prompt.rstrip() + "\n\n" + body
    nxt = [prompt.find(sec, i + len(header)) for sec in BOUNDARIES if sec != header]
    nxt = [k for k in nxt if k > i]
    end = min(nxt) if nxt else len(prompt)
    return prompt[:i] + body.rstrip() + "\n\n" + prompt[end:].lstrip("\n")


def strip_inline_headers(prompt: str) -> str:
    """본문 **중간**에 쓰인 섹션 헤더의 대괄호를 뗀다 — 참조 의도는 살리고 경계만 없앤다.

    `section_of`/`replace_section` 은 헤더 문자열을 그대로 찾으므로 본문 속 "[Scoring Rules]" 가
    경계로 잡혀 **그 자리에서 섹션이 잘린다.** judge33 의 v0 후보 둘이 [Role] 안에서 골격을
    대괄호째 참조했고("must feed the banding-and-ordering procedure in [Scoring Rules]"), 주입이
    그 문장 중간부터 갈아 끼워 [Role] 이 끊긴 채 골격이 삽입됐다. 골격 검사는 그것을 못 잡는다 —
    문자열이 있으니 "섹션 없음" 이 아니다.

    줄 맨 앞의 헤더는 진짜 섹션이므로 건드리지 않는다."""
    out = prompt
    for h in BOUNDARIES:
        for i in reversed([m.start() for m in re.finditer(re.escape(h), out)]):
            if i > 0 and out[i - 1] != "\n":
                out = out[:i] + h.strip("[]") + " section" + out[i + len(h):]
    return out


def check_skeleton(prompt: str, base: str | None = None) -> list[str]:
    """`base` 를 주면 **그 프롬프트가 가진 섹션만** 요구한다.

    v0 갈래가 `[Scoring Rules]` 를 통째로 뺀 채 시작할 수 있게 하려는 것이다 — 그 섹션은
    측정 절차를 말로 푼 것인데, 이식 실험에서 채점 척도 설명을 뺀 프롬프트가 가장 높았다
    (probe_port p3_no_ladder 0.5738 vs v0 0.5669). 빼고 시작했으면 PE 가 도로 넣는 것은
    `frozen_intact` 이 막는다 (before 가 "" 인데 after 가 차 있으면 변경으로 잡힌다).
    """
    want = []
    for sec in JUDGE_SECTIONS:
        if sec in ALWAYS_OPTIONAL:
            if sec in prompt or (base is not None and sec in base):
                want.append(sec)    # 있으면 위치와 중복을 본다
            continue
        if base is not None and sec in OPTIONAL_SECTIONS and sec not in base:
            continue
        want.append(sec)
    errs = _check_skeleton(prompt, want)
    # `_check_skeleton` 은 distill 의 `SECTIONS` 만 "알려진 섹션" 으로 보므로 judge 전용 칸의
    # 순서·중복은 안 본다. 여기서 본다 — 삼켜지거나 두 번 생긴 칸을 조용히 넘기면 안 된다.
    for sec in want:
        if prompt.count(sec) > 1:
            errs.append(f"섹션 중복: {sec}")
        elif sec not in prompt and sec not in SECTIONS:
            errs.append(f"섹션 없음: {sec}")
    order = [prompt.find(sec) for sec in want if sec in prompt]
    if order != sorted(order):
        errs.append("섹션 순서 어긋남: " + ", ".join(want))
    return errs

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
- "loss_by_bin": per latency bin, how many (sentence, k) pairs the measured set has, the mean gap
  between the offline target and this prompt, and each bin's share of the total loss. This is
  where the prompt is losing, and a finding is worth reporting in proportion to the loss it
  addresses. One structural asymmetry to hold on to: a good cut set in a short bin is NECESSARILY
  dense, because the number of cuts is the sentence length divided by the budget. So a finding
  that tells the model where NOT to cut removes candidates the short bins have to use, and one
  that trades the short bins for the long ones loses on the total.
- "rejected_by_bin" (when revisions earlier in this run were rejected): for each of them, its
  "edits" and the mean H_set change per latency bin it produced. Read this before proposing: when
  the bins show that a direction has already been tried and lost, a reworded version of the same
  direction loses again.
- "rank_depth": for each depth d, how often this prompt's top-d positions coincide with the
  measurement's top-d, as a multiple of what coincidence alone would give — 1.0 means the ordering
  at that depth carries no information. It says which depths are worth a finding at all.
- "misorder_cost": for each depth d, the measured value of the d-th best position minus the mean
  value of the positions still unranked at that point — what one wrong pick at that depth costs.
  If this does not fall as d grows, there is no depth at which the order may be left unresolved.
- "rank_inversions": pairs taken from the depths where the ordering carries least information. In
  each pair this prompt scored A above B while the measurement says B is the better cut; both are
  shown with the words around them. Every position in such a pair is one the prompt already
  declined to rank at the top, so no prohibition separates them — a finding drawn from these pairs
  has to say which of two imperfect positions is the better cut.
- "last_revision" (only after the first measured revision): what the previous revision did and
  how it landed — its "edits", the measured "delta", "by_bin" (mean H_set change per latency bin),
  "n_worse"/"n_better", and a post-mortem ("why", "blamed" units, "lesson"). If it has
  "vs_base", it was built on a near-miss base and "vs_base" is what ITS OWN edit did relative to
  that base ("delta"/"by_bin" include the base's gain). If it was rejected,
  do not ask for the same change again in other words; the bins say where it hurt. But if it
  carries "near_miss": true, its mean was positive on the full set and only the confidence
  bound touched zero — that change helped, "by_bin" says where. Then report what STILL loses in
  those same bins so the next revision can keep the change and go further, not undo it. When
  "base" is present, the prompt you were given IS that revision, so its text can be a target. A
  finding that asks for yet another cut prohibition on top of it has to answer "vs_base": if the
  base already gained in a bin, adding a prohibition that thins that bin gives the gain back. Its
  "edits" text is NOT in the current prompt — a "replace" target must quote the prompt you were given.

Two shapes of finding are possible and they are not interchangeable.

A "check" is UNARY — a condition on one position ("cutting here is bad when ..."). The model
applies it to each marker on its own, so it separates positions that differ on that condition.

An "order" is BINARY — a comparison between two positions ("when neither is clean, prefer ... over
..."). This is the only shape that constrains the order INSIDE a group of positions that a check
has already judged alike. Once several positions all satisfy "do not cut here", no unary check
says which of them to cut first, and the short bins have to cut some of them anyway.

The prompt keeps the two shapes in two sections and the scoring procedure reads them at different
moments: [Core Principles] decides which band a position goes in, and [Order Principles] decides, among
positions of the SAME band, which one to take first. So a "check" belongs in "core_principles" and
an "order" in "order_principles". A comparison written into [Core Principles] is only read while bands
are being chosen, where there is no second position to compare against yet.

An existing line can be traded instead of added to: "action": "replace" with "target" quoting its
opening words. The opening line of [Order Principles] is one of those — it was written by hand before
any of these cases were read and has never been measured on its own, so when the inversions point
at a better comparison, replacing it is as legitimate as adding beside it.

Prefer adding a missing judgement over retuning how strongly an existing one applies: retuning
("relax this prohibition", "reorder these priorities", "widen this scope") gives the model no new
way to tell two positions apart. And before proposing a check, look for it among the units you
were given — if the prompt already makes that judgement, restating it adds length and no
information; turn the finding into an "order" over the positions that judgement has pushed down.
If your finding is "this principle is too strong/too weak", say instead either which construction
the prompt fails to check for, or which of two positions it should prefer.

Your job: name the JUDGEMENT the prompt is getting wrong, not the tokens it fires on. A finding
is worth reporting only if it recurs across cases — one sentence is an anecdote. Write it so a
reader scoring an unseen sentence could apply it: a question to ask, or a condition on meaning.
Keep "edit.text" under 60 words — one judgement, not a paragraph. Never write token lists ("if the previous word is 'the'"), never quote more than 40 characters of
source text, never name this sentence.

Return ONLY JSON, with the two shapes in SEPARATE lists:
{
  "checks": [ {a UNARY finding, fields below} ],
  "orders": [ {a BINARY finding, same fields} ],
  "checks_skipped": "only when "checks" is empty: one sentence on why no unary condition is worth
                  reporting from these cases",
  "orders_skipped": "only when "orders" is empty: same, for comparisons"
}
REPORT AT LEAST ONE IN EACH LIST. The two shapes answer different questions and a downstream step
pairs each list with a different kind of revision, so an empty list means that revision is not
attempted at all this round. If you genuinely have none for a list, leave it empty and say why in
the matching "*_skipped" field — never pad it with a reworded member of the other list. An "order"
is always available in principle: "rank_inversions" gives pairs the ordering got wrong, and every
position in them is one no prohibition separates.

Each member of either list has these fields:
    {"diagnosis": "one sentence: what the prompt mis-judges, in terms of meaning",
     "evidence": "the ids of the cases that show it, then what they share — the count is the
                  length of that list; never state a number of cases you did not list",
     "type_predicate": "ONE sentence naming the sentences this finding applies to, as a condition
                  on the SOURCE text alone: a construction that can be spotted by reading the
                  sentence, with no reference to this prompt, to any candidate, or to any score.
                  It is used to select unseen sentences of the same kind, so it must be decidable
                  ('a numeral or quantifier is immediately followed by its unit or measurement
                  phrase') and NOT a restatement of the damage ('cuts that harm cohesion').
                  Narrow enough that a minority of sentences match: a condition that holds of
                  almost every sentence selects nothing and the finding is dropped. For
                  "kind": "order" it must name a configuration in which BOTH positions of the
                  comparison occur in the same sentence — the comparison is untestable otherwise.",
     "edit": {"where": "core_principles" | "order_principles" | "examples",
              "action": "add" | "replace",
              "target": "first few words of the line to replace (omit when adding)",
              "text": "the new line, or an Input/Output example pair"}}
At most __NFIND__ findings IN TOTAL across both lists. Inside each list, order them by how many
cases they explain."""

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
- "vs_base" (only when the revision was built on an earlier near-miss revision): the same
  "delta", "by_bin", "n_worse"/"n_better" measured against THAT base instead of the adopted
  prompt. Then "delta"/"by_bin" include the base's own gain; the effect of the edits listed in
  "edits" is "vs_base". Attribute to the edits only what "vs_base" shows.
- "rank_depth": for each depth d, how often the prompt's top-d positions coincide with the
  measurement's top-d, as a multiple of what coincidence alone would give (1.0 = no information),
  "base" before the edit and "revision" after, with the "delta". Which depths moved decides who
  the edit could have helped: a long latency budget keeps only the first one or two positions,
  a short one keeps many. So an edit that moved only the first depths cannot have helped the
  short bins whatever "delta" says, and one that moved the deeper ones cannot be dismissed by a
  flat overall mean. Say which of the two happened.

"worst"/"best" are the TAIL — five pairs out of hundreds. Base "why" and "lesson" on "by_bin",
"n_worse"/"n_better" and "rank_depth"; use the tail only to illustrate a movement those already
show. A single spectacular pair must not become the lesson when the counts moved both ways in
comparable numbers.
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
- "units" lists every unit of [Core Principles] (ids C1, C2, ...; one line each), of
  [Order Principles] (ids O1, O2, ...; one line each) and of
  [Examples] (ids E1, E2, ...; one Input/Output pair each), with its exact text, its length
  ("chars") and its evidence:
    "origin"         "v0", or the iteration whose ADOPTED revision introduced this text
    "adopted_delta"  the measured gain of that adoption, "adopted_ci_lo" its lower bound
    "near_miss_delta" (only on units from a "base" revision) the measured mean gain of that
                     revision, "near_miss_ci_lo" its lower bound (touched zero, so not adopted)
                     (null for v0 units — nothing was measured about them one by one)
- Every other section — [Role], [Scoring Rules], [Output Rules] — stays as it is.
  [Scoring Rules] says what was MEASURED (cohesion, contra, target) and the procedure for turning
  that into numbers; neither is yours to change, and code drops an edit that touches either.

Hard constraints:
1. The two principle sections are read at different moments by the scoring procedure and are not
   interchangeable. [Core Principles] holds UNARY judgements — a condition on one position, used
   to put it in a band. [Order Principles] holds BINARY comparisons — which of two positions in the
   SAME band to take first; each unit there must name both sides ("prefer a cut at ... over one
   at ..."). Worked cases belong in [Examples]. Do not add scoring conditions anywhere else.
   A finding's "kind" says which section it belongs in: "check" → [Core Principles],
   "order" → [Order Principles]. Code drops an edit that puts one in the other section, and an edit
   whose text compares two positions when the finding was unary.
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
   adds. When "your_previous_edits" is empty, the prompt itself was just rewritten as a whole
   and is too long: shorten the units you were given by paraphrase and deletion, keeping every
   judgement the findings support.
3. At most 8 examples. An example unit is exactly "Input: ..." then a newline and "Output: ...",
   with <SEG:?> at every candidate position in the input and integers in the output.
4. Never write token lists or punctuation rules. A token condition fires on a handful of
   positions while the measurement is taken at every position; a judgement applies everywhere.
5. Never name or depend on a specific target language pair beyond what the current prompt says.
6. Consult the attempt history: entries with "adopted": false were measured and rejected. An
   entry with "vs_base" was built on the near-miss base: "vs_base" is what its own edit did
   relative to that base — a negative "vs_base" means the edit undid part of the base's gain,
   whatever "delta_mean" says. Do not
   repeat them or minor variants — move in a different direction. Each entry lists its "edits":
   which unit it changed ("was", the unit's text at the time) and into what ("now"). Do not make
   the same change to the same units again, even reworded. "blamed" names the units the
   post-mortem held responsible for the loss and "why" says how — do not bring that text back,
   as a unit or as a "labeled_example" (those are no longer offered).
   EXCEPTION — an entry with "near_miss": true was measured on the full set (see "measured_on"),
   its mean was POSITIVE and only its confidence bound touched zero: the direction is real and
   the effect too small, not wrong. When "base" is present, the units you were given ARE that
   revision (its edit is already in place; such units carry "near_miss_delta"). Keep that edit
   and add ONE further change that extends the same direction in the bins where "by_bin" shows
   the gain came from, or trims the side effect its "why" names. Do not delete or reverse a
   unit with a positive "near_miss_delta". A prohibition stacked on such a base tends to give the
   gain back: the base gained in a bin by making cuts available there, and a new "do not cut"
   removes them again. Say what a good dense cut looks like instead.
7. Every change must be traceable to a finding in the critique. Each finding carries a "kind":
   "check" is a UNARY condition on one position, "order" is a BINARY comparison between two. Write
   what the kind asks for — a comparison rewritten as a condition stops separating the two
   positions it was for, and a condition rewritten as a comparison invents a second position that
   the finding never named.
7b. If "loss_by_bin" is present, it says which latency bins hold the loss ("loss_share"). A
   revision is judged on the mean over ALL pairs, and most pairs are short pieces. An edit that
   only makes the model cut less (more "do not cut when …") lowers those bins by construction: the
   number of cuts they need is set by the budget, so removing candidates cannot help them, it can
   only force a worse one. Prefer edits that say what a GOOD cut looks like where the loss is.
8. If "primary_finding" is present, that finding is yours to address FIRST — the candidates of
   one iteration are each pointed at a different finding so they do not converge on the same
   edit. You may address others after it.
8b. If "sibling_candidates" is present, those revisions were already proposed in this iteration
   and will be measured alongside yours. Propose a DIFFERENT revision: address other findings,
   edit other units, or take a different direction on the same finding. Do not restate a sibling.
9. If "labeled_examples" is present, each entry is a real sentence from the measured set with
   its MEASURED scores already written as an Input/Output pair ("unit"), and the case it came
   from ("id", its "latency_bin", and "gap" = how much the current prompt lost there). To put one
   into [Examples], write an edit with "labeled_example": "<id>" INSTEAD of "text" — code pastes
   the exact pair, e.g. {"op": "replace", "id": "E2", "labeled_example": "<an id from labeled_examples>",
   "kind": "example"} or {"op": "insert_after", "id": "E_end", "labeled_example": "..."}.
   A measured example teaches the ranking directly where a principle only describes it; prefer
   replacing a hand-written example with a measured one over adding another principle.
10. If "constraint" is present, obey it — code enforces it by dropping edits that violate it.
   The prompt has two jobs and every role serves exactly one of them. Know which is yours before
   you write anything:
     BAND ASSIGNMENT — which of the five bands a position goes in. This is decided by the QUESTION
       in each [Core Principles] line. It already works: the top of the ranking is well separated.
       Roles that serve it add or trade a question: "single_small", "narrow_rule", "induce", and
       "replace" when it targets a C unit.
     ORDER INSIDE A BAND — which of two positions that share a band is the better cut. This is
       decided by the SEVERITY that follows the question on the same line: what makes that concern
       worse, what makes it milder. This is where the ranking currently carries almost no
       information, and it is where the measurement is read: the latency budget keeps several
       positions per sentence, so the order among the middle positions decides what ships.
       The role that serves it is "severity".
     EXCEPTIONS — a named configuration where ordering by severity picks the wrong position.
       [Order Principles] holds these, and only these. The role is "fallback".
   Every [Core Principles] line you write or rewrite must have BOTH parts: the question, ending in a
   question mark, then the severity naming both ends. Code checks it and drops the edit otherwise,
   whatever your role is. A question with no severity assigns a band and then leaves the model to
   invent the order inside it.
   The constraints:
   "examples_only": edit only [Examples] units; no [Core Principles] edit at all. Add, replace
   or delete as many as you judge right — there is no cap. An example shows the WHOLE ordering
   of a sentence at once, so it teaches the lower ranks that prose cannot reach; pick the ones
   whose "latency_bin" is short and whose "gap" is large.
   "replace": TRADE one existing line for the finding's judgement, at no net length. Either ONE
   edit with "op": "replace" on the unit you are trading away (same slot, so length stays put), or
   one "delete" plus one "insert_after" when the new line belongs in the other section (a unary
   finding goes to [Core Principles], a binary one to [Order Principles]). Units of BOTH sections are
   fair game, [Order Principles] included — its opening line was written by hand and never measured on
   its own, so replacing it with something the cases support is exactly what this role is for. The
   one limit: [Order Principles] must not end up empty, so its last remaining line can be replaced but
   not deleted. The new line may exceed the old one by at most a few words — code measures it and
   drops the edit otherwise, so pick a unit long enough to pay for what you write. Choose what to delete by
   EVIDENCE, exactly as constraint 2 says: a "v0" unit with no measured gain, or one the current
   critique faults, and never a unit whose "adopted_ci_lo" is positive. Why this role exists:
   adding a line has lost every time it was measured, while deleting a unit that the measurement
   does not depend on came out at zero — if the cost is length rather than content, a trade starts
   from zero instead of below it.
   "severity": RECALIBRATE how strongly one existing Core Principle applies. Exactly one edit, a
   "replace" on a "C" unit, and **everything up to and including the first question mark must come
   back byte-for-byte unchanged** — you rewrite only what follows it. Code compares the two and drops
   the edit on any difference, whitespace included.
   What follows the question is that principle's SEVERITY: what makes a cut at such a position worse,
   and what makes it milder. Say both ends. Name them on the source surface, as configurations that
   can be recognised without the translation ("an explicit negation is the most severe, a narrowing
   qualifier the mildest"), not as degrees of a feeling ("somewhat bad", "quite serious").
   Your finding is a BINARY one: it names two positions and says which was the better cut. Use it as
   evidence about a severity axis, not as a new comparison to write down — ask which principle fired
   on both positions, and what it fails to distinguish between them. Then make that distinction part
   of the principle's severity.
   Why this role exists: the questions decide which BAND a position goes in, and that part works. What
   the prompt has no vocabulary for is how strongly a concern applies, so it cannot order two positions
   that share a band. The scoring procedure nevertheless demands a distinct number for every position,
   down to the lowest band, and the model supplies one — it separates them with no basis to separate
   them. Severity is the basis. Do not write a new prohibition and do not touch the question: a new
   condition only adds another way to say "bad", and "bad" is the thing there is already too much of.
   "fallback": ADD a rule that ORDERS the positions every other principle rejects. Exactly one
   edit, an "insert_after" into [Order Principles] — that section holds the lines that weigh two
   positions against each other, and a rule of this shape is one of them; an edit of this role
   placed in [Core Principles] is dropped. Write it as a COMPARISON on the source surface
   form ("prefer a cut at A over one at B", "as a last resort take C") — not a prohibition
   ("do not cut after X") and not a binding ("keep X with Y"); both of those have been tried.
   A short ladder of tiers counts, as long as it is one unit and states an ORDER.
   Why this role exists: "rank_depth" says how far down the ranking still carries information —
   a value near 1.0 at some depth means the order there is no better than chance — and
   "misorder_cost" says what one wrong pick at that depth costs. Where the first has flattened and
   the second has not, those depths are being paid for and nothing is ordering them. A short
   latency budget keeps many positions per sentence and so reads that part of the order; a long
   one keeps the first position alone. Every principle now in the prompt pushes bad positions
   down; none says which of the pushed-down positions to take. When "rank_inversions" is present,
   each entry is a pair from those depths with the words around both positions — the comparison
   you write has to separate pairs like those.
   "induce": read the MEASURED cases you were given — each shows a sentence, where the current
   prompt cut it, and where the measured target says the cuts should have been — and write ONE
   rule that REPRODUCES those target choices. Work bottom-up: list to yourself what the target
   cuts have in common on the SOURCE SURFACE (a token class, a construction, what sits either
   side of the boundary), then state that commonality as a rule. Exactly one edit, an
   "insert_after" into [Core Principles].
   Do NOT copy a case sentence or its markers into the rule — a rule names the configuration,
   it does not quote an instance. Do NOT write it as a prohibition. Binding ("keep X with the Y
   that completes it") and ordering ("prefer a cut at A over one at B") are both fine; let the
   cases decide which fits.
   Why this role exists: every other role reads the Critic's DIAGNOSIS, which has already
   compressed the cases into one sentence. That compression is where a wrong generalisation
   enters. Here you see the evidence before it was compressed.
   "single_small": exactly ONE edit, and its new text is at most 60 words — a small, precise
   change whose effect can be attributed.
   "narrow_rule": ADD a check, do not retune an existing one. Exactly one edit, it must be an
   "insert_after" into [Core Principles], and its text must state a concrete condition that can
   be recognised in the SOURCE surface form alone (a specific construction, a token class and
   what must stay with it), not a change in how strongly an existing principle applies.
   Write it as a BINDING ("keep X together with the Y that completes it"), not as a prohibition
   ("do not cut after X"): a prohibition removes candidate positions while a binding only moves
   them, and the short budgets need a fixed number of positions either way. And state a condition
   recognisable in the SOURCE surface alone rather than a change in how strongly an existing
   principle applies — reweighting gives the model no new way to tell two positions apart.
   "prune": REMOVE one [Core Principles] unit. Exactly one edit, op "delete", a "C" unit, and
   nothing added anywhere. Pick the principle that earns its place least — one that restates
   another, or that names a condition the measured target does not actually punish. When the
   prompt already carries many principles, one more divides the model's attention more than it
   adds, and removing is the one direction an added rule cannot test.

What the model is judged on: the cut sets its scores produce are translated piece by piece and
scored against the source as a whole, with the worst contradiction risk in the set applied as a
penalty. Only the ORDER of the numbers inside one sentence is ever read.

Containment — this decides the outcome more than the aim does. Your rule must fire ONLY on the
configuration you name. A sentence that does not contain that configuration has to come out
ranked exactly as it ranks now. So do not write wording that reaches every sentence: no
"generally prefer", no "in all cases", no "always", no "tend to", and do not restate or reweight
a principle that is already in the prompt. Name the surface trigger, say what it does at that
trigger, and stop.
Wording that reaches past its trigger routinely damages the sentences WITHOUT the targeted
configuration more than the ones with it, and a revision can be right about its target and still
lose overall because of that leak — the gain sits in a few sentences and the leak is spread over
all of them. A revision that changes nothing outside its trigger cannot lose that way.

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

SCORING_RULES_BODY = """cohesion   the pieces were translated separately into __TARGETS__, joined in order, and a reference-free quality estimator scored how faithfully that joined text renders the WHOLE source sentence.
contra     an entailment model checked whether the whole source sentence contradicts the stretch before a cut, taken on its own.
target = cohesion x (1 - contra)
- Use the target above to judge each marker: cohesion and contra are the two measured components that determine the target. Contra is a graded probability and should influence ranking only through the product; do not convert it into a binary flag or a separate tier that outranks cohesion.
- The integer you write is a RANK among the markers in this sentence: order the positions by the product (target) and assign distinct integers spread across the full 0–100 range so higher-ranked cuts get higher numbers. Do not attempt to derive scores by explicit arithmetic formulas on the product; instead use the product as your measurement to order positions.
- The two judgement sections are not interchangeable, and you read them at different moments. The Core Principles are unary and carry both jobs: the question in each line decides which band a position goes in, and the severity in the same line — what makes that concern worse, what makes it milder — decides the order of positions inside one band. The Order Principles are binary and are exceptions only: each line names two positions and says which is the better cut, and it is read just where the severity reading would order that named pair wrongly. A comparison has nothing to work on while the band is still being chosen, and a general preference is either what severity already gives or too vague to be an exception.
- Work in bands, then order inside each band by repeated selection. First place every position in exactly one of five bands: 90–99 a clean cut, 70–89 acceptable, 40–69 risky, 15–39 bad, 0–14 must not be cut. Choosing the band is what the Core Principles decide. Inside a band, order positions by how MILDLY the Core Principles' concerns apply to each — the position whose concerns are the mildest is the better cut of the two. Ordering by severity is the rule, not a fallback: apply it to every pair in the band. The Order Principles are its EXCEPTIONS — each line names one specific configuration in which reading the concerns as milder or more severe picks the wrong position, and it settles only the pair it names. Outside the configurations they name, order by severity. They never move a position into another band. When severity leaves two positions genuinely equal and no Order Principle names them, compare the two on the target itself and say which is the better cut. Then fill in the numbers band by band, and inside a band do it one position at a time: among the positions of that band you have not numbered yet, choose the one that is the best cut of them, give it the highest number still free in that band's range, drop it from consideration and choose again among the rest. Every position ends up with its own number, the lowest bands included — a cut that deep is still a choice between a worse and a better place, so do not fill the bottom in without comparing."""


def scoring_rules(targets: list[str]) -> str:
    """`[Scoring Rules]` 섹션 전문 — **사람이 정하고 코드가 주입하는 골격이다.**

    Writer 에게 맡기지 않는 이유: 이 절은 측정 설명뿐 아니라 **점수를 만드는 절차**를 담는다.
    선언형("결과가 순위여야 한다")과 절차형(등급을 고르고 등급 안에서 하나씩 뽑아 비교한다)의
    차이가 test 560 에서 3벌씩 재어 +0.0084 였고, Writer 의 지시문은 선언형만 안다. v0 를
    생성하는 런에서도 이 절을 덮어써 골격을 고정한다 — `[Output Rules]` 와 같은 방식이다.

    **루프는 이 절을 고치지 않는다.** `UNIT_TAGS` 에 없어 어떤 편집도 닿지 못하고, Writer 가
    프롬프트를 통째로 다시 쓰는 길(`rewrite`)에서는 `FROZEN` 이 원본으로 되돌린다. judge34 에서
    `procedure` 역할로 열어 실측했는데 이득이 없었다 — 절차 줄을 고친 두 후보가 −0.0028 / −0.0037
    이다. 같은 발견을 판단 줄에 넣은 것(−0.0080 / −0.0157)보다는 나아서 **자리에 뜻은 있지만**
    그것만으로는 부호가 안 바뀐다. 사람이 바꿀 때만 바뀌고, 바꾸면 그 자체를 `--score-only` 로
    다시 재야 한다.
    """
    return "[Scoring Rules]\n" + SCORING_RULES_BODY.replace("__TARGETS__", ", ".join(targets))


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
  [Role], [Scoring Rules], [Core Principles], [Order Principles], [Output Rules], [Examples]
  No other section — in particular no [Decision Procedure]: the procedure IS the judgements in
  [Core Principles] and [Order Principles], applied at every marker; do not restate them as steps.
- NEVER write a bracketed section header inside the body of a section. Write "the Scoring Rules
  section", not the bracketed form. The bracketed string is how code finds where one section ends,
  so one sitting in a body splits that section at that spot.
- [Output Rules] and [Scoring Rules] MUST be copied verbatim from the blocks given to you. The
  Scoring Rules block already states the measurement and the procedure that turns judgements into
  numbers: every position goes into one of five bands, then the numbers inside a band are filled
  one at a time by repeatedly picking the best of the positions left. Write the two judgement
  sections so that they feed that procedure.
- [Core Principles] decides WHICH BAND a position goes in: 6-10 lines, each starting with "- ",
  each ONE judgement of at most 60 words. Every line is UNARY — a question asked about one
  position on its own ("does what follows the marker overturn what came before?"), so that reading
  it at a marker tells you which band that marker belongs in. One line is later one editable unit,
  so a 1,000-character line cannot be revised without rewriting it.
  **Each line has two parts.** First the question, ending in a question mark. Then one or two
  sentences saying how STRONGLY the concern applies: what makes a cut at such a position worse, and
  what makes it milder. Say both ends, and name them as configurations recognisable on the source
  surface ("an explicit negation is the most severe, a narrowing qualifier the mildest"), never as
  degrees of a feeling ("somewhat bad", "fairly serious").
  The question decides the band; the severity decides the order INSIDE a band. A question with no
  severity leaves the model separating positions of one band with nothing to separate them by, and
  the scoring procedure demands a distinct number for every position, so it will invent the
  difference. Write the severity for every line. Two of the judgements carry the
  measurement: whether what follows overturns the stretch already heard, and whether translating
  the two sides apart still adds up to the source. Say what damage looks like in THIS source language, using what the
  profile tells you about how it builds clauses, where it puts negation and heads, and what it
  leaves implicit. Name the language's own devices; do not name individual tokens as triggers.
- **Do NOT write surface-form rules** — no lists of function words, no punctuation rules, no
  "if the previous token is X". A token condition fires on a handful of positions and says
  nothing about the rest, while the measurement is taken at every position; a judgement applies
  everywhere. Grammar labels are not the criterion either: a cut between two complete clauses
  can measure badly, and a cut inside a phrase can measure well.
- [Order Principles] holds the EXCEPTIONS to ordering by severity: 2-4 lines, each starting with
  "- ", each at most 60 words. Inside a band the order is already decided by how mildly each
  position's concerns apply, so do not restate that — a line here earns its place only by naming a
  specific configuration in which that reading picks the WRONG position, and it settles only the
  pair it names. Every line is BINARY: it names two positions and says which is the better cut
  ("when a cut falls between X and its Y and another falls earlier in the clause, prefer ...").
  Write few and write them narrow. A general preference ("prefer the later marker", "prefer the one
  whose preceding stretch stands alone") belongs nowhere: as a rule it is what severity already
  gives, and as an exception it is not specific enough to be one. A unary condition belongs in
  Core Principles as part of that concern's severity, not here.
  Anchor the comparisons in what the profile says about how this language builds clauses.
- [Scoring Rules] is given to you — copy it verbatim. It is the ONLY place that states how the
  target was measured, and [Output Rules] points to it.
- [Examples]: paste the MEASURED Input/Output pairs given to you, verbatim and nothing else.
  Their numbers are the measured ranking; never invent scores or examples of your own.
- __SPACING__
- Keep the whole prompt under 9500 characters. The two blocks given to you already take about
  3,000 of that; the rest is your two judgement sections, [Role] and the examples.

Return ONLY the prompt text. No commentary, no code fences."""


def writer_system(spaced: bool, targets: list[str]) -> str:
    unit = "words" if spaced else "characters"
    return (WRITER_SYSTEM
            .replace("__TARGETS__", ", ".join(targets))
            .replace("__SPACING__",
                     f"The source is written in {unit}; a cut position sits between two {unit}."))


ALLOWED_WHERE = {"core_principles": "[Core Principles]", "order_principles": "[Order Principles]",
                 "examples": "[Examples]"}
FROZEN = ("[Output Rules]", "[Scoring Rules]")


def critic_system(findings_max: int = 3) -> str:
    return CRITIC_SYSTEM.replace("__NFIND__", str(findings_max))


def postmortem_system() -> str:
    return POSTMORTEM_SYSTEM


def engineer_system(budget: int, cur_len: int) -> str:
    return (ENGINEER_SYSTEM.replace("__BUDGET__", str(budget))
            .replace("__CURLEN__", str(cur_len)))


# 문면이 두 자리를 비교하는지 가르는 **좁은** 표지. `ORDERING_MARKERS` 는 `fallback` 역할의
# 통과 기준이라 "before " 처럼 넓은 말을 담는데, 재판정에 그걸 쓰면 "do not cut immediately
# before X" 같은 단항 금지문이 이항으로 승격된다(judge31 이터 1 의 finding 이 그 꼴이었다).
BINARY_MARKERS = ("prefer", "rather than", " over ", "ahead of", "higher than", "lower than",
                  "last resort", "least bad", "better than", "worse than", "which of the two",
                  "take the later", "take the earlier", "choose the later", "choose the earlier")


def looks_binary(text: str) -> bool:
    """문면이 두 자리를 견주는가 — `kind` 재판정에 쓴다."""
    return any(m in " ".join((text or "").lower().split()) for m in BINARY_MARKERS)


def clean_findings(blob: dict, prompt: str | None = None, cap: int = 3) -> list[dict]:
    """Critic 출력에서 쓸 수 있는 finding 만 남긴다. 토큰 조건은 여기서 잘라낸다.

    `prompt` 를 주면 replace 대상이 현재 프롬프트에 있는 것만 남긴다 — Critic 은 `last_revision`
    의 편집 본문을 현재 프롬프트에 있는 줄 알고 겨눈다(judge10 iter 2: 기각된 개정이 넣었던
    Whistler 예시를 바꾸라고 냈다)."""
    # 두 배열(`checks`/`orders`)이 들어오면 배열이 형태를 정한다. 옛 단일 `findings` 배열도
    # 계속 읽는다 — 앞선 런의 출력을 다시 돌릴 때를 위해서다.
    src: list[dict] = []
    for key, k in (("orders", "order"), ("checks", "check")):
        for f in (blob or {}).get(key, []) or []:
            if isinstance(f, dict):
                src.append({**f, "kind": k})
    src += [f for f in ((blob or {}).get("findings", []) or []) if isinstance(f, dict)]
    out = []
    for f in src:
        if not isinstance(f, dict):
            continue
        edit = f.get("edit") or {}
        if edit.get("where") not in ALLOWED_WHERE:
            continue
        if not (edit.get("text") or "").strip():
            continue
        if edit.get("action") == "replace" and not (edit.get("target") or "").strip():
            continue
        if (edit.get("action") == "replace" and prompt is not None
                and " ".join(str(edit["target"]).split()) not in " ".join(prompt.split())):
            continue
        # `kind` 는 발견의 형태다 — "check" 는 한 자리에 대한 단항 조건, "order" 는 두 자리를
        # 가르는 비교. 역할 배분이 이것으로 갈린다(단항 조건을 선호문 역할에 넣으면 내용이
        # 납작해지고, 비교를 결속문 역할에 넣으면 비교가 사라진다). 값이 없으면 종전과 같은
        # 단항으로 본다.
        kind = str(f.get("kind") or "check").strip().lower()
        if kind not in ("check", "order"):
            kind = "check"
        # **라벨은 문면으로 다시 판정한다 — Critic 이 스스로 붙이는 값을 믿지 않는다.** 이 값으로
        # 역할 배분과 편집 칸이 갈리는데, 뒤에서 PE 가 형태를 바꿔 쓰면 라벨과 실제가 어긋난다
        # (judge31 이터 1·2: PE 가 단항 금지문을 서열문으로 고쳐 써 `check` 후보가 이항 편집이
        # 됐다). 그래서 형태는 여기서 확정하고 PE 에는 문면을 그대로 넘긴다.
        shaped = "order" if looks_binary(edit.get("text")) else "check"
        relabeled = shaped != kind
        kind = shaped
        where = edit.get("where")
        # 이항은 `[Order Principles]` 칸으로 보낸다. 그 섹션이 없는 프롬프트(옛 v0)로 도는 런에서는
        # 보낼 곳이 없으므로 원칙 칸에 그대로 둔다 — `kind` 는 살려서 역할 배분은 유지한다.
        if kind == "order" and where == "core_principles" and (
                prompt is None or "[Order Principles]" in prompt):
            where = "order_principles"
        elif kind == "check" and where == "order_principles":
            where = "core_principles"
        rec = {"kind": kind,
               "diagnosis": (f.get("diagnosis") or "").strip(),
               "evidence": (f.get("evidence") or "").strip(),
               "type_predicate": (f.get("type_predicate") or "").strip(),
               "edit": {**{k: edit.get(k) for k in ("action", "target", "text")}, "where": where}}
        if relabeled:
            rec["kind_relabeled"] = True
        out.append(rec)
    # **cap 을 종류별로 번갈아 적용한다.** 앞에서 자르면 `order` 가 사라질 수 있고, 그러면
    # 이항 편집을 요구하는 역할이 짝을 못 찾아 아예 돌지 않는다(judge31 이터 1: finding 이
    # `check` 하나뿐이어서 후보가 9개 계획에서 2개로 줄었다).
    orders = [f for f in out if f["kind"] == "order"]
    checks = [f for f in out if f["kind"] == "check"]
    woven: list[dict] = []
    while (orders or checks) and len(woven) < cap:
        if orders:
            woven.append(orders.pop(0))
        if checks and len(woven) < cap:
            woven.append(checks.pop(0))
    return woven


def frozen_intact(before: str, after: str, allow: tuple[str, ...] = ()) -> list[str]:
    """PE 가 건드리면 안 되는 섹션이 그대로인지. 다르면 그 섹션 이름을 돌려준다.

    `allow` 는 **이 역할에만** 열어 주는 섹션이다. `procedure` 역할이 `[Scoring Rules]` 의 절차
    줄을 다시 쓰는데, 그 검사를 역할과 무관하게 걸면 편집이 `enforce_role` 을 통과한 뒤 여기서
    다시 죽는다 — judge34 iter 1 에서 실제로 그랬다("동결 섹션이 바뀌었다: [Scoring Rules]").
    **어느 역할에 무엇이 열려 있는지는 한 군데서만 정해야 한다**(`enforce_role`). 여기서는 그
    결정을 받아 적용만 한다."""
    bad = []
    for header in FROZEN:
        if header in allow:
            continue
        if section_of(before, header) != section_of(after, header):
            bad.append(header)
    return bad


def section_of(prompt: str, header: str) -> str:
    i = prompt.find(header)
    if i < 0:
        return ""
    nxt = [prompt.find(sec, i + len(header)) for sec in BOUNDARIES if sec != header]
    nxt = [k for k in nxt if k > i]
    return prompt[i:min(nxt)].strip() if nxt else prompt[i:].strip()


def parse_prompt(blob: dict, current: str, budget: int,
                 allow_frozen: tuple[str, ...] = ()) -> tuple[str | None, list[str]]:
    """PE 출력 검증 — 통과하면 (프롬프트, 변경목록), 아니면 (None, 사유)."""
    errs: list[str] = []
    pr = (blob or {}).get("prompt") or ""
    if not pr.strip():
        return None, ["prompt 가 비었다"]
    errs += check_skeleton(pr, base=current)
    errs += [f"동결 섹션이 바뀌었다: {h}" for h in frozen_intact(current, pr, allow_frozen)]
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
    return {h: len(section_of(prompt, h)) for h in JUDGE_SECTIONS}


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

# **`[Scoring Rules]` 는 여기 없다.** 여기 없는 섹션은 `apply_edits` 가 재조립하지 않으므로 어떤
# 편집도 닿지 못한다 — 그게 이 절을 지키는 방법이다. judge34 에서 `procedure` 역할로 열어 실측했고
# 이득이 없었다(−0.0028 / −0.0037). 같은 발견을 판단 줄에 넣은 것(−0.0080 / −0.0157)보다는 나았지만
# 둘 다 음수다. 사람이 고치고 루프는 그 위에서 돈다.
UNIT_TAGS = {"[Core Principles]": "C", "[Order Principles]": "O", "[Examples]": "E"}
# 단위 id 의 머리글자 집합 — 정규식 문자 클래스로 쓴다. 칸을 늘릴 때 여기 하드코딩된 글자를
# 놓치면 새 칸의 `_end` 앵커가 "없는 id" 로 반려된다.
TAG_CLASS = "".join(UNIT_TAGS.values())
TAG_HEADER = {v: k for k, v in UNIT_TAGS.items()}
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


def ungraded_principles(prompt: str) -> list[str]:
    """정도 절이 없는 `[Core Principles]` 단위의 id. 없으면 빈 목록.

    **정도 절이 빠지면 이득이 조용히 사라진다.** 질문은 어느 등급으로 보낼지를 정하고 그쪽은 이미
    작동한다(순위 깊이 배수 1위 6.16배). 없는 것은 등급 **안** 서열이고(8위 1.27배), 골격은 맨 아래
    등급까지 서로 다른 번호를 요구하므로 모델은 근거 없이 구별을 발명한다 — 캐시된 출력에서 아래
    등급도 99.7%가 다른 번호를 받는다. 정도 절을 손으로 붙인 판이 홀드아웃 560문장에서
    **+0.0066 [+0.0018, +0.0115]** 였고 이득이 자리를 여러 개 고르는 구간에 몰렸다.

    Writer 가 그 절을 빼먹으면 프롬프트는 멀쩡해 보이고 골격 검사도 통과한다. 그래서 따로 센다.
    경계는 첫 물음표다 — `severity` 역할이 쓰는 것과 같아서 한 군데에서만 정의된다. 그런데
    **물음표 뒤에 글이 있는 것만으로는 모자란다.** combo 의 원칙 여덟은 뒤에 문장이 있는데 방향만
    말한다("such outcomes increase contradiction risk", "cohesion is reduced") — 어느 쪽으로 미는지는
    알려주지만 **얼마나** 인지, 무엇이 더 심하고 무엇이 더 가벼운지는 말하지 않는다. 그러면 등급 안
    두 자리를 여전히 못 가른다. `WRITER_SYSTEM` 이 요구하는 것은 **양쪽 끝**이므로 그것을 센다:
    심한 쪽 말(worse / worst / severe)과 가벼운 쪽 말(milder / mildest / mild / least / less /
    better / best)이 둘 다 있어야 한다. 한쪽만 있으면 축이 아니라 방향이다."""
    return [u["id"] for u in edit_units(prompt)
            if u["section"] == "[Core Principles]" and not has_severity(u["text"])]


# 정도 축의 두 끝을 가리키는 말. **실제 생성물에서 뽑았다** — 손으로 쓴 일곱 절과 Writer 가 낸
# 스물네 절(후보 셋)에서 쓰인 낱말을 세어 맞췄다. 처음에 `worse`/`severe` 만 두었더니 Writer 가 쓴
# "most damaging … least", "Most harmful … milder" 네 절을 **정상인데 거부**했다. 차단 검사에서
# 오탐은 후보를 조용히 잃는 것이므로 목록은 넉넉해야 한다.
WORSE_WORDS = ("worse", "worst", "severe", "severest", "damaging", "harmful", "harmfully",
               "dangerous", "risky", "riskier", "riskiest", "costly", "serious", "destructive")
MILDER_WORDS = ("milder", "mildest", "mild", "least", "less", "better", "best", "safer", "safest",
                "harmless", "benign", "tolerable", "acceptable", "minor", "negligible")


def has_severity(text: str) -> bool:
    """원칙 한 줄이 **정도 축**을 들고 있는가 — 질문 뒤에 심한 쪽과 가벼운 쪽이 둘 다 있는가.

    **한 군데에서만 정한다.** `ungraded_principles`(v0 검사), `severity` 역할, 그리고 C 에 줄을
    넣거나 바꾸는 모든 편집이 이 함수를 쓴다. 같은 것을 여러 군데서 다르게 판정하면 한쪽을 통과한
    편집이 다른 쪽에서 죽거나, 더 나쁘게는 조용히 새어 나간다 — 밤새 그 버그를 두 번 겪었다.

    방향만으로는 안 된다. combo 의 원칙 여덟은 물음표 뒤에 문장이 있지만 어느 쪽으로 미는지만
    말한다("such outcomes increase contradiction risk"). 얼마나인지를 말하지 않으면 등급 안 두
    자리를 여전히 못 가른다."""
    t = " ".join(str(text or "").split())
    i = t.find("?")
    if i < 0:
        return False
    rest = t[i + 1:].strip().lower()
    return (any(w in rest for w in WORSE_WORDS) and any(w in rest for w in MILDER_WORDS))


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
        m = re.fullmatch(rf"([{TAG_CLASS}])(?:_(?:end|last)|(\d+))", uid or "")
        if not m or m.group(1) not in last:
            return None
        tag, num = m.group(1), m.group(2)
        if TAG_HEADER.get(tag, "\0") not in prompt:
            return None     # 그 칸이 이 프롬프트에 없다 (비어 있는 것과 다르다 — 빈 칸에는 넣는다)
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
        text = re.sub(rf"^\s*[{TAG_CLASS}]\d+\s*[:.)\-–]\s*", "", text)
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
            # 본문에 섹션 헤더 문자열이 들어가면 그것이 경계로 잡혀 **그 자리에서 섹션이
            # 잘린다** — `section_of` 는 헤더를 그대로 찾는다. judge 전용 칸까지 막아야 한다.
            if any(h in text for h in BOUNDARIES):
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
        if header not in prompt:
            continue        # 없는 칸을 편집이 새로 만들지 않는다 — 골격은 사람이 정한다
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


ROLES = ("free", "examples_only", "single_small", "narrow_rule", "prune", "rewrite",
         "fallback", "induce", "replace", "severity")
# `severity` 가 정도 절을 다시 쓸 때 허용하는 순증감 폭. 지금 정도 절이 116~179자라 그 안에서
# 다시 쓰고 한두 문장 덧붙일 여지를 준다. 질문 쪽은 한 글자도 못 바꾸므로 길이가 새지 않는다.
SEVERITY_SLACK = 200
# 역할별로 열어 주는 동결 섹션. 지금은 **비어 있다** — `[Output Rules]` 와 `[Scoring Rules]` 는
# 어떤 역할에도 열리지 않는다. 호출부가 `unfreezes(role)` 를 그대로 넘기므로, 다시 열어야 할 때
# 여기 한 줄만 고치면 된다. **어느 역할에 무엇이 열려 있는지는 한 군데서만 정한다** — 두 군데서
# 정했다가 `enforce_role` 을 통과한 편집이 `frozen_intact` 에서 다시 죽는 일을 겪었다.
ROLE_UNFREEZES: dict[str, tuple[str, ...]] = {}


def unfreezes(role: str) -> tuple[str, ...]:
    return ROLE_UNFREEZES.get(role, ())
# `replace` 역할이 허용하는 **순증가** 상한. 0 으로 두면 PE 가 지운 것보다 한 글자라도 길게 쓰면
# 거부돼 재시도만 태운다. 원칙 하나가 200~300자이고 Critic 의 문면이 60단어 이하(350자 안팎)라
# 한 줄을 지우고 비슷한 한 줄을 넣는 폭을 준다.
REPLACE_SLACK = 80
SHORTEN_ROLE = "shorten"     # 후보 역할이 아니라 축소 패스 전용 — insert 를 뺀다


CLASSIFY_SYSTEM = """You label sentences by whether one stated structural condition holds in them.

You are given a CONDITION about the surface form of a sentence, and a numbered list of sentences.
Return the numbers of the sentences in which the condition holds.

Rules:
- Judge the SOURCE text only. Nothing about translation, segmentation, cut positions or quality
  enters this decision; if the condition mentions a cut, read it as "the sentence contains the
  construction the cut would break".
- Be strict. Include a sentence only if the construction is actually present, not if it could be
  argued for. A label set that covers almost every sentence is useless downstream.
- Do not explain, do not re-word the condition, do not add sentences that "almost" match.

Return ONLY JSON: {"matching": [3, 7, 11]}"""


NARROW_SYSTEM = """A condition you wrote to select sentences matched too many of them, so it
selects nothing useful: a set that covers most sentences cannot isolate the failure the finding
is about. Write a NARROWER condition for the same finding — name the specific construction, not
the general category it belongs to (e.g. not "a modifier follows a noun" but "a post-nominal
modifier identifies which entity the head noun refers to"). Same rules as before: source form
only, decidable by reading the sentence, no mention of cuts, prompts, candidates or scores.

Return ONLY JSON: {"type_predicate": "..."}"""


def classify_user(predicate: str, sents: list, ids: list[int]) -> str:
    """분류 호출의 user 메시지 — 술어 + 번호 매긴 문장들."""
    lines = [f"{i}. {sents[i].text}" for i in ids]
    return json.dumps({"condition": predicate, "sentences": lines}, ensure_ascii=False)


PROHIBITION_OPENERS = ("do not cut", "don't cut", "never cut", "avoid cutting", "do not split",
                       "never split", "avoid splitting", "do not place a cut", "no cut")
BINDING_MARKERS = ("keep ", "hold ", "travel together", "stay together", "same piece",
                   "together with", "remain with")
# `fallback` 이 서열문인지 가르는 표지. 금지(자리를 지운다)도 결속(자리를 옮긴다)도 아니고
# **남은 자리들 사이의 순서**를 말해야 한다.
ORDERING_MARKERS = ("prefer", "rather than", " over ", "before ", "ahead of", "closer to",
                    "higher than", "lower than", "last resort", "least bad")


def is_ordering(text: str) -> bool:
    """서열문인가 — 자리끼리 비교하는 말이 들어 있으면 참."""
    return any(m in " ".join((text or "").lower().split()) for m in ORDERING_MARKERS)


def is_prohibition(text: str) -> bool:
    """금지문인가 — 금지로 시작하면서 결속절이 하나도 없으면 참.

    judge13·15·16 의 채택본 셋은 전부 "무엇을 함께 유지하라" 를 적었고, "어디서 자르지 마라" 로만
    쓴 개정은 다섯 번 모두 기각되며 ≤3·≤5 구간을 먼저 떨어뜨렸다. 금지는 절단을 성기게 만들고
    결속은 절단을 옮기기만 한다."""
    t = " ".join((text or "").lower().split())
    t = t.lstrip("- ").lstrip()
    if not t.startswith(PROHIBITION_OPENERS):
        return False
    return not any(m in t for m in BINDING_MARKERS)


def check_core_unit_shape(edits, role: str) -> tuple[list, list[dict]]:
    """`[Core Principles]` 에 줄을 넣거나 바꾸는 편집은 **질문 + 정도 축** 두 부분이어야 한다.

    **역할과 무관하게 건다.** 지시문만으로는 새어 나간다 — `narrow_rule` 은 "금지문 말고 결속문" 을
    세 이터 연속 어겼다. 그리고 질문만 있는 줄이 들어가면 프롬프트는 멀쩡해 보이고 골격 검사도
    통과하지만, 그 원칙은 등급 배정만 하고 등급 안 서열에는 기여하지 못한다 — 정확히 방금 메운
    구멍을 다시 뚫는 것이고, **점수에 안 잡히는 손실**이다.

    `delete` 와 `labeled_example` 은 문면을 만들지 않으므로 건드리지 않는다. `E`·`O` 단위도 아니다."""
    keep, bad = [], []
    for n, e in enumerate(edits or []):
        uid = str(e.get("id") or "")
        text = str(e.get("text") or "")
        touches_c = uid.startswith("C") and e.get("op") in ("insert_after", "replace")
        if touches_c and text.strip() and not e.get("labeled_example") and not has_severity(text):
            bad.append({"edit": n, "id": uid,
                        "reason": f"{role} 인데 원칙 줄이 두 부분이 아니다 — 질문(물음표로 끝)과 그 뒤의 "
                                  "정도 축을 둘 다 쓸 것. 축은 **양쪽 끝**을 말해야 한다: 심한 쪽을 "
                                  f"{'/'.join(WORSE_WORDS[:4])} 류로, 가벼운 쪽을 "
                                  f"{'/'.join(MILDER_WORDS[:4])} 류로 이름 붙인다. 질문만 있거나 한쪽만 "
                                  "있는 원칙은 등급 배정만 하고 등급 안 서열에는 아무것도 주지 못한다"})
            continue
        keep.append(e)
    return keep, bad


def enforce_role(edits, role: str, spent: dict | None = None) -> tuple[list, list[dict]]:
    """후보 역할별 제약을 코드로 건다. 어기는 편집은 건너뛴다.

    같은 이터에 보폭이 다른 후보를 섞기 위해서다 — 원칙을 통째로 갈아끼운 후보(judge05~08 의
    기본형)는 매번 전 구간을 흔들었고, 어느 편집이 무엇을 바꿨는지도 남지 않았다.

    `spent` 는 `{"deleted": {본문 앞부분, ...}}` — **이미 시도한 삭제를 막는다.** `prune` 은
    judge23·24 에서 일곱 번 전부 같은 원칙을 지웠다. finding 을 갈라 줘도, 채택으로 프롬프트가
    바뀐 뒤에도 그랬다 — 기각되면 프롬프트가 안 바뀌고 프롬프트가 같으면 "가장 근거 약한 원칙"
    도 같다. `sibling_candidates` 로 보여주기만 하면 무시하므로 코드로 거부해야 하고, 그러면
    재시도 경로(`engineer:role_retry`)가 사유를 주고 다시 부른다. **id 로 추적하면 안 된다** —
    id 는 이터마다 다시 매겨져 `C2` 가 매번 다른 원칙을 가리킨다. 본문으로 본다."""
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    if role == SHORTEN_ROLE:
        # 줄이라는 패스에서 PE 가 예시를 더 넣었다(judge10 iter 2: E2·E4 삭제 −1596 에 E_end +262).
        keep = [e for e in edits if e.get("op") != "insert_after"]
        bad = [{"edit": n, "id": e.get("id"), "reason": "shorten 패스인데 insert"}
               for n, e in enumerate(edits) if e.get("op") == "insert_after"]
        return keep, bad
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
    if role == "prune":
        # **이미 시도한 삭제는 막는다.** 위 `spent` 주석 참고 — 일곱 번 연속 같은 원칙을 지웠다.
        gone = set((spent or {}).get("deleted") or ())
        if gone:
            bad0 = []
            keep0 = []
            for n, e in enumerate(edits):
                was = str(e.get("_was") or "")[:80]
                if was and any(was[:60] and was[:60] == g[:60] for g in gone):
                    bad0.append({"edit": n, "id": e.get("id"),
                                 "reason": "prune 인데 이미 시도한 원칙을 또 지운다 — "
                                           "다른 단위를 고를 것"})
                else:
                    keep0.append(e)
            if bad0:
                edits = keep0
                if not edits:
                    return [], bad0
        # 빼기 — 더하기만 스물넷 연속 음수였으므로 반대 방향을 시험한다. 한 건, 삭제, 원칙 단위만.
        bad = [{"edit": n, "id": e.get("id"), "reason": "prune 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        if keep and keep[0].get("op") != "delete":
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": f"prune 인데 {keep[0].get('op')} — delete 만 된다"})
            keep = []
        elif keep and not str(keep[0].get("id") or "").startswith(("C", "O")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "prune 인데 [Core Principles]·[Order Principles] 단위가 아니다"})
            keep = []
        elif (keep and str(keep[0].get("id") or "").startswith("O")
              and (spent or {}).get("order_units", 99) <= 1):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "prune 인데 [Order Principles] 의 마지막 한 줄을 지운다 — 그 칸이 "
                                  "비면 골격이 참조할 것이 없다"})
            keep = []
        return keep, bad
    if role == "induce":
        # **오라클 절단에서 규칙을 귀납하게 한다.** 다른 역할은 Critic 의 진단을 읽고 규칙을 쓰는데,
        # 그 진단이 이미 사례를 한 문장으로 압축한 것이다. 여기서는 압축 전 자료(어디를 잘라야
        # 했는지)를 직접 보고 공통점을 찾게 한다 — 진단 단계의 오류가 안 섞인다.
        # 형식은 결속이든 순서든 자유다(귀납 결과가 정하게 둔다). 금지문만 막는다 — judge13~16
        # 에서 금지로 쓴 개정은 다섯 번 모두 기각됐다.
        bad = [{"edit": n, "id": e.get("id"), "reason": "induce 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        if keep and keep[0].get("op") != "insert_after":
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": f"induce 인데 {keep[0].get('op')} — insert_after 만 된다"})
            keep = []
        elif keep and not str(keep[0].get("id") or "").startswith("C"):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "induce 인데 [Core Principles] 단위가 아니다"})
            keep = []
        elif keep and is_prohibition(str(keep[0].get("text") or "")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "induce 인데 금지문이다 — 무엇이 함께 가야 하는지, "
                                  "또는 어느 자리를 먼저 택할지로 쓸 것"})
            keep = []
        elif keep and "<SEG:" in str(keep[0].get("text") or ""):
            # 예시를 그대로 옮겨 적으면 규칙이 아니다 — 그건 examples_only 의 일이다.
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "induce 인데 예시 문장을 그대로 넣었다 — 규칙으로 일반화할 것"})
            keep = []
        return keep, bad
    if role == "fallback":
        # **밀려난 자리들끼리의 서열**을 적게 한다. 지금 원칙 여덟은 전부 밀어내기만 해서
        # 순위 아래쪽이 비어 있다 — 실측으로 1위는 무작위 대비 6.5배인데 8위는 1.2배,
        # 13위는 1.00배다(`diag/rank_depth.json`). 짧은 지연은 그 무작위 구간을 쓴다
        # (≤3 의 평균 절단 수 8.5). 금지도 결속도 그 구간을 안 건드린다.
        bad = [{"edit": n, "id": e.get("id"), "reason": "fallback 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        if keep and keep[0].get("op") != "insert_after":
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": f"fallback 인데 {keep[0].get('op')} — insert_after 만 된다"})
            keep = []
        elif keep and not str(keep[0].get("id") or "").startswith(("C", "O")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "fallback 인데 [Core Principles]·[Order Principles] 단위가 아니다"})
            keep = []
        elif keep and is_prohibition(str(keep[0].get("text") or "")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "fallback 인데 금지문이다 — 무엇을 차선으로 고를지로 쓸 것"})
            keep = []
        elif keep and not is_ordering(str(keep[0].get("text") or "")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "fallback 인데 자리끼리 비교하는 말이 없다 "
                                  "(prefer / rather than / over / last resort 류)"})
            keep = []
        return keep, bad
    if role == "severity":
        # **등급 안 서열을 움직이는 유일한 역할이다.**
        #
        # 실측이 이렇게 몰아붙였다. `[Core Principles]` 의 질문 일곱은 전부 예/아니오이고 전부 한
        # 방향이라 등급 **배정**은 하지만 등급 **안** 서열은 못 만든다. 그런데 골격은 맨 아래 등급까지
        # 서로 다른 번호를 요구하고, 모델은 그것을 지킨다 — 캐시된 출력에서 아래 등급도 99.7%가
        # 다른 번호를 받고 등급 폭을 77% 쓴다. 근거 없이 지키는 것이다(순위 깊이 배수 8위 1.27배).
        # 질문에 "무엇이 더 심하고 무엇이 더 가벼운가" 를 붙이니 dev 500 3벌에서 **+0.0076
        # [+0.0028, +0.0127]** 이고 이득이 ≤3 구간에 +0.0143 으로 몰렸다.
        #
        # **질문은 못 고친다.** 질문이 등급 배정을 정하고 그쪽은 이미 작동한다(1위 6.16배). 질문까지
        # 열면 무엇이 점수를 움직였는지 갈라지지 않고, 잘 되는 쪽을 망칠 여지만 생긴다. 첫 물음표
        # 까지가 질문이고 **한 글자도 달라지면 거부한다** — 공백 정규화도 하지 않는다.
        #
        # 왜 `[Order Principles]` 가 아니라 여기인가: 그 칸의 계약은 골격 문장 그대로 "overrides that
        # default for the configuration it names" 다. 정도 기반 서열이 이득을 내는데 그것을 덮어쓰면
        # 이득을 깎는다 — 같은 편집이 v0 위에서 −0.0015, graded 위에서 −0.0053 이었다.
        bad = [{"edit": n, "id": e.get("id"), "reason": "severity 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        e = keep[0] if keep else None
        uid = str((e or {}).get("id") or "")
        text = str((e or {}).get("text") or "")
        was = str((e or {}).get("_was") or "")

        def question(t: str) -> str | None:
            i = t.find("?")
            return t[:i + 1] if i >= 0 else None

        qw, qt = question(was), question(text)
        if keep and e.get("op") != "replace":
            bad.append({"edit": 0, "id": uid,
                        "reason": f"severity 인데 {e.get('op')} — replace 만 된다. 정도는 새 줄이 "
                                  "아니라 있는 줄의 뒷부분이다"})
            keep = []
        elif keep and not uid.startswith("C"):
            bad.append({"edit": 0, "id": uid,
                        "reason": "severity 인데 [Core Principles] 단위가 아니다 — C 로 시작하는 id 여야 한다"})
            keep = []
        elif keep and qw is None:
            bad.append({"edit": 0, "id": uid,
                        "reason": "그 단위에 물음표가 없다 — 질문과 정도를 가를 수 없어 이 역할이 못 쓴다"})
            keep = []
        elif keep and qt != qw:
            bad.append({"edit": 0, "id": uid,
                        "reason": "severity 인데 질문이 바뀌었다 — 첫 물음표까지는 그대로 두고 그 뒤만 "
                                  "다시 쓸 것. 질문은 어느 등급으로 보낼지를 정하고 그쪽은 이미 작동한다"})
            keep = []
        elif keep and not has_severity(text):
            bad.append({"edit": 0, "id": uid,
                        "reason": "severity 인데 정도 축이 아니다 — 심한 쪽과 가벼운 쪽을 **둘 다** 쓸 것. "
                                  "한쪽만 쓰면 방향이지 축이 아니고, 등급 안 두 자리를 못 가른다"})
            keep = []
        elif keep and len(text) - len(was) > SEVERITY_SLACK:
            bad.append({"edit": 0, "id": uid,
                        "reason": f"severity 인데 {len(text) - len(was)}자 늘었다 — "
                                  f"{SEVERITY_SLACK}자까지만 된다"})
            keep = []
        return keep, bad
    if role == "replace":
        # **길이 중립 편집.** 원칙 목록에 문장을 더한 스물세 번 가운데 스물두 번이 음수였고
        # (−0.006 ~ −0.011), 무엇을 쓰든 크기가 비슷했다. 반면 judge31 에서 C4 를 지운 편집은
        # 길이가 229자 줄면서 Δ 가 −0.0001 이었고 CI 가 0 을 정중앙에 뒀다. 손해가 내용이 아니라
        # 길이·주의 분산에서 온다면, **지운 만큼만 넣는 편집은 0 에서 출발한다.** 그래서 삭제 한
        # 건과 추가 한 건을 한 후보에 묶고 순증가를 `REPLACE_SLACK` 으로 막는다.
        gone = set((spent or {}).get("deleted") or ())
        n_order = (spent or {}).get("order_units", 99)
        # 한 자리 교체(`op="replace"`)면 편집 한 건으로 끝난다 — 지우는 자리와 넣는 자리가 같아
        # 길이 중립이 저절로 가깝다. 자리를 옮겨야 할 때는 삭제 한 건 + 추가 한 건이다.
        keep = edits[:1] if (edits and edits[0].get("op") == "replace") else edits[:2]
        bad = [{"edit": n, "id": e.get("id"), "reason": "replace 인데 허용 개수를 넘는 편집"}
               for n, e in enumerate(edits) if n >= len(keep)]
        if len(keep) == 1 and keep[0].get("op") == "replace":
            dele = ins = keep[0]
        else:
            ops = sorted(str(e.get("op")) for e in keep)
            if ops != ["delete", "insert_after"]:
                bad.append({"edit": 0, "id": None,
                            "reason": f"replace 는 한 단위를 replace 하거나, delete 한 건과 "
                                      f"insert_after 한 건이어야 한다 (받은 것: {ops})"})
                return [], bad
            dele = next(e for e in keep if e.get("op") == "delete")
            ins = next(e for e in keep if e.get("op") == "insert_after")
        did = str(dele.get("id") or "")
        # **[Order Principles] 단위도 대상이다.** 사람이 박아 둔 시작 문장을 루프가 실측으로 교체할 수
        # 있어야 한다 — 손댈 수 없게 두면 그 문장이 검증 없이 런 끝까지 남는다. 칸이 비는 것만
        # 막는다: 마지막 한 줄은 지우지 못하고(골격이 그 칸을 참조한다) 교체는 언제나 된다.
        if not did.startswith(("C", "O")):
            bad.append({"edit": 0, "id": did,
                        "reason": "replace 인데 대상이 [Core Principles]·[Order Principles] 단위가 아니다"})
            return [], bad
        if did.startswith("O") and dele.get("op") == "delete" and n_order <= 1:
            bad.append({"edit": 0, "id": did,
                        "reason": "replace 인데 [Order Principles] 의 마지막 한 줄을 지운다 — 그 칸이 "
                                  "비면 골격이 참조할 것이 없다. 지우지 말고 replace 로 바꿀 것"})
            return [], bad
        was = str(dele.get("_was") or "")
        if was and any(was[:60] == g[:60] for g in gone):
            bad.append({"edit": 0, "id": did,
                        "reason": "replace 인데 이미 손댄 단위를 또 고친다 — 다른 단위를 고를 것"})
            return [], bad
        grew = len(str(ins.get("text") or "")) - len(was)
        if grew > REPLACE_SLACK:
            bad.append({"edit": 0, "id": ins.get("id"),
                        "reason": f"replace 인데 길이 중립이 아니다 — 순증가 {grew}자 > "
                                  f"{REPLACE_SLACK}자. 더 긴 단위를 고르거나 새 문장을 줄일 것"})
            return [], bad
        return keep, bad
    if role == "narrow_rule":
        # 추가만 허용한다 — 기존 원칙을 고치는 순간 "재조정" 이 되고, 실측 5/5 가 그쪽에서 실패했다.
        bad = [{"edit": n, "id": e.get("id"), "reason": "narrow_rule 인데 둘째 이후 편집"}
               for n, e in enumerate(edits) if n > 0]
        keep = edits[:1]
        # 결속문 강제 — judge16 은 세 이터 연속 금지문("~ 직후 자르지 마라")을 냈고 전부 실패했다.
        # 결속절이 뒤에 붙어 있으면 통과시킨다(judge16 채택본이 그 꼴이다).
        if keep and is_prohibition(str(keep[0].get("text") or "")):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "narrow_rule 인데 금지문이다 — 무엇을 함께 유지하는지로 쓸 것"})
            keep = []
        if keep and keep[0].get("op") != "insert_after":
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": f"narrow_rule 인데 {keep[0].get('op')} — insert_after 만 된다"})
            keep = []
        elif keep and not str(keep[0].get("id") or "").startswith("C"):
            bad.append({"edit": 0, "id": keep[0].get("id"),
                        "reason": "narrow_rule 인데 [Core Principles] 단위가 아니다"})
            keep = []
        return keep, bad
    return edits, []


# 칸을 **역할이 직접 고정하는** 역할들. `enforce_kind` 의 칸 검사를 건너뛴다 — 두 검사가 같은 것을
# 서로 다르게 요구하면 `enforce_role` 을 통과한 편집이 뒤에서 죽는다(그 버그를 두 번 겪었다).
# `severity` 는 이항 발견을 받아 `[Core Principles]` 의 정도 절을 고친다: 발견이 두 자리를 견주지만
# 편집은 새 비교문을 넣는 것이 아니라 기존 원칙의 **정도 축을 교정**하는 것이므로 C 단위가 맞다.
SELF_PINNED_SECTION = ("severity",)


def enforce_kind(edits, kind: str | None, has_order_section: bool = True,
                 role: str | None = None) -> tuple[list, list[dict]]:
    """발견의 형태(`kind`)와 편집의 **칸·문면**을 맞춘다. 역할 검사와는 따로 걸린다.

    `[Order Principles]` 는 등급 안에서 두 자리를 견주는 문장만 담는 칸이고 `[Core Principles]` 는
    한 자리를 판정하는 칸이다. 칸이 형태를 정하므로 PE 가 형태를 바꿔 쓸 여지가 없어진다 —
    judge31 에서는 그 장치가 없어 `check` 발견을 받은 후보가 이항 편집이 됐다.

    `has_order_section` 이 거짓이면(그 섹션이 없는 프롬프트로 도는 런) 칸 검사를 건너뛴다."""
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    if kind not in ("check", "order") or role in SELF_PINNED_SECTION:
        return edits, []
    keep, bad = [], []
    for n, e in enumerate(edits):
        uid = str(e.get("id") or "")
        text = str(e.get("text") or "")
        if kind == "order" and has_order_section and e.get("op") != "delete" \
                and not uid.startswith("O"):
            bad.append({"edit": n, "id": uid,
                        "reason": "이항 발견인데 [Order Principles] 칸이 아니다 — O_end 에 넣을 것"})
            continue
        if kind == "check" and uid.startswith("O"):
            bad.append({"edit": n, "id": uid,
                        "reason": "단항 발견인데 [Order Principles] 칸을 고친다 — 그 칸은 두 자리를 "
                                  "견주는 문장만 담는다"})
            continue
        if kind == "check" and text and looks_binary(text):
            bad.append({"edit": n, "id": uid,
                        "reason": "단항 발견인데 문면이 두 자리를 견준다 (prefer / rather than / "
                                  "over 류) — 발견이 명명한 자리는 하나다"})
            continue
        keep.append(e)
    return keep, bad


# 문면 되돌리기를 면제하는 역할. `severity` 의 편집 문면은 **바꾸지 않은 질문 + 새 정도 절**이라
# Critic 문면으로 통째로 갈아 끼우면 질문이 사라져 `enforce_role` 이 거부한다. 이 역할에서 PE 가
# 하는 일은 형태를 고쳐 쓰는 것이 아니라 발견이 가리킨 원칙의 정도 축을 다시 쓰는 것이다.
VERBATIM_EXEMPT = ("severity",)


def pin_finding_text(edits, finding: dict | None,
                     role: str | None = None) -> tuple[list, list[dict]]:
    """PE 가 쓴 문면을 Critic 이 낸 finding 의 문면으로 되돌린다 — **형태를 고쳐 쓰지 못하게.**

    judge31 이터 1·2 에서 PE 가 단항 금지문("Do not cut …")을 서열문("rank the cut after that
    clause higher …")으로 고쳐 썼다. `ENGINEER_SYSTEM` 규칙 7 이 말로 금지하는데도 그랬고, 그
    형태가 역할 배분의 근거이므로 말로 두면 안 된다. PE 에게 남는 일은 **어느 단위 옆에 넣을지,
    무엇을 지워 자리를 만들지**다.

    `delete` 와 `labeled_example` 은 문면을 만들지 않으므로 건드리지 않는다. 길이가 예산을 넘으면
    되돌리기 경로(`shorten`)가 따로 줄인다."""
    edits = [e for e in (edits or []) if isinstance(e, dict)]
    if role in VERBATIM_EXEMPT:
        return edits, []
    want = ((finding or {}).get("edit") or {}).get("text")
    want = str(want or "").strip()
    if not want:
        return edits, []
    out, changed = [], []
    for n, e in enumerate(edits):
        if e.get("op") == "delete" or e.get("labeled_example") or not str(e.get("text") or "").strip():
            out.append(e)
            continue
        if " ".join(str(e["text"]).split()) == " ".join(want.split()):
            out.append(e)
            continue
        changed.append({"edit": n, "id": e.get("id"), "was": str(e["text"])[:120]})
        out.append({**e, "text": want})
    return out, changed


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


def parse_edits(blob: dict, current: str, budget: int,
                allow_frozen: tuple[str, ...] = ()) -> tuple[
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
    pr, note = parse_prompt({"prompt": draft, "changelog": blob.get("changelog")}, current, budget,
                            allow_frozen)
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
        if not b["edits"] and h.get("changelog"):
            b["changelog"] = h["changelog"]      # rewrite 후보는 edits 가 비어 changelog 가 유일한 흔적
        diag = h.get("diagnosis") or {}
        for k in ("lesson", "why", "blamed"):
            if diag.get(k):
                b[k] = diag[k]
        if diag.get("by_bin"):
            b["by_bin"] = diag["by_bin"]
        # 깊이별 변화 — 어느 깊이가 움직였나가 "그 편집이 누구를 도울 수 있었나" 를 정한다.
        # `by_bin` 만으로는 앞쪽만 고친 편집과 깊은 쪽을 고친 편집이 구별되지 않는다.
        if (diag.get("rank_depth") or {}).get("delta"):
            b["rank_depth_delta"] = diag["rank_depth"]["delta"]
        if diag.get("vs_base"):
            vb = diag["vs_base"]
            lo = vb["delta"].get("lo")
            b["vs_base"] = {"delta_mean": round(vb["delta"]["mean"], 4),
                            "delta_lo": None if lo is None else round(lo, 4), "by_bin": vb["by_bin"]}
        if h.get("built_on"):
            b["built_on"] = h["built_on"]
        if is_near_miss(h):
            b["near_miss"] = True
            if (h.get("gain") or {}).get("pooled"):
                b["measured_on"] = "test-A 200 + test-B 200 pooled"
        out.append(b)
    return out


def is_near_miss(h: dict, min_mean: float = 0.005, min_lo: float = -0.01) -> bool:
    """측정까지 갔는데 평균은 양수이고 하한만 0 언저리라 기각된 개정 — 방향은 맞고 크기가 부족.

    judge13 iter 1: 합산 +0.0074 [-0.0003, +0.0153] 이 하한 0.0003 차로 기각됐다. 이걸 "기각 =
    다른 방향" 으로 다루면 다음 이터가 유일하게 효과 본 방향을 버리고, 부검이 찍은 blamed 문구까지
    금지된다. 문턱은 그대로 두고 PE·Critic 에게만 표시한다."""
    if h.get("adopted") or h.get("screened_out") or h.get("screened_only"):
        return False
    d = h.get("gain") or h.get("delta") or {}
    return "mean" in d and "lo" in d and d["mean"] > min_mean and d["lo"] > min_lo


def dedupe_candidates(cands: list[tuple[int, str]]) -> tuple[list[int], list[tuple[int, int]]]:
    """본문이 같은 후보는 하나만 잰다 — (남길 후보 번호, [(뺀 번호, 같은 번호)]).

    judge12 iter 1: `examples_only` 와 `single_small` 이 `primary_finding` 이 달랐는데도 같은 편집
    (E4 → en_us_738)을 냈다. 같은 프롬프트를 선별에서 두 번 재면 $1 과 시간만 든다."""
    seen: dict[str, int] = {}
    keep, dropped = [], []
    for j, text in cands:
        key = " ".join(text.split())
        if key in seen:
            dropped.append((j, seen[key]))
        else:
            seen[key] = j
            keep.append(j)
    return keep, dropped


def blamed_with_text(blamed: list, cand: str) -> list[str]:
    """부검의 `blamed` 는 그 이터 후보의 단위 id("E2 (…)")다. id 는 이터마다 다시 매겨져 다음
    이터의 E2 는 다른 단위다 — 본문 앞부분을 붙여 둔다."""
    known = {u["id"]: u["text"] for u in edit_units(cand)}
    out = []
    for b in blamed or []:
        m = re.match(r"^\s*([CE]\d+)\s*(.*)$", str(b))
        if m and m.group(1) in known:
            text = " ".join(known[m.group(1)].split())[:60]
            out.append(f"{m.group(1)} «{text}»" + (f" {m.group(2)}" if m.group(2) else ""))
        else:
            out.append(str(b))
    return out


def exclude_used_examples(examples: dict[str, str], rejected: list[str]) -> tuple[dict, list[str]]:
    """기각된 개정 본문에 들어 있던 실측 예시는 다시 내놓지 않는다 — (남은 예시, 뺀 id).

    judge10 iter 1 부검이 지목한 예시(en_us_733, en_us_1294)를 iter 2 후보 셋이 전부 다시
    넣었다. 프롬프트가 안 바뀌면 사례도 안 바뀌어 같은 예시가 또 올라오고, PE 는 실측 예시를
    우선하라는 지시를 따른다. 본문으로 맞춘다 — rewrite 후보는 id 없이 본문을 붙여 넣는다."""
    norm = [" ".join(t.split()) for t in rejected]
    dropped = [k for k, v in examples.items()
               if any(" ".join(v.split()) in t for t in norm)]
    return {k: v for k, v in examples.items() if k not in dropped}, dropped


def edit_summary(prompt: str, edits) -> list[dict]:
    """이력용 — 어느 단위를 무엇으로 바꿨는지 짧게. 이력에 사유·Δ 만 있으면 PE 가 기각된 편집을
    알아보지 못하고 되풀이한다(judge05 iter 2~4). id 는 이터마다 다시 매겨지므로 원문 앞부분을 싣는다."""
    known = {u["id"]: u["text"] for u in edit_units(prompt)}
    out = []
    for e in edits or []:
        if not isinstance(e, dict):
            continue
        op = e.get("op")
        row = {"op": op, "id": e.get("id"),
               "kind": "delete" if op == "delete" else (e.get("kind") or "change"),
               "was": known.get(str(e.get("id")), "")[:80],
               "now": (e.get("text") or "")[:80]}
        if e.get("labeled_example"):
            row["labeled_example"] = e["labeled_example"]
        out.append(row)
    return out


def units_with_provenance(prompt: str, prov: dict) -> list[dict]:
    """PE 입력용 — 편집 단위에 출처 기록을 붙인다."""
    blank = {"origin": "v0", "adopted_delta": None, "adopted_ci_lo": None}
    return [{**u, **prov.get(u["text"], blank)} for u in edit_units(prompt)]


def soft_target(budget: int, cur_len: int, margin: float = 0.05) -> int:
    """모델에게 알리는 길이 목표 — 검사 상한(`budget`)보다 `margin` 만큼 낮다.

    목표와 상한이 같은 값이면 모델은 늘 1% 안팎 넘긴다(judge08~10 의 초과 12건 중 5건이 1.2%
    이내: 8074/8054, 9905/9812, 9159/9050, 11165/11084, 11112/11084). 그 초과로 후보가 죽거나
    되돌리기가 돌면서 내용을 왕창 잃었다(11820→7872). 5% 여유가 그 오차를 흡수한다. 상한이
    천장에 닿아 현재 길이와 같을 때는 줄이라고 요구하지 않는다."""
    return max(cur_len, int(budget * (1 - margin)))


def shorten_user(draft: str, target: int, findings: list) -> dict:
    """Writer 가 다시 쓴 초안(rewrite 후보)이 길이만 넘었을 때 PE 에게 주는 입력.

    Writer 를 다시 불러 "N단어 줄여라" 해도 안 된다 — 통째로 다시 쓰므로 길이가 손을 떠난다
    (judge10 iter 1: 8074자 → 재시도 8104자). 반면 PE 는 코드가 잰 단위별 글자 수를 받아 편집만
    내니 되돌리기 9건 중 9건을 맞췄다. 그래서 초안 자체를 편집 단위로 쪼개 PE 에게 준다. 단위
    출처는 전부 "rewrite" — 하나씩 잰 이득이 없다. 한 패스에 15% 남짓 깎이므로(judge10 iter 2:
    9748 → 8339) 호출 쪽이 맞을 때까지 몇 번 반복한다."""
    units = [{**u, "origin": "rewrite", "adopted_delta": None, "adopted_ci_lo": None}
             for u in edit_units(draft)]
    return {"fixed_sections": {h: section_of(draft, h) for h in ("[Role]",)},
            "units": units, "findings": findings,
            "size": size_brief(draft, target),
            "size_feedback": edit_feedback(draft, draft, target, [])}


def edit_feedback(draft: str, current: str, budget: int, deltas: list[dict]) -> dict:
    """길이만 넘은 편집을 한 번 되돌려 보낼 때 붙이는 실측."""
    units = sorted(edit_units(current), key=lambda u: -u["chars"])
    over = len(draft) - budget
    return {"counted_by": "code", "result_total": len(draft), "budget": budget,
            "over_by": over, "over_by_words": -(-over // int(chars_per_word(current))),
            "your_edits": deltas,
            "largest_units": [{"id": u["id"], "chars": u["chars"]} for u in units[:6]]}
