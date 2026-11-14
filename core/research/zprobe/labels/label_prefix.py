"""Prefix labeler (prefix16): the labeler sees only what the model will see.

For every source sentence, a mechanical list of candidate points (prefix_points.candidate_points) is
walked in order. At each point the LLM gets the source words heard so far (1..k) and the translation
committed so far, and answers {"add": text or "", "src": [source word numbers]}. Each answer passes a
step gate (range, lookahead against k, open-piece shape); a lookahead violation becomes a wait (empty
item) at once, a shape/format violation is re-asked once at that point and then becomes a wait. When the
sentence is over, a regret check shows the LLM the whole source (and the reference, if any) with the
piece boundaries and asks which pieces should not have gone out; a regretted piece becomes a wait and
the points after it are re-asked. The record keeps one item per candidate point, empty ones included -
those are the wait labels.

Output: --out jsonl {"sid","lang","tgt","schedule":[[after,text],...],"src_words":[[...],...],"attempts",
"regret_rounds","calls"}; <out>.fail.jsonl {"sid","reason"}; <out>.usage.jsonl one line per call with
kind "step" or "regret" (cost-watch). Resume: sids in --out or .fail.jsonl are skipped; a sentence cut by
a crash restarts from its first point. Exit 2 = budget stop, 3 = API errors on > 1% of rows.

usage: PYTHONPATH=. .venv/bin/python core/research/zprobe/labels/label_prefix.py --limit 50 \
    --sentences models/zprobe/data/prefix14/sentences.jsonl --out models/zprobe/data/prefix16/pilot.jsonl
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from label_schedule import LANG_NAME  # noqa: E402
from schedule_gate import words  # noqa: E402

RULES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "prefix_rules_v1.txt")
RULES = {"path": RULES_PATH}

END_NOTE = "The source sentence ENDS here: emit everything that is still missing and close the sentence."
MORE_NOTE = "More source may follow; you do not know how much."


def build_step_prompt(row: dict, k: int, committed: str, is_end: bool, note: str = "") -> str:
    """Prompt for one decision point: words 1..k, the committed text, the rules. Never the reference, never
    words after k."""
    tokens = words(row["src"], row["lang"])[:k]
    numbered = " ".join(f"{i}:{t}" for i, t in enumerate(tokens, 1))
    rules = open(RULES["path"], encoding="utf-8").read().strip()
    lines = [f"Source ({LANG_NAME[row['lang']]}) heard so far, words numbered from 1:", numbered,
             END_NOTE if is_end else MORE_NOTE,
             f"Translation ({LANG_NAME[row['tgt']]}) committed so far: {committed.strip() or '(nothing yet)'}",
             rules]
    if note:
        lines.append(note)
    lines.append('Answer with JSON only: {"add": "...", "src": [i, j, ...]} - "src" lists the numbers of the heard '
                 'source words that "add" translates (all <= the last word shown); "add": "" with "src": [] when '
                 'nothing can be added yet.')
    return "\n".join(lines)


def build_regret_prompt(row: dict, pieces: list[tuple[int, str]]) -> str:
    """After the sentence: whole source, reference if any, the pieces numbered - which ones were a mistake?"""
    numbered = " ".join(f"[{i}] {t.strip()}" for i, (_, t) in enumerate(pieces, 1))
    lines = [f"A simultaneous interpreter produced this translation piece by piece, each piece committed before "
             f"hearing the rest of the source. The pieces are numbered in order.",
             f"Source ({LANG_NAME[row['lang']]}): {row['src']}"]
    if row.get("ref"):
        lines.append(f"Reference translation ({LANG_NAME[row['tgt']]}), for meaning only: {row['ref']}")
    lines += [f"Pieces ({LANG_NAME[row['tgt']]}): {numbered}",
              "Knowing the whole source now, list the pieces that should NOT have been emitted at that moment: "
              "(i) the piece left no natural place for content that came later (word order is now blocked), "
              "(ii) its meaning turned out wrong once the rest was heard (negation, tense, what a modifier attaches to), "
              "(iii) it contains content that had not been heard yet. A piece that is merely awkward but correct is not "
              "a regret. The last piece may be listed only for (ii) or (iii).",
              'Answer with JSON only: {"regret": [piece numbers], "note": "one short sentence, or empty"}']
    return "\n".join(lines)


def parse_add(reply) -> tuple[str, set[int]]:
    """{"add": str, "src": [int]} -> (text, src set); ValueError("format: ...") otherwise. Blank add -> ("", set())."""
    if not isinstance(reply, dict) or "add" not in reply:
        raise ValueError("format: expected {'add': ..., 'src': [...]}")
    add = reply["add"]
    if not isinstance(add, str):
        raise ValueError(f"format: 'add' must be a string: {add!r}")
    if not add.strip():
        return "", set()
    src = reply.get("src")
    if not isinstance(src, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in src):
        raise ValueError(f"format: 'src' must be a list of ints: {src!r}")
    return add.strip(), set(src)


def parse_regret(reply, n_pieces: int) -> tuple[list[int], str]:
    """{"regret": [int], "note": str} -> (piece numbers within 1..n, note); ValueError on bad shape."""
    if not isinstance(reply, dict) or not isinstance(reply.get("regret"), list):
        raise ValueError("format: expected {'regret': [...], 'note': ...}")
    nums = sorted({x for x in reply["regret"] if isinstance(x, int) and not isinstance(x, bool) and 1 <= x <= n_pieces})
    return nums, str(reply.get("note") or "")
