"""The addition-validity pieces of build_conv_pairs, importable without its CLI: the RULES text,
the format / openness / leak checks, and the language-name map. build_conv_pairs.py and
eval_models.py both run these, so a rule change moves every consumer at once."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_prefix import closes, content  # noqa: E402

LANGNAME = {"ko": "Korean", "en": "English"}

RULES = {
    "en": ("English: natural spoken English a simultaneous interpreter would actually say.\n"
           "Two priorities, in this order:\n"
           "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, "
           "or content word whose source appears only in a later phrase.\n"
           "2. Naturalness: the additions read, joined together, as one fluent English sentence. Additions "
           "before the last must stay syntactically open - no sentence-final period. When emitting a phrase's "
           "content right away would force stiff phrasing, leave that addition \"\" (wait) and fold the content "
           "naturally into the NEXT addition instead. Waiting is always allowed; fillers and stiffness are not."),
    "ko": ("Korean: natural spoken Korean a simultaneous interpreter would actually say, with connective endings "
           "(-고, -며, -는데, -면서, -지만, -어서/아서, -도록) and particles; never -다, -요, -니다 or a period before the "
           "end; end the sentence politely in 해요체 (-어요/-아요/-예요/-이에요).\n"
           "Two priorities, in this order:\n"
           "1. STRICT no-lookahead: addition i may use ONLY information from phrases 1..i - no entity, number, or "
           "content word whose source appears only in a later phrase.\n"
           "2. Naturalness: the additions read, joined together, as one fluent Korean sentence. When emitting a "
           "phrase's content right away would force stiff phrasing or a filler tail, leave that addition \"\" "
           "(wait) and fold the content naturally into the NEXT addition instead. Waiting is always allowed; "
           "fillers and stiffness are not."),
}


def leaked(adds, hp, tgt):
    if not hp:
        return []
    full = content(hp[-1] or "", tgt)
    for i in range(len(hp) - 1):
        if not hp[i]:
            continue
        have = content(hp[i], tgt)
        extra = sorted(content(" ".join(adds[: i + 1]), tgt) - have & full)
        if extra:
            return extra
    return []


def make_check(hy):
    def check(row, adds):
        tgt = row["tgt"]
        if not isinstance(adds, list) or len(adds) != len(row["chunks"]):
            return f"need exactly {len(row['chunks'])} additions"
        if not isinstance(adds[-1], str) or not adds[-1].strip():
            return "the last addition is empty"
        body = [x.strip() for x in adds if isinstance(x, str) and x.strip()]
        for x in body[:-1]:
            if closes(x, tgt):
                return f"addition before the end closes the sentence: {x!r}"
        if not closes(body[-1], tgt):
            return f"the last addition does not close the sentence: {body[-1]!r}"
        extra = leaked(adds, hy.get(row["sid"]) or [], tgt)
        if extra:
            return ("these words are said before their source phrase has been heard (no-lookahead violation): "
                    + ", ".join(extra) + " - move them to the addition of the phrase that contains them")
        return None
    return check
