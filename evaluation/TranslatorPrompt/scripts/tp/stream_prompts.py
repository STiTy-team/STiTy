"""2.1 동시통역 조건. 문장을 <SEG> 조각으로 나눠 차례로 번역하고, 한 번 낸 조각 번역은 고치지 않는다.

모든 조건이 같은 문맥을 받는다 — 같은 문장의 앞 조각 원문과 이미 내보낸 번역(<prev>). 서버가 앞 세그먼트
(원문, 번역) 쌍을 문맥으로 넘기는 것과 같은 모양이다. 조건끼리는 system 문면만 다르다.

P0  2.3 의 S0(출력 규칙 + <src>)을 조각에 그대로 쓴다. 조각을 문장처럼 번역한다 — 기준
P1  + 동시통역 지시: 아직 안 들은 말을 짐작해 완성하지 말고, 뒤에 무엇이 와도 이어질 수 있게.
    한국어는 열린 어미(…는데, …고), 명사구, 주제 먼저 꺼내기
P2  P1 + 조각 번역 예시 (평가 강연과 무관한 문장으로 직접 쓴 것)
P3  P1 + 짐작 없이 옮길 수 없으면 빈 출력 허용. 빈 출력이면 그 조각은 다음 조각과 합쳐 다시 보낸다
H1  P1 + 부분 보류: 한국어 자리가 뒤에 올 말에 달린 부분만 원문 그대로 <hold> 에 남기고 나머지를 <out> 으로 낸다.
    남긴 원문은 다음 조각 앞에 붙여 다시 보낸다. 마지막 조각에서는 남기지 않는다 (서버는 문장 끝을 안다)
H2  H1 + 보류 예시
H2F H2 와 같되 마지막 조각은 따로 부른다: 이미 내보낸 번역과 남은 원문 전부를 주고 문장을 끝맺게 한다.
    H2 에서 마지막 조각을 열린 채 두거나 또 보류해 서술어가 사라진 것을 막으려는 것. 마지막 앞 단계는 H2 기록을
    그대로 쓴다 (stream_finish.py) — 차이는 마지막 호출 하나뿐이다
H3  한 프롬프트로 마지막 조각까지 처리하는 부분 보류. H2 와 다른 점은 셋이다.
    ① 앞에서 미룬 원문을 <held> 로 새 조각(<src>)과 나눠 준다
    ② <prev> 에 조각 원문 전체와 그중 아직 번역하지 않은 부분을 함께 보여 준다 (H2 는 번역된 부분만 계산해
       보여 줘서, 원문을 통째로 남기고 번역도 낸 경우 원문 칸이 비었다)
    ③ <hold> 에 남긴 말은 <out> 에서 옮기지 말라는 규칙. 마지막 조각이면 user 에 그렇게 적고, 같은 system
       문면이 문장을 맺는 규칙까지 담는다
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tp.quality_prompts import NAME, base_system  # noqa: E402

CONDITIONS = ["P0", "P1", "P2", "P3", "H1", "H2", "H3"]
HOLD_CONDS = ("H1", "H2")

PREV_NOTE = ("Earlier pieces of the same sentence may be given inside <prev>, each with the translation "
             "that the listener has already seen. They are context only: do not translate or repeat them.")

SIMUL = ("You are working as a simultaneous interpreter. The text inside <src> is only the next piece of a "
         "sentence the speaker is still saying, and more words may follow. Translate only what this piece "
         "says. Do not guess or complete what has not been said yet, and do not phrase it so that the rest "
         "of the sentence could contradict it. The listener already saw the translation in <prev>, which "
         "cannot be changed, so continue on from it. Write Korean that can be continued naturally whatever "
         "comes next: prefer open endings such as '…는데', '…고', '…면서', noun phrases, or bringing the "
         "topic forward first (for example '제가 보여드리려는 건'), rather than a finished sentence ending in "
         "'…다' or '…요'. Finish the sentence only when the piece itself clearly ends it.")

# 평가 강연(Gehry TED)과 겹치지 않게 직접 쓴 강연·발표 말투 예시. (앞 조각들, 이번 조각, 번역)
EXAMPLES = [
    ([], "So what we found in the second experiment", "그래서 두 번째 실험에서 저희가 발견한 건"),
    ([("So what we found in the second experiment", "그래서 두 번째 실험에서 저희가 발견한 건")],
     "was that people didn't really care about the price.", "사람들이 가격에는 별로 신경 쓰지 않았다는 겁니다."),
    ([], "If you look at the map on the left,", "왼쪽 지도를 보시면,"),
    ([], "The reason I started this company", "제가 이 회사를 시작한 이유는"),
    ([("I grew up in a small town", "저는 작은 마을에서 자랐는데")],
     "where nobody had ever", "그곳에선 아무도"),
]

DEFER = ("If the piece is too short or too unclear to translate without guessing what comes next (for "
         "example it stops right after 'the', 'to' or 'that'), reply with an empty line instead. It will "
         "be joined with the next piece and sent again. Never reply empty when the piece ends the sentence.")

HOLD = ("Like a professional interpreter, hold back what you cannot place yet. Korean puts verbs last and "
        "modifiers first, so the Korean for some words depends on what the speaker has not said yet (for example "
        "an object whose description may follow, or a verb whose object is still coming). Translate now only the "
        "part whose Korean wording and position you are sure of, and keep the rest in the original English. This "
        "format replaces the one-line output rule above. Reply in exactly this format and nothing else:\n<out>Korean to show the listener now (may be empty)</out>\n"
        "<hold>English words from <src> that you did not translate, copied exactly (may be empty)</hold>\n"
        "The held words come back to you at the start of the next <src>, and you translate them then. Every word "
        "of <src> must be either translated in <out> or copied into <hold>; never drop anything. Do not hold "
        "what you can already translate safely: holding delays the listener.")

LAST = "This is the last piece of the sentence: hold nothing and translate everything that is left."

HOLD_EXAMPLES = [
    ([], "We finally found the cause", "저희는 마침내", "found the cause"),
    ([("We finally", "저희는 마침내")], "found the cause of the delay in the shipment.",
     "배송이 늦어진 원인을 찾았습니다.", ""),
    ([], "She asked me to bring", "그녀가 저에게", "asked me to bring"),
    ([], "If you look at the map on the left,", "왼쪽 지도를 보시면,", ""),
]

FINISH = ("This is the end of the sentence. The listener has already seen the Korean in <prev>, which cannot be "
          "changed. <src> holds everything that has not been translated yet: words held back from earlier pieces "
          "and the last piece. Translate all of it so that, read right after the translation in <prev>, it "
          "completes one natural Korean sentence. Leave nothing out, especially verbs: the Korean verb usually "
          "belongs here at the end. Finish with a proper sentence ending (for example '…습니다' or '…어요'). Do not "
          "repeat what is already in <prev>.")

EMPTY_MARKS = {"", '""', "''", "(empty)", "<empty>", "[empty]", "（空）"}


def prev_block(prev: list[tuple[str, str]]) -> str:
    if not prev:
        return ""
    return "<prev>\n" + "\n".join(f"{s}\t=> {t}" for s, t in prev) + "\n</prev>\n"


def build(cond: str, src: str, tgt: str, piece: str, prev: list[tuple[str, str]],
          last: bool = False) -> tuple[str, str]:
    system = base_system(src, tgt) + "\n\n" + PREV_NOTE
    if cond in ("P1", "P2", "P3"):
        system += "\n\n" + SIMUL
    if cond == "P2":
        system += "\n\nExamples:\n" + "\n\n".join(
            f"{prev_block(p)}<src>{s}</src>\n{t}" for p, s, t in EXAMPLES)
    if cond == "P3":
        system += "\n\n" + DEFER
    if cond in HOLD_CONDS:
        system += "\n\n" + HOLD
    if cond == "H2":
        system += "\n\nExamples:\n" + "\n\n".join(
            f"{prev_block(p)}<src>{s}</src>\n<out>{o}</out>\n<hold>{h}</hold>" for p, s, o, h in HOLD_EXAMPLES)
    if cond not in CONDITIONS:
        raise ValueError(cond)
    what = "sentence" if cond == "P0" else "piece"
    user = f"{prev_block(prev)}Translate the {what} in <src> into {NAME[tgt]}.\n<src>{piece}</src>"
    if cond in HOLD_CONDS and last:
        user += "\n" + LAST
    return system, user


H3_RULES = (
    "Like a professional interpreter, hold back what you cannot place yet. Korean puts verbs last and modifiers "
    "first, so the Korean for some words depends on what the speaker has not said yet (for example an object whose "
    "description may follow, or a verb whose object is still coming).\n"
    "Input: <prev> lists the earlier pieces of this sentence. Each line shows the English you were given, the "
    "Korean the listener already saw (it cannot be changed), and the English you held back. <held> repeats the "
    "English still held back, which you must now handle. <src> is the new piece.\n"
    "Reply in exactly this format and nothing else (it replaces the one-line output rule above):\n"
    "<out>Korean to show the listener now (may be empty)</out>\n"
    "<hold>English words from <held> or <src> that you still hold back, copied exactly (may be empty)</hold>\n"
    "Rules:\n"
    "- Every English word in <held> and <src> goes to exactly one place: translated in <out>, or copied into "
    "<hold>. Never both, never neither. Do not translate a word in <out> and also hold it.\n"
    "- Hold only what you cannot place safely yet: holding delays the listener.\n"
    "- <out> continues right after the Korean already shown in <prev>. Do not repeat it.\n"
    "- For a piece that is not the last one, prefer Korean that stays open for what follows (…는데, …고, noun "
    "phrases) rather than a finished sentence.\n"
    "- When the message says it is the last piece: hold nothing. Translate everything left in <held> and <src>, "
    "include the verb, and finish the sentence with a proper ending such as '…습니다' or '…어요'.")

H3_EXAMPLES = [
    ([], "", "We finally found the cause", False, "저희는 마침내", "found the cause"),
    ([("We finally found the cause", "저희는 마침내", "found the cause")], "found the cause",
     "of the delay in the shipment.", True, "배송이 늦어진 원인을 찾았습니다.", ""),
    ([], "", "She asked me to bring", False, "그녀가", "asked me to bring"),
    ([], "", "If you look at the map on the left,", False, "왼쪽 지도를 보시면,", ""),
]

LAST_H3 = "This is the last piece of the sentence."


def prev_block3(prev: list[tuple[str, str, str]]) -> str:
    if not prev:
        return ""
    return "<prev>\n" + "\n".join(f"given: {s} | shown: {t} | held back: {h or '-'}" for s, t, h in prev) + "\n</prev>\n"


def h3_user(prev, held: str, piece: str, last: bool) -> str:
    return (prev_block3(prev) + f"<held>{held}</held>\n<src>{piece}</src>" + ("\n" + LAST_H3 if last else ""))


def build_h3(src: str, tgt: str, held: str, piece: str, prev: list[tuple[str, str, str]],
             last: bool) -> tuple[str, str]:
    system = base_system(src, tgt) + "\n\n" + SIMUL + "\n\n" + H3_RULES + "\n\nExamples:\n" + "\n\n".join(
        h3_user(p, h, s, l) + f"\n<out>{o}</out>\n<hold>{k}</hold>" for p, h, s, l, o, k in H3_EXAMPLES)
    user = f"Translate into {NAME[tgt]}.\n" + h3_user(prev, held, piece, last)
    return system, user


def build_finish(src: str, tgt: str, rest: str, prev: list[tuple[str, str]]) -> tuple[str, str]:
    system = base_system(src, tgt) + "\n\n" + PREV_NOTE + "\n\n" + FINISH
    user = f"{prev_block(prev)}Finish the sentence: translate <src> into {NAME[tgt]}.\n<src>{rest}</src>"
    return system, user


def is_empty(hyp: str) -> bool:
    return hyp.strip() in EMPTY_MARKS


def _words(s: str) -> list[str]:
    return [w for w in (re.sub(r"[^\w'-]", "", x).lower() for x in s.split()) if w]


def parse_hold(raw: str, src: str) -> dict:
    """<out>/<hold> 를 읽는다. hold 는 원문 어절로만 이뤄져야 한다 — 아니면 버리고 invalid 로 표시한다
    (없는 말을 다음 조각에 실어 보내면 안 된다). 태그가 없으면 출력 전체를 out 으로 본다."""
    mo = re.search(r"<out>(.*?)</out>", raw, re.S)
    mh = re.search(r"<hold>(.*?)</hold>", raw, re.S)
    if not mo:
        return {"out": raw.strip(), "hold": "", "parse_fail": True, "hold_invalid": False}
    out, hold = mo.group(1).strip(), (mh.group(1).strip() if mh else "")
    invalid = False
    if hold:
        pool = _words(src)
        for w in _words(hold):
            if w in pool:
                pool.remove(w)
            else:
                invalid = True
                break
    return {"out": out, "hold": "" if invalid else hold, "parse_fail": False, "hold_invalid": invalid}


def covered(src: str, hold: str) -> str:
    """src 에서 hold 를 뺀, 이번에 번역된 원문. 정확히 부분 문자열이면 잘라 내고 아니면 어절 단위로 뺀다."""
    if not hold:
        return src
    if hold in src:
        return " ".join(src.replace(hold, " ", 1).split())
    pool = _words(hold)
    keep = []
    for x in src.split():
        w = re.sub(r"[^\w'-]", "", x).lower()
        if w in pool:
            pool.remove(w)
        else:
            keep.append(x)
    return " ".join(keep)
