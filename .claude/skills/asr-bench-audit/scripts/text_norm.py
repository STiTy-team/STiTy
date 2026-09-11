# -*- coding: utf-8 -*-
"""채점 전 표기 정규화 — 숫자와 단위.

백엔드마다 **표기 규칙이 다르다.** Nemotron 은 아라비아 숫자를 한 번도 출력하지
않고("이천십일년") Qwen3 는 단위를 약어로 쓴다("15m"). 참조는 숫자와 단위어를
쓰므로, 그대로 채점하면 **인식이 맞아도 오답**이 된다(실측: Nemotron ko CER
0.1013 중 0.023 이 표기 차이였다).

그래서 채점 직전에 양쪽을 같은 표기로 맞춘다. 세 백엔드에 같은 변환을 적용하므로
어느 쪽에도 유리하지 않다.

## 숫자 — 참조에 있는 수만, 문맥이 맞을 때만

가설 전체에서 한글 수사를 찾아 바꾸면 오검출이 난다. 한글 수사는 일상 음절과
겹치기 때문이다 — "데이"의 이, "사육"의 사·육, "일부"의 일. 실제로 첫 판에서
`데이포트로 -> 데2포트로`, `타이브레이크 -> 타2브레2크` 가 나왔다.

그래서 **참조에 실제로 있는 숫자**에 대응하는 표기만 대상으로 삼고,
  · 앞 글자가 수사면 더 긴 수의 일부이므로 건드리지 않는다
  · 뒤 글자가 수사이고 (형태+그 글자)가 더 긴 수사의 접두면 건드리지 않는다
  · **한 글자 수사(일·이·육…)는 뒤에 단위나 조사가 올 때만** 되돌린다
영어 `one` 은 대명사와 겹쳐 아예 건너뛴다.

가설이 **틀린 수**를 말했으면 형태가 안 맞아 치환되지 않는다 — 오답을 정답으로
바꿔 주지 않는다.

## 단위 — 양쪽 대칭

`15m` 과 `15미터` 는 같은 인식 결과다. 약어를 단위어로 펴서 양쪽에 같이 적용한다
(긴 것부터 — 킬로미터를 미터보다 먼저 봐야 한다).
"""
from __future__ import print_function
import re
import unicodedata

SINO = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SMALL = ["", "십", "백", "천"]
BIG = ["", "만", "억", "조"]
SINO_CH = set("영공일이삼사오육륙칠팔구십백천만억조")
# 숫자 뒤에 올 수 있는 단위/조사 — 한 글자 수사는 이 문맥에서만 되돌린다
COUNTER = set("월일년시분초대개명번차위세미터킬그램퍼센트원장쪽회권마리살층도%")
PARTICLE = set("이가은는을를에의로와과도")

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]

# 약어 -> 단위어. 긴 것부터 본다.
UNIT_KO = [("km", "킬로미터"), ("cm", "센티미터"), ("mm", "밀리미터"),
           ("kg", "킬로그램"), ("m", "미터"), ("%", "퍼센트")]
UNIT_EN = [("%", " percent")]


def ko_read(n):
    """정수 -> 한자어 읽기 후보. 1만 은 '만'/'일만' 둘 다 쓰인다."""
    if n == 0:
        return ["영", "공"]

    def four(x, strip_one):
        s = ""
        for i, ch in enumerate(reversed(str(x))):
            d = int(ch)
            if d == 0:
                continue
            head = SINO[d]
            if strip_one and d == 1 and i > 0:
                head = ""
            s = head + SMALL[i] + s
        return s

    groups, x = [], n
    while x > 0:
        groups.append(x % 10000)
        x //= 10000
    out = []
    for strip in (True, False):
        s = ""
        for i, g in enumerate(groups):
            if g == 0:
                continue
            part = four(g, strip)
            if i > 0 and part == "일" and strip:
                part = ""
            s = part + BIG[i] + s
        if s and s not in out:
            out.append(s)
    return out


def _en_small(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return (TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + _en_small(n % 100) if n % 100 else "")
    return None


def en_read(n):
    out = []
    if n < 1000:
        out.append(_en_small(n))
    elif n < 1000000:
        th, rest = n // 1000, n % 1000
        out.append(_en_small(th) + " thousand" + (" " + _en_small(rest) if rest else ""))
        if 1100 <= n < 10000:                      # 2011 -> twenty eleven
            hi, lo = n // 100, n % 100
            out.append(_en_small(hi) + (" " + _en_small(lo) if lo else " hundred"))
    return [o for o in out if o]


def canon_numbers_ko(ref, hyp):
    nums = [int(m.group()) for m in re.finditer(r"\d+", ref)]
    forms = [(f, n) for n in nums for f in ko_read(n)]
    allforms = set(f for f, _ in forms)
    forms.sort(key=lambda x: -len(x[0]))
    for f, n in forms:
        idx = 0
        while True:
            i = hyp.find(f, idx)
            if i < 0:
                break
            before = hyp[i - 1] if i > 0 else ""
            j = i + len(f)
            after = hyp[j] if j < len(hyp) else ""
            nxt = hyp[j:j + 2].lstrip()[:1]        # 공백 건너뛴 다음 글자
            if before in SINO_CH:
                ok = False                          # 더 긴 수의 일부
            elif after in SINO_CH and any(g.startswith(f + after) for g in allforms):
                ok = False                          # 더 긴 수사의 접두
            elif len(f) == 1 and not (after in COUNTER or after in PARTICLE
                                      or nxt in COUNTER):
                ok = False                          # 한 글자 수사는 단위/조사 앞에서만
            else:
                ok = True
            if ok:
                hyp = hyp[:i] + str(n) + hyp[j:]
                idx = i + len(str(n))
            else:
                idx = i + 1
    return hyp


def canon_numbers_en(ref, hyp):
    low = hyp.lower()
    for m in re.finditer(r"\d+", ref):
        n = int(m.group())
        if n == 1:
            continue                                # 'one' 은 대명사와 겹친다
        for f in sorted(en_read(n), key=len, reverse=True):
            pat = re.compile(r"\b" + f.replace(" ", r"[\s-]+") + r"\b", re.I)
            if pat.search(low):
                hyp = pat.sub(str(n), hyp)
                low = hyp.lower()
    return hyp


def canon_units(text, lang):
    """약어를 단위어로 편다. 숫자 뒤에 붙은 것만 본다."""
    table = UNIT_KO if lang == "ko" else UNIT_EN
    for ab, full in table:
        if ab == "%":
            text = re.sub(r"(\d)\s*%", r"\1" + full, text)
        else:
            text = re.sub(r"(\d)\s*" + re.escape(ab) + r"(?![a-zA-Z])",
                          r"\1" + full, text)
    return text


def canon_pair(ref, hyp, lang):
    """채점 직전 (참조, 가설) 정규화. 같은 변환을 양쪽에 적용한다."""
    if lang == "ko":
        hyp = canon_numbers_ko(ref, hyp)
    else:
        hyp = canon_numbers_en(ref, hyp)
    return canon_units(ref, lang), canon_units(hyp, lang)


# ── 채점 ─────────────────────────────────────────────────────────────────
# 채점과 정규화를 한 모듈에 두는 이유: 둘이 갈라지면 "보고서가 인용하는 값"이
# 어느 구현으로 나온 것인지 추적할 수 없게 된다(실제로 그런 적이 있다).

def _norm(text):
    text = unicodedata.normalize("NFKC", (text or "").lower())
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _edit_distance(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(ref, hyp, unit):
    """CER(cer) / WER(wer). NFKC + 소문자화 + 문장부호 제거 후 편집거리."""
    r, h = _norm(ref), _norm(hyp)
    if unit == "cer":
        r, h = r.replace(" ", ""), h.replace(" ", "")
        seq_r, seq_h = list(r), list(h)
    else:
        seq_r, seq_h = r.split(), h.split()
    if not seq_r:
        return None
    return _edit_distance(seq_r, seq_h) / len(seq_r)


def score_pair(ref, hyp, unit, lang):
    """(표기 정규화 후, 원값). 본 수치는 정규화 후를 쓰고 원값을 같이 남긴다."""
    raw = score(ref, hyp, unit)
    try:
        r2, h2 = canon_pair(ref, hyp, lang)
    except Exception:                                       # noqa: BLE001
        return raw, raw
    return score(r2, h2, unit), raw
