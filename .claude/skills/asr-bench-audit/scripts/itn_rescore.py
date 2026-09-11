# -*- coding: utf-8 -*-
"""숫자 표기 차이를 걷어내고 재채점 — 오검출 방지 규칙 적용본.

한글 수사는 일상 음절과 겹친다('데이'의 이, '사육'의 사/육). 그래서
  · 참조에 실제로 있는 숫자에 대응하는 형태만 본다
  · 앞 글자가 수사면 더 긴 수의 일부이므로 건드리지 않는다
  · 한 글자짜리 수사(일·이·육…)는 **뒤에 단위나 조사가 올 때만** 숫자로 되돌린다
세 백엔드에 같은 변환을 적용한다.
"""
import json, re, unicodedata

def _norm(t):
    t = unicodedata.normalize("NFKC", (t or "").lower())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()

def _ed(a, b):
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
    r, h = _norm(ref), _norm(hyp)
    if unit == "cer":
        r, h = r.replace(" ", ""), h.replace(" ", "")
        sr, sh = list(r), list(h)
    else:
        sr, sh = r.split(), h.split()
    return _ed(sr, sh) / len(sr) if sr else None

SINO = ["", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"]
SMALL = ["", "십", "백", "천"]
BIG = ["", "만", "억", "조"]
SINO_CH = set("영공일이삼사오육륙칠팔구십백천만억조")
# 숫자 뒤에 올 수 있는 단위/조사 — 한 글자 수사는 이 문맥에서만 되돌린다
COUNTER = set("월일년시분초대개명번차위세미터킬그램퍼센트원장쪽회권마리살층도%")
PARTICLE = set("이가은는을를에의로와과도")

def ko_read(n):
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
    groups = []
    x = n
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

ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
        "eighty", "ninety"]

def en_small(n):
    if n < 20:
        return ONES[n]
    if n < 100:
        return (TENS[n // 10] + (" " + ONES[n % 10] if n % 10 else "")).strip()
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + en_small(n % 100) if n % 100 else "")
    return None

def en_read(n):
    out = []
    if n < 1000:
        out.append(en_small(n))
    elif n < 1000000:
        th, rest = n // 1000, n % 1000
        out.append(en_small(th) + " thousand" + (" " + en_small(rest) if rest else ""))
        if 1100 <= n < 10000:
            hi, lo = n // 100, n % 100
            out.append(en_small(hi) + (" " + en_small(lo) if lo else " hundred"))
    return [o for o in out if o]

def canon_ko(ref, hyp):
    nums = [int(m.group()) for m in re.finditer(r"\d+", ref)]
    forms = []
    for n in nums:
        for f in ko_read(n):
            forms.append((f, n))
    allforms = {f for f, _ in forms}
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
            nxt = hyp[j:j + 2].lstrip()[:1]      # 공백 건너뛴 다음 글자
            ok = True
            if before in SINO_CH:
                ok = False                        # 더 긴 수의 일부
            elif after in SINO_CH and any(g.startswith(f + after) for g in allforms):
                ok = False                        # 더 긴 수사의 접두
            elif len(f) == 1 and not (after in COUNTER or after in PARTICLE
                                      or nxt in COUNTER):
                ok = False                        # 한 글자 수사는 단위/조사 앞에서만
            if ok:
                hyp = hyp[:i] + str(n) + hyp[j:]
                idx = i + len(str(n))
            else:
                idx = i + 1
    return hyp

def canon_en(ref, hyp):
    low = hyp.lower()
    for m in re.finditer(r"\d+", ref):
        n = int(m.group())
        if n == 1:
            continue                              # 'one' 은 대명사와 겹친다
        for f in sorted(en_read(n), key=len, reverse=True):
            pat = re.compile(r"\b" + f.replace(" ", r"[\s-]+") + r"\b", re.I)
            if pat.search(low):
                hyp = pat.sub(str(n), hyp)
                low = hyp.lower()
    return hyp

R = {
 "qwen3_ko":   "/home/skkai/bench-results/qwen3_partial/qwen3_ko.json",
 "voxtral_ko": "/home/skkai/bench-results/laal2_20260911/voxtral_sd0_ko.json",
 "nemo_ko":    "/home/skkai/bench-results/nemo_rc3/nemotron_rc3_ko.json",
 "qwen3_en":   "/home/skkai/bench-results/qwen3_partial/qwen3_en.json",
 "voxtral_en": "/home/skkai/bench-results/laal2_20260911/voxtral_sd0_en.json",
 "nemo_en":    "/home/skkai/bench-results/nemo_rc3/nemotron_rc3_en.json",
}

print("%-12s %-5s %9s %10s %9s  %s" % ("백엔드", "단위", "원래", "정규화후", "차이", "바뀐 클립"))
for k, p in R.items():
    unit = "cer" if k.endswith("ko") else "wer"
    canon = canon_ko if k.endswith("ko") else canon_en
    rows = json.load(open(p))["rows"]
    base, fixed, touched = [], [], 0
    for r in rows:
        if r.get(unit) is None:
            continue
        base.append(r[unit])
        h = canon(r["reference"], r["transcript"])
        if h != r["transcript"]:
            touched += 1
        fixed.append(score(r["reference"], h, unit))
    b, f = sum(base) / len(base), sum(fixed) / len(fixed)
    print("%-12s %-5s %9.4f %10.4f %+9.4f  %d/50" % (k, unit, b, f, f - b, touched))

print("\n오검출 점검 — 바뀐 전사 전부 출력(ko, 각 백엔드 2건):")
for k in ("nemo_ko", "qwen3_ko", "voxtral_ko"):
    rows = json.load(open(R[k])["rows"] if False else json.load(open(R[k]))["rows"]) if False else json.load(open(R[k]))["rows"]
    shown = 0
    for r in rows:
        h = canon_ko(r["reference"], r["transcript"])
        if h != r["transcript"] and shown < 2:
            print("  [%s] REF %s" % (k, r["reference"][:70]))
            print("  %s      후  %s" % (" " * len(k), h[:70]))
            shown += 1
