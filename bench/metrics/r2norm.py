"""r2 채점 정규화: 정답과 전사에 똑같이 적용해, 모델이 틀리지 않았는데 틀린 것으로 잡히는 표기 차이를 없앤다.

데이터셋 설정의 `dataset.r2norm` 으로 켠다(환경변수 STITY_R2NORM 이 있으면 그것이 우선한다). 값은 all, 또는 쉼표로 고른 부분(fillers, variants, dialect, numbers)이다.
숫자 맞춤(numbers.py)보다 먼저 원문에 적용한다. 목록과 근거는 Confluence [8차][북극성 전사 정답 리뉴얼(r2)] 에 있다.

- fillers: 뜻 없는 군말을 지운다. 감탄·반응(아, ah, oh, 응)은 번역에 나가는 말이라 지우지 않는다.
- variants: 같은 말의 다른 표기를 한 꼴로 맞춘다(것/거, 예요/에요 …). 줄임말 JMT 는 풀어 쓴 존맛탱, 철자로 읽은 J-M-T 와 같게 본다.
- dialect: 북극성 081·087 의 부산 사투리 어미를 표준어 꼴로 맞춘다. 낱말 자체가 다른 것(직이다/찍다, 마)은 넣지 않는다.
- numbers: 숫자 맞춤이 놓치는 꼴을 먼저 풀어 둔다(이차 -> 2 차, 한잔 -> 한 잔).
"""
import re

FILLERS = {"um", "umm", "uh", "uhm", "er", "erm", "hmm", "mm", "mhm", "음", "어", "어어", "흠"}
VARIANTS = [("것", "거"), ("예요", "에요"), ("구요", "고요"), ("같애", "같아"), ("쫌", "좀"), ("뭘로", "뭐로"), ("크래쉬", "크래시"), ("존맛탱", "JMT")]
WORD_VARIANTS = {"ok": "okay", "j-m-t": "JMT"}
DIALECT = {"가시믄": "가시면", "가가": "가서", "넘어갔제": "넘어갔지", "되는갑다": "되는가 보다", "드이소": "드세요",
           "직입니더": "직입니다"}
PUNCT = ".,!?…~;:\"'“”‘’()[]—–-"
SINO = {"일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
NATIVE = {"한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
CHA = re.compile(r"^([일이삼사오육칠팔구])차(.*)$")
GLUED = re.compile(r"^(다섯|여섯|일곱|여덟|아홉|한|두|세|네|열)(시간|마리|잔|병|명|번|시|개|살|분)(.*)$")
PARTS = ("fillers", "variants", "dialect", "numbers")


def parts(value):
    value = (value or "").strip().lower()
    if value in ("1", "all", "true", "yes"):
        return set(PARTS)
    chosen = {p.strip() for p in value.split(",") if p.strip()}
    unknown = chosen - set(PARTS)
    if unknown:
        raise ValueError(f"r2norm: unknown parts {sorted(unknown)}")
    return chosen


def _token(tok, use):
    lead = tok[:len(tok) - len(tok.lstrip(PUNCT))]
    tail = tok[len(tok.rstrip(PUNCT)):]
    core = tok[len(lead):len(tok) - len(tail)] if tail else tok[len(lead):]
    if not core:
        return tok
    if "fillers" in use and core.lower() in FILLERS:
        return ""
    if "dialect" in use and core in DIALECT:
        core = DIALECT[core]
    if "variants" in use:
        core = WORD_VARIANTS.get(core.lower(), core)
        for a, b in VARIANTS:
            core = core.replace(a, b)
    if "numbers" in use:
        m = CHA.match(core)
        if m:
            core = f"{SINO[m.group(1)]} 차{m.group(2)}"
        else:
            m = GLUED.match(core)
            if m:
                core = f"{m.group(1)} {m.group(2)}{m.group(3)}"
    return lead + core + tail


def apply(text, use):
    if not text or not use:
        return text
    return " ".join(t for t in (_token(w, use) for w in text.split()) if t)
