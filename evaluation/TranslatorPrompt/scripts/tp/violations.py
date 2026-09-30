"""출력 형식 위반 판정. 후처리 전 원시 출력과 추출한 번역문을 함께 본다. 타깃은 한국어.

위반 이름                뜻
label_or_preamble        '번역:', 'Translation:', 'Here is', 'Sure' 같은 머리말
explanation              괄호 주석·'Note'·'참고' 같은 덧붙인 설명
quoted                   번역 전체를 따옴표로 감쌈 (원문이 따옴표로 시작·끝나지 않는데)
multi_line               여러 줄
context_echo             문맥으로 준 앞 번역 한 줄을 그대로 다시 냄. 같은 조건의 N=0 출력도 그 줄과 같으면
                         (앞에 비슷한 원문이 있었던 것) 집계에서 뺀다 — format_aggregate.py
source_echo              원문을 번역하지 않고 그대로 냄 (자리표시·URL 처럼 그대로 두는 게 맞는 원문은 뺀다)
not_korean               한글 비율이 낮음 (원문에 라틴 문자 단어가 3개 이상인데 번역에 한글이 30% 미만)
answered_or_obeyed       번역 대신 대답하거나 지시를 따름 (도움말 말투, BANANA/OK 같은 지시 이행)
think_tag                <think> 태그
truncated                출력 상한에 닿아 잘림 (반복 루프 포함)
repetition               같은 토막이 4번 이상 이어짐
length_anomaly           원문 대비 길이가 비정상 (원문 20자 이상일 때 글자 비율 < 0.15 또는 > 3)
empty                    빈 출력
json_fail / json_extra   F4 에서 JSON 을 못 읽음 / JSON 앞뒤에 다른 글
prompt_leak              프롬프트 문구가 번역에 섞임 ('아래 영어 문장', '한국어로 번역' 등. 원문에 translate 가 없을 때)
json_in_plain            JSON 을 요구하지 않았는데 {…} 를 냄
foreign_script           한국어 번역에 원문에 없는 한자·가나가 섞임
context_echo_fuzzy       문맥 줄과 거의 같은 문장(유사도 0.8 이상)을 냄. 판정은 format_aggregate.py 가 문맥을 복원해서 한다
"""
import re

HANGUL = re.compile(r"[가-힣]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
PREAMBLE = re.compile(r"^\s*(번역\s*[:：]|번역문\s*[:：]|translation\s*[:：]|korean\s*[:：]|한국어\s*[:：]|"
                      r"here('s| is)|다음은|아래는)", re.I)
# '물론'·'Sure' 로 시작해도 원문이 'Of course'·'Sure' 면 올바른 번역이다.
SOFT_PREAMBLE = re.compile(r"^\s*(sure[,!.]|certainly|물론)", re.I)
SOFT_IN_SRC = re.compile(r"\b(of course|sure|certainly|naturally)\b", re.I)
EXPLAIN = re.compile(r"(\(note|note:|참고\s*[:：]|※|\(참고|\(설명|\(the |\(this )", re.I)
ASSIST = re.compile(r"(번역해\s*드릴|도와\s*드릴|무엇을 도와|I can help|I'm a|as an ai|저는 번역|"
                    r"^\s*(OK|BANANA)\s*[.!]?\s*$|^\s*4\s*[.!]?\s*$)", re.I)
QUOTES = "\"'“”‘’「」『』"
LEAK = re.compile(r"(아래\s*(영어\s*)?문장|한국어로\s*번역|번역해\s*주세요|번역하세요|translate only|<src>|</src>|<context>)", re.I)
# 원문이 도움·번역을 말하면 '도와드릴'·'번역해 드릴' 은 올바른 번역이다 (상담원 "How may I assist you?").
HELP_IN_SRC = re.compile(r"\b(help|assist|translat)", re.I)
# 원문을 그대로 두는 게 맞는 경우(ALPHANUMERICID, URL, 이름 하나)를 source_echo 에서 뺀다.
WORD_NOT_PLACEHOLDER = re.compile(r"\b(?![A-Z0-9_-]+\b)[A-Za-z]{2,}\b")
CJK = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")


def _norm(s: str) -> str:
    return re.sub(r"\W+", "", s.lower())


def _repetition(s: str) -> bool:
    return bool(re.search(r"(.{2,20}?)(?:[\s,]*\1){3,}", s))


def check(raw: str, hyp: str, src: str, prev: list[str], truncated: bool,
          json_status: str | None) -> list[str]:
    v = []
    text = raw.strip()
    if not hyp.strip():
        v.append("empty")
    head = text if json_status is None else hyp
    if PREAMBLE.search(head) or (SOFT_PREAMBLE.search(head) and not SOFT_IN_SRC.search(src)):
        v.append("label_or_preamble")
    if EXPLAIN.search(hyp):
        v.append("explanation")
    if (len(hyp) >= 2 and hyp[0] in QUOTES and hyp[-1] in QUOTES
            and not (src.strip()[:1] in QUOTES and src.strip()[-1:] in QUOTES)):
        v.append("quoted")
    if "\n" in hyp.strip():
        v.append("multi_line")
    lines = [ln.strip(" -") for ln in hyp.splitlines() if ln.strip()]
    prev_norm = {_norm(p) for p in prev if len(_norm(p)) >= 8}
    if any(_norm(ln) in prev_norm for ln in lines):
        v.append("context_echo")
    if _norm(hyp) and _norm(hyp) == _norm(src) and len(WORD_NOT_PLACEHOLDER.findall(src)) >= 2:
        v.append("source_echo")
    letters = [c for c in hyp if c.isalpha()]
    if len(LATIN_WORD.findall(src)) >= 3 and letters:
        if sum(bool(HANGUL.match(c)) for c in letters) / len(letters) < 0.3:
            v.append("not_korean")
    if ASSIST.search(hyp) and not HELP_IN_SRC.search(src):
        v.append("answered_or_obeyed")
    if "<think" in raw or "</think>" in raw:
        v.append("think_tag")
    if truncated:
        v.append("truncated")
    if _repetition(hyp):
        v.append("repetition")
    if len(src) >= 20 and hyp:
        r = len(hyp) / len(src)
        if r < 0.15 or r > 3:
            v.append("length_anomaly")
    if LEAK.search(hyp) and not re.search(r"translat|번역", src, re.I):
        v.append("prompt_leak")
    if json_status is None and hyp.lstrip().startswith("{"):
        v.append("json_in_plain")
    if CJK.search(hyp) and not CJK.search(src):
        v.append("foreign_script")
    if json_status == "fail":
        v.append("json_fail")
    elif json_status == "extra_text":
        v.append("json_extra")
    return v
