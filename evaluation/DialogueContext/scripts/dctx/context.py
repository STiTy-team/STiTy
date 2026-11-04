"""정준 요청, 프롬프트·DeepL 문맥 직렬화, 출력 정리. DESIGN.md 의 "조건" 절을 그대로 옮긴 것이다."""
import re

STRATEGIES = ("SRC", "TGT", "SRC_TGT", "SPK_SRC_TGT")
BASELINE = "NONE"
LANG_NAME = {"ko": "Korean", "en": "English"}

SYSTEM = (
    "You are a real-time dialogue translator.\n"
    "Translate only the CURRENT UTTERANCE from SOURCE LANGUAGE to TARGET LANGUAGE.\n"
    "Use CONTEXT only to resolve ambiguity and maintain discourse consistency.\n"
    "Do not translate or repeat the context.\n"
    "Preserve all information in the current utterance and do not add information that is "
    "not supported by it or the context.\n"
    "Preserve speaker intent, register, pronoun/reference consistency, and natural "
    "conversational style.\n"
    "Output only the translation."
)

USER_TEMPLATE = (
    "SOURCE LANGUAGE: {source}\n"
    "TARGET LANGUAGE: {target}\n"
    "\n"
    "CONTEXT (previous turns, oldest first):\n"
    "{context}\n"
    "\n"
    "CURRENT UTTERANCE{speaker}:\n"
    "{utterance}"
)


def speaker_labels(turns: list[dict]) -> dict:
    """대화 안 첫 등장 순서로 Speaker 1, Speaker 2 ..."""
    labels = {}
    for turn in sorted(turns, key=lambda t: t["turn_id"]):
        if turn["speaker"] not in labels:
            labels[turn["speaker"]] = f"Speaker {len(labels) + 1}"
    return labels


def canonical_request(*, source_language: str, target_language: str, utterance: str,
                      strategy: str, n: int, previous: list[dict], labels: dict,
                      current_speaker: str | None) -> dict:
    """previous 는 오래된 것부터의 앞 턴 전체 ({speaker, ko, en} 또는 {speaker, src, tgt}).
    뒤에서 n 개를 잘라 전략에 맞는 칸만 채운다."""
    items = []
    if strategy != BASELINE and n > 0:
        for turn in previous[-n:]:
            src = turn.get("src", turn.get(source_language))
            tgt = turn.get("tgt", turn.get(target_language))
            items.append({
                "speaker": labels[turn["speaker"]] if strategy == "SPK_SRC_TGT" else None,
                "src": src if strategy in ("SRC", "SRC_TGT", "SPK_SRC_TGT") else None,
                "tgt": tgt if strategy in ("TGT", "SRC_TGT", "SPK_SRC_TGT") else None,
            })
    return {
        "source_language": source_language,
        "target_language": target_language,
        "current_utterance": utterance,
        "current_speaker": labels[current_speaker] if strategy == "SPK_SRC_TGT" else None,
        "context_strategy": strategy,
        "context_n": n,
        "context_items": items,
    }


def context_block(request: dict) -> str:
    """CONTEXT 블록 본문. DeepL 의 context 로도 이 문자열을 그대로 쓴다. 문맥이 없으면 빈 문자열."""
    lines = []
    for item in request["context_items"]:
        if item["speaker"]:
            lines.append(f"[{item['speaker']}]")
        if item["src"] is not None:
            lines.append(f"SRC: {item['src']}")
        if item["tgt"] is not None:
            lines.append(f"TGT: {item['tgt']}")
    return "\n".join(lines)


def user_prompt(request: dict) -> str:
    speaker = request["current_speaker"]
    return USER_TEMPLATE.format(
        source=LANG_NAME.get(request["source_language"], request["source_language"]),
        target=LANG_NAME.get(request["target_language"], request["target_language"]),
        context=context_block(request) or "(none)",
        speaker=f" ({speaker})" if speaker else "",
        utterance=request["current_utterance"],
    )


_QUOTES = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("「", "」")]
_LABEL = re.compile(
    r"^\s*(?:\*\*)?(?:translation|english translation|english|tgt|target|"
    r"current utterance[^:\n]*)(?:\*\*)?\s*:\s*(?:\*\*)?\s*", re.IGNORECASE)
_SPEAKER_TAG = re.compile(r"^\s*\[Speaker \d+\]\s*")
_EXPLAIN = re.compile(
    r"\((?:note|lit\.|literally|explanation|or\b|i\.e\.)|\b(?:note|explanation|translation)\s*:|"
    r"here(?:'s| is) the translation", re.IGNORECASE)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _unquote(s: str) -> str:
    for left, right in _QUOTES:
        if len(s) >= 2 and s.startswith(left) and s.endswith(right) and left not in s[1:-1]:
            return s[1:-1].strip()
    return s


def clean_output(raw: str) -> tuple[str, str | None]:
    """(hypothesis, format_violation). 여러 줄은 고르지 않고 줄바꿈만 공백으로 바꾼 채 사유를 남긴다."""
    reasons = []
    s = raw or ""
    if "<think>" in s or "</think>" in s:
        # 생각 모드를 껐을 때 Qwen 이 내는 빈 <think></think> 는 위반으로 치지 않는다
        if re.sub(r"</?think>", "", "".join(_THINK.findall(s)) or s.split("</think>")[0]).strip():
            reasons.append("think_block")
        s = _THINK.sub("", s)
        s = s.split("</think>")[-1]
    s = s.strip()
    for _ in range(3):
        before = s
        s = _unquote(s)
        s = _SPEAKER_TAG.sub("", s)
        s = _LABEL.sub("", s).strip()
        if s == before:
            break
    lines = [line.strip() for line in s.splitlines() if line.strip()]
    if len(lines) > 1:
        reasons.append(f"multi_line:{len(lines)}")
        s = " ".join(lines)
    s = _unquote(s.strip())
    if _EXPLAIN.search(s):
        reasons.append("explanation")  # 고치지 않고 표시만 한다
    if not s:
        reasons.append("empty")
    return s, "; ".join(reasons) or None
