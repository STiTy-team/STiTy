"""번역 요청 문면. 두 모델이 같은 system / user 를 받는다.

문맥은 앞 문장들에 대해 그 조건의 모델이 낸 번역(한국어)이다. 문맥 형태는 둘이다.

- ``tgt``: 앞 번역만 (1.1 기본)
- ``src_tgt``: 앞 문장마다 영어 원문과 그 번역을 쌍으로 (DialogueContext 의 ``SRC_TGT`` 와 같은 구성)

두 형태는 문맥 설명 한 구절과 문맥 블록만 다르고 나머지 문면은 같다.
"""

CONTEXT_MODES = ("tgt", "src_tgt")

_SYSTEM = ("You are a translation engine for a live talk, translating English into Korean one "
           "sentence at a time. {what} may be given as context only: never "
           "translate them again and never repeat them. Use them only to keep terms, names, "
           "style and politeness level consistent and to resolve references in the CURRENT "
           "sentence. Translate the CURRENT sentence and nothing else, even if it looks "
           "incomplete. Output only the Korean translation, with no explanation and no quotes.")
_WHAT = {"tgt": "Earlier Korean translations",
         "src_tgt": "Earlier English sentences with their Korean translations"}
SYSTEMS = {m: _SYSTEM.format(what=w) for m, w in _WHAT.items()}
SYSTEM = SYSTEMS["tgt"]


def user_message(current: str, previous_translations: list[str], previous_sources: list[str] | None = None,
                 mode: str = "tgt") -> str:
    if not previous_translations:
        return f"Translate the following English sentence into Korean.\n\n{current}"
    if mode == "tgt":
        ctx = "\n".join(f"- {t}" for t in previous_translations)
        head = "Your Korean translations of the preceding sentences of this talk, oldest first"
    elif mode == "src_tgt":
        if previous_sources is None or len(previous_sources) != len(previous_translations):
            raise ValueError("src_tgt needs one source per previous translation")
        ctx = "\n".join(f"EN: {s}\nKO: {t}" for s, t in zip(previous_sources, previous_translations))
        head = "The preceding sentences of this talk with your Korean translations, oldest first"
    else:
        raise ValueError(f"unknown context mode {mode!r}")
    return (f"{head} (context only — do NOT translate or repeat these):\n"
            f"{ctx}\n\n"
            "Translate ONLY the English sentence below into Korean.\n\n"
            f"{current}")


def clean(raw: str) -> str:
    """채점용. 앞뒤 공백·따옴표만 걷는다. 여러 줄 출력은 그대로 두고 flag 로 따로 센다."""
    return raw.strip().strip('"').strip("“”").strip()
