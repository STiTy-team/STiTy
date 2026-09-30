"""번역 요청 문면. 두 모델이 같은 system / user 를 받는다.

문맥은 앞 문장들에 대해 **그 조건의 모델이 낸 번역(한국어)만**이다. 원문은 넣지 않는다.
"""

SYSTEM = ("You are a translation engine for a live talk, translating English into Korean one "
          "sentence at a time. Earlier Korean translations may be given as context only: never "
          "translate them again and never repeat them. Use them only to keep terms, names, "
          "style and politeness level consistent and to resolve references in the CURRENT "
          "sentence. Translate the CURRENT sentence and nothing else, even if it looks "
          "incomplete. Output only the Korean translation, with no explanation and no quotes.")


def user_message(current: str, previous_translations: list[str]) -> str:
    if not previous_translations:
        return f"Translate the following English sentence into Korean.\n\n{current}"
    ctx = "\n".join(f"- {t}" for t in previous_translations)
    return ("Your Korean translations of the preceding sentences of this talk, oldest first "
            "(context only — do NOT translate or repeat these):\n"
            f"{ctx}\n\n"
            "Translate ONLY the English sentence below into Korean.\n\n"
            f"{current}")


def clean(raw: str) -> str:
    """채점용. 앞뒤 공백·따옴표만 걷는다. 여러 줄 출력은 그대로 두고 flag 로 따로 센다."""
    return raw.strip().strip('"').strip("“”").strip()
