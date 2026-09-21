import json
import re

from core.utils.langs import CODE_TO_NAME

from .dialogue import join

SYSTEM = """You are the translation engine of a live conversation. Each request gives you what was said before and one new piece of speech. Translate only the new piece.

What was said before is context. Never translate it again and never repeat its translation. Use it only to resolve what the new piece leaves implicit: who or what is referred to, omitted subjects and objects, gender, politeness level, and names and terms already used.

The new piece is streaming speech recognition output. It may be an incomplete fragment that is finished in the next piece, and it may contain recognition errors. Translate exactly what it says. Do not complete it, and do not borrow words from the context to fill it in.

Reply with one JSON object and nothing else: {"translation": "<the new piece in the target language>"}"""


def name(code: str) -> str:
    return CODE_TO_NAME.get(code, code or "unknown language")


def shown_translation(entry: dict, target_lang: str) -> str | None:
    if entry.get("translation"):
        return entry["translation"]
    if entry.get("lang") == target_lang:
        return entry["text"]
    return None


def turns(context: list[dict], target_lang: str) -> list[dict]:
    merged: list[dict] = []
    for entry in context:
        key = (entry.get("speaker") or "", entry.get("lang") or "")
        if merged and merged[-1]["key"] == key:
            merged[-1]["entries"].append(entry)
        else:
            merged.append({"key": key, "entries": [entry]})
    for turn in merged:
        turn["speaker"], turn["lang"] = turn["key"]
        shown = [shown_translation(entry, target_lang) for entry in turn["entries"]]
        turn["text"] = join([entry["text"] for entry in turn["entries"]], turn["lang"])
        turn["translation"] = (join(shown, target_lang)
                               if None not in shown and turn["lang"] != target_lang else "")
    return merged


def speaker_lines(context: list[dict], target_lang: str) -> str:
    lines = []
    for turn in turns(context, target_lang):
        lines.append(f"{turn['speaker'] or '-'} ({name(turn['lang'])}): {turn['text']}")
        if turn["translation"]:
            lines.append(f"= {turn['translation']}")
    return "\n".join(lines)


def said(context: list[dict], fallback_lang: str) -> tuple[str, str]:
    langs = {entry.get("lang") or fallback_lang for entry in context}
    if len(langs) > 1:
        return " ".join(f"[{name(entry.get('lang') or fallback_lang)}] {entry['text']}"
                        for entry in context), ""
    lang = langs.pop()
    return join([entry["text"] for entry in context], lang), f" (in {name(lang)})"


def seen(context: list[dict], target_lang: str) -> str | None:
    shown = [shown_translation(entry, target_lang) for entry in context]
    if None in shown or not any(entry.get("translation") for entry in context):
        return None
    return join(shown, target_lang)


def by_speaker(context: list[dict], speaker: str, source_lang: str,
               target_lang: str) -> str:
    target = name(target_lang)
    last = context[-1].get("speaker") if context else ""
    if not speaker:
        who = "New piece"
    elif speaker == last:
        who = f"New piece — {speaker} keeps talking"
    else:
        who = f"New piece — speaker change, now {speaker}"
    return (f"Conversation so far, oldest first. Lines starting with \"=\" are the {target} "
            f"translations the listener has already seen.\n\n"
            f"{speaker_lines(context, target_lang)}"
            f"\n\n{who}, in {name(source_lang)}. Translate it into {target}:")


def as_one_block(context: list[dict], source_lang: str, target_lang: str) -> str:
    target = name(target_lang)
    text, where = said(context, source_lang)
    block = (f"What was said before{where}, oldest first. Speakers are not identified: this "
             f"may be one person, or several people taking turns.\n{text}")
    shown = seen(context, target_lang)
    if shown is not None:
        block += f"\n\nWhat the listener has already seen in {target}:\n{shown}"
    return block + f"\n\nNew piece, in {name(source_lang)}. Translate it into {target}:"


def messages(text: str, target_lang: str, source_lang: str, context: list[dict],
             speaker: str) -> tuple[str, str]:
    if not context:
        lead = f"New piece, in {name(source_lang)}. Translate it into {name(target_lang)}:"
    elif speaker or any(entry.get("speaker") for entry in context):
        lead = by_speaker(context, speaker, source_lang, target_lang)
    else:
        lead = as_one_block(context, source_lang, target_lang)
    return SYSTEM, f"{lead}\n{text}"


def extract(reply: str, speaker: str = "") -> str:
    body = reply.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", body, re.S)
    if fenced:
        body = fenced.group(1)
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return ""
    if not isinstance(value, dict) or not isinstance(value.get("translation"), str):
        return ""
    translation = value["translation"].strip()
    if speaker:
        translation = re.sub(rf"^{re.escape(speaker)}\s*[:：]\s*", "", translation)
    return translation.strip().strip('"').strip()
