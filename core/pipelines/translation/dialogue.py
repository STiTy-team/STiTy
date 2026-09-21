from string import ascii_uppercase

from core.utils.langs import CHAR_UNIT_LANGS


DEFAULT_COUNT = 10
DEFAULT_MAX_CHARS = 500


def speaker_label(index: int) -> str:
    return ascii_uppercase[index % 26] * (index // 26 + 1)


def join(parts: list[str], lang: str) -> str:
    return ("" if lang in CHAR_UNIT_LANGS else " ").join(parts)


def recent(context: list | None, target: str, count: int = DEFAULT_COUNT,
           max_chars: int = DEFAULT_MAX_CHARS) -> list[dict]:
    entries = [dict(entry) if isinstance(entry, dict) else {"text": str(entry)}
               for entry in (context or [])]
    entries = [entry for entry in entries if str(entry.get("text") or "").strip()]
    kept: list[dict] = []
    used = 0
    for entry in reversed(entries[-count:] if count > 0 else []):
        used += len(entry["text"])
        if used > max_chars:
            break
        kept.append(entry)
    kept.reverse()
    if any(entry.get("lang") != target and not entry.get("translation") for entry in kept):
        for entry in kept:
            entry.pop("translation", None)
    return kept


class Dialogue:

    def __init__(self):
        self.group = None
        self.finals: list[dict] = []
        self.labels: dict[str, str] = {}

    def open(self, *, group: str) -> None:
        if group != self.group:
            self.group, self.finals, self.labels = group, [], {}

    def label(self, speaker: str) -> str:
        if not speaker:
            return ""
        if speaker not in self.labels:
            self.labels[speaker] = speaker_label(len(self.labels))
        return self.labels[speaker]

    def said(self, original: str, translation: str, *, lang: str, target: str,
             speaker: str = "") -> None:
        if original.strip():
            self.finals.append({"text": original.strip(), "lang": lang, "target": target,
                                "translation": (translation or "").strip(),
                                "speaker": self.label(speaker)})

    def context(self, target: str) -> list[dict]:
        entries = []
        for final in self.finals:
            entry = {"text": final["text"], "lang": final["lang"]}
            if final["speaker"]:
                entry["speaker"] = final["speaker"]
            if (final["translation"] and final["target"] == target
                    and final["lang"] != target):
                entry["translation"] = final["translation"]
            entries.append(entry)
        return entries
