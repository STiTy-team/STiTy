import json
import re
import time
from dataclasses import dataclass, field

from core.utils import logging

from . import translators
from .api import ApiTranslation
from .local_qwen_3_5_4b.translator import ends_a_sentence, language_name

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the voice of a live interpreter. The source sentence reaches you in \
pieces. Each reply is spoken aloud at once and can never be taken back.

Reply with one JSON object and nothing else:
{"text": "...", "finished": true or false, "corrected": true or false}

"text" is what to say next, in the target language:
1. Never repeat what was already spoken. "already spoken" followed by "text" must read as one \
sentence.
2. Translate everything the source has said so far, verbs included. Hold back only what the \
source has not said yet: never guess a verb, a negation, a refusal or a yes/no it has not reached.
3. When the request says the sentence ends with this piece, complete the sentence: give the verb \
and the sentence ending, and close it with punctuation.
4. If the source now means something different from what was already spoken, fix it briefly in \
speech ("- actually, ...") and set "corrected" to true.
5. Earlier sentences are context only. Use them for pronouns, omitted subjects and names. Keep \
the same translation for the same term and the same politeness level. Never carry their words \
into this sentence.
6. Leave "text" empty only for a bare filler or noise ("uh", "um", "haha"). Otherwise say your \
best guess.
7. "text" holds only the words to speak: no quotes, no notes, no JSON.

Set "finished" to true when the source sentence is complete.

Example, English into Korean (the Korean verb comes last, so it waits for the source verb):
piece "So if we push the launch" -> {"text": "그러면 출시를", "finished": false, "corrected": false}
piece "to next week, is that okay?" -> {"text": "다음 주로 미뤄도 괜찮을까요?", "finished": true, "corrected": false}

Example, Korean into English:
piece "그 계약은 아직" -> {"text": "That contract is still", "finished": false, "corrected": false}
piece "검토 중이라서 사인 못 해요." -> {"text": "under review, so we can't sign yet.", "finished": true, "corrected": false}"""

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "continuation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "finished": {"type": "boolean"},
                "corrected": {"type": "boolean"},
            },
            "required": ["text", "finished", "corrected"],
            "additionalProperties": False,
        },
    },
}
MIN_REPEAT_CHARS = 4

SCRIPTS = {
    "ko": re.compile(r"[가-힣]"),
    "ja": re.compile(r"[\u3040-\u30ff]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
}
JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
TEXT_FIELD = re.compile(r'"text"\s*:\s*"?(.*?)"?\s*(?:,\s*"(?:finished|corrected)"|\}|$)', re.DOTALL)



def written_in(text: str, lang: str) -> bool:
    if lang in SCRIPTS:
        return bool(SCRIPTS[lang].search(text))
    return not any(script.search(text) for script in SCRIPTS.values())


def checked_source(text: str, source_lang: str | None, target_lang: str) -> str | None:
    if source_lang == target_lang and not written_in(text, target_lang):
        return None
    return source_lang



def without_repeat(said: str, text: str) -> str:
    for size in range(min(len(said), len(text)), MIN_REPEAT_CHARS - 1, -1):
        if said.endswith(text[:size]):
            return text[size:].strip()
    return text


@dataclass
class OpenSentence:
    source_lang: str
    target_lang: str
    sources: list[str] = field(default_factory=list)
    spoken: list[str] = field(default_factory=list)

    def source(self) -> str:
        return " ".join(self.sources)

    def said(self) -> str:
        return " ".join(part for part in self.spoken if part)


@translators.register("continuation")
class ContinuationTranslation(ApiTranslation):
    SETTINGS = {
        **ApiTranslation.SETTINGS,
        "max_open_pieces": ("max_open_pieces", int),
        "new_sentence_after_sec": ("new_sentence_after_sec", float),
    }

    async def load(self) -> None:
        await super().load()
        self.context_turns = self.settings.get("context_turns", 3)
        self.max_open_pieces = self.settings.get("max_open_pieces", 4)
        self.new_sentence_after_sec = self.settings.get("new_sentence_after_sec", 3.0)
        self.extra_body = {"response_format": RESPONSE_FORMAT, **self.extra_body}
        self.corrections = 0
        self.start()

    def start(self, **_) -> None:
        self.history: list[tuple[str, str]] = []
        self.open: OpenSentence | None = None
        self.last_piece_at = 0.0

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        text = text.strip()
        if not text or not target_lang:
            return "", ""
        source = checked_source(text, source_lang, target_lang)
        if source == target_lang:
            return text, source_lang
        sentence = self._sentence_for(source or "", target_lang)
        sentence.sources.append(text)
        closing = ends_a_sentence(text) or len(sentence.sources) >= self.max_open_pieces
        try:
            reply = self._parse(await self._ask(self._prompt_for(sentence, text, closing)))
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            log.warning("[TRANS-ERROR] %s", e)
            sentence.spoken.append("")
            return "", ""

        spoken = without_repeat(sentence.said(), reply["text"])
        sentence.spoken.append(spoken)
        if reply["corrected"]:
            self.corrections += 1
        finished = reply["finished"] or closing
        log.info("[CONTINUATION] piece=%d finished=%s corrected=%s said=%r",
                 len(sentence.sources), finished, reply["corrected"], spoken)
        if finished:
            self._close()
        return spoken, source_lang or ""

    async def close(self) -> None:
        log.info("[CONTINUATION] corrections=%d", self.corrections)
        await super().close()

    def _sentence_for(self, source_lang: str, target_lang: str) -> OpenSentence:
        now = time.monotonic()
        paused = now - self.last_piece_at > self.new_sentence_after_sec
        self.last_piece_at = now
        same_direction = (self.open is not None and self.open.source_lang == source_lang
                          and self.open.target_lang == target_lang)
        if paused or not same_direction:
            self._close()
            self.open = OpenSentence(source_lang, target_lang)
        return self.open

    def _close(self) -> None:
        if self.open is not None and self.open.sources:
            self.history.append((self.open.source(), self.open.said()))
        self.open = None

    def _prompt_for(self, sentence: OpenSentence, newest: str, closing: bool) -> list[dict]:
        source = language_name(sentence.source_lang or None)
        target = language_name(sentence.target_lang)
        lines = [f"Translate {source} speech into {target}."]
        earlier = self.history[-self.context_turns:] if self.context_turns else []
        if earlier:
            lines.append("\nEarlier sentences, oldest first (context only):")
            lines.extend(f"- {self._for_model(original)} => {translation}"
                         for original, translation in earlier)
        lines.append("\nCurrent sentence so far:")
        lines.append(f"source: {self._for_model(sentence.source())}")
        lines.append(f"newest piece: {self._for_model(newest)}")
        lines.append(f"already spoken: {sentence.said() or '(nothing yet)'}")
        lines.append("\nThe sentence ends with this piece." if closing
                     else "\nThe sentence may continue after this piece.")
        return [{"role": "system", "content": self._system_prompt(SYSTEM_PROMPT)},
                {"role": "user", "content": "\n".join(lines)}]

    @staticmethod
    def _parse(reply: str) -> dict:
        reply = reply.split("</think>")[-1].strip()
        found = JSON_OBJECT.search(reply)
        try:
            data = json.loads(found.group(0)) if found else {}
        except json.JSONDecodeError:
            data = {}
        if isinstance(data.get("text"), str):
            return {"text": data["text"].strip(),
                    "finished": bool(data.get("finished")),
                    "corrected": bool(data.get("corrected"))}
        field = TEXT_FIELD.search(reply)
        if field is None and "{" not in reply:
            return {"text": reply.strip('"'), "finished": False, "corrected": False}
        return {"text": field.group(1).strip() if field else "",
                "finished": '"finished": true' in reply,
                "corrected": '"corrected": true' in reply}
