import json
import re
from dataclasses import dataclass, field

from core.utils import logging

from . import translators
from .api import ApiTranslation
from .local_qwen_3_5_4b.translator import ends_a_sentence, language_name

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the voice of a live interpreter. Speech reaches you in pieces, and \
every piece you write is spoken aloud at once and can never be taken back.

Reply with one JSON object and nothing else:
{"text": "...", "finished": true or false, "corrected": true or false}

Rules for "text", the words to say next in the target language:
1. What was already spoken is fixed. Never repeat it. Write only what comes next, so that \
the already-spoken words followed by your text read as one sentence.
2. If the source sentence is not finished yet, say only what is certain so far: topics, \
objects, time, place. Do not state a main verb, a negation, a refusal or a yes/no unless the \
source already contains it. End so the sentence can still continue.
3. If the source now means something different from what was already spoken, correct it \
briefly and naturally ("- actually, ...", "- or rather, ...") and set "corrected" to true. \
Getting the meaning right matters more than sounding smooth.
4. The earlier sentences are context only. Use them for pronouns, omitted subjects, names and \
terms, and keep every term translated the way you translated it before.
5. Use an empty "text" only when the piece is a bare filler or noise ("uh", "um", "아", "음"). \
For anything with content, say your best guess rather than nothing.

Set "finished" to true when the source sentence is complete."""

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


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
    }

    async def load(self) -> None:
        await super().load()
        self.context_turns = self.settings.get("context_turns", 3)
        self.max_open_pieces = self.settings.get("max_open_pieces", 4)
        self.corrections = 0
        self.start()

    def start(self, **_) -> None:
        self.history: list[tuple[str, str]] = []
        self.open: OpenSentence | None = None

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        text = text.strip()
        if not text or not target_lang:
            return "", ""
        if source_lang == target_lang:
            return text, source_lang
        sentence = self._sentence_for(source_lang or "", target_lang)
        sentence.sources.append(text)
        try:
            reply = self._parse(await self._ask(self._prompt_for(sentence, text)))
        except Exception as e:  # noqa: BLE001
            self.failed += 1
            log.warning("[TRANS-ERROR] %s", e)
            sentence.spoken.append("")
            return "", ""

        spoken = reply["text"]
        sentence.spoken.append(spoken)
        if reply["corrected"]:
            self.corrections += 1
        finished = reply["finished"] or ends_a_sentence(text)
        log.info("[CONTINUATION] piece=%d finished=%s corrected=%s said=%r",
                 len(sentence.sources), finished, reply["corrected"], spoken)
        if finished or len(sentence.sources) >= self.max_open_pieces:
            self._close()
        return spoken, source_lang or ""

    async def close(self) -> None:
        log.info("[CONTINUATION] corrections=%d", self.corrections)
        await super().close()

    def _sentence_for(self, source_lang: str, target_lang: str) -> OpenSentence:
        same_direction = (self.open is not None and self.open.source_lang == source_lang
                          and self.open.target_lang == target_lang)
        if not same_direction:
            self._close()
            self.open = OpenSentence(source_lang, target_lang)
        return self.open

    def _close(self) -> None:
        if self.open is not None and self.open.sources:
            self.history.append((self.open.source(), self.open.said()))
        self.open = None

    def _prompt_for(self, sentence: OpenSentence, newest: str) -> list[dict]:
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
        if not isinstance(data.get("text"), str):
            return {"text": reply.strip().strip('"'), "finished": False, "corrected": False}
        return {"text": data["text"].strip(),
                "finished": bool(data.get("finished")),
                "corrected": bool(data.get("corrected"))}
