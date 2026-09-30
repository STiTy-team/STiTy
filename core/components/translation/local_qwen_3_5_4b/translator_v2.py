import re

from core.utils import logging, scripts
from core.utils.glossary import Glossary

from .. import translators
from .translator import (
    SYSTEM_PROMPT,
    Qwen35Translation,
    ends_a_sentence,
    language_name,
    without_reasoning,
)

log = logging.getLogger(__name__)

LEAK_SHARE = 0.15
MAX_PIECES = 6
META_ANSWER_RE = re.compile(
    r"^\s*(?:please (?:translate|provide)|sure[,!.]|okay,? i understand|"
    r"here(?:'s| is) the translation|i can help|i'm sorry,? but i|"
    r"you are a translation engine|translate (?:the following|only)|earlier turns)\b",
    re.IGNORECASE)


def _style(value) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping of target language -> instruction, "
                         f"got {type(value).__name__}")
    return {str(k): str(v) for k, v in value.items()}


@translators.register("qwen3.5:v2")
class Qwen35TranslationV2(Qwen35Translation):
    SETTINGS = {
        **Qwen35Translation.SETTINGS,
        "glossary": ("glossary", str),
        "style": ("style", _style),
        "leak_retry": ("leak_retry", bool),
        "meta_guard": ("meta_guard", bool),
        "continuation": ("continuation", bool),
    }

    async def load(self) -> None:
        await super().load()
        name = self.settings.get("glossary")
        self.glossary = Glossary.load(name) if name else None
        self.sentence = {}

    def start(self, **_) -> None:
        self.sentence: dict[tuple, list[tuple[str, str]]] = {}

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[str] | None = None) -> tuple[str, str]:
        if not text.strip() or not target_lang:
            return "", ""
        if source_lang == target_lang:
            return text, source_lang
        pieces = self.sentence.setdefault((source_lang, target_lang), []) \
            if self._continuing() else []
        if len(pieces) >= MAX_PIECES:
            pieces.clear()
        prompt = self._prompt_v2(text.strip(), target_lang, source_lang, context or [], pieces)
        try:
            reply = await self._reply(prompt, text, target_lang, source_lang, context or [])
        except Exception as e:  # noqa: BLE001
            log.warning("[TRANS-ERROR] %s", e)
            return "", ""
        if self._continuing():
            pieces.append((text.strip(), reply))
            if ends_a_sentence(text):
                pieces.clear()
        return reply, source_lang or ""

    def _continuing(self) -> bool:
        return bool(self.settings.get("continuation"))

    async def _reply(self, prompt: list[dict], text: str, target_lang: str,
                     source_lang: str | None, context: list[str]) -> str:
        reply = without_reasoning(await self._chat(prompt))
        if self.settings.get("meta_guard") and META_ANSWER_RE.match(reply) \
                and not META_ANSWER_RE.match(text):
            log.info("[TRANS-META] reply=%r", reply[:80])
            reply = without_reasoning(await self._chat(self._plain_prompt(text, target_lang,
                                                                          source_lang)))
            if META_ANSWER_RE.match(reply):
                return ""
        if self.settings.get("leak_retry") and self._leaks(reply, target_lang):
            retry = without_reasoning(await self._chat(
                self._plain_prompt(text, target_lang, source_lang, strict=True)))
            log.info("[TRANS-LEAK] first=%r retry=%r", reply[:80], retry[:80])
            if scripts.foreign_share(retry, target_lang) < scripts.foreign_share(reply, target_lang):
                reply = retry
        return reply

    @staticmethod
    def _leaks(reply: str, target_lang: str) -> bool:
        return scripts.foreign_share(reply, target_lang) > LEAK_SHARE

    def _instructions(self, text: str, target_lang: str, source_lang: str | None) -> list[str]:
        extra = []
        style = self.settings.get("style", {}).get(target_lang)
        if style:
            extra.append(style)
        if self.glossary is not None and source_lang:
            pairs = self.glossary.pairs_in(text, source_lang, target_lang)
            if pairs:
                listed = "\n".join(f"- {source} -> {target}" for source, target in pairs)
                extra.append(f"Use these fixed translations for names and terms:\n{listed}")
        return extra

    def _plain_prompt(self, text: str, target_lang: str, source_lang: str | None, *,
                      strict: bool = False) -> list[dict]:
        source, target = language_name(source_lang), language_name(target_lang)
        request = f"Translate the following {source} text into {target}.\n\n{text}"
        extra = self._instructions(text, target_lang, source_lang)
        if strict:
            extra.append(f"Write only in {target}. Do not use Chinese characters or any "
                         f"other language except for names.")
        if extra:
            request = "\n\n".join(extra) + "\n\n" + request
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request}]

    def _prompt_v2(self, text: str, target_lang: str, source_lang: str | None,
                   context: list[str], pieces: list[tuple[str, str]]) -> list[dict]:
        if not pieces:
            prompt = self._prompt(text, target_lang, source_lang, context)
            extra = self._instructions(text, target_lang, source_lang)
            if extra:
                prompt[1]["content"] = "\n\n".join(extra) + "\n\n" + prompt[1]["content"]
            return prompt
        source, target = language_name(source_lang), language_name(target_lang)
        said = " ".join(src for src, _ in pieces)
        shown = " ".join(tgt for _, tgt in pieces if tgt)
        request = (f"The speaker is in the middle of a {source} sentence. The part already "
                   f"spoken and its {target} translation, which is already on screen and "
                   f"cannot change:\n{source}: {said}\n{target}: {shown}\n\n"
                   f"The speaker continues with:\n{text}\n\n"
                   f"Write only the {target} words that continue the translation on screen "
                   f"to cover the new part. Do not repeat what is on screen.")
        extra = self._instructions(said + " " + text, target_lang, source_lang)
        if extra:
            request = "\n\n".join(extra) + "\n\n" + request
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request}]
