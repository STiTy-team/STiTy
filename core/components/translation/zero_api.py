"""zero-api: translates one line through an OpenAI-compatible chat API, with no extras.

One user message per line: no system prompt, no earlier lines as context, no fallback
model, no error handling. Connecting, the request itself and usage/cost logging are
inherited from the `api` translator, so every call still lands in the usage log.
"""

from . import translators
from .api import ApiTranslation
from .local_qwen_3_5_4b.translator import language_name, without_reasoning


@translators.register("zero-api")
class ZeroApiTranslation(ApiTranslation):
    SETTINGS = {
        "url": ("url", str),
        "model": ("model", str),
        "key_env": ("key_env", str),
        "reasoning_effort": ("reasoning_effort", str),
        "extra_body": ("extra_body", dict),
        "usage_path": ("usage_path", str),
    }

    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str | None = None,
        context: list[str] | None = None,
    ) -> tuple[str, str]:
        request = (
            f"Translate the following {language_name(source_lang)} text into "
            f"{language_name(target_lang)}. Output only the translation.\n\n{text}"
        )
        reply = await self._chat([{"role": "user", "content": request}], self.model)
        return without_reasoning(reply), source_lang or ""
