from pathlib import Path

from ..registry import Component


class Translator(Component):

    usage_log: Path | None = None

    async def translate(self, text: str, target_lang: str,
                        source_lang: str | None = None,
                        context: list[dict] | None = None,
                        speaker: str = "") -> tuple[str, str]:
        raise NotImplementedError

    def usage(self) -> dict | None:
        return None
