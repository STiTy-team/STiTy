from ..registry import Registry

translators = Registry("translation")

from .base import Translator  # noqa: E402
from .deepl import DeepLTranslation  # noqa: E402
from .gemini import GeminiTranslation  # noqa: E402
from .gpt import GPTTranslation  # noqa: E402
from .local import LocalTranslation  # noqa: E402
__all__ = ["translators", "Translator", "DeepLTranslation", "GeminiTranslation",
           "GPTTranslation", "LocalTranslation"]
