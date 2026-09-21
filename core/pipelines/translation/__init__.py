from ..registry import Registry

translators = Registry("translation")

from .base import Translator  # noqa: E402
from .local import LocalTranslation  # noqa: E402
__all__ = ["translators", "Translator", "LocalTranslation"]
