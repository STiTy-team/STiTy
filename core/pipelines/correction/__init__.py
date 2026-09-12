from core.pipelines.registry import Registry

correctors = Registry("correction")

from .base import Corrector  # noqa: E402

__all__ = ["correctors", "Corrector"]
