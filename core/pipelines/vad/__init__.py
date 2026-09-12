from ..registry import Registry

detectors = Registry("vad")

from .base import Detector, Speech  # noqa: E402
from .silero import SileroVad  # noqa: E402
__all__ = ["detectors", "Detector", "Speech", "SileroVad"]
