from ..registry import Registry

transcribers = Registry("transcription")

from .base import Transcriber  # noqa: E402
from .qwen3 import Qwen3Transcription  # noqa: E402
__all__ = ["transcribers", "Transcriber", "Qwen3Transcription"]
