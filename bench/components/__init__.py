"""Importing this package fires the registry decorators."""
from . import transcription, translation  # noqa: F401

__all__ = ["transcription", "translation"]
