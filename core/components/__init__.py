from .registry import Component, Partial, Record, Speech, Transcribed, Translated, discover

discover(__name__)

__all__ = ["Component", "Partial", "Record", "Speech", "Transcribed", "Translated"]
