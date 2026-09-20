from .registry import Component, Final, Partial, Speech, Transcribed, discover

discover(__name__, packages=True)

__all__ = ["Component", "Final", "Partial", "Speech", "Transcribed"]
