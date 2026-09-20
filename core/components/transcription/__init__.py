from ..registry import Registry, discover

transcribers = Registry("transcription")

discover(__name__)
