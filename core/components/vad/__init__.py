from ..registry import Registry, discover

detectors = Registry("vad")

discover(__name__)
