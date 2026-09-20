from ..registry import Registry, discover

translators = Registry("translation")

discover(__name__)
