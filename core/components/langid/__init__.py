from ..registry import Registry, discover

langids = Registry("langid")

discover(__name__)
