from ..registry import Registry, discover

correctors = Registry("correction")

discover(__name__)
