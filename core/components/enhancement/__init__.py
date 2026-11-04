from ..registry import Registry, discover

enhancers = Registry("enhancement")

discover(__name__)
