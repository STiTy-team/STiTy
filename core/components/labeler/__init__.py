from ..registry import Registry, discover

labelers = Registry("labeler")

discover(__name__)
