from ..registry import Registry

pipelines = Registry("pipeline")

from .base import Pipeline  # noqa: E402
from .cascade import CascadePipeline  # noqa: E402

__all__ = ["pipelines", "Pipeline", "CascadePipeline"]
