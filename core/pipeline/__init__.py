from core.components.registry import Registry, discover
from core.errors import ConfigError
from core.utils.config import as_component

from .base import Pipeline, Processor, RunsPer, Stage

pipelines = Registry("pipeline")

discover(__name__)

__all__ = ["Pipeline", "Processor", "RunsPer", "Stage", "pipelines", "validate", "build",
           "build_part", "build_processor", "describe", "stages_of"]

DEFAULT_PARTS = {"mixer": "mixdown"}


def _registries(pipeline_name: str) -> list[tuple]:
    cls = pipelines.get(pipeline_name)
    return ([(registry, True) for registry in cls.REQUIRED]
            + [(registry, False) for registry in cls.OPTIONAL])


def validate(cfg_stity) -> dict:
    spec = cfg_stity.pipeline
    options = dict(spec.options)
    parts = {}

    for registry, required in _registries(spec.name):
        raw = options.pop(registry.kind, None)
        if raw is None:
            if required:
                raise ConfigError(
                    f"pipeline {spec.name!r} cannot run without a {registry.kind}; "
                    f"its options have no {registry.kind!r} key."
                )
            continue
        try:
            name, part_options = as_component(raw)
        except ValueError as e:
            raise ConfigError(f"pipeline.{registry.kind}: {e}") from None
        parts[registry.kind] = {
            "name": name,
            "kwargs": registry.get(name).validate(part_options, kind=registry.kind),
        }

    leftover = options
    return {"pipeline": pipelines.get(spec.name).validate(leftover), "parts": parts}


def build(cfg) -> Pipeline:
    resolved = cfg.stity.resolved
    by_kind = {registry.kind: registry
               for registry, _ in _registries(cfg.stity.pipeline.name)}
    parts = {kind: by_kind[kind].get(part["name"])(part["kwargs"], cfg=cfg)
             for kind, part in resolved["parts"].items()}
    return pipelines.get(cfg.stity.pipeline.name)(
        resolved["pipeline"], parts=parts, cfg=cfg)


def stages_of(cfg) -> tuple[Stage, ...]:
    return pipelines.get(cfg.stity.pipeline.name).STAGES


async def build_part(cfg, kind: str):
    by_kind = {registry.kind: registry
               for registry, _ in _registries(cfg.stity.pipeline.name)}
    part = cfg.stity.resolved["parts"].get(kind)
    if part is None and kind in DEFAULT_PARTS:
        part = {"name": DEFAULT_PARTS[kind], "kwargs": {}}
    if part is None:
        return None
    built = by_kind[kind].get(part["name"])(part["kwargs"], cfg=cfg)
    await built.load()
    return built


async def build_processor(cfg, stage: Stage) -> Processor:
    parts = {kind: await build_part(cfg, kind) for kind in stage.parts}
    return stage.processor({kind: part for kind, part in parts.items() if part is not None})


def describe(cfg) -> dict:
    resolved = cfg.stity.resolved
    return {"pipeline": cfg.stity.pipeline.name, **resolved["pipeline"],
            **{kind: {"backend": part["name"], **part["kwargs"]}
               for kind, part in resolved["parts"].items()}}
