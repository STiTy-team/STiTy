from pathlib import Path

from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from core import config as core_config
from core.config import PipelineConfig
from core.utils.config import ConfigBody
from core.utils.paths import get_project_root

CONFIG_DIR = get_project_root() / "server" / "configs"
BASE_FILE = CONFIG_DIR / "application.yml"
LOCAL_OVERLAY = CONFIG_DIR / "application.local.yml"


class CorsSettings(BaseModel):
    allow_origins: list[str] = ["http://localhost:8081"]
    allow_origin_regex: str | None = None
    allow_methods: list[str] = ["*"]
    allow_headers: list[str] = ["*"]


class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    cors: CorsSettings = CorsSettings()


class RecordingSettings(BaseModel):
    enabled: bool = False
    dir: Path = get_project_root() / "server" / "logs" / "recordings"


class StitySettings(BaseModel):
    pipeline: str


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STITY_", env_nested_delimiter="__")

    server: ServerSettings = ServerSettings()
    recording: RecordingSettings = RecordingSettings()
    stity: StitySettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        **_: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        files = [
            f
            for f in [
                CONFIG_DIR / "application.yml",
                CONFIG_DIR / "application.local.yml",
            ]
            if f.is_file()
        ]

        yaml = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=files,
            deep_merge=True,
        )

        return init_settings, env_settings, yaml


class ServerConfig(ConfigBody):
    server: ServerSettings
    recording: RecordingSettings
    pipeline_name: str
    stity: PipelineConfig


def load_config() -> ServerConfig:
    settings = AppSettings()
    return ServerConfig.parse(
        {
            "server": settings.server.model_dump(),
            "recording": settings.recording.model_dump(),
            "pipeline_name": settings.stity.pipeline,
            "stity": core_config.load_pipeline(settings.stity.pipeline),
        },
        root="server config",
    )
