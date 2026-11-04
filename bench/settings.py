import socket
from pathlib import Path
from typing import Annotated, Self

from pydantic import AfterValidator, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.errors import ConfigError
from core.integrations.s3 import Bucket

ExpandedPath = Annotated[Path, AfterValidator(Path.expanduser)]


class BenchSettings(BaseSettings):
    model_config = SettingsConfigDict(
        alias_generator=str.upper,
        env_ignore_empty=True,
        str_strip_whitespace=True,
        extra="ignore",
    )

    stity_data_root: ExpandedPath
    stity_host: str = Field(default_factory=socket.gethostname)
    stity_job_id: str | None = None
    stity_s3_bucket: str | None = None
    stity_s3_prefix: str = "stity/"
    discord_webhook_url: str | None = None
    xdg_cache_home: ExpandedPath = Path.home() / ".cache"

    @classmethod
    def load(cls) -> Self:
        try:
            return cls()
        except ValidationError as e:
            problems = "\n".join(f"  {error['loc'][0]}: {error['msg']}" for error in e.errors())
            raise ConfigError(
                f"environment variables are missing or invalid:\n{problems}"
            ) from None

    def bucket(self) -> Bucket | None:
        return Bucket(self.stity_s3_bucket, self.stity_s3_prefix) if self.stity_s3_bucket else None
