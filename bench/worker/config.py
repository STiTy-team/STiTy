import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from core.utils.paths import get_project_root
from pydantic import Field, field_validator
from pydantic_settings import NoDecode

from ..settings import BenchSettings, ExpandedPath

DEFAULT_COMMAND = "make bench CONFIG={pipeline} DATASET={dataset}"
DEFAULT_BACKOFF_SEC = [60, 600, 3600, 10800]
MACHINE_ID_FILE = Path("/etc/machine-id")


def machine_id_of_this_computer() -> str:
    if MACHINE_ID_FILE.is_file():
        return MACHINE_ID_FILE.read_text().strip()
    return socket.gethostname()


class WorkerSettings(BenchSettings):
    stity_s3_bucket: str
    stity_machine_id: str = Field(default_factory=machine_id_of_this_computer)
    stity_worker_repo: ExpandedPath = get_project_root()
    stity_worker_command: str = DEFAULT_COMMAND
    stity_worker_poll_sec: float = 60
    stity_worker_kill_grace_sec: float = 30
    stity_worker_backoff_sec: Annotated[list[float], NoDecode] = DEFAULT_BACKOFF_SEC
    xdg_state_home: ExpandedPath = Path.home() / ".local" / "state"

    @field_validator("stity_worker_backoff_sec", mode="before")
    @classmethod
    def split_commas(cls, value: object) -> object:
        return value.split(",") if isinstance(value, str) else value


@dataclass
class Config:
    host: str
    machine_id: str
    repo: Path
    command: str
    state_dir: Path
    worktree: Path
    poll_sec: float = 60
    kill_grace_sec: float = 30
    backoff_sec: list[float] = field(default_factory=lambda: list(DEFAULT_BACKOFF_SEC))

    @property
    def job_logs(self) -> Path:
        return self.state_dir / "jobs"

    @property
    def running_job_file(self) -> Path:
        return self.state_dir / "running-job.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "worker.lock"

    @classmethod
    def from_settings(cls, settings: WorkerSettings) -> "Config":
        return cls(
            host=settings.stity_host,
            machine_id=settings.stity_machine_id,
            repo=settings.stity_worker_repo,
            command=settings.stity_worker_command,
            state_dir=settings.xdg_state_home / "stity",
            worktree=settings.xdg_cache_home / "stity" / "worktree",
            poll_sec=settings.stity_worker_poll_sec,
            kill_grace_sec=settings.stity_worker_kill_grace_sec,
            backoff_sec=settings.stity_worker_backoff_sec,
        )
