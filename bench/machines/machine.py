from datetime import datetime, timedelta
from typing import Annotated, Literal, NamedTuple
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, BaseModel, Field, model_validator

from core.integrations.s3 import Bucket, Conflict

from .queue import History, Queue

FOLDER = "machines/"
CONNECTED_WITHIN = timedelta(minutes=5)
MINUTES_PER_DAY = 24 * 60

HealthState = Literal["idle", "outside_window", "running", "backoff", "paused", "stopping"]


def minutes_of(clock: str) -> int:
    hours, _, minutes = clock.partition(":")
    return int(hours) * 60 + int(minutes)


def clock_time(value: str) -> str:
    hours, separator, minutes = value.partition(":")
    if not (separator and hours.isdigit() and minutes.isdigit() and len(minutes) == 2):
        raise ValueError(f"{value!r} is not a time like 09:30")
    if int(minutes) >= 60 or minutes_of(value) > MINUTES_PER_DAY:
        raise ValueError(f"{value!r} is not between 00:00 and 24:00")
    return f"{int(hours):02d}:{minutes}"


def time_zone_name(value: str) -> str:
    try:
        ZoneInfo(value)
    except (KeyError, ValueError):
        raise ValueError(f"unknown timezone {value!r}") from None
    return value


ClockTime = Annotated[str, AfterValidator(clock_time)]
TimeZoneName = Annotated[str, AfterValidator(time_zone_name)]


class WeeklyBlock(BaseModel):
    day: int = Field(ge=0, le=6)
    start: ClockTime
    end: ClockTime

    @model_validator(mode="after")
    def ends_after_it_starts(self) -> "WeeklyBlock":
        if minutes_of(self.end) <= minutes_of(self.start):
            raise ValueError(f"a block must end after it starts ({self.start}–{self.end})")
        return self


class MachineSettings(BaseModel):
    timezone: TimeZoneName = "Asia/Seoul"
    blocks: list[WeeklyBlock] = Field(default_factory=list)
    paused: bool = False


DEFAULT_SETTINGS = MachineSettings(
    blocks=[WeeklyBlock(day=day, start="00:00", end="04:00") for day in range(7)]
)


class GpuInfo(BaseModel):
    index: int
    name: str
    memory_used_mb: float
    memory_total_mb: float
    util: float | None = None


class WindowInfo(BaseModel):
    open: bool
    ends_at: datetime | None = None
    next_opens_at: datetime | None = None


class Backoff(BaseModel):
    level: int
    until: datetime


class Health(BaseModel):
    host: str
    machine_id: str
    worker_commit: str | None = None
    worker_started_at: datetime
    updated_at: datetime
    state: HealthState
    job: str | None = None
    window: WindowInfo | None = None
    backoff: Backoff | None = None
    gpus: list[GpuInfo] = Field(default_factory=list)
    disk_free_gb: float | None = None
    disk_total_gb: float | None = None
    last_error: str | None = None


class HostTaken(Exception):
    pass


class StillConnected(Exception):
    pass


class StoredSettings(NamedTuple):
    settings: MachineSettings
    etag: str | None


class Notes(NamedTuple):
    text: str
    etag: str | None


class Machine:
    def __init__(self, bucket: Bucket, host: str):
        self.bucket = bucket
        self.host = host
        self.folder = f"{FOLDER}{host}/"
        self.history = History(bucket, f"{self.folder}history.json")
        self.queue = Queue(bucket, host, f"{self.folder}queue/", self.history)

    def health(self) -> Health | None:
        stored = self.bucket.get_json(f"{self.folder}health.json")
        return None if stored is None else Health.model_validate(stored.data)

    def write_health(self, health: Health) -> None:
        self.bucket.put_json(f"{self.folder}health.json", health.model_dump(mode="json"))

    def connected(self, now: datetime, health: Health | None = None) -> bool:
        health = health or self.health()
        return health is not None and now - health.updated_at < CONNECTED_WITHIN

    def settings(self) -> StoredSettings:
        stored = self.bucket.get_json(f"{self.folder}settings.json")
        if stored is None:
            return StoredSettings(MachineSettings(), None)
        return StoredSettings(MachineSettings.model_validate(stored.data), stored.etag)

    def save_settings(self, settings: MachineSettings, etag: str | None) -> str:
        return self.bucket.put_json(
            f"{self.folder}settings.json",
            settings.model_dump(mode="json"),
            if_match=etag,
            if_absent=etag is None,
        )

    def notes(self) -> Notes:
        stored = self.bucket.get_bytes(f"{self.folder}notes.md")
        return (
            Notes("", None) if stored is None else Notes(stored.data.decode("utf-8"), stored.etag)
        )

    def save_notes(self, text: str, etag: str | None) -> str:
        return self.bucket.put_bytes(
            f"{self.folder}notes.md",
            text.encode("utf-8"),
            if_match=etag,
            if_absent=etag is None,
            content_type="text/markdown; charset=utf-8",
        )

    def register(self, machine_id: str) -> None:
        health = self.health()
        if health is not None and health.machine_id != machine_id:
            raise HostTaken(
                f"host name {self.host!r} belongs to another machine ({health.machine_id}); set "
                f"STITY_HOST to another name"
            )
        try:
            self.save_settings(DEFAULT_SETTINGS, etag=None)
        except Conflict:
            pass

    def remove(self, now: datetime) -> None:
        if self.connected(now):
            raise StillConnected(f"{self.host} is connected; stop its worker first")
        for key in self.bucket.list(self.folder):
            self.bucket.delete(key)


def all_machines(bucket: Bucket) -> list[Machine]:
    hosts = {key.removeprefix(FOLDER).split("/", 1)[0] for key in bucket.list(FOLDER)}
    return [Machine(bucket, host) for host in sorted(hosts)]
