import secrets
from datetime import datetime, timedelta
from pathlib import PurePosixPath
from typing import Callable, Literal, NamedTuple

from pydantic import BaseModel, Field

from core.integrations.s3 import Bucket, Conflict

from ..store import RemoteRun

KEEP_FINISHED = 100

JobState = Literal["queued", "running", "stopping", "cancelling"]
Outcome = Literal["window_end", "machine_full", "lost", "stopped"]
FinalOutcome = Literal["done", "failed", "cancelled"]


class Attempt(BaseModel):
    host: str
    started_at: datetime
    commit: str | None = None
    ended_at: datetime | None = None
    outcome: Outcome | None = None
    ran_sec: float | None = None
    error: str | None = None


class Job(BaseModel):
    id: str
    branch: str
    pipeline: str
    pipeline_yaml: str
    dataset: str
    dataset_yaml: str
    submitted_at: datetime
    state: JobState = "queued"
    attempts: list[Attempt] = Field(default_factory=list)


class Finished(BaseModel):
    job: Job
    outcome: FinalOutcome
    ended_at: datetime
    run: RemoteRun | None = None
    error: str | None = None


def same_attempt(entry: dict, finished: Finished) -> bool:
    job = entry.get("job") or {}
    return job.get("id") == finished.job.id and len(job.get("attempts") or []) == len(
        finished.job.attempts
    )


class History:
    def __init__(self, bucket: Bucket, key: str):
        self.bucket = bucket
        self.key = key

    def recent(self) -> list[Finished]:
        stored = self.bucket.get_json(self.key)
        return [] if stored is None else [Finished.model_validate(entry) for entry in stored.data]

    def record(self, finished: Finished) -> None:
        while True:
            stored = self.bucket.get_json(self.key)
            earlier = stored.data if stored else []
            if any(same_attempt(entry, finished) for entry in earlier):
                return
            entries = [finished.model_dump(mode="json"), *earlier]
            try:
                self.bucket.put_json(
                    self.key,
                    entries[:KEEP_FINISHED],
                    if_match=stored.etag if stored else None,
                    if_absent=stored is None,
                )
                return
            except Conflict:
                continue


class IllegalTransition(Exception):
    pass


class QueuedJob(NamedTuple):
    job: Job
    etag: str


def require(job: Job, *states: str) -> None:
    if job.state not in states:
        raise IllegalTransition(f"job {job.id} is {job.state}, expected {' or '.join(states)}")


class Queue:
    def __init__(self, bucket: Bucket, host: str, folder: str, history: History | None = None):
        self.bucket = bucket
        self.host = host
        self.folder = folder
        self.history = history
        self.seen: dict[str, QueuedJob] = {}

    def key(self, job_id: str) -> str:
        return f"{self.folder}{job_id}.json"

    def submit(
        self,
        *,
        branch: str,
        pipeline: str,
        pipeline_yaml: str,
        dataset: str,
        dataset_yaml: str,
        now: datetime | None = None,
    ) -> Job:
        now = now or self.bucket.now()
        latest = max((queued.job.submitted_at for queued in self.jobs()), default=None)
        if latest and now <= latest:
            now = latest + timedelta(microseconds=1)
        job = Job(
            id=f"{now:%Y%m%dT%H%M%S}-{secrets.token_hex(2)}",
            branch=branch,
            pipeline=pipeline,
            pipeline_yaml=pipeline_yaml,
            dataset=dataset,
            dataset_yaml=dataset_yaml,
            submitted_at=now,
        )
        self.bucket.put_json(self.key(job.id), job.model_dump(mode="json"), if_absent=True)
        return job

    def get(self, job_id: str) -> QueuedJob | None:
        stored = self.bucket.get_json(self.key(job_id))
        return None if stored is None else QueuedJob(Job.model_validate(stored.data), stored.etag)

    def jobs(self) -> list[QueuedJob]:
        found = [
            self._unchanged_or_get(PurePosixPath(key).stem, etag)
            for key, etag in self.bucket.etags(self.folder).items()
        ]
        present = [queued for queued in found if queued is not None]
        self.seen = {queued.job.id: queued for queued in present}
        return sorted(present, key=lambda queued: (queued.job.submitted_at, queued.job.id))

    def _unchanged_or_get(self, job_id: str, etag: str) -> QueuedJob | None:
        seen = self.seen.get(job_id)
        return seen if seen is not None and seen.etag == etag else self.get(job_id)

    def start(self, queued: QueuedJob, now: datetime) -> QueuedJob:
        job = queued.job
        require(job, "queued")
        previous = job.attempts[-1] if job.attempts else None
        if previous and previous.ended_at:
            now = max(now, previous.ended_at)
        started = job.model_copy(
            update={
                "state": "running",
                "attempts": [*job.attempts, Attempt(host=self.host, started_at=now)],
            }
        )
        return self.save(started, queued.etag)

    def requeue(
        self,
        queued: QueuedJob,
        outcome: Outcome,
        now: datetime,
        ran_sec: float | None = None,
        error: str | None = None,
    ) -> QueuedJob:
        job = queued.job
        require(job, "running", "stopping")
        last = job.attempts[-1]
        ended_at = max(now, last.started_at)
        if ran_sec is None:
            ran_sec = (ended_at - last.started_at).total_seconds()
        closed = last.model_copy(
            update={
                "ended_at": ended_at,
                "outcome": outcome,
                "ran_sec": round(ran_sec, 1),
                "error": error,
            }
        )
        waiting = job.model_copy(
            update={"state": "queued", "attempts": [*job.attempts[:-1], closed]}
        )
        return self.save(waiting, queued.etag)

    def record_commit(self, queued: QueuedJob, commit: str) -> QueuedJob:
        attempts = queued.job.attempts
        with_commit = attempts[-1].model_copy(update={"commit": commit})
        return self.save(
            queued.job.model_copy(update={"attempts": [*attempts[:-1], with_commit]}), queued.etag
        )

    def close(
        self,
        job: Job,
        outcome: FinalOutcome,
        now: datetime,
        *,
        run: RemoteRun | None = None,
        error: str | None = None,
        etag: str | None = None,
    ) -> Finished:
        self.bucket.delete(self.key(job.id), if_match=etag)
        attempts = job.attempts
        if attempts and attempts[-1].ended_at is None:
            last = attempts[-1]
            ended_at = max(now, last.started_at)
            ran_sec = round((ended_at - last.started_at).total_seconds(), 1)
            attempts = [*attempts[:-1], last.model_copy(update={"ended_at": ended_at, "ran_sec": ran_sec})]
        finished = Finished(
            job=job.model_copy(update={"attempts": attempts}),
            outcome=outcome,
            ended_at=now,
            run=run,
            error=error,
        )
        if self.history is not None:
            self.history.record(finished)
        return finished

    def discard(self, job_id: str) -> None:
        self.bucket.delete(self.key(job_id))

    def cancel(self, job_id: str, *, worker_alive: bool) -> Job | None:
        def cancel_once(queued: QueuedJob) -> Job | None:
            if queued.job.state == "queued" or not worker_alive:
                self.close(queued.job, "cancelled", self.bucket.now(), etag=queued.etag)
                return None
            cancelling = queued.job.model_copy(update={"state": "cancelling"})
            return self.save(cancelling, queued.etag).job

        return self.change_until_saved(job_id, cancel_once)

    def stop(self, job_id: str) -> Job | None:
        def stop_once(queued: QueuedJob) -> Job:
            require(queued.job, "running", "stopping")
            return self.save(queued.job.model_copy(update={"state": "stopping"}), queued.etag).job

        return self.change_until_saved(job_id, stop_once)

    def move(self, job_id: str, to: "Queue") -> Job | None:
        def move_once(queued: QueuedJob) -> Job:
            require(queued.job, "queued")
            try:
                to.bucket.put_json(
                    to.key(job_id), queued.job.model_dump(mode="json"), if_absent=True
                )
            except Conflict:
                raise IllegalTransition(f"job {job_id} is already on {to.host}") from None
            try:
                self.bucket.delete(self.key(job_id), if_match=queued.etag)
            except Conflict:
                to.discard(job_id)
                raise
            return queued.job

        return self.change_until_saved(job_id, move_once)

    def put_back_left_running(self, now: datetime) -> list[Job]:
        put_back = []
        for queued in self.jobs():
            try:
                if queued.job.state == "cancelling":
                    self.close(queued.job, "cancelled", now, etag=queued.etag)
                elif queued.job.state in ("running", "stopping"):
                    put_back.append(self.requeue(queued, "lost", now).job)
            except Conflict:
                continue
        return put_back

    def save(self, job: Job, etag: str) -> QueuedJob:
        etag = self.bucket.put_json(self.key(job.id), job.model_dump(mode="json"), if_match=etag)
        saved = QueuedJob(job, etag)
        self.seen[job.id] = saved
        return saved

    def change_until_saved(
        self, job_id: str, change: Callable[[QueuedJob], Job | None]
    ) -> Job | None:
        while True:
            queued = self.get(job_id)
            if queued is None:
                raise KeyError(f"no job {job_id} on {self.host}")
            try:
                return change(queued)
            except Conflict:
                continue

