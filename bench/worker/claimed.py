from enum import Enum

from core.integrations.s3 import Conflict
from core.utils import logging

from ..machines.queue import FinalOutcome, Outcome, Queue, QueuedJob
from ..store import RemoteRun
from .clock import BucketClock

log = logging.getLogger(__name__)

WRITE_ATTEMPTS = 5


class Instruction(Enum):
    KEEP_RUNNING = "keep_running"
    STOP = "stop"
    CANCEL = "cancel"


class ClaimedJob:
    def __init__(self, queue: Queue, clock: BucketClock, queued: QueuedJob):
        self.queue = queue
        self.clock = clock
        self.queued = queued

    @property
    def job(self):
        return self.queued.job

    def record_commit(self, commit: str) -> None:
        try:
            self.queued = self.queue.record_commit(self.queued, commit)
        except Conflict:
            pass

    def check(self) -> Instruction:
        current = self.queue.get(self.job.id)
        if current is None or current.job.state == "cancelling":
            return Instruction.CANCEL
        self.queued = current
        if current.job.state == "stopping":
            return Instruction.STOP
        return Instruction.KEEP_RUNNING

    def put_back(self, outcome: Outcome, ran_sec: float, error: str | None = None) -> bool:
        for _ in range(WRITE_ATTEMPTS):
            try:
                self.queued = self.queue.requeue(
                    self.queued, outcome, self.clock.now(), ran_sec, error
                )
                log.info("[JOB-REQUEUED] %s outcome=%s ran=%.0fs", self.job.id, outcome, ran_sec)
                return True
            except Conflict:
                if self.check() is Instruction.CANCEL:
                    self.end("cancelled")
                    return False
                if self.queued.job.state == "queued":
                    return True
        return False

    def end(self, outcome: FinalOutcome, run: RemoteRun | None = None, error: str | None = None) -> None:
        self.queue.close(self.job, outcome, self.clock.now(), run=run, error=error)
        log.info("[JOB-END] %s outcome=%s run=%s", self.job.id, outcome, run and run.folder)
