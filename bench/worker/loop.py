import fcntl
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Callable, TextIO

from core.integrations.s3 import Bucket, Conflict
from core.utils import logging

from .. import notify
from ..machines import timetable
from ..machines.machine import CONNECTED_WITHIN, Backoff, Health, Machine, MachineSettings
from ..machines.queue import IllegalTransition, Job
from ..machines.timetable import Window
from . import checkout, gpu
from .claimed import ClaimedJob, Instruction
from .clock import BucketClock
from .config import Config
from .process import JobProcess, RunningJobFile

log = logging.getLogger(__name__)

ERROR_CHARS = 1500
STOP_SIGNAL_WAIT_SEC = 2
WATCH_STEP_SEC = 0.5
CRASHED_REASON = "worker가 죽었다가 다시 켜졌어요"
HEALTH_HEARTBEAT = CONNECTED_WITHIN * 0.6


class AlreadyRunning(Exception):
    pass


class EndReason(Enum):
    EXITED = auto()
    CANCELLED = auto()
    WINDOW_END = auto()
    WORKER_STOPPING = auto()
    STOP_REQUESTED = auto()


PUT_BACK = {
    EndReason.WINDOW_END: ("window_end", "이 머신의 시간대가 끝났어요"),
    EndReason.WORKER_STOPPING: ("stopped", "worker가 꺼졌어요 (재부팅이나 systemctl stop)"),
    EndReason.STOP_REQUESTED: ("stopped", "Queue 페이지에서 멈추라고 했어요"),
}


@dataclass
class Ending:
    reason: EndReason
    exit_code: int | None = None


def lock_machine(lock_file: Path) -> TextIO:
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_file, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise AlreadyRunning(f"another worker already runs on this machine ({lock_file})") from None
    return handle


class Worker:
    def __init__(self, bucket: Bucket, config: Config):
        self.bucket = bucket
        self.config = config
        self.machine = Machine(bucket, config.host)
        self.clock = BucketClock(bucket)
        self.running_job = RunningJobFile(config.running_job_file)
        self.started_at = self.clock.now()
        self.commit = checkout.head(config.repo)
        self.stopping = threading.Event()
        self.backoff_level = 0
        self.backoff_until: datetime | None = None
        self.last_error: str | None = None
        self.health_shown: dict | None = None
        self.health_written_at: datetime | None = None
        self.unsettled: Callable[[], None] | None = None

    def stop(self, *_signal_args) -> None:
        self.stopping.set()

    def run(self) -> None:
        self.lock = lock_machine(self.config.lock_file)
        self.machine.register(self.config.machine_id)
        self._recover_after_crash()
        log.info("[WORKER-START] host=%s commit=%s", self.config.host, self.commit)
        while not self.stopping.is_set():
            try:
                self.tick()
                self.last_error = None
            except Exception as e:
                self._remember_error(e)
            self.stopping.wait(self.config.poll_sec)
        self._write_health("stopping")
        log.info("[WORKER-STOP] host=%s", self.config.host)

    def tick(self) -> None:
        if self.unsettled is not None:
            self.unsettled()
            self.unsettled = None
        now = self.clock.now()
        settings = self.machine.settings().settings
        window = timetable.current_window(settings, now)
        waiting_state = self._waiting_state(settings, now)
        if waiting_state:
            self._write_health(waiting_state, settings=settings)
            return
        claimed = self._claim_next(now, in_window=window is not None)
        if claimed is None:
            self._write_health("idle" if window else "outside_window", settings=settings)
            return
        self.run_job(claimed, window, settings)

    def run_job(self, claimed: ClaimedJob, window: Window | None, settings: MachineSettings) -> None:
        job = claimed.job
        try:
            cwd, commit = checkout.worktree_for(self.config.repo, self.config.worktree, job)
        except checkout.CheckoutError as e:
            error = str(e)
            self._settle_until_done(lambda: self._not_started(claimed, error))
            return
        claimed.record_commit(commit)

        job_process = JobProcess.start(job, self.config.command, cwd, self.config.job_logs)
        self.running_job.remember(job, job_process)
        notify.job_started(
            job,
            host=self.config.host,
            commit=commit,
            window_ends=None if job.run_now or window is None else window.end,
            timezone=settings.timezone,
        )
        self._write_health("running", job, settings=settings)
        try:
            ending = self._watch(claimed, job_process, window)
        finally:
            job_process.stop(self.config.kill_grace_sec)
            self.running_job.forget()
        self._settle_until_done(lambda: self._settle(claimed, job_process, ending))

    def _settle_until_done(self, settle: Callable[[], None]) -> None:
        self.unsettled = settle
        settle()
        self.unsettled = None

    def _not_started(self, claimed: ClaimedJob, error: str) -> None:
        claimed.end("failed", error=error)
        notify.job_not_started(claimed.job, host=self.config.host, error=error)

    def _watch(self, claimed: ClaimedJob, job_process: JobProcess, window: Window | None) -> Ending:
        next_check_in = time.monotonic() + self.config.poll_sec
        while True:
            exit_code = job_process.exit_code()
            if exit_code is not None:
                return Ending(EndReason.EXITED, exit_code)
            if self.stopping.is_set():
                return Ending(EndReason.WORKER_STOPPING)
            if not claimed.job.run_now and (window is None or self.clock.now() >= window.end):
                return Ending(EndReason.WINDOW_END)
            if time.monotonic() < next_check_in:
                self.stopping.wait(WATCH_STEP_SEC)
                continue
            next_check_in = time.monotonic() + self.config.poll_sec
            try:
                ending, window = self._check_in(claimed, job_process)
            except Exception as e:
                self._remember_error(e)
                continue
            if ending:
                return ending

    def _check_in(
        self, claimed: ClaimedJob, job_process: JobProcess
    ) -> tuple[Ending | None, Window | None]:
        instruction = claimed.check()
        if instruction is Instruction.CANCEL:
            return Ending(EndReason.CANCELLED), None
        if instruction is Instruction.STOP:
            return Ending(EndReason.STOP_REQUESTED), None
        settings = self.machine.settings().settings
        self._write_health("running", claimed.job, settings)
        return None, timetable.current_window(settings, self.clock.now())

    def _settle(self, claimed: ClaimedJob, job_process: JobProcess, ending: Ending) -> None:
        if ending.reason is EndReason.EXITED and ending.exit_code == 0:
            self.backoff_level, self.backoff_until = 0, None
            claimed.end("done", run=job_process.uploaded_run())
            return
        if ending.reason is EndReason.EXITED and self.stopping.wait(STOP_SIGNAL_WAIT_SEC):
            ending = Ending(EndReason.WORKER_STOPPING)
        match ending.reason:
            case EndReason.EXITED:
                self._settle_failure(claimed, job_process, ending.exit_code)
            case EndReason.CANCELLED:
                claimed.end("cancelled")
            case _:
                outcome, reason = PUT_BACK[ending.reason]
                ran_sec = job_process.elapsed_sec()
                if claimed.put_back(outcome, ran_sec):
                    notify.job_requeued(
                        claimed.job, host=self.config.host, reason=reason, ran_sec=ran_sec
                    )

    def _settle_failure(self, claimed: ClaimedJob, job_process: JobProcess, exit_code: int) -> None:
        gpus = gpu.snapshot()
        if not gpu.mostly_in_use(gpus):
            log_tail = "\n".join(job_process.last_lines())[-ERROR_CHARS:]
            error = f"exit {exit_code}\n{log_tail}"
            claimed.end("failed", error=error)
            notify.job_failed(claimed.job, host=self.config.host, error=error)
            return
        claimed.put_back("machine_full", job_process.elapsed_sec(), error="GPUs held by others")
        delay_sec = self._back_off()
        notify.machine_full(
            claimed.job,
            host=self.config.host,
            gpus=gpu.describe(gpus),
            retry_in=notify.korean_duration(delay_sec),
        )

    def _back_off(self) -> float:
        self.backoff_level += 1
        steps = self.config.backoff_sec
        delay_sec = steps[min(self.backoff_level, len(steps)) - 1]
        self.backoff_until = self.clock.now() + timedelta(seconds=delay_sec)
        log.warning("[MACHINE-FULL] backoff level %d, %ds", self.backoff_level, delay_sec)
        return delay_sec

    def _waiting_state(self, settings: MachineSettings, now: datetime) -> str | None:
        if settings.paused:
            return "paused"
        if self.backoff_until and self.backoff_until > now:
            return "backoff"
        return None

    def _claim_next(self, now: datetime, *, in_window: bool) -> ClaimedJob | None:
        waiting = [queued for queued in self.machine.queue.jobs() if queued.job.state == "queued"]
        urgent = next((queued for queued in waiting if queued.job.run_now), None)
        picked = urgent or (waiting[0] if waiting and in_window else None)
        if picked is None:
            return None
        try:
            started = self.machine.queue.start(picked, now)
        except (Conflict, IllegalTransition):
            return None
        job = started.job
        log.info("[JOB-START] %s %s on %s", job.id, job.pipeline, job.dataset)
        return ClaimedJob(self.machine.queue, self.clock, started)

    def _recover_after_crash(self) -> None:
        leftover = self.running_job.kill_leftover(self.config.kill_grace_sec)
        if leftover:
            log.warning("[WORKER-RECOVER] killed job %s left running", leftover)
        for job in self.machine.queue.put_back_left_running(self.clock.now()):
            log.warning("[WORKER-RECOVER] put back job %s, lost when this worker died", job.id)
            last_attempt = job.attempts[-1]
            notify.job_requeued(
                job, host=self.config.host, reason=CRASHED_REASON, ran_sec=last_attempt.ran_sec
            )

    def _remember_error(self, error: Exception) -> None:
        self.last_error = f"{type(error).__name__}: {error}"
        log.warning("[WORKER-ERROR] %s", self.last_error)

    def _write_health(
        self,
        state: str,
        job: Job | None = None,
        settings: MachineSettings | None = None,
    ) -> None:
        now = self.clock.now()
        backing_off = self.backoff_until and self.backoff_until > now
        shown = dict(
            state=state,
            job=job.id if job else None,
            window=timetable.window_info(settings or self.machine.settings().settings, now),
            backoff=Backoff(level=self.backoff_level, until=self.backoff_until)
            if backing_off
            else None,
            last_error=self.last_error,
        )
        heartbeat_due = (
            self.health_written_at is None
            or now - self.health_written_at >= HEALTH_HEARTBEAT
        )
        if shown == self.health_shown and not heartbeat_due:
            return
        disk = shutil.disk_usage(self.config.repo)
        health = Health(
            host=self.config.host,
            machine_id=self.config.machine_id,
            worker_commit=self.commit,
            worker_started_at=self.started_at,
            updated_at=now,
            gpus=gpu.snapshot(),
            disk_free_gb=round(disk.free / 2**30, 1),
            disk_total_gb=round(disk.total / 2**30, 1),
            **shown,
        )
        self.machine.write_health(health)
        self.health_shown, self.health_written_at = shown, now
