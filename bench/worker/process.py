import json
import os
import re
import signal
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..machines.queue import Job
from ..store import RemoteRun

UPLOADED_LINE = re.compile(r"UPLOADED\]? runs/([^/\s]+)/([^/\s]+)/([^/\s]+)/")
TAIL_LINES = 30


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def kill_group(pgid: int, grace_sec: float, reap: Callable[[], object] = lambda: None) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        reap()
        if not group_alive(pgid):
            return
        time.sleep(0.2)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@dataclass
class JobProcess:
    process: subprocess.Popen
    log_path: Path
    started_monotonic: float

    @classmethod
    def start(cls, job: Job, command: str, cwd: Path, log_dir: Path) -> "JobProcess":
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{job.id}.log"
        command_line = command.format(pipeline=job.pipeline, dataset=job.dataset, job=job.id)
        with open(log_path, "a", encoding="utf-8") as log:
            log.write(f"$ {command_line}\n")
            log.flush()
            process = subprocess.Popen(
                ["bash", "-c", command_line],
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
                env={**os.environ, "STITY_JOB_ID": job.id},
            )
        return cls(process, log_path, time.monotonic())

    @property
    def pgid(self) -> int:
        return self.process.pid

    def exit_code(self) -> int | None:
        return self.process.poll()

    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_monotonic

    def stop(self, grace_sec: float) -> None:
        if self.exit_code() is not None:
            return
        kill_group(self.pgid, grace_sec, reap=self.process.poll)
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    def last_lines(self) -> list[str]:
        with open(self.log_path, encoding="utf-8", errors="replace") as log:
            return [line.rstrip("\n") for line in deque(log, maxlen=TAIL_LINES)]

    def uploaded_run(self) -> RemoteRun | None:
        with open(self.log_path, encoding="utf-8", errors="replace") as log:
            runs = [RemoteRun(*match.groups()) for line in log if (match := UPLOADED_LINE.search(line))]
        return runs[-1] if runs else None


class RunningJobFile:
    def __init__(self, path: Path):
        self.path = path

    def remember(self, job: Job, process: JobProcess) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"job": job.id, "pgid": process.pgid}))

    def forget(self) -> None:
        self.path.unlink(missing_ok=True)

    def kill_leftover(self, grace_sec: float) -> str | None:
        if not self.path.is_file():
            return None
        leftover = json.loads(self.path.read_text())
        self.forget()
        if not group_alive(leftover["pgid"]):
            return None
        kill_group(leftover["pgid"], grace_sec)
        return leftover["job"]
