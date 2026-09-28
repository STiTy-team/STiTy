"""Child processes that live exactly as long as this one.

Every tool runs from a uv project, so `uv_command` builds the command: the program, its
positional arguments, then one `--flag value` per keyword (`max_model_len=4096` becomes
`--max-model-len 4096`, `True` a bare flag, `None`/`False` nothing).

`run` runs one such command to completion from the repository root and raises
`ProcessError` if it fails.

`ManagedProcess.start` runs that command, waits until `ready()` says it is serving, and
ties the child's life to ours: it is terminated at normal exit and, on Linux, through
the kernel's parent-death signal, even when we are killed outright. The signal is set
up in the parent: between fork and exec the child only calls an already-resolved
function, because loading a library there can deadlock on a lock another thread held. Its output goes to a log file
whose path is in every startup error.
"""
import asyncio
import atexit
import ctypes
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Awaitable, Callable

from core.errors import ProcessError
from core.utils import logging
from core.utils.paths import get_project_root

log = logging.getLogger(__name__)

PR_SET_PDEATHSIG = 1


def uv_command(project: str | Path, program: str, *args, **flags) -> list[str]:
    argv = ["uv", "run", "--project", str(project), program, *map(str, args)]
    for name, value in flags.items():
        if value is None or value is False:
            continue
        argv.append(f"--{name.replace('_', '-')}")
        if value is not True:
            argv.append(str(value))
    return argv


def run(project: str | Path, program: str, *args, **flags) -> None:
    argv = uv_command(project, program, *args, **flags)
    log.info("[PROCESS] running %s", " ".join(argv))
    sys.stdout.flush()
    code = subprocess.run(argv, cwd=get_project_root()).returncode
    if code:
        raise ProcessError(f"{' '.join(argv)} exited with code {code}")


def _parent_death_signal() -> Callable[[], None] | None:
    if not sys.platform.startswith("linux"):
        return None
    prctl = ctypes.CDLL(None, use_errno=True).prctl
    return lambda: prctl(PR_SET_PDEATHSIG, signal.SIGTERM)


_die_with_parent = _parent_death_signal()


class ManagedProcess:

    def __init__(self, name: str, argv: list[str], *, ready: Callable[[], Awaitable[bool]],
                 startup_timeout_sec: float = 600.0, poll_sec: float = 2.0):
        self.name = name
        self.argv = argv
        self.ready = ready
        self.startup_timeout_sec = startup_timeout_sec
        self.poll_sec = poll_sec
        self.log_path = Path(tempfile.gettempdir()) / f"{name}.log"
        self.process: asyncio.subprocess.Process | None = None

    async def start(self) -> None:
        log.info("[PROCESS] starting %s: %s (log: %s)", self.name, " ".join(self.argv), self.log_path)
        with open(self.log_path, "ab") as log_file:
            self.process = await asyncio.create_subprocess_exec(
                *self.argv, stdout=log_file, stderr=asyncio.subprocess.STDOUT,
                preexec_fn=_die_with_parent)
        atexit.register(self._terminate)
        try:
            await asyncio.wait_for(self._wait_ready(), self.startup_timeout_sec)
        except TimeoutError:
            await self.stop()
            raise ProcessError(f"{self.name} was not ready within {self.startup_timeout_sec}s; "
                               f"see {self.log_path}") from None
        log.info("[PROCESS] %s ready (pid %d)", self.name, self.process.pid)

    async def _wait_ready(self) -> None:
        while not await self.ready():
            if self.process.returncode is not None:
                raise ProcessError(f"{self.name} exited with code {self.process.returncode}; "
                                   f"see {self.log_path}")
            await asyncio.sleep(self.poll_sec)

    async def stop(self, timeout_sec: float = 15.0) -> None:
        if self.process is None or self.process.returncode is not None:
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout_sec)
        except TimeoutError:
            self.process.kill()
            await self.process.wait()

    def _terminate(self) -> None:
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass  # Ctrl-C reached the whole process group; the child is already gone
