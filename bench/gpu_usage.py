import os
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from core.utils import logging
from core.utils.json import JsonlWriter, read_jsonl
from core.utils.paths import get_project_root

log = logging.getLogger(__name__)

MIB = 1024 * 1024
INTERVAL_SEC = 1.0
STOP_GRACE_SEC = 5.0


class GpuSampler:
    """Runs the sampler as its own process, so an NVML crash or hang cannot take the run with it."""

    def __init__(self, timeline: Path):
        self.timeline = timeline
        self._process: subprocess.Popen | None = None
        self._failure: str | None = None

    def start(self) -> "GpuSampler":
        root = str(get_project_root())
        path = os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")]))
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "bench.gpu_usage", str(self.timeline), str(os.getpid())],
                cwd=root,
                env={**os.environ, "PYTHONPATH": path},
                stdin=subprocess.DEVNULL,
            )
        except OSError as e:
            self._failure = f"GPU sampler did not start: {e}"
            log.warning("[GPU] %s", self._failure)
        return self

    def stop(self) -> None:
        if self._process is None:
            return
        try:
            self._process.send_signal(signal.SIGTERM)
            self._process.wait(timeout=STOP_GRACE_SEC)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        except OSError:
            pass
        if self._process.returncode not in (0, -signal.SIGTERM):
            self._failure = f"GPU sampler exited with code {self._process.returncode}"
            log.warning("[GPU] %s", self._failure)
        self._process = None

    def scores(self) -> tuple[dict, dict]:
        try:
            metrics, unavailable = summarize(
                list(read_jsonl(self.timeline)) if self.timeline.is_file() else []
            )
        except Exception as e:
            log.warning("[GPU] could not read %s: %s", self.timeline, e)
            return {}, {"gpu": f"could not read the GPU timeline: {e}"}
        if not metrics and self._failure:
            return {}, {"gpu": self._failure}
        return metrics, unavailable


def summarize(lines: list[dict]) -> tuple[dict, dict]:
    reasons = [line["unavailable"] for line in lines if "unavailable" in line]
    samples = [line for line in lines if "ours_mib" in line]
    if not samples:
        return {}, {"gpu": reasons[0] if reasons else "no GPU sample was taken"}
    by_process: dict[str, int] = defaultdict(int)
    for sample in samples:
        for name, used in sample["processes"].items():
            by_process[name] = max(by_process[name], used)
    metrics = {
        "gpu_peak_mib": max(s["ours_mib"] for s in samples),
        "gpu_mean_mib": round(sum(s["ours_mib"] for s in samples) / len(samples)),
        "gpu_other_peak_mib": max(s["other_mib"] for s in samples),
        "gpu_by_process_mib": dict(sorted(by_process.items())),
    }
    utils = [s["util"] for s in samples if s.get("util") is not None]
    if not utils:
        return metrics, {"gpu_util_mean": "the driver reported no per-process utilization"}
    return {**metrics, "gpu_util_mean": round(sum(utils) / len(utils), 3)}, {}


class _Nvml:
    def __init__(self):
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.devices = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
        self._util_seen = [0] * len(self.devices)

    def processes(self, device) -> dict[int, float]:
        found: dict[int, float] = {}
        for listing in (
            self.nvml.nvmlDeviceGetComputeRunningProcesses,
            self.nvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            try:
                procs = listing(device)
            except self.nvml.NVMLError:
                continue
            for proc in procs:
                if proc.usedGpuMemory:
                    found[proc.pid] = max(found.get(proc.pid, 0.0), proc.usedGpuMemory / MIB)
        return found

    def util(self, index: int, pids: set[int]) -> float | None:
        try:
            samples = self.nvml.nvmlDeviceGetProcessUtilization(self.devices[index], self._util_seen[index])
        except self.nvml.NVMLError:
            return None
        if samples:
            self._util_seen[index] = max(s.timeStamp for s in samples)
        return min(1.0, sum(s.smUtil for s in samples if s.pid in pids) / 100)


def _process_tree(root_pid: int) -> dict[int, str]:
    import psutil

    root = psutil.Process(root_pid)
    pids = {}
    for proc in [root, *root.children(recursive=True)]:
        try:
            pids[proc.pid] = proc.name()
        except psutil.Error:
            continue
    pids.pop(os.getpid(), None)
    return pids


def _sample(nvml: _Nvml, root_pid: int, started: float) -> dict:
    ours = _process_tree(root_pid)
    ours_mib = other_mib = 0.0
    by_process: dict[str, float] = defaultdict(float)
    utils, devices = [], []
    for index, device in enumerate(nvml.devices):
        device_ours = 0.0
        for pid, used in nvml.processes(device).items():
            if pid in ours:
                device_ours += used
                by_process[ours[pid]] += used
            else:
                other_mib += used
        ours_mib += device_ours
        if device_ours and (util := nvml.util(index, set(ours))) is not None:
            utils.append(util)
        memory = nvml.nvml.nvmlDeviceGetMemoryInfo(device)
        devices.append({"index": index, "used_mib": round(memory.used / MIB), "total_mib": round(memory.total / MIB)})
    return {
        "t": round(time.monotonic() - started, 1),
        "ours_mib": round(ours_mib),
        "other_mib": round(other_mib),
        "util": round(sum(utils) / len(utils), 3) if utils else None,
        "processes": {name: round(used) for name, used in by_process.items()},
        "devices": devices,
    }


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(timeline: Path, root_pid: int) -> None:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    writer = JsonlWriter(timeline)
    try:
        nvml = _Nvml()
    except Exception as e:
        writer.write({"unavailable": f"NVML unavailable: {e}"})
        return
    if not nvml.devices:
        writer.write({"unavailable": "no NVIDIA GPU"})
        return
    started, warned = time.monotonic(), False
    while _alive(root_pid):
        try:
            writer.write(_sample(nvml, root_pid, started))
        except Exception as e:
            if not warned:
                print(f"[GPU] sampling failed, keeps trying: {e}", file=sys.stderr, flush=True)
                warned = True
        time.sleep(INTERVAL_SEC)


if __name__ == "__main__":
    main(Path(sys.argv[1]), int(sys.argv[2]))
