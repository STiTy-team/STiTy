import shutil
import subprocess

from ..machines.machine import GpuInfo

QUERY = "index,name,memory.used,memory.total,utilization.gpu"
FULL_AT_FRACTION = 0.5


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def snapshot() -> list[GpuInfo]:
    if shutil.which("nvidia-smi") is None:
        return []
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        index, name, used, total, util = (part.strip() for part in line.split(","))
        gpus.append(
            GpuInfo(
                index=int(index),
                name=name,
                memory_used_mb=_number(used) or 0.0,
                memory_total_mb=_number(total) or 0.0,
                util=_number(util),
            )
        )
    return gpus


def mostly_in_use(gpus: list[GpuInfo]) -> bool:
    total = sum(gpu.memory_total_mb for gpu in gpus)
    used = sum(gpu.memory_used_mb for gpu in gpus)
    return total > 0 and used / total >= FULL_AT_FRACTION


def describe(gpus: list[GpuInfo]) -> str:
    return ", ".join(
        f"GPU{gpu.index} {gpu.memory_used_mb:.0f}/{gpu.memory_total_mb:.0f} MB" for gpu in gpus
    )
