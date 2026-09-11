import os
from pathlib import Path

from .errors import BenchConfigError

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
LOGS_DIR = BENCH_DIR / "logs"
ITEMS_DIR = BENCH_DIR / "items"

DATA_ROOT_ENV = "STITY_DATA_ROOT"


def data_root(*, create: bool = False) -> Path:
    """create=True is for converters, which populate the root.

    Reads keep create=False so a typo in the variable surfaces as an error rather
    than silently making an empty directory and reporting "dataset not found".
    """
    raw = os.environ.get(DATA_ROOT_ENV)
    if not raw:
        raise BenchConfigError(
            f"{DATA_ROOT_ENV} is not set. Point it at the directory holding the "
            f"converted datasets, e.g. export {DATA_ROOT_ENV}=~/datasets"
        )
    root = Path(raw).expanduser()
    if create:
        root.mkdir(parents=True, exist_ok=True)
    if not root.is_dir():
        raise BenchConfigError(f"{DATA_ROOT_ENV}={root} is not a directory")
    return root


def dataset_dir(name: str) -> Path:
    root = data_root()
    path = root / name
    if not path.is_dir():
        available = sorted(p.name for p in root.iterdir() if (p / "dataset.yml").is_file())
        raise BenchConfigError(
            f"dataset {name!r} not found under {root} (available: {available or 'none'}). "
            f"Run: bash datasets/{name}/install.sh"
        )
    return path


def resolve_model_path(raw: str) -> str:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        candidate = PROJECT_ROOT / raw
        if candidate.exists():
            return str(candidate)
    return str(path) if path.exists() else raw


def ensure_output_dirs() -> None:
    for d in (RESULTS_DIR, LOGS_DIR, ITEMS_DIR):
        d.mkdir(parents=True, exist_ok=True)
