import hashlib
import json
from pathlib import Path


def source_items(path: str | Path) -> tuple[Path, Path]:
    candidate = Path(path).resolve()
    items = candidate if candidate.is_file() else candidate / "items.jsonl"
    if not items.is_file():
        raise ValueError(f"items.jsonl not found: {items}")
    return items.parent, items


def read_rows(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{lineno} is not a JSON object")
            rows.append(value)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_output(path: Path, names, *, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in names if (path / name).exists()]
    if existing and not overwrite:
        raise ValueError(f"output already exists; pass --overwrite: {path}")
    for target in existing:
        target.unlink()
