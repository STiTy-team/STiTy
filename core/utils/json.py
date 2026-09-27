import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from core.errors import DataError


def _dumps(data: Any, **options) -> str:
    return json.dumps(data, ensure_ascii=False, default=str, **options)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, data: Any) -> None:
    Path(path).write_text(_dumps(data, indent=2), encoding="utf-8")


def read_jsonl(path: str | Path, *, strict: bool = False) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if strict:
                    raise DataError(f"{path}:{lineno}: invalid JSON ({e})") from e


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(_dumps(row) + "\n")


class JsonlWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a", encoding="utf-8")

    def write(self, row: Any) -> None:
        self._file.write(_dumps(row) + "\n")
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
