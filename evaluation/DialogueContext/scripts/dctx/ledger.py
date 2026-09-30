"""결과·비용 jsonl 에 한 줄씩 붙인다. S1 을 여러 프로세스로 돌릴 때도 줄이 섞이지 않게 flock 을 잡는다."""
import fcntl
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

REPORT_EVERY = 25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield f
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def append(path: Path, row: dict) -> None:
    with locked(path) as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # 죽으면서 반쯤 쓴 마지막 줄
    return rows


def _spent(f) -> float:
    f.seek(0)
    total = 0.0
    for line in f:
        try:
            total += float(json.loads(line).get("cost") or 0)
        except (ValueError, AttributeError):
            pass
    return total


class CostLedger:
    """호출마다 usage·추정 비용·누적 비용을 api_usage.jsonl 에 남긴다.
    누적은 파일에서 다시 더하므로 재시작해도, 여러 프로세스가 같이 써도 맞는다."""

    def __init__(self, path: Path, budget_usd: float):
        self.path = path
        self.budget = budget_usd
        self.calls = 0

    def spent(self) -> float:
        with locked(self.path) as f:
            return _spent(f)

    def over_budget(self) -> bool:
        return self.spent() >= self.budget

    def record(self, *, cost: float, **detail) -> float:
        with locked(self.path) as f:
            cumulative = _spent(f) + cost
            line = {"at": now(), **detail, "cost": round(cost, 8),
                    "cumulative_cost": round(cumulative, 8)}
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
        self.calls += 1
        if self.calls % REPORT_EVERY == 0:
            print(f"[cost] {self.calls} api calls in this process, cumulative estimated "
                  f"${cumulative:.4f} (budget ${self.budget})", flush=True)
        return cumulative
