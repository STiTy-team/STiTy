import argparse
import json
from pathlib import Path

from core.utils import langs
from core.utils.metrics import Utterance, comet
from core.utils.stream import read


class Languages:

    def __init__(self, lang: str, target: str):
        self.lang = lang
        self.target = target

    def expected_target(self, src_lang: str) -> str:
        return self.lang if langs.norm_code(src_lang) == self.target else self.target


def score(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    languages = Languages(**summary["config"]["languages"])
    items = [Utterance.from_row(row) for row in read(run_dir / "items.jsonl")
             if row.get("status") == "ok"]

    values, unavailable = comet.corpus(items, languages=languages)
    summary["unavailable"].pop("comet", None)
    summary["metrics"].update(values)
    summary["unavailable"].update(unavailable)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                            encoding="utf-8")
    return values or unavailable


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.comet",
        description="bench 실행 하나에 COMET 을 채점해 summary.json 에 더한다",
    )
    parser.add_argument("run_dir", type=Path, help="bench/runs/<이름>")
    args = parser.parse_args(argv)
    print(json.dumps(score(args.run_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
