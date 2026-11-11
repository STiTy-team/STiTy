"""Score a finished run again for another target language.

A two-way conversation run (dataset config with `lang`) is scored by bench for
`target` only: the turns spoken in `lang`. This scores the other direction from the
same items.jsonl -- the turns translated back into `lang` -- and writes it next to
the run, so `python -m bench.metrics.comet --run-dir <out>` can add COMET.

    PYTHONPATH=. bench/.venv/bin/python bench/tools/rescore_target.py --run-dir <run> --target ko
"""

import argparse
import json
from pathlib import Path

from bench.metrics import score
from bench.metrics.translation import translation_sentences
from core.utils.json import read_jsonl, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--target", required=True)
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / f"target-{args.target}"
    out.mkdir(exist_ok=True)
    rows = list(read_jsonl(run / "items.jsonl"))
    for row in rows:
        if row.get("status") == "ok":
            row.update(score.score_row(row, args.target))
    metrics, unavailable = score.score_run(rows, args.target)
    unavailable["comet"] = "scored separately by python -m bench.metrics.comet"
    write_jsonl(out / "comet_inputs.jsonl", translation_sentences(rows, args.target))
    base = json.loads((run / "summary.json").read_text())
    summary = {
        "name": f"{base['name']} -> {args.target}",
        "target": args.target,
        "metrics": metrics,
        "unavailable": unavailable,
        "counts": base.get("counts"),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(out)


if __name__ == "__main__":
    main()
