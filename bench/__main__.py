import argparse
import json
import logging
import sys

from . import config, paths
from .errors import BenchError

logger = logging.getLogger("bench")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m bench",
        description="Run a STiTy benchmark from a YAML config.",
    )
    p.add_argument("--config", required=True, help="path to the run config (YAML)")
    p.add_argument("--limit", type=int, default=None, help="override dataset.limit")
    p.add_argument("--name", default=None, help="override the run name")
    p.add_argument("--top-k", type=int, default=None, dest="top_k",
                   help="override logs.top_k")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and validate the config, print it, and exit before "
                        "loading the model")
    p.add_argument("--no-resume", action="store_true",
                   help="ignore any existing items file and start over")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(levelname)s\t%(message)s",
                        level=getattr(logging, args.log_level))

    try:
        cfg = config.load(
            args.config,
            overrides={"limit": args.limit, "name": args.name, "top_k": args.top_k},
        )
    except BenchError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(json.dumps({
            "name": cfg.name,
            "fingerprint": cfg.fingerprint(),
            "overrides": cfg.overrides,
            "resolved": cfg.resolved(),
        }, indent=2, ensure_ascii=False, sort_keys=False))
        return 0

    # Imported here, not at module level: it pulls in torch and vLLM, so --dry-run
    # and the config errors above stay fast and GPU-free.
    from . import driver

    paths.ensure_output_dirs()
    try:
        return driver.main(cfg, resume=not args.no_resume)
    except BenchError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
