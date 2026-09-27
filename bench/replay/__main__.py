"""`python -m bench.replay`: the bench dashboard and session replay, on one local page."""

from argparse import Namespace
from pathlib import Path

from core.errors import STiTyError
from core.utils import cli

from .server import PORT, serve
from .session import DEFAULT_TOP_K


def main(args: Namespace) -> int:
    initial = Path(args.run_dir).name if args.run_dir else None
    try:
        return serve(initial, port=args.port, top_k=args.top_k)
    except STiTyError as e:
        print(e)
        return e.exit_code
    except OSError as e:
        print(f"port {args.port} is not available: {e}")
        return 1


if __name__ == "__main__":
    args = cli.parse(
        [
            {
                "name": "run-dir",
                "default": None,
                "help": "bench/runs/<name> 을 주면 대시보드 대신 그 실행의 세션 재생으로 연다. "
                "떠 있는 페이지에서 다른 실행으로 언제든 바꿔 볼 수 있다",
            },
            {
                "name": "port",
                "type": int,
                "default": PORT,
                "help": f"들을 포트 (기본 {PORT})",
            },
            {
                "name": "top-k",
                "type": int,
                "default": DEFAULT_TOP_K,
                "help": f"세션 재생에서 실패·빈 전사 항목은 전부, 나머지는 WER 최악 순으로 "
                f"몇 개까지 보여줄지 (기본 {DEFAULT_TOP_K})",
            },
        ],
        prog="python -m bench.replay",
        description="bench 실행들을 비교하는 대시보드와, 녹음된 세션을 다시 재생하는 페이지",
    )
    raise SystemExit(main(args))
