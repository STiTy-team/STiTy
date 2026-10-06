"""`python -m bench.manager.server`: the API the manager web app talks to."""

from argparse import Namespace

from core.errors import STiTyError
from core.utils import cli, env

from .app import PORT, serve
from .session import DEFAULT_TOP_K


def main(args: Namespace) -> int:
    try:
        return serve(port=args.port, top_k=args.top_k)
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
        prog="python -m bench.manager.server",
        description="manager 웹 앱이 부르는 API: queue, 설정, 실행 비교, 세션 재생",
    )
    env.load()
    raise SystemExit(main(args))
