"""백엔드 WebSocket 서버 진입점.

    python server.py --backend nemotron --right-context 13 --port 8765

엔진은 --backend 로 고른다. 엔진 모듈은 지연 import 한다 - conda env 마다 설치된
라이브러리가 다르므로, 쓰지 않는 백엔드의 import 가 서버를 죽이면 안 된다.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from protocol import BackendServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["nemotron"], required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--lang", default="ko-KR")
    ap.add_argument("--right-context", type=int, default=13,
                    help="Nemotron att_context_size 의 right. 0/1/3/6/13 = 80ms~1.12s")
    ap.add_argument("--log-file")
    args = ap.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(format="%(asctime)s %(levelname)s\t%(message)s",
                        level=logging.INFO, handlers=handlers)

    if args.backend == "nemotron":
        from engine_nemotron import NemotronEngine, CHUNK_BY_RIGHT

        note = f"right_context={args.right_context} chunk={CHUNK_BY_RIGHT.get(args.right_context)}s"

        def factory():
            return NemotronEngine(right_context=args.right_context, lang=args.lang)
    else:
        raise SystemExit(f"unknown backend {args.backend}")

    BackendServer(factory, host=args.host, port=args.port,
                  backend_name=args.backend, config_note=note).run()


if __name__ == "__main__":
    main()
