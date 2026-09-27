"""The replay server: the dashboard at `/`, one run's session replay at `/replay`,
and the JSON and audio both pages fetch."""

import json
import mimetypes
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.errors import STiTyError

from ..config import get_runs_dir
from . import runs
from .session import (
    AUDIO_PREFIX,
    DEFAULT_TOP_K,
    clip_samples,
    item_data,
    item_payload,
    page,
    run_clips,
    wav_bytes,
)

PORT = 9130
STATIC = Path(__file__).with_name("static")
STATIC_PREFIX = "/static/"
ITEM_PREFIX = "/item/"
DATA_PREFIX = "/data/"
DISTRIBUTIONS_PREFIX = "/api/distributions/"


def serve(initial_run: str | None, *, port: int = PORT, top_k: int = DEFAULT_TOP_K) -> int:
    runs.resolve_run(None)  # fails fast if there are no runs yet

    clips_by_run: dict[str, dict] = {}
    encoded: dict[tuple[str, str], bytes] = {}

    def clips_for(run_name: str) -> dict:
        if run_name in clips_by_run:
            return clips_by_run[run_name]
        run_dir = get_runs_dir() / run_name
        clips = run_clips(run_dir)
        # A run without its summary is still going: its rows keep growing, and the
        # summary will bring the manifest in, so only a finished run's index is kept.
        if (run_dir / "summary.json").is_file():
            clips_by_run[run_name] = clips
        return clips

    def clip(run_name: str, item_id: str) -> bytes | None:
        clips = clips_for(run_name)
        if item_id not in clips:
            return None
        key = (run_name, item_id)
        if key not in encoded:
            try:
                run_dir = get_runs_dir() / run_name
                encoded[key] = wav_bytes(clip_samples(run_dir, item_id, clips[item_id]))
            except (STiTyError, OSError, RuntimeError) as e:
                print(f"audio for {item_id} ({run_name}) is unreadable: {e}")
                return None
        return encoded[key]

    def run_and_item(rest: str) -> tuple[str | None, str]:
        run_name, _, item_id = rest.partition("/")
        run_name = urllib.parse.unquote(run_name)
        return (run_name if run_name in runs.list_runs() else None), urllib.parse.unquote(item_id)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            # No cache headers at all leaves reload behavior up to each browser's
            # heuristics; these pages change every run and every code edit, so
            # never let a cached copy answer instead of the server.
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, data) -> None:
            if data is None:
                self.send_error(404, "no such run or item")
                return
            self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json")

        def _send_media(self, payload: bytes, content_type: str) -> None:
            start, end = 0, len(payload) - 1
            partial = False
            asked = self.headers.get("Range", "")
            if asked.startswith("bytes="):
                first, _, last = asked[len("bytes=") :].partition("-")
                try:
                    if first:
                        start = int(first)
                        end = int(last) if last else end
                    elif last:
                        start = max(0, len(payload) - int(last))
                except ValueError:
                    start, end = 0, len(payload) - 1
                else:
                    partial = True
            if partial and (start >= len(payload) or start > end):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(payload)}")
                self.end_headers()
                return

            end = min(end, len(payload) - 1)
            chunk = payload[start : end + 1]
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(chunk)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            qs = urllib.parse.parse_qs(parsed.query)
            requested = (qs.get("run") or [None])[0]

            if path in ("/", "/index.html"):
                self._send((STATIC / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
                return

            if path.startswith(STATIC_PREFIX):
                name = urllib.parse.unquote(path[len(STATIC_PREFIX) :])
                target = (STATIC / name).resolve()
                if STATIC.resolve() not in target.parents or not target.is_file():
                    self.send_error(404, "no such file")
                    return
                kind = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if kind.startswith("text/") or kind.endswith("javascript"):
                    kind += "; charset=utf-8"
                self._send(target.read_bytes(), kind)
                return

            if path == "/api/overview":
                self._json(runs.overview())
                return

            if path.startswith(DISTRIBUTIONS_PREFIX):
                run_name = urllib.parse.unquote(path[len(DISTRIBUTIONS_PREFIX) :])
                self._json(runs.distributions(run_name) if run_name in runs.list_runs() else None)
                return

            if path.startswith(AUDIO_PREFIX):
                run_name = runs.resolve_run(requested)
                name = urllib.parse.unquote(path[len(AUDIO_PREFIX) :])
                item_id = name[:-4] if name.endswith(".wav") else name
                data = clip(run_name, item_id)
                if data is None:
                    self.send_error(404, "no audio for that item")
                    return
                self._send_media(data, "audio/wav")
                return

            if path.startswith(DATA_PREFIX):
                run_name, item_id = run_and_item(path[len(DATA_PREFIX) :])
                self._json(item_data(get_runs_dir() / run_name, item_id) if run_name else None)
                return

            if path.startswith(ITEM_PREFIX):
                run_name, item_id = run_and_item(path[len(ITEM_PREFIX) :])
                self._json(item_payload(get_runs_dir() / run_name, item_id) if run_name else None)
                return

            if path == "/replay":
                try:
                    body = page(get_runs_dir() / runs.resolve_run(requested), top_k=top_k)
                except STiTyError as e:
                    self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8")
                    return
                self._send(body, "text/html; charset=utf-8")
                return

            self.send_error(404, "no such page")

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://localhost:{port}/"
        if initial_run:
            url += f"replay?run={urllib.parse.quote(runs.resolve_run(initial_run))}"
        n = len(runs.list_runs())
        print(f"{url}  ({n} run{'s' if n != 1 else ''}, ctrl-c to stop)", flush=True)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - no browser here is not a failure
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0
