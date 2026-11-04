"""The manager's API: the queue, configs, run overviews and session replay data and audio."""

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from core.errors import STiTyError

from ...config import get_runs_dir
from ...settings import BenchSettings
from . import dataset_manager, runs
from .dataset_manager import DatasetError
from .queue import QueueApi
from .session import (
    DEFAULT_TOP_K,
    clip_samples,
    item_data,
    item_payload,
    payload,
    run_clips,
    wav_bytes,
)

PORT = 29130
DISTRIBUTIONS_PREFIX = "/api/distributions/"
REPLAY_API = "/api/replay"
DATASETS_API = "/api/datasets"
MAX_BODY = 64 * 1024


def serve(*, port: int = PORT, top_k: int = DEFAULT_TOP_K) -> int:
    settings = BenchSettings.load()
    bucket = settings.bucket()
    if bucket is None:
        runs.resolve_run(None)  # fails fast if there are no runs yet

    dm = dataset_manager.DatasetManager(settings.stity_data_root)

    clips_by_run: dict[str, dict] = {}
    encoded: dict[tuple[str, str], bytes] = {}

    def forget_cached_audio(run_name: str) -> None:
        clips_by_run.pop(run_name, None)
        for key in [key for key in encoded if key[0] == run_name]:
            encoded.pop(key)

    queue_api = QueueApi(bucket, on_pulled=forget_cached_audio)

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

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)

        def _json(self, data, status: int = 200) -> None:
            if data is None:
                self.send_error(404, "no such run or item")
                return
            self._send(
                json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json", status
            )

        def do_POST(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)

            if path.startswith(DATASETS_API):
                length = int(self.headers.get("Content-Length") or 0)
                if length > dataset_manager.MAX_RECORD_BYTES:
                    self._json({"error": "take 가 너무 크다"}, 413)
                    return
                raw = self.rfile.read(length)
                self._datasets_api("POST", path[len(DATASETS_API) :], qs, raw)
                return

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                self._json({"ok": False, "error": "request too large"}, 413)
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json({"ok": False, "error": "body is not JSON"}, 400)
                return
            status, reply = queue_api.handle_post(path, body)
            self._json(reply, status)

        def do_HEAD(self) -> None:
            # <audio>/<video> probe a resource with HEAD before ranged GETs; every
            # response path already goes through _send/_send_media, which skip the
            # body write for self.command == "HEAD", so the same routing answers both.
            self.do_GET()

        def do_DELETE(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)
            if path.startswith(DATASETS_API):
                self._datasets_api("DELETE", path[len(DATASETS_API) :], qs, None)
                return
            self.send_error(404, "no such endpoint")

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
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload[start : end + 1])

        def _replay_api(self, rest: str, qs: dict) -> None:
            run_name = (qs.get("run") or [None])[0]
            item_id = (qs.get("item") or [""])[0]
            if rest == "":
                try:
                    data = payload(get_runs_dir() / runs.resolve_run(run_name), top_k=top_k)
                except STiTyError as e:
                    self._json({"error": str(e)}, 404)
                    return
                self._json(data)
                return
            if run_name not in runs.list_runs():
                self._json({"error": f"no run named {run_name!r}"}, 404)
                return
            run_dir = get_runs_dir() / run_name
            if rest == "/item":
                found = item_payload(run_dir, item_id)
            elif rest == "/data":
                found = item_data(run_dir, item_id)
            elif rest == "/audio":
                found = clip(run_name, item_id)
                if found is not None:
                    self._send_media(found, "audio/wav")
                    return
            else:
                self._json({"error": "no such endpoint"}, 404)
                return
            if found is None:
                self._json({"error": f"{run_name} has no item {item_id!r}"}, 404)
                return
            self._json(found)

        def _datasets_api(self, method: str, rest: str, qs: dict, body: bytes | None) -> None:
            one = {k: v[0] for k, v in qs.items()}
            try:
                if method == "GET" and rest == "":
                    return self._json(dm.list_datasets())
                if method == "POST" and rest == "":
                    return self._json(dm.create_dataset(one.get("name", ""), one.get("languages", "")), 201)

                if rest == "/record" and method == "POST":
                    dataset = dm.pick(one.get("ds", ""))
                    return self._json(
                        dm.record(
                            dataset,
                            body or b"",
                            prefix=one.get("prefix", ""),
                            speakers=int(one.get("speakers") or 0),
                            src_lang=one.get("src_lang", ""),
                            speaker=one.get("speaker", ""),
                        )
                    )

                dataset = dm.pick(one.get("ds", ""))
                item_id = one.get("id", "")
                if method == "GET" and rest == "/shape":
                    return self._json({"name": one.get("ds"), "path": str(dataset), **dm.shape(dataset)})
                if method == "GET" and rest == "/items":
                    return self._json(
                        dm.items(
                            dataset,
                            offset=max(0, int(one.get("offset") or 0)),
                            q=one.get("q", ""),
                            order=one.get("order", ""),
                        )
                    )
                if method == "GET" and rest == "/item":
                    return self._json(dm.item(dataset, item_id))
                if method == "GET" and rest == "/audio":
                    return self._send_media(dm.audio_bytes(dataset, item_id), "audio/wav")
                if method == "GET" and rest == "/verify":
                    return self._json(dm.verify(dataset))
                if method == "DELETE" and rest == "/audio":
                    return self._json(dm.drop_audio(dataset, item_id))
                if method == "DELETE" and rest == "/item":
                    return self._json(dm.drop_item(dataset, item_id))
            except DatasetError as e:
                self._json({"error": e.message}, e.code)
                return
            except (OSError, RuntimeError, ValueError) as e:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
                return
            self.send_error(404, "no such endpoint")

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            path, qs = parsed.path, urllib.parse.parse_qs(parsed.query)

            answered = queue_api.handle_get(path)
            if answered is not None:
                status, data = answered
                self._json(data, status)
                return

            if path.startswith(REPLAY_API):
                self._replay_api(path[len(REPLAY_API) :], qs)
                return

            if path.startswith(DATASETS_API):
                self._datasets_api("GET", path[len(DATASETS_API) :], qs, None)
                return

            if path == "/api/overview":
                self._json(runs.overview())
                return

            if path.startswith(DISTRIBUTIONS_PREFIX):
                run_name = urllib.parse.unquote(path[len(DISTRIBUTIONS_PREFIX) :])
                self._json(runs.distributions(run_name) if run_name in runs.list_runs() else None)
                return

            self.send_error(404, "no such endpoint")

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        n = len(runs.list_runs()) if get_runs_dir().is_dir() else 0
        print(f"manager API on http://127.0.0.1:{port}/api  ({n} run{'s' if n != 1 else ''})", flush=True)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0
