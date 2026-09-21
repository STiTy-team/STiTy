import argparse
import io
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.errors import AudioError, DataError
from core.utils import audio as audio_mod
from core.utils import env, logging

PORT = 9130
TEMPLATE = Path(__file__).with_name("replay.html")
RUNS_DIR = Path(__file__).resolve().parent / "runs"
AUDIO_PREFIX = "/audio/"


def audio_index(summary: dict) -> dict[str, dict]:
    declared = str((summary.get("dataset") or {}).get("root") or "")
    if not declared:
        return {}
    candidates = [Path(declared)]
    root = env.path("STITY_DATA_ROOT")
    if root is not None:
        candidates.append(root / Path(declared).name)

    for ds_root in candidates:
        manifest = ds_root / "manifest.jsonl"
        if not manifest.is_file():
            continue
        index = {}
        for raw in logging.read_stream(manifest):
            item_id = raw.get("id")
            if not item_id or not raw.get("audio"):
                continue
            index[str(item_id)] = {
                "path": ds_root / str(raw["audio"]),
                "offset": raw.get("offset"),
                "duration": raw.get("duration"),
            }
        if index:
            return index
    return {}


def wav_bytes(entry: dict) -> bytes:
    import soundfile as sf

    audio = audio_mod.load_window(entry["path"], offset=entry.get("offset"),
                                  duration=entry.get("duration"))
    buffer = io.BytesIO()
    sf.write(buffer, audio, audio_mod.SAMPLING_RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def payload(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        siblings = sorted(d.name for d in RUNS_DIR.glob("*") if d.is_dir())
        raise DataError(f"{run_dir} is not a directory. Runs in {RUNS_DIR}: "
                        + (", ".join(siblings) if siblings else "none yet"))
    path = run_dir / "events.jsonl"
    if not path.is_file():
        raise DataError(f"{path} is missing -- replay reads a run's event stream")
    events = list(logging.read_stream(path))
    if not events:
        raise DataError(f"{path} has no events")

    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in logging.read_stream(run_dir / "items.jsonl")}
    try:
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        summary = {}

    items: dict[str, dict] = {}
    order: list[str] = []
    src = target = ""
    current: str | None = None
    events_before_any_item: list[dict] = []
    for event in events:
        item_id = event.get("item") or current
        if not item_id:
            events_before_any_item.append(event)
            continue
        current = item_id
        if item_id not in items:
            items[item_id] = {"id": item_id, "events": [], "duration": 0.0,
                              "group": event.get("session") or ""}
            order.append(item_id)
            items[item_id]["events"].extend(events_before_any_item)
            events_before_any_item.clear()
        items[item_id]["events"].append(event)
        if event.get("type") == "item_open":
            items[item_id]["duration"] = float(event.get("audio_sec") or 0.0)
            src = src or event.get("src_lang") or ""
            target = target or event.get("target_lang") or ""
    if events_before_any_item and order:
        items[order[-1]]["events"].extend(events_before_any_item)

    for item_id, item in items.items():
        wer = (rows.get(item_id) or {}).get("wer")
        if wer is not None:
            item["wer"] = wer
        played_past_the_clip = [e.get("audio") or 0.0 for e in item["events"]]
        item["duration"] = max([item["duration"]] + played_past_the_clip)

    clips = audio_index(summary)
    for item_id, item in items.items():
        if item_id in clips:
            item["audio_url"] = AUDIO_PREFIX + urllib.parse.quote(item_id) + ".wav"

    return {
        "run": {"name": summary.get("name") or run_dir.name,
                "stamp": summary.get("stamp") or "",
                "status": summary.get("status") or "",
                "dir": run_dir.name,
                "src_lang": src, "target_lang": target,
                "n_audio": sum(1 for i in items.values() if i.get("audio_url"))},
        "items": [items[i] for i in order],
    }


def page(run_dir: Path) -> bytes:
    return render(payload(run_dir))


def render(data: dict) -> bytes:
    body = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{data['run']['name']} — session replay</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n").encode("utf-8")


def serve(run_dir: Path, port: int = PORT) -> int:
    data = payload(run_dir)
    body = render(data)
    summary_path = Path(run_dir) / "summary.json"
    try:
        clips = audio_index(json.loads(summary_path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        clips = {}
    encoded: dict[str, bytes] = {}

    def clip(item_id: str) -> bytes | None:
        if item_id not in clips:
            return None
        if item_id not in encoded:
            try:
                encoded[item_id] = wav_bytes(clips[item_id])
            except (AudioError, OSError, RuntimeError) as e:
                print(f"audio for {item_id} is unreadable: {e}")
                return None
        return encoded[item_id]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_media(self, payload: bytes, content_type: str) -> None:
            start, end = 0, len(payload) - 1
            partial = False
            asked = self.headers.get("Range", "")
            if asked.startswith("bytes="):
                first, _, last = asked[len("bytes="):].partition("-")
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
            chunk = payload[start:end + 1]
            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(chunk)))
            if partial:
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{len(payload)}")
            self.end_headers()
            self.wfile.write(chunk)

        def do_GET(self) -> None:
            path = urllib.parse.urlparse(self.path).path
            if path.startswith(AUDIO_PREFIX):
                name = urllib.parse.unquote(path[len(AUDIO_PREFIX):])
                data = clip(name[:-4] if name.endswith(".wav") else name)
                if data is None:
                    self.send_error(404, "no audio for that item")
                    return
                self._send_media(data, "audio/wav")
                return
            self._send(body, "text/html; charset=utf-8")

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://localhost:{port}"
        n = data["run"]["n_audio"]
        heard = f", {n} clips" if n else ", no audio (dataset not found)"
        print(f"{url}  ({Path(run_dir).name}{heard}, ctrl-c to stop)", flush=True)
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 - no browser here is not a failure
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.replay",
                                     description="녹음된 세션을 브라우저에서 다시 재생한다")
    parser.add_argument("run_dir", help="bench/runs/<name>-<stamp>")
    args = parser.parse_args(argv)
    try:
        return serve(Path(args.run_dir))
    except DataError as e:
        print(e)
        return 1
    except OSError as e:
        print(f"port {PORT} is not available: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
