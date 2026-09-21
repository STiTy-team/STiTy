import argparse
import io
import json
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

from core.errors import AudioError, DataError
from core.utils import audio as audio_mod
from core.utils import env, stream

ITEM_METRIC_KEYS = ("wer", "cer", "sentence_bleu", "avg_fsl_sec", "laal_ms",
                    "n_segments", "route_errors")

PORT = 9130
TEMPLATE = Path(__file__).with_name("replay.html")
RUNS_DIR = Path(__file__).resolve().parent / "runs"
AUDIO_PREFIX = "/audio/"
ITEM_PREFIX = "/item/"
DEFAULT_TOP_K = 10


def list_runs() -> list[str]:
    """Run directory names under RUNS_DIR, most recently finished first."""
    def stamp(d: Path) -> float:
        summary = d / "summary.json"
        return summary.stat().st_mtime if summary.is_file() else 0.0

    dirs = [d for d in RUNS_DIR.glob("*")
            if d.is_dir() and (d / "events.jsonl").is_file()]
    return [d.name for d in sorted(dirs, key=stamp, reverse=True)]


def _read_summary(run_dir: Path) -> dict:
    try:
        return json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pipeline_yaml(summary: dict) -> str:
    """The resolved pipeline config, re-rendered as YAML.

    `summary["config"]["stity"]` already holds every value the run actually used
    (including ones the source file left to a default) -- re-dumping it beats
    reading `configs/pipelines/<name>.yml` back off disk, which only shows what
    was written and can drift from what the run historically saw.
    """
    stity = ((summary.get("config") or {}).get("stity")) or {}
    if not stity:
        return ""
    return yaml.dump(stity, sort_keys=False, default_flow_style=False, allow_unicode=True)


def run_catalog() -> list[dict]:
    """Every run with its dataset identity and config, so the client can offer it
    as a compare target without a second round trip once picked.

    `manifest_sha256` is a hash of the dataset's raw manifest.jsonl -- two runs
    share it only when their item ids are the same set, regardless of pipeline or
    which target language each was scored against.
    """
    out = []
    for name in list_runs():
        summary = _read_summary(RUNS_DIR / name)
        out.append({"name": name,
                    "dataset_sha": (summary.get("dataset") or {}).get("manifest_sha256") or "",
                    "pipeline_yaml": _pipeline_yaml(summary),
                    "summary_metrics": summary.get("metrics") or {}})
    return out


def choose_items(rows: dict[str, dict], *, top_k: int) -> set[str]:
    """Every failed/empty-hypothesis item, plus the worst-WER items up to top_k total.

    A failure is never crowded out by the ranking -- it is the reason the replay
    got opened. See bench/README.md's replay contract.
    """
    failed = [r for r in rows.values()
              if r.get("status") != "ok" or not (r.get("hypothesis") or "").strip()]
    failed_ids = {r["id"] for r in failed}
    scored = sorted(
        (r for r in rows.values() if r["id"] not in failed_ids and r.get("wer") is not None),
        key=lambda r: r["wer"], reverse=True)
    chosen = failed + scored[:max(0, top_k - len(failed))]
    return {r["id"] for r in chosen}


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
        for raw in stream.read(manifest):
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


def _finalize_item(item_id: str, events: list[dict], duration: float, *,
                   rows: dict[str, dict], clips: dict[str, dict], run_name: str) -> dict:
    item = {"id": item_id, "events": events, "duration": duration}
    row = rows.get(item_id) or {}
    wer = row.get("wer")
    if wer is not None:
        item["wer"] = wer
    metrics = {k: row[k] for k in ITEM_METRIC_KEYS if row.get(k) is not None}
    if metrics:
        item["metrics"] = metrics
    played_past_the_clip = [e.get("audio") or 0.0 for e in events]
    item["duration"] = max([item["duration"]] + played_past_the_clip)
    if item_id in clips:
        item["audio_url"] = (AUDIO_PREFIX + urllib.parse.quote(item_id)
                             + ".wav?run=" + urllib.parse.quote(run_name))
    return item


def item_payload(run_dir: Path, item_id: str) -> dict | None:
    """One item's data from a run, regardless of whether it made that run's own cut.

    Backs the "compare with" picker: the item on screen may not be among the
    other run's worst-`top_k`, so it has to be fetched on demand instead of
    riding along in that run's own page load.
    """
    run_dir = Path(run_dir)
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        return None

    events: list[dict] = []
    duration = 0.0
    found = False
    for event in stream.read(events_path):
        if event.get("item") != item_id:
            continue
        found = True
        events.append(event)
        if event.get("type") == "item_open":
            duration = float(event.get("audio_sec") or 0.0)
    if not found:
        return None

    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in stream.read(run_dir / "items.jsonl")}
    clips = audio_index(_read_summary(run_dir))
    return _finalize_item(item_id, events, duration,
                          rows=rows, clips=clips, run_name=run_dir.name)


def payload(run_dir: Path, *, top_k: int = DEFAULT_TOP_K) -> dict:
    """One run's replay data: the worst-`top_k` items by WER, plus every failure.

    Reads `items.jsonl` first to decide which items make the cut, then walks
    `events.jsonl` once keeping only their events -- a run with hundreds of items
    never has to materialize (or ship to the browser) the events of the ones
    nobody asked to see.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        siblings = list_runs()
        raise DataError(f"{run_dir} is not a directory. Runs in {RUNS_DIR}: "
                        + (", ".join(siblings) if siblings else "none yet"))
    events_path = run_dir / "events.jsonl"
    if not events_path.is_file():
        raise DataError(f"{events_path} is missing -- replay reads a run's event stream")

    rows = {}
    if (run_dir / "items.jsonl").is_file():
        rows = {r.get("id"): r for r in stream.read(run_dir / "items.jsonl")}
    summary = _read_summary(run_dir)

    # An empty set means "no ranking data" (no items.jsonl) -- show everything
    # rather than nothing.
    chosen_ids = choose_items(rows, top_k=top_k) if rows else set()

    items: dict[str, dict] = {}
    order: list[str] = []
    src = target = ""
    n_events = 0
    for event in stream.read(events_path):
        n_events += 1
        item_id = event.get("item")
        if not item_id or (chosen_ids and item_id not in chosen_ids):
            continue
        if item_id not in items:
            items[item_id] = {"id": item_id, "events": [], "duration": 0.0,
                              "group": event.get("session") or ""}
            order.append(item_id)
        items[item_id]["events"].append(event)
        if event.get("type") == "item_open":
            items[item_id]["duration"] = float(event.get("audio_sec") or 0.0)
            src = src or event.get("src_lang") or ""
            target = target or event.get("target_lang") or ""
    if n_events == 0:
        raise DataError(f"{events_path} has no events")

    clips = audio_index(summary)
    items = {item_id: _finalize_item(item_id, item["events"], item["duration"],
                                     rows=rows, clips=clips, run_name=run_dir.name)
             for item_id, item in items.items()}

    # Every item's id and score, with no events -- cheap enough to send in full so
    # the picker can offer the rest of the run without loading their event streams.
    all_items = [{"id": r["id"], "wer": r.get("wer")} for r in rows.values()]

    return {
        "run": {"name": summary.get("name") or run_dir.name,
                "stamp": summary.get("stamp") or "",
                "status": summary.get("status") or "",
                "dir": run_dir.name,
                "src_lang": src, "target_lang": target,
                "dataset_sha": (summary.get("dataset") or {}).get("manifest_sha256") or "",
                "n_audio": sum(1 for i in items.values() if i.get("audio_url")),
                "n_total": len(rows) if rows else len(order),
                "n_shown": len(order),
                "top_k": top_k,
                "pipeline_yaml": _pipeline_yaml(summary),
                "summary_metrics": summary.get("metrics") or {}},
        "items": [items[i] for i in order],
        "all_items": all_items,
        "runs": run_catalog(),
    }


def page(run_dir: Path, *, top_k: int = DEFAULT_TOP_K) -> bytes:
    return render(payload(run_dir, top_k=top_k))


def render(data: dict) -> bytes:
    body = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{data['run']['name']} — session replay</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n").encode("utf-8")


def resolve_run(requested: str | None) -> str:
    """The requested run if it exists, else the most recently finished one."""
    runs = list_runs()
    if not runs:
        raise DataError(f"no runs yet in {RUNS_DIR}")
    if requested and requested in runs:
        return requested
    return runs[0]


def serve(initial_run: str | None, *, port: int = PORT, top_k: int = DEFAULT_TOP_K) -> int:
    default_run = resolve_run(initial_run)  # fails fast if RUNS_DIR is empty

    summaries: dict[str, dict] = {}
    clips_by_run: dict[str, dict] = {}
    encoded: dict[tuple[str, str], bytes] = {}

    def summary_for(run_name: str) -> dict:
        if run_name not in summaries:
            summaries[run_name] = _read_summary(RUNS_DIR / run_name)
        return summaries[run_name]

    def clips_for(run_name: str) -> dict:
        if run_name not in clips_by_run:
            clips_by_run[run_name] = audio_index(summary_for(run_name))
        return clips_by_run[run_name]

    def clip(run_name: str, item_id: str) -> bytes | None:
        clips = clips_for(run_name)
        if item_id not in clips:
            return None
        key = (run_name, item_id)
        if key not in encoded:
            try:
                encoded[key] = wav_bytes(clips[item_id])
            except (AudioError, OSError, RuntimeError) as e:
                print(f"audio for {item_id} ({run_name}) is unreadable: {e}")
                return None
        return encoded[key]

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            # No cache headers at all leaves reload behavior up to each browser's
            # heuristics; this page changes every run and every code edit, so
            # never let a cached copy answer instead of the server.
            self.send_header("Cache-Control", "no-store")
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
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            requested = (qs.get("run") or [None])[0]
            run_name = requested if requested in list_runs() else default_run

            if parsed.path.startswith(AUDIO_PREFIX):
                name = urllib.parse.unquote(parsed.path[len(AUDIO_PREFIX):])
                item_id = name[:-4] if name.endswith(".wav") else name
                data = clip(run_name, item_id)
                if data is None:
                    self.send_error(404, "no audio for that item")
                    return
                self._send_media(data, "audio/wav")
                return

            if parsed.path.startswith(ITEM_PREFIX):
                other_run, _, item_id = parsed.path[len(ITEM_PREFIX):].partition("/")
                other_run = urllib.parse.unquote(other_run)
                item_id = urllib.parse.unquote(item_id)
                item = (item_payload(RUNS_DIR / other_run, item_id)
                       if other_run in list_runs() else None)
                if item is None:
                    self.send_error(404, "that run has no such item")
                    return
                self._send(json.dumps(item, ensure_ascii=False).encode("utf-8"),
                          "application/json")
                return

            try:
                body = page(RUNS_DIR / run_name, top_k=top_k)
            except DataError as e:
                self._send(str(e).encode("utf-8"), "text/plain; charset=utf-8")
                return
            self._send(body, "text/html; charset=utf-8")

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://localhost:{port}/?run={urllib.parse.quote(default_run)}"
        runs = list_runs()
        print(f"{url}  ({len(runs)} run{'s' if len(runs) != 1 else ''} to pick from, "
              f"ctrl-c to stop)", flush=True)
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
    parser.add_argument("run_dir", nargs="?", default=None,
                        help="bench/runs/<name> (기본값: 가장 최근에 끝난 실행). "
                             "떠 있는 페이지에서 다른 실행으로 언제든 바꿔 볼 수 있다")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"실패·빈 전사 항목은 전부, 나머지는 WER 최악 순으로 몇 개까지 "
                             f"보여줄지 (기본 {DEFAULT_TOP_K})")
    args = parser.parse_args(argv)
    initial = Path(args.run_dir).name if args.run_dir else None
    try:
        return serve(initial, top_k=args.top_k)
    except DataError as e:
        print(e)
        return 1
    except OSError as e:
        print(f"port {PORT} is not available: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
