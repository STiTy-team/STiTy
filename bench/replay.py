import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core.errors import DataError
from core.utils import logging

PORT = 3000
TEMPLATE = Path(__file__).with_name("replay.html")


def payload(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
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

    return {
        "run": {"name": summary.get("name") or run_dir.name,
                "stamp": summary.get("stamp") or "",
                "status": summary.get("status") or "",
                "dir": run_dir.name,
                "src_lang": src, "target_lang": target},
        "items": [items[i] for i in order],
    }


def page(run_dir: Path) -> bytes:
    data = payload(run_dir)
    body = TEMPLATE.read_text(encoding="utf-8").replace(
        "__DATA__", json.dumps(data, ensure_ascii=False))
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{data['run']['name']} — session replay</title>\n</head>\n<body>\n"
        f"{body}\n</body>\n</html>\n").encode("utf-8")


def serve(run_dir: Path, port: int = PORT) -> int:
    body = page(run_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            pass

    ThreadingHTTPServer.allow_reuse_address = True
    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://localhost:{port}"
        print(f"{url}  ({Path(run_dir).name}, ctrl-c to stop)", flush=True)
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
