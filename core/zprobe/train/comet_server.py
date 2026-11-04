"""Tiny COMET scoring server for the stity-env training loop (comet lives in .venv).
POST /score  {"items":[{"src":..,"mt":..,"ref":..},...], "model": "da"|"kiwi"} -> {"scores":[...]}
run: .venv/bin/python comet_server.py --port 8777"""
import argparse, json
from http.server import BaseHTTPRequestHandler, HTTPServer
from comet import download_model, load_from_checkpoint
ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8777); a = ap.parse_args()
models = {"da": load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")),
          "kiwi": load_from_checkpoint(download_model("Unbabel/wmt22-cometkiwi-da"))}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        m = models[body.get("model", "da")]
        out = m.predict(body["items"], batch_size=32, gpus=1, progress_bar=False).scores
        b = json.dumps({"scores": [float(s) for s in out]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
print("comet server ready", flush=True); HTTPServer(("127.0.0.1", a.port), H).serve_forever()
