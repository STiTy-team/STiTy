#!/usr/bin/env bash
# Opens a Cloudflare quick tunnel to the conversation server and, once the server answers through
# it, prints a QR code that switches a mobile development build to that server.
set -euo pipefail

port="${1:-8765}"

if ! command -v cloudflared >/dev/null; then
  echo "cloudflared not found: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" >&2
  exit 1
fi

log="$(mktemp -t stity-tunnel.XXXXXX)"
cloudflared tunnel --no-autoupdate --url "http://localhost:${port}" >"$log" 2>&1 &
tunnel_pid=$!
trap 'kill "$tunnel_pid" 2>/dev/null; rm -f "$log"' EXIT

until origin="$(grep -oEm1 'https://[a-z0-9-]+\.trycloudflare\.com' "$log")"; do
  kill -0 "$tunnel_pid" 2>/dev/null || { cat "$log" >&2; exit 1; }
  sleep 1
done

echo "[tunnel] ${origin} — waiting for the server to answer through it"
until curl -sf "${origin}/api/v1/healthz" >/dev/null; do
  kill -0 "$tunnel_pid" 2>/dev/null || { cat "$log" >&2; exit 1; }
  sleep 5
done

server="wss://${origin#https://}"
link="stity-dev://dev-server?url=${server}"
echo
uvx --quiet --from 'qrcode==8.*' qr --ascii "$link"
echo "[tunnel] Scan with the phone camera to switch the STiTy (Dev) app to this server,"
echo "[tunnel] or enter it in Profile → Developer → Server: ${server}"
echo

wait "$tunnel_pid"
