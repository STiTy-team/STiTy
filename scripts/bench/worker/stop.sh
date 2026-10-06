#!/usr/bin/env bash
set -euo pipefail

SERVICE=stity-worker.service
UNIT_FILE=/etc/systemd/system/$SERVICE
CONFIG_DIR=/etc/stity-worker

usage() {
  cat <<USAGE
Stop the queue worker and remove it from systemd. Run with sudo.

  sudo scripts/bench/worker/stop.sh

A running job goes back to the queue. The unit and $CONFIG_DIR (settings and AWS key) are
removed, so the worker does not come back at boot. To cut the key off everywhere, also
deactivate it in IAM. Start it again with scripts/bench/worker/start.sh.
USAGE
  exit "${1:-0}"
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

case "${1:-}" in
  "") ;;
  -h|--help) usage ;;
  *) usage 1 ;;
esac
[[ $EUID -eq 0 ]] || fail "run with sudo"
if [[ ! -f "$UNIT_FILE" ]]; then
  echo "[not installed] $SERVICE"
  exit 0
fi
systemctl disable --now "$SERVICE"
rm -rf "$UNIT_FILE" "$CONFIG_DIR"
systemctl daemon-reload
systemctl reset-failed "$SERVICE" 2>/dev/null || true
echo "[stopped] $SERVICE"
