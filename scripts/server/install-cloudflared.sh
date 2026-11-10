#!/usr/bin/env bash
# Downloads the cloudflared release binary for this machine into .bin/ at the repository root.
set -euo pipefail

bin_dir="$(cd "$(dirname "$0")/../.." && pwd)/.bin"
release="https://github.com/cloudflare/cloudflared/releases/latest/download"

case "$(uname -m)" in
  x86_64 | amd64) arch=amd64 ;;
  aarch64 | arm64) arch=arm64 ;;
  armv7l | armv6l) arch=arm ;;
  i386 | i686) arch=386 ;;
  *) echo "unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

mkdir -p "$bin_dir"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

case "$(uname -s)" in
  Linux)
    curl -fsSL "${release}/cloudflared-linux-${arch}" -o "$tmp/cloudflared"
    ;;
  Darwin)
    curl -fsSL "${release}/cloudflared-darwin-${arch}.tgz" | tar -xz -C "$tmp"
    ;;
  *) echo "unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

chmod +x "$tmp/cloudflared"
"$tmp/cloudflared" --version >/dev/null
mv "$tmp/cloudflared" "$bin_dir/cloudflared"
"$bin_dir/cloudflared" --version
