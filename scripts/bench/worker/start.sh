#!/usr/bin/env bash
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true

SERVICE=stity-worker.service
UNIT_FILE=/etc/systemd/system/$SERVICE
UNIT_TEMPLATE="$(dirname "${BASH_SOURCE[0]}")/$SERVICE"
CONFIG_DIR=/etc/stity-worker
INSTALLED_ENV_FILE=$CONFIG_DIR/worker.env
INSTALLED_CREDENTIALS=$CONFIG_DIR/aws-credentials
REQUIRED_TOOLS=(uv git make nvidia-smi)
REQUIRED_VARS=(STITY_S3_BUCKET STITY_DATA_ROOT)
SECRET_VARS=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN AWS_SHARED_CREDENTIALS_FILE AWS_PROFILE)
SETTLE_SEC=15

usage() {
  cat <<USAGE
Install the queue worker as a systemd service and start it. Run with sudo.

  sudo scripts/bench/worker/start.sh --env-file <path> --aws-credentials <path> [--user <name>] [--repo <clone>]

--env-file         settings (STITY_*, AWS_DEFAULT_REGION, DISCORD_WEBHOOK_URL), plain KEY=value lines, no AWS keys
--aws-credentials  the AWS key in the standard credentials format, under [default]
--user             defaults to the user who ran sudo
--repo             defaults to the clone this script is in

Both files are copied into $CONFIG_DIR (root only). On a later run, leave out a file to keep the
installed one: '--aws-credentials new.ini' alone swaps the key. A running job goes back to the queue.
Stop and remove everything with scripts/bench/worker/stop.sh.
USAGE
  exit "${1:-0}"
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

check_user_and_repo() {
  [[ -n "$user" && -n "$repo" ]] || usage 1
  [[ "$user" != root ]] || fail "run the worker as a normal user: pass --user"
  id "$user" >/dev/null 2>&1 || fail "no user $user"
  [[ -f "$repo/bench/worker/__main__.py" ]] || fail "$repo is not a STiTy clone with the worker"
  [[ "$(stat -c %U "$repo")" == "$user" ]] || fail "$repo is not owned by $user, so git would refuse it"
}

check_env_file() {
  local file="$1" var
  [[ -f "$file" ]] || fail "no env file $file"
  for var in "${REQUIRED_VARS[@]}"; do
    grep -q "^$var=.\+" "$file" || fail "$file does not set $var"
  done
  for var in "${SECRET_VARS[@]}"; do
    ! grep -q "^$var=" "$file" || fail "$file sets $var; AWS keys belong in --aws-credentials only"
  done
  ! grep -q '^export ' "$file" || fail "$file has 'export' lines; systemd reads plain KEY=value lines only"
}

check_credentials() {
  local file="$1" key
  [[ -f "$file" ]] || fail "no credentials file $file"
  grep -q '^\[default\]' "$file" || fail "$file has no [default] section"
  for key in aws_access_key_id aws_secret_access_key; do
    grep -q "^$key *= *[^ ]" "$file" || fail "$file does not set $key"
  done
}

pick() {
  local given="$1" installed="$2" what="$3"
  if [[ -n "$given" ]]; then
    echo "$given"
  elif [[ -f "$installed" ]]; then
    echo "$installed"
  else
    fail "nothing installed yet: pass $what"
  fi
}

find_login_path() {
  local home
  home="$(getent passwd "$user" | cut -d: -f6)"
  login_path="$home/.local/bin:$home/.cargo/bin:$(runuser -l "$user" -c 'printf %s "$PATH"')"
}

check_tools() {
  local tool
  for tool in "${REQUIRED_TOOLS[@]}"; do
    PATH="$login_path" command -v "$tool" >/dev/null \
      || fail "$tool is not on $user's PATH ($login_path)"
  done
}

install_files() {
  install -d -m 700 -o root -g root "$CONFIG_DIR"
  [[ "$env_file" == "$INSTALLED_ENV_FILE" ]] \
    || install -m 600 -o root -g root "$env_file" "$INSTALLED_ENV_FILE"
  [[ "$credentials" == "$INSTALLED_CREDENTIALS" ]] \
    || install -m 600 -o root -g root "$credentials" "$INSTALLED_CREDENTIALS"
}

write_unit() {
  local unit
  unit="$(<"$UNIT_TEMPLATE")"
  unit="${unit//@USER@/$user}"
  unit="${unit//@REPO@/$repo}"
  unit="${unit//@PATH@/$login_path}"
  unit="${unit//@UV@/$(PATH="$login_path" command -v uv)}"
  unit="${unit//@CONFIG_DIR@/$CONFIG_DIR}"
  [[ "$unit" != *@*@* ]] || fail "$UNIT_TEMPLATE has a placeholder this script does not fill"
  printf '%s\n' "$unit" > "$UNIT_FILE"
}

invocation() { systemctl show -p InvocationID --value "$SERVICE"; }

confirm_running() {
  local started
  started="$(invocation)"
  sleep "$SETTLE_SEC"
  if ! systemctl is-active --quiet "$SERVICE" || [[ "$(invocation)" != "$started" ]]; then
    journalctl -u "$SERVICE" -n 30 --no-pager >&2 || true
    fail "$SERVICE did not stay up; the log is above"
  fi
}

remind_to_delete() {
  local file
  for file in "$given_env_file" "$given_credentials"; do
    [[ -z "$file" || "$file" == "$CONFIG_DIR"/* ]] || echo "[reminder] installed a copy; delete the original: rm '$file'"
  done
}

user="${SUDO_USER:-}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
given_env_file= given_credentials=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) user="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --env-file) given_env_file="$2"; shift 2 ;;
    --aws-credentials) given_credentials="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) usage 1 ;;
  esac
done

[[ $EUID -eq 0 ]] || fail "run with sudo"
repo="$(cd "$repo" && pwd)"
check_user_and_repo
env_file="$(pick "$given_env_file" "$INSTALLED_ENV_FILE" --env-file)"
credentials="$(pick "$given_credentials" "$INSTALLED_CREDENTIALS" --aws-credentials)"
check_env_file "$env_file"
check_credentials "$credentials"
find_login_path
check_tools
install_files
write_unit
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
systemctl restart "$SERVICE"
echo "[starting] $SERVICE as $user from $repo"
confirm_running
echo "[running] $SERVICE -- journalctl -u stity-worker -f"
remind_to_delete
