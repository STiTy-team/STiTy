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
APT_PACKAGES=(git make curl)
OLDEST_SYSTEMD=231
FETCH_TIMEOUT_SEC=60
REQUIRED_VARS=(STITY_S3_BUCKET STITY_DATA_ROOT)
KEY_VARS=(AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN)
REFUSED_VARS=(AWS_SHARED_CREDENTIALS_FILE AWS_PROFILE)
SETTLE_SEC=15

usage() {
  cat <<USAGE
Set this machine up as a queue worker and start it. Run with sudo.
Installs git, make and curl (apt) and uv (for --user) when they are missing.

  sudo scripts/bench/worker/start.sh --env-file <path> [--user <name>] [--repo <clone>]

--env-file  plain KEY=value lines: STITY_*, AWS_DEFAULT_REGION, DISCORD_WEBHOOK_URL, and the
            machine's AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY
--user      defaults to the user who ran sudo
--repo      defaults to the clone this script is in

The AWS key goes into $INSTALLED_CREDENTIALS and the rest into $INSTALLED_ENV_FILE, both
root only; then the key lines are removed from the file you passed. On a later run the file
only needs what changes: its values replace the installed ones, so a file holding just the two
AWS lines swaps the key. Without --env-file the installed settings are used as they are.
A running job goes back to the queue. Stop and remove everything with scripts/bench/worker/stop.sh.
USAGE
  exit "${1:-0}"
}

fail() { echo "[ERROR] $*" >&2; exit 1; }

contains() {
  local wanted="$1" item
  shift
  for item in "$@"; do [[ "$item" == "$wanted" ]] && return 0; done
  return 1
}

check_user_and_repo() {
  [[ -n "$user" && -n "$repo" ]] || usage 1
  [[ "$user" != root ]] || fail "run the worker as a normal user: pass --user"
  id "$user" >/dev/null 2>&1 || fail "no user $user"
  [[ -f "$repo/bench/worker/__main__.py" ]] || fail "$repo is not a STiTy clone with the worker"
  [[ "$(stat -c %U "$repo")" == "$user" ]] || fail "$repo is not owned by $user, so git would refuse it"
}

unquote() {
  local value="$1"
  [[ "$value" =~ ^\"(.*)\"$ || "$value" =~ ^\'(.*)\'$ ]] && value="${BASH_REMATCH[1]}"
  printf '%s' "$value"
}

read_env_file() {
  local file="$1" line name
  [[ -f "$file" ]] || fail "no env file $file"
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "${line//[[:space:]]/}" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" != export\ * ]] || fail "$file has 'export' lines; systemd reads plain KEY=value lines only"
    [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || fail "$file has a line that is not KEY=value"
    name="${line%%=*}"
    ! contains "$name" "${REFUSED_VARS[@]}" || fail "$file sets $name; the service sets where the key is"
    if contains "$name" "${KEY_VARS[@]}"; then
      keys[$name]="$(unquote "${line#*=}")"
    else
      contains "$name" "${setting_names[@]}" || setting_names+=("$name")
      settings[$name]="${line#*=}"
    fi
  done < "$file"
}

collect() {
  [[ -f "$INSTALLED_ENV_FILE" ]] && read_env_file "$INSTALLED_ENV_FILE"
  [[ -z "$given_env_file" ]] || read_env_file "$given_env_file"
  [[ -f "$INSTALLED_ENV_FILE" || -n "$given_env_file" ]] || fail "nothing installed yet: pass --env-file"
  local var
  for var in "${REQUIRED_VARS[@]}"; do
    [[ -n "${settings[$var]:-}" ]] || fail "no $var in the settings"
  done
  if [[ -n "${keys[AWS_ACCESS_KEY_ID]:-}${keys[AWS_SECRET_ACCESS_KEY]:-}" ]]; then
    [[ -n "${keys[AWS_ACCESS_KEY_ID]:-}" && -n "${keys[AWS_SECRET_ACCESS_KEY]:-}" ]] \
      || fail "$given_env_file needs both AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
  elif [[ ! -s "$INSTALLED_CREDENTIALS" ]]; then
    fail "no AWS key installed yet: add AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to the env file"
  fi
}

check_systemd() {
  local version
  version="$(systemctl --version | awk 'NR == 1 { print $2 }')"
  (( ${version%%.*} >= OLDEST_SYSTEMD )) \
    || fail "systemd $version is too old; the key handling needs $OLDEST_SYSTEMD or newer (Ubuntu 18.04+)"
}

install_packages() {
  local missing=() package
  for package in "${APT_PACKAGES[@]}"; do
    command -v "$package" >/dev/null || missing+=("$package")
  done
  (( ${#missing[@]} == 0 )) && return 0
  command -v apt-get >/dev/null || fail "install ${missing[*]} first; there is no apt-get here"
  echo "[installing] ${missing[*]}"
  { apt-get update -qq && DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=a \
    apt-get install -y -qq "${missing[@]}"; } >/dev/null 2>&1 || fail "apt-get could not install ${missing[*]}"
}

install_uv() {
  PATH="$login_path" command -v uv >/dev/null && return 0
  echo "[installing] uv for $user"
  runuser -l "$user" -c 'curl -LsSf https://astral.sh/uv/install.sh | sh -s -- -q'
}

check_git_fetch() {
  runuser -u "$user" -- env GIT_TERMINAL_PROMPT=0 GIT_SSH_COMMAND="ssh -o BatchMode=yes" \
    timeout "$FETCH_TIMEOUT_SEC" git -C "$repo" fetch -q origin \
    || fail "git fetch in $repo fails for $user without a prompt (git says why above); the worker fetches every job's branch, so $user needs an SSH key without a passphrase or a read-only deploy key, and GitHub's host key accepted once"
}

make_data_root() {
  local data_root
  data_root="$(unquote "${settings[STITY_DATA_ROOT]}")"
  [[ -d "$data_root" ]] || runuser -u "$user" -- mkdir -p "$data_root"
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

write_root_only() {
  install -m 600 -o root -g root /dev/null "$1"
  cat > "$1"
}

install_files() {
  local name
  install -d -m 700 -o root -g root "$CONFIG_DIR"
  for name in "${setting_names[@]}"; do
    printf '%s=%s\n' "$name" "${settings[$name]}"
  done | write_root_only "$INSTALLED_ENV_FILE"
  [[ -n "${keys[AWS_ACCESS_KEY_ID]:-}" ]] || return 0
  {
    printf '[default]\naws_access_key_id = %s\naws_secret_access_key = %s\n' \
      "${keys[AWS_ACCESS_KEY_ID]}" "${keys[AWS_SECRET_ACCESS_KEY]}"
    [[ -z "${keys[AWS_SESSION_TOKEN]:-}" ]] || printf 'aws_session_token = %s\n' "${keys[AWS_SESSION_TOKEN]}"
  } | write_root_only "$INSTALLED_CREDENTIALS"
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

remove_keys_from_given_file() {
  [[ -n "$given_env_file" && ${#keys[@]} -gt 0 ]] || return 0
  local kept
  kept="$(grep -vE "^($(IFS='|'; echo "${KEY_VARS[*]}"))=" "$given_env_file" || true)"
  printf '%s\n' "$kept" > "$given_env_file"
  echo "[key removed] the AWS key lines are gone from $given_env_file; the rest of it is kept"
}

user="${SUDO_USER:-}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
given_env_file=
while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) user="$2"; shift 2 ;;
    --repo) repo="$2"; shift 2 ;;
    --env-file) given_env_file="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) usage 1 ;;
  esac
done

[[ $EUID -eq 0 ]] || fail "run with sudo"
repo="$(cd "$repo" && pwd)"
check_user_and_repo
declare -A settings=() keys=()
setting_names=()
collect
check_systemd
install_packages
find_login_path
install_uv
find_login_path
check_tools
check_git_fetch
make_data_root
install_files
remove_keys_from_given_file
write_unit
systemctl daemon-reload
systemctl enable "$SERVICE" >/dev/null
systemctl restart "$SERVICE"
echo "[starting] $SERVICE as $user from $repo"
confirm_running
echo "[running] $SERVICE -- journalctl -u stity-worker -f"
