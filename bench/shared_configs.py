import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import NamedTuple

import yaml

from core import config as core_config
from core.errors import ConfigError
from core.integrations.s3 import Bucket, Conflict
from core.utils import env, logging
from core.utils.paths import get_project_root

from . import config, store
from .settings import BenchSettings

log = logging.getLogger(__name__)

KINDS = ("pipeline", "dataset")
FOLDER = "configs/"
SYNCED_FILE = ".synced.json"


class SharedConfig(NamedTuple):
    text: str
    etag: str


def key(kind: str, name: str) -> str:
    return f"{FOLDER}{kind}s/{name}.yml"


def split_key(key_in_bucket: str) -> tuple[str, str] | None:
    parts = key_in_bucket.removeprefix(FOLDER).split("/")
    if len(parts) != 2 or not parts[1].endswith(".yml"):
        return None
    kind = parts[0].removesuffix("s")
    return (kind, parts[1].removesuffix(".yml")) if kind in KINDS else None


def with_meta(text: str, meta: dict) -> str:
    """The config text with its top-level `meta:` block replaced; every other line stays as written."""
    def line(k: str, v: object) -> str:
        style = {"allow_unicode": True, "width": float("inf")}
        if isinstance(v, list):
            return f"  {k}: {yaml.safe_dump(v, default_flow_style=True, **style)}"
        return f"  {yaml.safe_dump({k: v}, default_flow_style=False, **style)}"

    block = "meta:\n" + "".join(line(k, v) for k, v in meta.items()) if meta else "meta: {}\n"
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("meta:")), None)
    if start is None:
        return block + text
    end = start + 1
    if not lines[start].partition("#")[0].removeprefix("meta:").strip():
        while end < len(lines) and (not lines[end].strip() or lines[end][0] in " \t"):
            end += 1
        while end > start + 1 and not lines[end - 1].strip():
            end -= 1
    return "".join(lines[:start]) + block + "".join(lines[end:])


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def refuse_taken_names(bucket: Bucket, identity: dict) -> None:
    for kind, mine in identity.items():
        claimed = store.read_ref(bucket, kind, mine["ref"])
        if claimed and claimed["hash"] != mine["hash"]:
            raise ConfigError(
                f"{kind} name {mine['ref']!r} already means other content (claimed by "
                f"{claimed['claimed_by']} at {claimed['claimed_at']}); give this one a new name "
                f"or raise meta.version"
            )


class SharedConfigs:
    def __init__(self, bucket: Bucket):
        self.bucket = bucket

    def names(self) -> list[tuple[str, str, str]]:
        found = []
        newest_first = sorted(self.bucket.modified(FOLDER).items(), key=lambda kv: kv[1], reverse=True)
        for key_in_bucket, modified_at in newest_first:
            kind_and_name = split_key(key_in_bucket)
            if kind_and_name:
                found.append((*kind_and_name, modified_at.isoformat(timespec="seconds")))
        return found

    def get(self, kind: str, name: str) -> SharedConfig | None:
        stored = self.bucket.get_bytes(key(kind, name))
        return None if stored is None else SharedConfig(stored.data.decode("utf-8"), stored.etag)

    def all(self) -> dict[str, dict[str, SharedConfig]]:
        configs = {kind: {} for kind in KINDS}
        for kind, name, _ in self.names():
            shared = self.get(kind, name)
            if shared is not None:
                configs[kind][name] = shared
        return configs

    def save(self, kind: str, name: str, text: str, etag: str | None) -> str:
        if kind not in KINDS:
            raise ConfigError(f"unknown config kind {kind!r} (known: {list(KINDS)})")
        refuse_taken_names(self.bucket, {kind: config.check_text(kind, name, text)})
        return self.bucket.put_bytes(
            key(kind, name),
            text.encode("utf-8"),
            if_match=etag,
            if_absent=etag is None,
            content_type="text/yaml; charset=utf-8",
        )

    def edit_meta(self, kind: str, name: str, etag: str, new_name: str, meta: dict) -> str:
        """Rewrites the config's `meta:` block and, when `new_name` differs, moves it there.

        A rename writes the new name first and removes the old one only if nobody changed it
        meanwhile; otherwise the new copy is taken back out and the conflict is raised.
        """
        current = self.get(kind, name)
        if current is None:
            raise ConfigError(f"{kind} config {name!r} no longer exists")
        if current.etag != etag:
            raise Conflict(f"{kind} config {name!r} changed since it was loaded")
        text = with_meta(current.text, meta)
        if new_name == name:
            return self.save(kind, name, text, etag)
        if self.get(kind, new_name) is not None:
            raise ConfigError(f"{kind} config {new_name!r} already exists")
        new_etag = self.save(kind, new_name, text, None)
        try:
            self.bucket.delete(key(kind, name), if_match=etag)
        except Conflict:
            self.bucket.delete(key(kind, new_name), if_match=new_etag)
            raise
        return new_etag


class LocalCopy:
    def __init__(self, root: Path):
        self.root = root
        self.synced_path = root / SYNCED_FILE
        self.synced: dict[str, str] = (
            json.loads(self.synced_path.read_text(encoding="utf-8"))
            if self.synced_path.is_file()
            else {}
        )

    def path(self, kind: str, name: str) -> Path:
        return self.root / f"{kind}s" / f"{name}.yml"

    def read(self, kind: str, name: str) -> str | None:
        path = self.path(kind, name)
        return path.read_text(encoding="utf-8") if path.is_file() else None

    def write(self, kind: str, name: str, text: str) -> None:
        path = self.path(kind, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def files(self) -> list[tuple[str, str]]:
        return [
            (kind, path.stem)
            for kind in KINDS
            for path in sorted((self.root / f"{kind}s").glob("*.yml"))
        ]

    def edited(self, kind: str, name: str, text: str) -> bool:
        return self.synced.get(f"{kind}/{name}") != text_hash(text)

    def mark_synced(self, kind: str, name: str, text: str) -> None:
        self.synced[f"{kind}/{name}"] = text_hash(text)

    def save_synced(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.synced_path.write_text(json.dumps(self.synced, indent=2, sort_keys=True) + "\n")


def pull(shared: SharedConfigs, local: LocalCopy) -> None:
    for kind, configs in shared.all().items():
        for name, remote in configs.items():
            pull_one(local, kind, name, remote)
    local.save_synced()


def pull_named(shared: SharedConfigs, local: LocalCopy, kind_and_names: list[str]) -> None:
    for kind_and_name in kind_and_names:
        kind, _, name = kind_and_name.partition("/")
        if kind not in KINDS or not name:
            raise ConfigError(f"{kind_and_name!r} is not <pipeline|dataset>/<name>")
        remote = shared.get(kind, name)
        if remote is None:
            print(f"missing  {kind} {name}: not in S3, the local copy is used if there is one")
            continue
        pull_one(local, kind, name, remote)
    local.save_synced()


def pull_one(local: LocalCopy, kind: str, name: str, remote: SharedConfig) -> None:
    mine = local.read(kind, name)
    if mine == remote.text:
        local.mark_synced(kind, name, mine)
    elif mine is None or not local.edited(kind, name, mine):
        local.write(kind, name, remote.text)
        local.mark_synced(kind, name, remote.text)
        print(f"pulled   {kind} {name}")
    elif not local.edited(kind, name, remote.text):
        print(f"unpushed {kind} {name}: edited here; push it to share")
    else:
        print(f"kept     {kind} {name}: edited here and in S3; push it, or delete the file and pull")


def is_file_path(ref: str) -> bool:
    return ref.endswith(".yml") or "/" in ref


def config_text(kind: str, ref: str, shared: SharedConfigs | None) -> tuple[str, str]:
    """The name and YAML of the config a run asked for.

    A path is read as is. A name comes from S3, or from `configs/` when there is no bucket
    or the run is a queued job, whose worker wrote the text submitted with it there.
    """
    if is_file_path(ref):
        path = Path(ref)
        if not path.is_file():
            raise ConfigError(f"no {kind} config file {ref}")
        return path.stem, path.read_text(encoding="utf-8")
    if shared is None:
        return ref, core_config.resolve(ref, kind).read_text(encoding="utf-8")
    remote = shared.get(kind, ref)
    if remote is None:
        raise ConfigError(
            f"no {kind} config {ref!r} in S3; push it, or pass configs/{kind}s/{ref}.yml to run "
            f"the local file"
        )
    mine = LocalCopy(get_project_root() / "configs").read(kind, ref)
    if mine is not None and mine != remote.text:
        log.warning(
            "[CONFIG] %s %s differs from configs/%ss/%s.yml; running the S3 one", kind, ref, kind, ref
        )
    return ref, remote.text


def load_for_run(pipeline: str, dataset: str, bucket: Bucket | None, *, queued_job: bool):
    shared = None if bucket is None or queued_job else SharedConfigs(bucket)
    pipeline_name, pipeline_yaml = config_text("pipeline", pipeline, shared)
    dataset_name, dataset_yaml = config_text("dataset", dataset, shared)
    return config.from_text(pipeline_name, pipeline_yaml, dataset_name, dataset_yaml)


def push(shared: SharedConfigs, local: LocalCopy) -> int:
    refused = 0
    for kind, name in local.files():
        mine = local.read(kind, name)
        remote = shared.get(kind, name)
        if remote is not None and remote.text == mine:
            local.mark_synced(kind, name, mine)
            continue
        if remote is not None and local.edited(kind, name, remote.text):
            print(f"refused  {kind} {name}: changed in S3 since your last pull; pull first")
            refused += 1
            continue
        try:
            shared.save(kind, name, mine, remote and remote.etag)
        except (ConfigError, Conflict) as e:
            print(f"refused  {kind} {name}: {e}")
            refused += 1
            continue
        local.mark_synced(kind, name, mine)
        print(f"pushed   {kind} {name}")
    local.save_synced()
    return refused


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.shared_configs",
        description="Sync configs/ with the configs in S3, which are the ones the team shares",
    )
    parser.add_argument("action", choices=("pull", "push", "list"))
    parser.add_argument(
        "only", nargs="*", metavar="KIND/NAME", help="pull only these, e.g. pipeline/mock"
    )
    args = parser.parse_args()
    action = args.action
    env.load()
    bucket = BenchSettings.load().bucket()
    if bucket is None:
        print("STITY_S3_BUCKET is not set; configs live in S3 (see .env.example)", file=sys.stderr)
        return 1
    shared = SharedConfigs(bucket)
    local = LocalCopy(get_project_root() / "configs")
    if action == "list":
        for kind, name, modified_at in shared.names():
            print(f"{modified_at}  {kind:<8}  {name}")
        return 0
    if action == "pull":
        if args.only:
            pull_named(shared, local, args.only)
        else:
            pull(shared, local)
        return 0
    return 1 if push(shared, local) else 0


if __name__ == "__main__":
    sys.exit(main())
