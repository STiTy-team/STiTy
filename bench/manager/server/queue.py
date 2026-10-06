import re
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Callable

from core.integrations.s3 import Bucket, Conflict
from core.utils.paths import get_project_root
from pydantic import ValidationError

from core import config as core_config
from core.errors import ConfigError

from ... import config, store
from ...shared_configs import KINDS, SharedConfigs, refuse_taken_names
from ...config import get_runs_dir
from ...machines.machine import Machine, MachineSettings, all_machines
from . import cache, run_list
from .git import GitError, Repo

NOT_CONFIGURED = (
    "S3 is not configured. Set STITY_S3_BUCKET and the AWS variables in .env "
    "(see .env.example), then restart make bench-manager."
)


class BadRequest(Exception):
    pass


class EditedFirst(Exception):
    def __init__(self, fresh: dict):
        super().__init__("someone else changed this first; showing what they saved")
        self.fresh = fresh


class QueueApi:
    def __init__(
        self,
        bucket: Bucket | None,
        repo_root: Path | None = None,
        on_pulled: Callable[[str], None] = lambda run_name: None,
    ):
        self.bucket = bucket
        self.on_pulled = on_pulled
        self.repo = Repo(repo_root or get_project_root())
        self.known_machines: dict[str, Machine] = {}

    def machine(self, host: str) -> Machine:
        if self.bucket is None:
            raise BadRequest(NOT_CONFIGURED)
        if host not in self.known_machines:
            self.known_machines[host] = Machine(self.bucket, host)
        return self.known_machines[host]

    def machine_view(self, machine: Machine, now: datetime) -> dict:
        notes = machine.notes()
        settings = machine.settings()
        health = machine.health()
        return {
            "host": machine.host,
            "connected": machine.connected(now, health),
            "health": None if health is None else health.model_dump(mode="json"),
            "notes": notes.text,
            "notes_etag": notes.etag,
            "settings": settings.settings.model_dump(mode="json"),
            "settings_etag": settings.etag,
            "jobs": [queued.job.model_dump(mode="json") for queued in machine.queue.jobs()],
            "history": [finished.model_dump(mode="json") for finished in machine.history.recent()],
        }

    @cache.cached
    def machines(self) -> dict:
        if self.bucket is None:
            return {"configured": False, "message": NOT_CONFIGURED}
        now = self.bucket.now()
        return {
            "configured": True,
            "now": now.isoformat(),
            "machines": [
                self.machine_view(self.machine(found.host), now)
                for found in all_machines(self.bucket)
            ],
        }

    @cache.cached
    def branches(self) -> dict:
        return self.repo.branches()

    @cache.cached
    def runs(self) -> dict:
        return run_list.all_runs(self.bucket)

    def submit(self, host: str, body: dict) -> dict:
        machine = self.machine(host)
        branch = str(body.get("branch", ""))
        self.repo.fetch()
        self.repo.latest_commit(branch)
        pipeline, dataset = body.get("pipeline") or {}, body.get("dataset") or {}
        names_and_text = {
            "pipeline": (str(pipeline.get("name", "")), str(pipeline.get("yaml", ""))),
            "dataset": (str(dataset.get("name", "")), str(dataset.get("yaml", ""))),
        }
        try:
            bench_config = config.from_text(*names_and_text["pipeline"], *names_and_text["dataset"])
        except (ConfigError, ValidationError) as e:
            raise BadRequest(str(e)) from None
        try:
            refuse_taken_names(self.bucket, bench_config.identity)
        except ConfigError as e:
            raise BadRequest(str(e)) from None
        job = machine.queue.submit(
            branch=branch,
            pipeline=names_and_text["pipeline"][0],
            pipeline_yaml=names_and_text["pipeline"][1],
            dataset=names_and_text["dataset"][0],
            dataset_yaml=names_and_text["dataset"][1],
        )
        return job.model_dump(mode="json")

    def shared_configs(self) -> SharedConfigs:
        if self.bucket is None:
            raise BadRequest(NOT_CONFIGURED)
        return SharedConfigs(self.bucket)

    @cache.cached
    def configs(self) -> dict:
        shared = self.shared_configs()
        modified = {(kind, name): at for kind, name, at in shared.names()}
        return {
            kind: {
                name: {
                    "yaml": config_file.text,
                    "etag": config_file.etag,
                    "modified_at": modified.get((kind, name)),
                    **describe(name, config_file.text),
                }
                for name, config_file in found.items()
            }
            for kind, found in shared.all().items()
        }

    def check_config(self, body: dict) -> dict:
        kind, name, text = str(body.get("kind", "")), str(body.get("name", "")), str(body.get("yaml", ""))
        if kind not in KINDS:
            raise BadRequest(f"unknown config kind {kind!r}")
        try:
            mine = config.check_text(kind, name, text)
        except (ConfigError, ValidationError) as e:
            return {"valid": False, "error": str(e)}
        claimed = self.bucket and store.read_ref(self.bucket, kind, mine["ref"])
        if not claimed or claimed["hash"] == mine["hash"]:
            return {"valid": True, "ref": mine["ref"], "taken": None}
        version = mine["version"] + 1
        while (other := store.read_ref(self.bucket, kind, config.ref(name, version))) and other[
            "hash"
        ] != mine["hash"]:
            version += 1
        return {
            "valid": True,
            "ref": mine["ref"],
            "taken": {"by": claimed["claimed_by"], "at": claimed["claimed_at"]},
            "free_version": version,
        }

    def save_config(self, kind: str, name: str, body: dict) -> dict:
        if kind not in KINDS:
            raise BadRequest(f"unknown config kind {kind!r}")
        shared = self.shared_configs()
        text = str(body.get("yaml", ""))
        try:
            etag = shared.save(kind, name, text, body.get("etag") or None)
        except ConfigError as e:
            raise BadRequest(str(e)) from None
        except Conflict:
            current = shared.get(kind, name)
            raise EditedFirst(
                {"yaml": current.text if current else "", "etag": current.etag if current else None}
            ) from None
        return {"name": name, "yaml": text, "etag": etag}

    def edit_config_meta(self, kind: str, name: str, body: dict) -> dict:
        if kind not in KINDS:
            raise BadRequest(f"unknown config kind {kind!r}")
        shared = self.shared_configs()
        new_name = str(body.get("name") or name)
        try:
            etag = shared.edit_meta(
                kind, name, str(body.get("etag") or ""), new_name, body.get("meta") or {}
            )
        except ConfigError as e:
            raise BadRequest(str(e)) from None
        except Conflict:
            current = shared.get(kind, name)
            raise EditedFirst(
                {"yaml": current.text if current else "", "etag": current.etag if current else None}
            ) from None
        return {"name": new_name, "etag": etag}

    def cancel(self, host: str, job_id: str) -> dict:
        machine = self.machine(host)
        worker_alive = machine.connected(self.bucket.now())
        job = machine.queue.cancel(job_id, worker_alive=worker_alive)
        return {"removed": job is None, "job": job and job.model_dump(mode="json")}

    def save_notes(self, host: str, body: dict) -> dict:
        machine = self.machine(host)
        text = str(body.get("notes", ""))
        try:
            etag = machine.save_notes(text, body.get("etag") or None)
        except Conflict:
            current = machine.notes()
            raise EditedFirst({"notes": current.text, "etag": current.etag}) from None
        return {"notes": text, "etag": etag}

    def save_settings(self, host: str, body: dict) -> dict:
        machine = self.machine(host)
        try:
            settings = MachineSettings.model_validate(body.get("settings") or {})
        except ValidationError as e:
            raise BadRequest(f"timetable: {e.errors()[0]['msg']}") from None
        try:
            etag = machine.save_settings(settings, body.get("etag") or None)
        except Conflict:
            current = machine.settings()
            raise EditedFirst(
                {"settings": current.settings.model_dump(mode="json"), "etag": current.etag}
            ) from None
        return {"settings": settings.model_dump(mode="json"), "etag": etag}

    def pull(self, body: dict) -> dict:
        if self.bucket is None:
            raise BadRequest(NOT_CONFIGURED)
        run = store.RemoteRun(str(body["dataset"]), str(body["pipeline"]), str(body["run_id"]))
        if run not in store.list_runs(self.bucket):
            raise BadRequest(f"no run {run.folder} in the bucket")
        try:
            run_dir = store.pull_run(self.bucket, run, get_runs_dir())
        except FileExistsError as e:
            raise BadRequest(str(e)) from None
        name = run_dir.relative_to(get_runs_dir()).as_posix()
        self.on_pulled(name)
        return {"run": name}

    def get_routes(self) -> list[tuple[str, Callable]]:
        return [
            (r"/api/machines", self.machines),
            (r"/api/branches", self.branches),
            (r"/api/configs", self.configs),
            (r"/api/runs", self.runs),
        ]

    def post_routes(self, body: dict) -> list[tuple[str, Callable]]:
        job = r"/api/machines/([^/]+)/jobs/([^/]+)"
        return [
            (r"/api/reload", lambda: {}),
            (r"/api/machines/([^/]+)/jobs", lambda host: self.submit(host, body)),
            (rf"{job}/cancel", self.cancel),
            (r"/api/machines/([^/]+)/notes", lambda host: self.save_notes(host, body)),
            (r"/api/machines/([^/]+)/settings", lambda host: self.save_settings(host, body)),
            (r"/api/runs/pull", lambda: self.pull(body)),
            (r"/api/configs/check", lambda: self.check_config(body)),
            (
                r"/api/configs/([^/]+)/([^/]+)/meta",
                lambda kind, name: self.edit_config_meta(kind, name, body),
            ),
            (r"/api/configs/([^/]+)/([^/]+)", lambda kind, name: self.save_config(kind, name, body)),
        ]

    def handle_get(self, path: str) -> tuple[int, object] | None:
        answer = dispatch(self.get_routes(), path)
        if answer is None:
            return None
        status, reply = answer
        return status, reply["data"] if status == 200 else reply

    def handle_post(self, path: str, body: dict) -> tuple[int, dict]:
        try:
            return dispatch(self.post_routes(body), path) or (
                404,
                {"ok": False, "error": "no such endpoint"},
            )
        finally:
            cache.clear()


def describe(name: str, text: str) -> dict:
    try:
        _, meta = core_config.split_meta(core_config.parse_yaml(text, name), name)
    except (ConfigError, ValidationError):
        return {"description": "", "tags": [], "version": 1}
    return {"description": meta.description, "tags": list(meta.tags), "version": meta.version}


def dispatch(routes: list[tuple[str, Callable]], path: str) -> tuple[int, dict] | None:
    for pattern, handler in routes:
        match = re.fullmatch(pattern, path)
        if match is None:
            continue
        try:
            groups = (urllib.parse.unquote(group) for group in match.groups())
            return 200, {"ok": True, "data": handler(*groups)}
        except EditedFirst as e:
            return 409, {"ok": False, "error": str(e), "fresh": e.fresh}
        except Conflict:
            return 409, {"ok": False, "error": "someone else changed this first; reload"}
        except (BadRequest, GitError, KeyError) as e:
            return 400, {"ok": False, "error": str(e).strip("'\"")}
        except Exception as e:
            return 500, {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return None
