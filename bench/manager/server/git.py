import os
import re
import threading
import time
from pathlib import Path

import git

from ... import source

FETCH_EVERY_SEC = 60
BRANCH_NAME = re.compile(r"[A-Za-z0-9._/-]+")
UNKNOWN_REVISION = (ValueError, git.BadName, git.BadObject, git.GitCommandError)


class GitError(Exception):
    pass


class Repo:
    def __init__(self, root: Path):
        self.repo = git.Repo(os.environ.get("GIT_DIR") or root)
        self.fetched_at = 0.0
        self.fetch_lock = threading.Lock()

    def fetch(self) -> None:
        with self.fetch_lock:
            if time.monotonic() - self.fetched_at < FETCH_EVERY_SEC:
                return
            try:
                source.fetch_origin(self.repo)
            except git.GitCommandError:
                pass
            self.fetched_at = time.monotonic()

    def current_branch(self) -> str | None:
        try:
            return self.repo.active_branch.name
        except TypeError:
            return None

    def branches(self) -> dict:
        self.fetch()
        remote_branches = [
            ref for ref in self.repo.remotes.origin.refs if ref.remote_head != "HEAD"
        ]
        newest_first = sorted(
            remote_branches, key=lambda ref: ref.commit.committed_datetime, reverse=True
        )
        return {
            "current": self.current_branch(),
            "branches": [
                {
                    "name": ref.remote_head,
                    "sha": ref.commit.hexsha,
                    "author": ref.commit.author.name,
                    "date": ref.commit.committed_datetime.isoformat(),
                    "subject": ref.commit.summary,
                }
                for ref in newest_first
            ],
        }

    def latest_commit(self, branch: str) -> git.Commit:
        if not BRANCH_NAME.fullmatch(branch or ""):
            raise GitError(f"{branch!r} is not a branch name")
        try:
            return self.repo.commit(f"origin/{branch}")
        except UNKNOWN_REVISION as e:
            raise GitError(f"branch {branch!r} is not on origin; push it first") from e

