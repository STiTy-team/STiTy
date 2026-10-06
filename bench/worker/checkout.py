import shutil
from pathlib import Path

import git

from .. import source
from ..machines.queue import Job

FILE_ADDED_WITH_THE_QUEUE = "bench/store.py"
UNKNOWN_REVISION = (ValueError, git.BadName, git.BadObject)


class CheckoutError(Exception):
    pass


def head(repo_root: Path) -> str | None:
    try:
        return git.Repo(repo_root).head.commit.hexsha
    except (git.GitError, ValueError):
        return None


def worktree_for(repo_root: Path, worktree: Path, job: Job) -> tuple[Path, str]:
    repo = git.Repo(repo_root)
    try:
        source.fetch_origin(repo)
    except git.GitCommandError as e:
        raise CheckoutError(f"fetching origin failed: {e.stderr.strip()}") from e
    try:
        commit = repo.commit(f"origin/{job.branch}")
    except UNKNOWN_REVISION as e:
        raise CheckoutError(f"branch {job.branch!r} is not on origin") from e
    try:
        commit.tree / FILE_ADDED_WITH_THE_QUEUE
    except KeyError as e:
        raise CheckoutError(
            f"branch {job.branch!r} is at {commit.hexsha[:12]}, which predates the shared bench "
            f"({FILE_ADDED_WITH_THE_QUEUE} is missing), so it would not upload its run"
        ) from e

    remove_worktree(repo, worktree)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    try:
        repo.git.worktree("add", "--detach", str(worktree), commit.hexsha)
    except git.GitCommandError as e:
        raise CheckoutError(f"creating the worktree failed: {e.stderr.strip()}") from e
    for kind, name, text in (
        ("pipelines", job.pipeline, job.pipeline_yaml),
        ("datasets", job.dataset, job.dataset_yaml),
    ):
        config_file = worktree / "configs" / kind / f"{name}.yml"
        config_file.parent.mkdir(parents=True, exist_ok=True)
        config_file.write_text(text, encoding="utf-8")
    return worktree, commit.hexsha


def remove_worktree(repo: git.Repo, path: Path) -> None:
    try:
        repo.git.worktree("remove", "--force", str(path))
    except git.GitCommandError:
        shutil.rmtree(path, ignore_errors=True)
    repo.git.worktree("prune")
