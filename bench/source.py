import os
from pathlib import Path

import git

from core.errors import ConfigError

WATCHED = ("core", "bench", "Qwen3-ASR")
NOT_RESULT_CHANGING = (":(exclude)bench/runs", ":(exclude,glob)**/*.md")
MAX_FILES = 20
FETCH_TIMEOUT_SEC = 20


def clean_commit(root: Path) -> str:
    try:
        repo = git.Repo(root, search_parent_directories=True)
        commit = repo.head.commit.hexsha
    except (git.GitError, ValueError):
        raise ConfigError(f"{root} is not a git checkout; a run must come from a commit") from None
    status = repo.git.status(
        "--porcelain", "--untracked-files=all", "--", *WATCHED, *NOT_RESULT_CHANGING
    )
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    if changed:
        shown = "\n".join(f"  {name}" for name in changed[:MAX_FILES])
        raise ConfigError(
            "uncommitted changes would make this run impossible to trace back to a commit:\n"
            f"{shown}\nCommit them first."
        )
    return commit


def fetch_origin(repo: git.Repo) -> None:
    ssh_without_prompts = f"{os.environ.get('GIT_SSH_COMMAND', 'ssh')} -o BatchMode=yes"
    with repo.git.custom_environment(GIT_TERMINAL_PROMPT="0", GIT_SSH_COMMAND=ssh_without_prompts):
        repo.remotes.origin.fetch(kill_after_timeout=FETCH_TIMEOUT_SEC)
