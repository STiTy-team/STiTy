"""Environment variables, read from the process environment.

`os.environ` is the standard way to read them and nothing improves on it, so this
stays thin: typed accessors and one convention per kind. The single piece worth a
library is reading a `.env` file, and that library is python-dotenv -- what Flask,
uvicorn and pydantic-settings all use for the same job.

Read through these rather than binding a module-level constant. A constant is
evaluated at import, before the program has done anything, so a value set afterwards
is invisible -- which is what makes `GOOGLE_TRANSLATE_API_KEY` stale in
streaming_websocket_server.py:439.

Which variables exist is not declared here. Only the two API keys are read by more
than one program; everything else means something to exactly one of them, so each
names its own.
"""
import os
from pathlib import Path


def load(path: str | Path = ".env", *, override: bool = False) -> bool:
    """Read a `.env` file into the process environment. Once, from an entry point.

    The shell wins by default, so `FOO=bar python -m thing` beats the file. Returns
    whether a file was found; a missing one is not an error.
    """
    from dotenv import load_dotenv

    return load_dotenv(path, override=override)


def get(name: str, default: str = "") -> str:
    """The value, stripped. `default` when unset or blank.

    Stripped because a value that came from a file or a heredoc carries a newline,
    and a key with a trailing newline fails authentication in a way that reads like
    a wrong key.
    """
    return os.environ.get(name, "").strip() or default


def flag(name: str) -> bool:
    """True only for `1`.

    Not `true`/`yes`/`on`: every flag already in this repo is compared against `"1"`
    exactly (the five `AST_*` toggles, `QWEN3_TEST_AUTO_SERVER`), and a second
    spelling that works in some places would be worse than one that works nowhere.
    """
    return os.environ.get(name, "").strip() == "1"


def path(name: str, default: Path | None = None) -> Path | None:
    """A filesystem path with `~` expanded. `default` when unset."""
    value = get(name)
    return Path(value).expanduser() if value else default
