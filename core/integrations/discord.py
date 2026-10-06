import logging

import httpx

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "DiscordBot (https://github.com/STiTy-team, 1)"}
TITLE_MAX_CHARS = 100


def send(webhook_url: str | None, message: str) -> None:
    if not webhook_url:
        return
    try:
        response = httpx.post(
            webhook_url,
            json={"content": message},
            headers=HEADERS,
            timeout=10,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("[DISCORD-FAILED] %s", e)


def send_thread(webhook_url: str | None, *, title: str, head: str, detail: str | None = None) -> None:
    """Opens a new forum post titled `title` with `head` as its first message, then
    posts `detail` as a reply inside that post's thread, if given. The webhook must
    point at a Forum channel — a webhook alone can't open a thread anywhere else."""
    if not webhook_url:
        return
    clipped_title = title if len(title) <= TITLE_MAX_CHARS else title[: TITLE_MAX_CHARS - 1] + "…"
    try:
        opened = httpx.post(
            webhook_url,
            params={"wait": "true"},
            json={"content": head, "thread_name": clipped_title},
            headers=HEADERS,
            timeout=10,
        )
        opened.raise_for_status()
        if not detail:
            return
        thread_id = opened.json()["channel_id"]
        reply = httpx.post(
            webhook_url,
            params={"thread_id": thread_id},
            json={"content": detail},
            headers=HEADERS,
            timeout=10,
        )
        reply.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("[DISCORD-FAILED] %s", e)
