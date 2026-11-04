import logging

import httpx

log = logging.getLogger(__name__)


def send(webhook_url: str | None, message: str) -> None:
    if not webhook_url:
        return
    try:
        response = httpx.post(
            webhook_url,
            json={"content": message},
            headers={
                "User-Agent": "DiscordBot (https://github.com/STiTy-team, 1)",
            },
            timeout=10,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("[DISCORD-FAILED] %s", e)
