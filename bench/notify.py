from datetime import datetime
from zoneinfo import ZoneInfo

from core.integrations import discord

from .machines.queue import Job
from .settings import BenchSettings

MAX_ERROR_CHARS = 1500

STATUS_HEADLINES = {
    "ok": ("✅", "정상 종료"),
    "degraded": ("⚠️", "일부 오류와 함께 종료"),
    "failed": ("❌", "실패로 종료"),
}


def number(value: float | None, digits: int) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def error_block(message: str) -> str:
    tail = message[-MAX_ERROR_CHARS:]
    note = "" if tail == message else f"(마지막 {MAX_ERROR_CHARS}자만 보여드려요)\n"
    return f"{note}```\n{tail}\n```"


def korean_duration(seconds: float | None) -> str:
    if seconds is None:
        return "알 수 없음"
    hours, rest = divmod(int(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if secs:
        parts.append(f"{secs}초")
    return " ".join(parts) or "0초"


def success(summary: dict, *, run_id: str | None = None) -> None:
    identity = summary.get("identity") or {}
    pipeline = (identity.get("pipeline") or {}).get("ref") or "-"
    dataset = (identity.get("dataset") or {}).get("ref") or "-"
    metrics = summary.get("metrics") or {}
    counts = summary.get("counts") or {}
    wall_sec = counts.get("wall_sec")
    icon, headline = STATUS_HEADLINES.get(summary.get("status"), ("ℹ️", "종료"))
    where = f"S3 run `{run_id}`" if run_id else "S3 에 올리지 않았어요 — 이 머신에만 있어요"
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"{icon} bench {headline} — {pipeline} · {dataset}",
        head=(
            f"item {counts.get('items', '-')}개 · 오류 {counts.get('errored', '-')}개 · "
            f"빈 출력 {counts.get('empty_transcription_output', '-')}개 · "
            f"{number(None if wall_sec is None else wall_sec / 60, 0)}분 걸림\n"
            f"{where}"
        ),
        detail=(
            f"WER↓ **{number(metrics.get('wer'), 4)}** · CER↓ **{number(metrics.get('cer'), 4)}** · "
            f"BLEU↑ **{number(metrics.get('bleu'), 2)}** · COMET↑ **{number(metrics.get('comet'), 4)}**\n"
            f"LAAL↓ {number(metrics.get('laal_ms'), 0)}ms · YAAL↓ {number(metrics.get('yaal_ms'), 0)}ms · "
            f"첫 segment↓ {number(metrics.get('avg_fsl_sec'), 2)}초"
        ),
    )


def failure(
    error: BaseException,
    *,
    pipeline: str | None = None,
    dataset: str | None = None,
    summary: dict | None = None,
) -> None:
    settings = BenchSettings.load()
    if settings.stity_job_id:
        return
    identity = (summary or {}).get("identity") or {}
    pipeline = (identity.get("pipeline") or {}).get("ref") or pipeline or "-"
    dataset = (identity.get("dataset") or {}).get("ref") or dataset or "-"
    message = f"{type(error).__name__}: {error}"
    discord.send_thread(
        settings.discord_webhook_url,
        title=f"❌ bench 실패했어요 — {pipeline} · {dataset}",
        head="로그를 보고 고친 뒤 다시 실행해 주세요",
        detail=error_block(message),
    )


def machine_full(job: Job, *, host: str, gpus: str, retry_in: str) -> None:
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"⏳ {host} 머신의 GPU가 가득 찼어요 — {job.pipeline} · {job.dataset}",
        head=(
            f"job `{job.id}` 실패 시점에 다른 프로세스가 GPU를 쓰고 있었어요\n"
            f"job은 queue로 돌려놨고, 이 머신은 {retry_in} 뒤에 다시 시도해요"
        ),
        detail=gpus,
    )


def job_started(
    job: Job,
    *,
    host: str,
    commit: str,
    window_ends: datetime | None,
    timezone: str,
) -> None:
    zone = ZoneInfo(timezone)
    until = (
        "바로 실행이라 시간표와 상관없이 끝까지 돌아요"
        if window_ends is None
        else f"이 시간대는 {window_ends.astimezone(zone):%m/%d %H:%M} ({zone.key}) 까지"
    )
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"▶️ {host} 에서 job을 시작했어요 — {job.pipeline} · {job.dataset}",
        head=f"branch `{job.branch}` (`{commit[:10]}`) · job `{job.id}` · {len(job.attempts)}번째 시도 · {until}",
    )


def job_not_started(job: Job, *, host: str, error: str) -> None:
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"❌ {host} 에서 job을 시작하지 못했어요 — {job.pipeline} · {job.dataset}",
        head=f"job `{job.id}` · 실패로 끝났어요. 고친 뒤 Queue 페이지에서 다시 넣어 주세요",
        detail=error_block(error),
    )


def job_failed(job: Job, *, host: str, error: str) -> None:
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"❌ {host} 에서 job이 실패했어요 — {job.pipeline} · {job.dataset}",
        head=f"job `{job.id}` · queue에서 뺐어요. 고친 뒤 다시 넣어 주세요",
        detail=error_block(error),
    )


def job_requeued(job: Job, *, host: str, reason: str, ran_sec: float | None) -> None:
    discord.send_thread(
        BenchSettings.load().discord_webhook_url,
        title=f"🔁 job을 queue로 돌려놨어요 — {job.pipeline} · {job.dataset}",
        head=(
            f"job `{job.id}` · `{host}` 에서 {korean_duration(ran_sec)} 돌았어요 · {reason}\n"
            f"다른 머신이나 다음 시간대에 처음부터 다시 돌아요"
        ),
    )
