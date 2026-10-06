import time
from datetime import datetime, timedelta

from botocore.exceptions import BotoCoreError, ClientError

from core.integrations.s3 import Bucket
from core.utils import logging

log = logging.getLogger(__name__)

RESYNC_SEC = 3600
RETRY_SYNC_SEC = 60
ELAPSED_CLOCK = getattr(time, "CLOCK_BOOTTIME", time.CLOCK_MONOTONIC)


def elapsed() -> float:
    return time.clock_gettime(ELAPSED_CLOCK)


class BucketClock:
    def __init__(self, bucket: Bucket):
        self.bucket = bucket
        self.sync()

    def sync(self) -> datetime:
        self.synced_at = self.bucket.now()
        self.synced_elapsed = elapsed()
        self.next_sync_elapsed = self.synced_elapsed + RESYNC_SEC
        return self.synced_at

    def now(self) -> datetime:
        if elapsed() >= self.next_sync_elapsed:
            try:
                return self.sync()
            except (BotoCoreError, ClientError) as e:
                log.warning("[CLOCK-SYNC-FAILED] %s", e)
                self.next_sync_elapsed = elapsed() + RETRY_SYNC_SEC
        return self.synced_at + timedelta(seconds=elapsed() - self.synced_elapsed)
