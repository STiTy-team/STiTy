import signal

from core.errors import ConfigError
from core.utils import logging

from ..machines.machine import HostTaken
from .config import Config, WorkerSettings
from .loop import AlreadyRunning, Worker

log = logging.getLogger(__name__)


def main() -> int:
    logging.configure()
    settings = WorkerSettings.load()
    worker = Worker(settings.bucket(), Config.from_settings(settings))
    signal.signal(signal.SIGTERM, worker.stop)
    signal.signal(signal.SIGINT, worker.stop)
    worker.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConfigError as e:
        log.error("[FAILED] %s", e)
        raise SystemExit(e.exit_code)
    except HostTaken as e:
        log.error("[FAILED] %s", e)
        raise SystemExit(ConfigError.exit_code)
    except AlreadyRunning as e:
        log.error("[FAILED] %s", e)
        raise SystemExit(1)
