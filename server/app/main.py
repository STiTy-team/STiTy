import os
import platform
from contextlib import asynccontextmanager
import fastapi
from fastapi import FastAPI
from pyfiglet import figlet_format
from core.utils import env, logging
from core.utils.paths import get_project_root
from app.api.errors import register_exception_handlers
from app.api.v1.routes import routers as v1_routers
from app.containers import Container

log = logging.getLogger(__name__)


def load_env() -> None:
    env.load(get_project_root() / ".env")


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = app.container.server_config()
    log.info(
        "Starting STiTy server (%s pipeline) on port %d",
        config.stity.pipeline.name,
        config.server.port,
    )

    await app.container.init_resources()

    log.info(
        "Started STiTy server (pid %d, Python %s, FastAPI %s)",
        os.getpid(),
        platform.python_version(),
        fastapi.__version__,
    )

    yield

    log.info("Shutting down STiTy server (pid %d)", os.getpid())
    await app.container.shutdown_resources()
    log.info("Stopped STiTy server")


def create_app() -> FastAPI:
    load_env()
    logging.configure()
    print(figlet_format("STiTy"))

    app = FastAPI(lifespan=lifespan)
    app.container = Container()

    register_exception_handlers(app)
    app.include_router(v1_routers, prefix="/api/v1")

    return app
