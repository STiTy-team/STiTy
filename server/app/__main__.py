import uvicorn

from app import config
from app.main import load_env


def main() -> None:
    load_env()
    server_config = config.load_config()

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host=server_config.server.host,
        port=server_config.server.port,
        log_config=None,
    )


if __name__ == "__main__":
    main()
