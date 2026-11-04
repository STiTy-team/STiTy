from dependency_injector import containers, providers

from app.config import load_config
from app.services.conversation import init_conversation_service
from app.services.health import HealthService


class Container(containers.DeclarativeContainer):
    wiring_config = containers.WiringConfiguration(packages=["app.api"])

    server_config = providers.Singleton(load_config)
    health_service = providers.Singleton(HealthService, config=server_config)

    conversation_service = providers.Resource(
        init_conversation_service,
        config=server_config,
    )
