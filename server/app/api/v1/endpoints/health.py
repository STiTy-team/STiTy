from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from app.api.v1.schema.health import HealthResult
from app.containers import Container
from app.services.health import HealthService

router = APIRouter(tags=["health"])


@router.get("/healthz", response_model=HealthResult)
@inject
def healthz(
    service: HealthService = Depends(Provide[Container.health_service]),
):
    return HealthResult(status=service.check(), pipeline=service.pipeline())
