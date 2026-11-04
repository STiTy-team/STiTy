from fastapi import APIRouter

from app.api.v1.endpoints import conversation, health

routers = APIRouter()
routers.include_router(health.router)
routers.include_router(conversation.router)
