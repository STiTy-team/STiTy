from pydantic import BaseModel


class HealthResult(BaseModel):
    status: str
    pipeline: dict
