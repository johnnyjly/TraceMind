from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]


def create_app() -> FastAPI:
    application = FastAPI(
        title="TraceMind",
        description="Evidence-driven incident diagnosis for microservices.",
        version="0.1.0",
    )

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok")

    return application


app = create_app()
