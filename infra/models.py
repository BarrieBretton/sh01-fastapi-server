# infra/models.py

from typing import Literal

from pydantic import BaseModel


InfraType = Literal[
    "render",
    "postgres",
    "b2",
]


class SlotSelectionRequest(BaseModel):
    slot: str
    dry_run: bool = True


class ActiveInfrastructure(BaseModel):
    render: str | None = None
    postgres: str | None = None
    b2: str | None = None


class InfraStatusResponse(BaseModel):
    active: ActiveInfrastructure
    available: dict[str, list[str]]