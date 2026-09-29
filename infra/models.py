from typing import Literal

from pydantic import BaseModel


InfraType = Literal["render", "postgres", "b2"]
RenderRole = Literal["n8n", "sh01"]


class SlotSelectionRequest(BaseModel):
    slot: str
    dry_run: bool = True


class RenderSlotSelectionRequest(BaseModel):
    slot: str
    dry_run: bool = True


class ActiveInfrastructure(BaseModel):
    render: str | None = None
    postgres: str | None = None
    b2: str | None = None


class InfraStatusResponse(BaseModel):
    active: ActiveInfrastructure
    available: dict[str, object]


class PostgresMigrationRequest(BaseModel):
    source: str
    destination: str
    dry_run: bool = True


class B2MigrationRequest(BaseModel):
    source: str
    destination: str
    prune_extra: bool = False
    dry_run: bool = True


class FailoverRequest(BaseModel):
    target_postgres: str | None = None
    target_render: str | None = None
    target_b2: str | None = None
    sync_b2: bool = False
    prune_b2_extra: bool = False
    switch_router: bool = True
    quiesce_source: bool = True
    dry_run: bool = True
