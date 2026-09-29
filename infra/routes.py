# infra/routes.py
import shutil

from fastapi import APIRouter, Depends, HTTPException

from .auth import require_control_plane_key

from .postgres import (
    postgres_health,
    postgres_verify,
    postgres_migrate,
)

from .models import (
    ActiveInfrastructure,
    InfraStatusResponse,
    SlotSelectionRequest,
    PostgresMigrationRequest
)
from .registry import registry
from .state import state


router = APIRouter(
    prefix="/infra",
    tags=["Infrastructure Control Plane"],
    dependencies=[
        Depends(require_control_plane_key)
    ],
)


@router.get("/health")
async def infra_health():
    return {
        "ok": True,
        "service": "infra-control-plane",
    }


@router.get(
    "/status",
    response_model=InfraStatusResponse,
)
async def infra_status():
    return InfraStatusResponse(
        active=ActiveInfrastructure(
            **state.all()
        ),
        available={
            "render": registry.slots(
                "render"
            ),
            "postgres": registry.slots(
                "postgres"
            ),
            "b2": registry.slots(
                "b2"
            ),
        },
    )


@router.get("/registry")
async def infra_registry():
    """
    Safe registry representation.

    Later we will explicitly redact any secret-bearing
    fields once provider configs are added.
    """

    return {
        "render": registry.slots(
            "render"
        ),
        "postgres": registry.slots(
            "postgres"
        ),
        "b2": registry.slots(
            "b2"
        ),
    }


def select_slot(
    infra_type: str,
    request: SlotSelectionRequest,
):
    if not registry.exists(
        infra_type,
        request.slot,
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown {infra_type} slot: "
                f"{request.slot}"
            ),
        )

    current = state.get_active(
        infra_type
    )

    result = {
        "infra_type": infra_type,
        "previous": current,
        "target": request.slot,
        "dry_run": request.dry_run,
    }

    if request.dry_run:
        return {
            **result,
            "changed": False,
        }

    state.set_active(
        infra_type,
        request.slot,
    )

    return {
        **result,
        "changed": True,
    }


@router.post("/select/render")
async def select_render(
    request: SlotSelectionRequest,
):
    return select_slot(
        "render",
        request,
    )


@router.post("/select/postgres")
async def select_postgres(
    request: SlotSelectionRequest,
):
    return select_slot(
        "postgres",
        request,
    )


@router.post("/select/b2")
async def select_b2(
    request: SlotSelectionRequest,
):
    return select_slot(
        "b2",
        request,
    )

@router.get("/diagnostics/postgres-tools")
async def postgres_tools_diagnostic():
    return {
        "pg_dump": shutil.which("pg_dump"),
        "pg_restore": shutil.which("pg_restore"),
        "psql": shutil.which("psql"),
    }

@router.get("/postgres/{slot}/health")
async def postgres_slot_health(
    slot: str,
):
    if not registry.exists(
        "postgres",
        slot,
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown postgres slot: {slot}",
        )

    try:
        return postgres_health(slot)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post("/postgres/{slot}/verify")
async def postgres_slot_verify(
    slot: str,
):
    if not registry.exists(
        "postgres",
        slot,
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown postgres slot: {slot}",
        )

    try:
        return postgres_verify(slot)

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@router.post("/postgres/migrate")
async def migrate_postgres(
    request: PostgresMigrationRequest,
):
    if not registry.exists(
        "postgres",
        request.source,
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown source postgres slot: {request.source}",
        )

    if not registry.exists(
        "postgres",
        request.destination,
    ):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown destination postgres slot: {request.destination}",
        )

    if request.dry_run:
        return {
            "source": request.source,
            "destination": request.destination,
            "dry_run": True,
            "would_migrate": True,
        }

    try:
        source_health = postgres_health(
            request.source
        )

        destination_health = postgres_health(
            request.destination
        )

        if not source_health["healthy"]:
            raise RuntimeError(
                "Source database is not healthy"
            )

        if not destination_health["healthy"]:
            raise RuntimeError(
                "Destination database is not healthy"
            )

        migration = postgres_migrate(
            request.source,
            request.destination,
        )

        verification = postgres_verify(
            request.destination
        )

        return {
            "dry_run": False,
            "migration": migration,
            "verification": verification,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )