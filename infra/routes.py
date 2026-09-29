# infra/routes.py

import shutil
import subprocess

from fastapi import APIRouter, Depends, HTTPException

from .auth import require_control_plane_key
from .models import (
    ActiveInfrastructure,
    InfraStatusResponse,
    PostgresMigrationRequest,
    SlotSelectionRequest,
)
from .postgres import (
    postgres_health,
    postgres_migrate,
    postgres_verify,
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

    Secret-bearing values are not returned here.
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

    def version(
        command: str,
    ) -> str | None:
        path = shutil.which(
            command
        )

        if not path:
            return None

        result = subprocess.run(
            [
                command,
                "--version",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        return (
            result.stdout.strip()
            or result.stderr.strip()
        )

    return {
        "pg_dump": {
            "path": shutil.which(
                "pg_dump"
            ),
            "version": version(
                "pg_dump"
            ),
        },
        "pg_restore": {
            "path": shutil.which(
                "pg_restore"
            ),
            "version": version(
                "pg_restore"
            ),
        },
        "psql": {
            "path": shutil.which(
                "psql"
            ),
            "version": version(
                "psql"
            ),
        },
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
            detail=(
                f"Unknown postgres slot: "
                f"{slot}"
            ),
        )

    try:
        return postgres_health(
            slot
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


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
            detail=(
                f"Unknown postgres slot: "
                f"{slot}"
            ),
        )

    try:
        return postgres_verify(
            slot
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc


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
            detail=(
                "Unknown source postgres slot: "
                f"{request.source}"
            ),
        )

    if not registry.exists(
        "postgres",
        request.destination,
    ):
        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown destination postgres slot: "
                f"{request.destination}"
            ),
        )

    if request.source == request.destination:
        raise HTTPException(
            status_code=400,
            detail=(
                "Source and destination "
                "Postgres slots must differ"
            ),
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
                "Source database is not healthy: "
                + source_health.get(
                    "stderr",
                    "",
                )
            )

        if not destination_health["healthy"]:
            raise RuntimeError(
                "Destination database is not healthy: "
                + destination_health.get(
                    "stderr",
                    "",
                )
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
        ) from exc