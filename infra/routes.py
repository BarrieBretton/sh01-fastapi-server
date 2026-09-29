# infra/routes.py

from fastapi import APIRouter, Depends, HTTPException

from .auth import require_control_plane_key
from .models import (
    ActiveInfrastructure,
    InfraStatusResponse,
    SlotSelectionRequest,
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