import shutil
import subprocess

from fastapi import APIRouter, Depends, HTTPException

from .auth import require_control_plane_key
from .b2ring import b2_compare, b2_health, b2_migrate
from .jobs import jobs
from .models import (
    ActiveInfrastructure,
    B2MigrationRequest,
    FailoverRequest,
    InfraStatusResponse,
    PostgresMigrationRequest,
    RenderSlotSelectionRequest,
    SlotSelectionRequest,
)
from .orchestrator import build_failover_plan, run_failover
from .postgres import postgres_compare, postgres_health, postgres_migrate, postgres_verify
from .registry import registry
from .render_provider import render_health, render_service_status
from .router_client import router_status
from .state import state


router = APIRouter(
    prefix="/infra",
    tags=["Infrastructure Control Plane"],
    dependencies=[Depends(require_control_plane_key)],
)


def _require_slot(kind: str, slot: str) -> None:
    if not registry.exists(kind, slot):
        raise HTTPException(status_code=404, detail=f"Unknown {kind} slot: {slot}")


@router.get("/health")
async def infra_health():
    return {
        "ok": True,
        "service": "infra-control-plane",
        "state": state.snapshot(),
    }


@router.get("/status", response_model=InfraStatusResponse)
async def infra_status():
    return InfraStatusResponse(
        active=ActiveInfrastructure(**state.all()),
        available={
            "render": {
                "n8n": registry.render_slots("n8n"),
                "sh01": registry.render_slots("sh01"),
            },
            "postgres": registry.slots("postgres"),
            "b2": registry.slots("b2"),
        },
    )


@router.get("/registry")
async def infra_registry():
    return {
        "render": {
            "n8n": registry.render_slots("n8n"),
            "sh01": registry.render_slots("sh01"),
        },
        "postgres": registry.slots("postgres"),
        "b2": registry.slots("b2"),
    }


def select_slot(infra_type: str, request: SlotSelectionRequest):
    _require_slot(infra_type, request.slot)
    current = state.get_active(infra_type)
    if request.dry_run:
        return {
            "infra_type": infra_type,
            "previous": current,
            "target": request.slot,
            "dry_run": True,
            "changed": False,
        }
    result = state.set_active(infra_type, request.slot, persist=True)
    return {**result, "dry_run": False, "changed": True}


@router.post("/select/render")
async def select_render(request: RenderSlotSelectionRequest):
    _require_slot("render", request.slot)
    actual_role = registry.render_role(request.slot)
    if actual_role != request.role:
        raise HTTPException(
            status_code=400,
            detail=f"Render slot {request.slot} has role={actual_role}, not role={request.role}",
        )
    current = state.get_active("render", role=request.role)
    if request.dry_run:
        return {
            "infra_type": "render",
            "role": request.role,
            "previous": current,
            "target": request.slot,
            "dry_run": True,
            "changed": False,
        }
    result = state.set_active(
        "render",
        request.slot,
        persist=True,
        role=request.role,
    )
    return {**result, "dry_run": False, "changed": True}


@router.post("/select/postgres")
async def select_postgres(request: SlotSelectionRequest):
    return select_slot("postgres", request)


@router.post("/select/b2")
async def select_b2(request: SlotSelectionRequest):
    return select_slot("b2", request)


@router.get("/diagnostics/postgres-tools")
async def postgres_tools_diagnostic():
    def version(command: str) -> str | None:
        path = shutil.which(command)
        if not path:
            return None
        result = subprocess.run(
            [command, "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or result.stderr.strip()

    return {
        "pg_dump": {"path": shutil.which("pg_dump"), "version": version("pg_dump")},
        "pg_restore": {"path": shutil.which("pg_restore"), "version": version("pg_restore")},
        "psql": {"path": shutil.which("psql"), "version": version("psql")},
    }


@router.get("/postgres/{slot}/health")
async def postgres_slot_health(slot: str):
    _require_slot("postgres", slot)
    try:
        return postgres_health(slot)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/postgres/{slot}/verify")
async def postgres_slot_verify(slot: str):
    _require_slot("postgres", slot)
    try:
        return postgres_verify(slot)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/postgres/compare")
async def compare_postgres(source: str, destination: str):
    _require_slot("postgres", source)
    _require_slot("postgres", destination)
    if source == destination:
        raise HTTPException(status_code=400, detail="Source and destination Postgres slots must differ")
    try:
        return postgres_compare(source, destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/postgres/migrate")
async def migrate_postgres(request: PostgresMigrationRequest):
    _require_slot("postgres", request.source)
    _require_slot("postgres", request.destination)
    if request.source == request.destination:
        raise HTTPException(status_code=400, detail="Source and destination Postgres slots must differ")
    if request.dry_run:
        return {
            "source": request.source,
            "destination": request.destination,
            "dry_run": True,
            "would_migrate": True,
        }
    try:
        source_health = postgres_health(request.source)
        destination_health = postgres_health(request.destination)
        if not source_health["healthy"]:
            raise RuntimeError("Source database is not healthy: " + source_health.get("stderr", ""))
        if not destination_health["healthy"]:
            raise RuntimeError("Destination database is not healthy: " + destination_health.get("stderr", ""))
        migration = postgres_migrate(request.source, request.destination)
        verification = postgres_verify(request.destination)
        return {"dry_run": False, "migration": migration, "verification": verification}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/render/{slot}/health")
async def render_slot_health(slot: str):
    _require_slot("render", slot)
    return render_health(slot)


@router.get("/render/{slot}/status")
async def render_slot_status(slot: str):
    _require_slot("render", slot)
    try:
        return render_service_status(slot)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/b2/{slot}/health")
async def b2_slot_health(slot: str):
    _require_slot("b2", slot)
    return b2_health(slot)


@router.get("/b2/compare")
async def compare_b2(source: str, destination: str):
    _require_slot("b2", source)
    _require_slot("b2", destination)
    if source == destination:
        raise HTTPException(status_code=400, detail="Source and destination B2 slots must differ")
    try:
        return b2_compare(source, destination)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/b2/migrate")
async def migrate_b2(request: B2MigrationRequest):
    _require_slot("b2", request.source)
    _require_slot("b2", request.destination)
    if request.source == request.destination:
        raise HTTPException(status_code=400, detail="Source and destination B2 slots must differ")
    if request.dry_run:
        return {
            "source": request.source,
            "destination": request.destination,
            "prune_extra": request.prune_extra,
            "dry_run": True,
            "would_migrate": True,
        }
    job = jobs.submit(
        "b2_migrate",
        request.model_dump(),
        b2_migrate,
        request.source,
        request.destination,
        request.prune_extra,
    )
    return {"accepted": True, "job": job}


@router.get("/router/{role}/status")
async def cloudflare_router_status(role: str):
    if role not in {"n8n", "sh01"}:
        raise HTTPException(status_code=404, detail=f"Unknown router role: {role}")
    try:
        return router_status(role)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# Compatibility alias for the original package. It means the n8n router.
@router.get("/router/status")
async def cloudflare_router_status_legacy():
    try:
        return router_status("n8n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/failover")
async def failover(request: FailoverRequest):
    try:
        plan = build_failover_plan(
            target_postgres=request.target_postgres,
            target_render=request.target_render,
            target_b2=request.target_b2,
            sync_b2=request.sync_b2,
            prune_b2_extra=request.prune_b2_extra,
            switch_router=request.switch_router,
            quiesce_source=request.quiesce_source,
        )
        if request.dry_run:
            return {"dry_run": True, "plan": plan}

        job = jobs.submit(
            "failover",
            request.model_dump(),
            run_failover,
            request.target_postgres,
            request.target_render,
            request.target_b2,
            request.sync_b2,
            request.prune_b2_extra,
            request.switch_router,
            request.quiesce_source,
        )
        return {"accepted": True, "job": job, "plan": plan}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job: {job_id}")
    return job
