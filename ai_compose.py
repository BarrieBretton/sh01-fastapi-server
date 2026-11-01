"""FastAPI routes and durable-job integration for AI composition."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ai_compose_engine import ComposeError, run_composition
from ai_compose_ir import EditPlan
from ai_compose_schema import AIComposeRequest, CompositionPlan, EditorialPolicy
from media_job_store import store
from media_resources import spawn_media_task, validate_high_resources

try:
    from media_capacity import MEDIA_RENDER_LOCK
except ImportError:  # pragma: no cover - useful when schema tooling is run alone
    MEDIA_RENDER_LOCK = asyncio.Lock()

logger = logging.getLogger("ai_compose_routes")
router = APIRouter(prefix="/media", tags=["AI Video Composition"])

AI_COMPOSE_JOB_KIND = "ai_compose"
DURABLE_TIMEOUT_SECONDS = max(5, int(os.getenv("AI_COMPOSE_DURABLE_TIMEOUT_SECONDS", "30")))
JOB_RETENTION_SECONDS = max(3600, int(os.getenv("AI_COMPOSE_JOB_RETENTION_SECONDS", "172800")))

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()
_background_tasks: set[asyncio.Task] = set()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _media_key() -> str:
    return (os.getenv("MEDIA_CLIP_API_KEY", "").strip() or os.getenv("TELEGRAM_STORAGE_API_KEY", "").strip())


def _require_key(value: str | None) -> None:
    expected = _media_key()
    if not expected:
        raise HTTPException(status_code=500, detail="MEDIA_CLIP_API_KEY is not configured")
    if not value or not hmac.compare_digest(value, expected):
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")


def _job_id(request_id: str) -> str:
    key = _media_key() or "ai-compose"
    return hmac.new(key.encode(), f"ai-compose:{request_id}".encode(), hashlib.sha256).hexdigest()[:32]


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


async def _persist(job: dict[str, Any], request: AIComposeRequest | None = None) -> bool:
    try:
        payload = {"request_id": job.get("request_id"), "request": request.model_dump(mode="json")} if request else None
        await asyncio.wait_for(
            asyncio.to_thread(
                store.put_job,
                str(job["job_id"]),
                AI_COMPOSE_JOB_KIND,
                str(job.get("status") or "unknown"),
                payload,
                _job_view(job),
                job.get("error"),
            ),
            timeout=DURABLE_TIMEOUT_SECONDS,
        )
        return True
    except Exception as exc:
        logger.warning("AI compose durable persist failed job=%s: %s", job.get("job_id"), exc)
        return False


async def _load_durable(job_id: str) -> dict[str, Any] | None:
    try:
        row = await asyncio.wait_for(asyncio.to_thread(store.get_job, job_id), timeout=DURABLE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="AI compose durable lookup timed out") from exc
    if not row or row.get("kind") != AI_COMPOSE_JOB_KIND:
        return None
    result = row.get("result")
    if isinstance(result, dict):
        output = dict(result)
    else:
        output = {"job_id": job_id, "status": row.get("status"), "error": row.get("error")}
    output["durable"] = True
    output["durable_updated_at"] = row.get("updated_at")
    return output


async def _prune_jobs() -> None:
    cutoff = time.time() - JOB_RETENTION_SECONDS
    async with _jobs_lock:
        stale = [
            job_id for job_id, job in _jobs.items()
            if job.get("status") in {"complete", "failed"}
            and float(job.get("_finished_epoch") or time.time()) < cutoff
        ]
        for job_id in stale:
            _jobs.pop(job_id, None)


def _track(task: asyncio.Task) -> None:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _execute(job_id: str, request: AIComposeRequest) -> None:
    async with _jobs_lock:
        job = _jobs[job_id]
    payload = {"request_id": job["request_id"], "request": request.model_dump(mode="json")}

    async def update(**values: Any) -> None:
        job.update(values)
        await _persist(job, request)

    try:
        async with MEDIA_RENDER_LOCK:
            job.update(status="running", stage="starting", started_at=_now())
            await _persist(job, request)
            await run_composition(request, job, update)
    except Exception as exc:
        logger.exception("AI compose failed job=%s", job_id)
        job.update(status="failed", stage="failed", error=str(exc)[:3000], finished_at=_now(), _finished_epoch=time.time())
        await _persist(job, request)


def make_dispatch_job(job_id: str, request: AIComposeRequest, attempt: int, lease_id: str) -> dict[str, Any]:
    """Called by media_dispatcher for the durable queue branch."""
    return {
        "job_id": job_id,
        "request_id": request.request_id or job_id,
        "status": "running",
        "stage": "queued",
        "progress": 0.0,
        "attempt": attempt,
        "created_at": _now(),
        "outputs": [],
        "status_url": f"{os.getenv('PUBLIC_API_BASE', '').rstrip('/')}/media/ai-compose-jobs/{job_id}",
        "_capacity_lease_id": lease_id,
    }


async def register_dispatch_job(job: dict[str, Any]) -> None:
    async with _jobs_lock:
        _jobs[job["job_id"]] = job


@router.post("/ai-compose")
async def create_ai_compose(request: AIComposeRequest, x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    _require_key(x_api_key)
    validate_high_resources(request.unlimited_resources)
    await _prune_jobs()
    request_id = (request.request_id or str(uuid.uuid4())).strip()
    job_id = _job_id(request_id)

    async with _jobs_lock:
        existing = _jobs.get(job_id)
        if existing and existing.get("status") in {"queued", "running", "complete"}:
            return JSONResponse(status_code=200 if existing.get("status") == "complete" else 202, content=_job_view(existing))
    durable = await _load_durable(job_id)
    if durable and durable.get("status") == "complete":
        return JSONResponse(status_code=200, content=durable)

    try:
        row = await asyncio.to_thread(
            store.queue_enqueue,
            job_id,
            AI_COMPOSE_JOB_KIND,
            {"request_id": request_id, "request": request.model_dump(mode="json")},
        )
        result = row.get("result")
        if isinstance(result, dict) and row.get("status") in {"complete", "failed", "interrupted"}:
            return JSONResponse(status_code=200, content=result)
        return JSONResponse(status_code=202, content={
            "job_id": job_id,
            "request_id": request_id,
            "status": row.get("status", "queued"),
            "stage": row.get("status", "queued"),
            "attempt": row.get("attempts", 0),
            "status_url": f"{os.getenv('PUBLIC_API_BASE', '').rstrip('/')}/media/ai-compose-jobs/{job_id}",
            "outputs": [],
            "error": None,
        })
    except Exception as exc:
        if os.getenv("AI_COMPOSE_LOCAL_FALLBACK", "true").lower() not in {"1", "true", "yes"}:
            raise HTTPException(status_code=503, detail=f"Canonical media queue unavailable: {exc}") from exc
        job = {
            "job_id": job_id,
            "request_id": request_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "attempt": 0,
            "created_at": _now(),
            "outputs": [],
            "status_url": f"{os.getenv('PUBLIC_API_BASE', '').rstrip('/')}/media/ai-compose-jobs/{job_id}",
            "queue_fallback": True,
        }
        async with _jobs_lock:
            _jobs[job_id] = job
        await _persist(job, request)
        _track(spawn_media_task(_execute(job_id, request), request.unlimited_resources))
        return JSONResponse(status_code=202, content=_job_view(job))


@router.get("/ai-compose-jobs/{job_id}")
async def get_ai_compose_job(job_id: str, x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    _require_key(x_api_key)
    await _prune_jobs()
    async with _jobs_lock:
        local = _jobs.get(job_id)
        if local:
            return _job_view(local)
    try:
        queued = await asyncio.to_thread(store.queue_get, job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
    if queued:
        if isinstance(queued.get("result"), dict):
            return queued["result"]
        return {"job_id": job_id, "request_id": queued.get("request_id"), "status": queued.get("status"), "stage": queued.get("status"), "attempt": queued.get("attempts", 0), "outputs": [], "error": queued.get("error")}
    durable = await _load_durable(job_id)
    if durable:
        return durable
    raise HTTPException(status_code=404, detail="Unknown or expired AI compose job")


@router.get("/ai-compose-capabilities")
async def ai_compose_capabilities(x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    _require_key(x_api_key)
    return {
        "workflow": "sh01/ai-compose@1",
        "local_director": bool(os.getenv("AI_COMPOSE_LLM_URL", "").strip() or os.getenv("LLAMA_SERVER_URL", "").strip()),
        "local_director_url_configured": bool(os.getenv("AI_COMPOSE_LLM_URL", "").strip() or os.getenv("LLAMA_SERVER_URL", "").strip()),
        "asset_sources": {"local_library": True, "pexels": bool(os.getenv("PEXELS_API_KEY", "")), "pixabay": bool(os.getenv("PIXABAY_API_KEY", "")), "unsplash": bool(os.getenv("UNSPLASH_ACCESS_KEY", "")), "freesound": bool(os.getenv("FREESOUND_API_KEY", ""))},
        "features": [
            "long-source-chunking",
            "word-timed-captions",
            "semantic-asset-requests",
            "explicit-editorial-policy",
            "versioned-edit-plan",
            "evidence-artifact",
            "partial-frame-video-layers",
            "rendered-motion-graphics",
            "animated-camera-keyframes",
            "xfade-transitions",
            "render-feature-contract",
            "motivated-camera-punch-ins",
            "adaptive-sfx",
            "optional-local-voiceover",
            "multi-variant-output",
            "technical-quality-review",
            "provenance-receipt",
            "preview-local-file-route",
        ],
        "animations": ["none", "fade", "slide_up", "slide_down", "slide_left", "slide_right", "float", "reveal"],
        "geometry": "normalized x/y/width/height coordinates",
        "encoder": os.getenv("AI_COMPOSE_VIDEO_CODEC", "auto"),
        "work_dir": os.getenv("AI_COMPOSE_WORK_DIR", "media_jobs"),
    }


@router.get("/ai-compose-schema")
async def ai_compose_schema(x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    _require_key(x_api_key)
    return {
        "request": AIComposeRequest.model_json_schema(),
        "plan": CompositionPlan.model_json_schema(),
        "editorial_policy": EditorialPolicy.model_json_schema(),
        "edit_plan": EditPlan.model_json_schema(),
    }


@router.get("/ai-compose-files/{job_id}/{filename}")
async def get_ai_compose_file(job_id: str, filename: str, x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    _require_key(x_api_key)
    if not re.fullmatch(r"variant_[0-9]{2}\.mp4", filename):
        raise HTTPException(status_code=400, detail="Invalid AI compose filename")
    root = (Path(os.getenv("AI_COMPOSE_WORK_DIR", "media_jobs")) / job_id).resolve()
    path = (root / filename).resolve()
    if not path.is_relative_to(root) or not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="AI compose file not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)

