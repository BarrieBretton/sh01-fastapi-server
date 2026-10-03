from __future__ import annotations

import asyncio
import os
import secrets
from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from .persistence import store


MEDIA_JOB_STORE_API_KEY = os.getenv("MEDIA_JOB_STORE_API_KEY", "").strip()
MEDIA_JOB_STORE_SERVER_ENABLED = (
    os.getenv("MEDIA_JOB_STORE_SERVER_ENABLED", "false").strip().lower()
    in {"1", "true", "yes", "on"}
)

router = APIRouter(prefix="/internal/media-jobs", tags=["Internal Media Job Store"])


def _require_media_job_key(
    x_media_job_key: str | None = Header(default=None, alias="X-MEDIA-JOB-KEY"),
) -> None:
    if not MEDIA_JOB_STORE_SERVER_ENABLED:
        raise HTTPException(status_code=404, detail="Not found")
    if not MEDIA_JOB_STORE_API_KEY:
        raise HTTPException(status_code=503, detail="MEDIA_JOB_STORE_API_KEY is not configured")
    if not x_media_job_key or not secrets.compare_digest(
        x_media_job_key,
        MEDIA_JOB_STORE_API_KEY,
    ):
        raise HTTPException(status_code=401, detail="Missing/invalid media job store key")


class MediaJobWrite(BaseModel):
    kind: str
    status: str
    payload: Any = None
    result: Any = None
    error: str | None = None


@router.get("/health")
async def media_job_store_health(
    x_media_job_key: str | None = Header(default=None, alias="X-MEDIA-JOB-KEY"),
):
    _require_media_job_key(x_media_job_key)
    healthy = await asyncio.to_thread(store._healthy_slots)
    if not healthy:
        raise HTTPException(status_code=503, detail="No healthy Postgres slots")
    return {"ok": True, "healthy_postgres_slots": healthy}


@router.put("/{job_id}")
async def put_media_job(
    job_id: str,
    body: MediaJobWrite,
    x_media_job_key: str | None = Header(default=None, alias="X-MEDIA-JOB-KEY"),
):
    _require_media_job_key(x_media_job_key)
    try:
        await asyncio.to_thread(
            store.put_job,
            job_id,
            body.kind,
            body.status,
            body.payload,
            body.result,
            body.error,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Durable media job write failed: {exc}") from exc
    return {"ok": True, "job_id": job_id, "status": body.status}


@router.get("/{job_id}")
async def get_media_job(
    job_id: str,
    x_media_job_key: str | None = Header(default=None, alias="X-MEDIA-JOB-KEY"),
):
    _require_media_job_key(x_media_job_key)
    try:
        job = await asyncio.to_thread(store.get_job, job_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Durable media job read failed: {exc}") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown media job")
    return {"ok": True, "job": job}
