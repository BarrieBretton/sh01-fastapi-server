"""Persistent one-at-a-time media queue consumer. Start on SH01 workers only."""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from typing import Any

from media_capacity import HEAVY_MEDIA_CAPACITY
from media_job_store import store
from media_resources import spawn_media_task, validate_high_resources

logger = logging.getLogger("media_dispatcher")
OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
POLL_SECONDS = max(1, int(os.getenv("MEDIA_QUEUE_POLL_SECONDS", "10")))
HEARTBEAT_SECONDS = max(5, int(os.getenv("MEDIA_QUEUE_HEARTBEAT_SECONDS", "20")))
_poller: asyncio.Task | None = None


async def _heartbeat(job_id: str, token: str, lost: asyncio.Event) -> None:
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            alive = await asyncio.to_thread(store.queue_heartbeat, job_id, OWNER, token)
        except Exception as exc:
            logger.error("Queue heartbeat unavailable for %s: %s", job_id, exc)
            lost.set()
            return
        if not alive:
            logger.error("Queue ownership lost for %s", job_id)
            lost.set()
            return


async def _run(row: dict[str, Any], lease_id: str) -> bool:
    job_id, kind, payload, token = row["id"], row["kind"], row["payload"], row["lease_token"]
    lost = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(job_id, token, lost))
    result: dict[str, Any] = {}
    try:
        if kind == "auto_edit":
            import auto_editor as m
            request = m.AutoEditRequest.model_validate(payload["request"])
            validate_high_resources(request.unlimited_resources)
            job = {"job_id": job_id, "request_id": payload["request_id"],
                   "status": "running", "stage": "queued", "attempt": row["attempts"],
                   "created_at": m._now(), "outputs": [],
                   "status_url": f"{m.PUBLIC_API_BASE}/media/auto-edit-jobs/{job_id}",
                   "_capacity_lease_id": lease_id}
            async with m._jobs_lock:
                m._jobs[job_id] = job
            await spawn_media_task(m._execute(job_id, request), request.unlimited_resources)
            result = m._job_view(job)
        elif kind == "clip_caption":
            import caption_router as m
            request = m.ClipCaptionRequest.model_validate(payload["request"])
            validate_high_resources(request.unlimited_resources)
            job = {"job_id": job_id, "request_id": payload["request_id"],
                   "status": "running", "stage": "queued", "attempt": row["attempts"],
                   "created_at": m.time.strftime("%Y-%m-%dT%H:%M:%SZ", m.time.gmtime()),
                   "token": m._request_token(payload["request_id"]), "video_url": None,
                   "status_url": f"{m.PUBLIC_API_BASE}/media/clip-caption-jobs/{job_id}",
                   "_capacity_lease_id": lease_id}
            async with m._jobs_lock:
                m._jobs[job_id] = job
            await spawn_media_task(m._execute(job_id, request), request.unlimited_resources)
            result = m._job_view(job)
        elif kind == "audio_image_video":
            import media_router as m
            request = m.AudioImageRenderRequest.model_validate(payload["request"])
            validate_high_resources(request.unlimited_resources)
            job = {"job_id": job_id, "queue_id": request.queue_id,
                   "status": "accepted", "token": m._render_token_for_queue(request.queue_id),
                   "video_url": None, "error": None, "_capacity_lease_id": lease_id}
            async with m._render_state_lock:
                m._render_jobs[job_id] = job
                m._queue_to_job[request.queue_id] = job_id
                m._active_job_id = job_id
            await spawn_media_task(m._run_render_job(job_id, request), request.unlimited_resources)
            result = m._job_payload(job)
        else:
            raise RuntimeError(f"Unsupported media kind: {kind}")
    except Exception as exc:
        logger.exception("Queue consumer failed %s", job_id)
        result = {"job_id": job_id, "status": "failed", "stage": "failed", "error": str(exc)[:3000]}
        await HEAVY_MEDIA_CAPACITY.release(lease_id)
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
    if lost.is_set():
        logger.error("Job %s completed locally but queue ownership was lost; stopping dispatcher", job_id)
        return False
    status = str(result.get("status", "failed"))
    if status not in {"complete", "failed", "interrupted"}:
        status = "failed"
        result.update(status=status, error=result.get("error") or "Worker ended without a terminal status")
    try:
        if not await asyncio.to_thread(store.queue_finish, job_id, OWNER, token, status, result):
            logger.error("Queue claim changed while finishing %s; stopping dispatcher", job_id)
            return False
    except Exception:
        logger.exception("Cannot finish queue job %s; stopping dispatcher", job_id)
        return False
    return True


async def consume() -> None:
    while True:
        try:
            lease = await HEAVY_MEDIA_CAPACITY.try_reserve(
                kind="queued_media", job_id="pending", endpoint="/media/queue")
            if lease is None:
                await asyncio.sleep(POLL_SECONDS)
                continue
            try:
                high_capable = os.getenv("MEDIA_ALLOW_UNLIMITED_RESOURCES", "false").lower() in {"1", "true", "yes"}
                row = await asyncio.to_thread(store.queue_claim, OWNER, high_capable)
                if row is None:
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                if not await _run(row, lease.lease_id):
                    return  # Fail closed if the canonical queue cannot confirm ownership.
            finally:
                await HEAVY_MEDIA_CAPACITY.release(lease.lease_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Media queue poll failed")
            await asyncio.sleep(max(5, POLL_SECONDS))


def start() -> None:
    global _poller
    lease_seconds = max(60, int(os.getenv("MEDIA_QUEUE_LEASE_SECONDS", "300")))
    if HEARTBEAT_SECONDS * 3 >= lease_seconds:
        raise RuntimeError("MEDIA_QUEUE_HEARTBEAT_SECONDS must be less than one third of the queue lease")
    if _poller is None or _poller.done():
        _poller = asyncio.create_task(consume(), name="sh01-media-queue")
