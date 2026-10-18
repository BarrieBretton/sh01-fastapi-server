"""Coordinate legacy synchronous media paths with the canonical queue."""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import uuid
from contextlib import asynccontextmanager

from fastapi import HTTPException

from media_job_store import store
from media_resources import require_memory_headroom

logger = logging.getLogger("media_admission")
OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"
HEARTBEAT_SECONDS = max(5, int(os.getenv("MEDIA_QUEUE_HEARTBEAT_SECONDS", "20")))


async def _heartbeat(job_id: str, token: str, lost: asyncio.Event):
    while True:
        await asyncio.sleep(HEARTBEAT_SECONDS)
        try:
            if not await asyncio.to_thread(store.queue_heartbeat, job_id, OWNER, token):
                lost.set()
                logger.error("Legacy media request lost global admission: %s", job_id)
                return
        except Exception:
            lost.set()
            logger.exception("Legacy media heartbeat failed: %s", job_id)
            return


@asynccontextmanager
async def heavy_request(kind: str):
    job_id = "legacy-" + uuid.uuid4().hex
    try:
        token = await asyncio.to_thread(store.queue_acquire_external, job_id, kind, OWNER)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Shared media queue unavailable: {exc}") from exc
    if not token:
        raise HTTPException(status_code=409, detail={
            "code": "heavy_media_busy",
            "message": "A media job is running or waiting in the shared queue",
            "retryable": True,
        }, headers={"Retry-After": "10"})
    lost = asyncio.Event()
    heartbeat = asyncio.create_task(_heartbeat(job_id, token, lost))
    result = {"job_id": job_id, "status": "complete"}
    try:
        await asyncio.to_thread(require_memory_headroom, kind)
        yield
        if lost.is_set():
            raise HTTPException(status_code=503, detail="Lost shared media admission during request")
    except BaseException as exc:
        result = {"job_id": job_id, "status": "failed", "error": str(exc)[:1000]}
        raise
    finally:
        heartbeat.cancel()
        try:
            await heartbeat
        except asyncio.CancelledError:
            pass
        try:
            await asyncio.to_thread(store.queue_finish, job_id, OWNER, token,
                                    result["status"], result)
        except Exception:
            logger.exception("Could not release shared admission for %s", job_id)
