from __future__ import annotations

import asyncio
import os
import secrets
import time
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException


RETRY_AFTER_SECONDS = max(1, int(os.getenv("HEAVY_MEDIA_RETRY_AFTER_SECONDS", "5")))


@dataclass(frozen=True)
class HeavyMediaLease:
    lease_id: str
    kind: str
    job_id: str
    request_id: str | None
    endpoint: str | None
    acquired_at_epoch: float

    @property
    def acquired_at(self) -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.acquired_at_epoch))


class HeavyMediaCapacity:
    """Single-slot, reject-on-busy capacity guard for RAM-heavy media work.

    This guard is intentionally process-local. SH01 currently runs one uvicorn worker,
    so this protects all heavy endpoints in that process. The infrastructure router
    is responsible for ensuring only one Render ring is operationally active.
    """

    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._active: HeavyMediaLease | None = None

    async def try_reserve(
        self,
        *,
        kind: str,
        job_id: str,
        request_id: str | None = None,
        endpoint: str | None = None,
    ) -> HeavyMediaLease | None:
        async with self._state_lock:
            if self._active is not None:
                return None
            lease = HeavyMediaLease(
                lease_id=secrets.token_hex(16),
                kind=str(kind),
                job_id=str(job_id),
                request_id=str(request_id) if request_id is not None else None,
                endpoint=str(endpoint) if endpoint is not None else None,
                acquired_at_epoch=time.time(),
            )
            self._active = lease
            return lease

    async def release(self, lease_id: str) -> bool:
        lease_id = str(lease_id or "")
        if not lease_id:
            return False
        async with self._state_lock:
            if self._active is None or self._active.lease_id != lease_id:
                return False
            self._active = None
            return True

    async def status(self) -> dict[str, Any]:
        async with self._state_lock:
            active = self._active
            if active is None:
                return {
                    "capacity": 1,
                    "busy": False,
                    "active_kind": None,
                    "active_job_id": None,
                    "active_request_id": None,
                    "active_endpoint": None,
                    "active_since": None,
                    "active_for_seconds": 0,
                    "retry_after_seconds": RETRY_AFTER_SECONDS,
                }
            return {
                "capacity": 1,
                "busy": True,
                "active_kind": active.kind,
                "active_job_id": active.job_id,
                "active_request_id": active.request_id,
                "active_endpoint": active.endpoint,
                "active_since": active.acquired_at,
                "active_for_seconds": round(max(0.0, time.time() - active.acquired_at_epoch), 3),
                "retry_after_seconds": RETRY_AFTER_SECONDS,
            }


HEAVY_MEDIA_CAPACITY = HeavyMediaCapacity()

# Defense-in-depth execution mutex. Known heavy endpoints reserve capacity before
# returning 202, then also enter this lock while executing. This prevents accidental
# parallel execution if a future endpoint is added incorrectly but still uses the lock.
MEDIA_RENDER_LOCK = asyncio.Lock()


async def reserve_heavy_media_or_raise(
    *,
    kind: str,
    job_id: str,
    request_id: str | None = None,
    endpoint: str | None = None,
) -> HeavyMediaLease:
    lease = await HEAVY_MEDIA_CAPACITY.try_reserve(
        kind=kind,
        job_id=job_id,
        request_id=request_id,
        endpoint=endpoint,
    )
    if lease is not None:
        return lease

    status = await HEAVY_MEDIA_CAPACITY.status()
    raise HTTPException(
        status_code=409,
        detail={
            "code": "heavy_media_busy",
            "message": "SH01 is already processing another RAM-heavy media job",
            "retryable": True,
            **status,
        },
        headers={"Retry-After": str(RETRY_AFTER_SECONDS)},
    )
