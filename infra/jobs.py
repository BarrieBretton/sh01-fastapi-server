import logging
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable

from .persistence import store

logger = logging.getLogger("infra.jobs")


class JobManager:
    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="infra-job")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def submit(
        self,
        kind: str,
        payload: dict[str, Any],
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        job_id = str(uuid.uuid4())
        job = {
            "id": job_id,
            "kind": kind,
            "status": "queued",
            "payload": payload,
            "result": None,
            "error": None,
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        with self._lock:
            self._jobs[job_id] = job
        store.put_job(job_id, kind, "queued", payload=payload)
        self._executor.submit(self._run, job_id, kind, payload, fn, args, kwargs)
        return dict(job)

    def _run(
        self,
        job_id: str,
        kind: str,
        payload: dict[str, Any],
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        self._update(job_id, status="running", error=None)
        store.put_job(job_id, kind, "running", payload=payload)
        try:
            result = fn(*args, **kwargs)
            self._update(job_id, status="succeeded", result=result, error=None)
            store.put_job(job_id, kind, "succeeded", payload=payload, result=result)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.error("Job %s failed: %s\n%s", job_id, error, traceback.format_exc())
            self._update(job_id, status="failed", error=error)
            store.put_job(job_id, kind, "failed", payload=payload, error=error)

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.update(changes)
            job["updated_at"] = self._now()

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])
        return store.get_job(job_id)


jobs = JobManager()
