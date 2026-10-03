from __future__ import annotations

import os
import time
from typing import Any

import requests


MODE = os.getenv("MEDIA_JOB_STORE_MODE", "remote").strip().lower()
CONTROL_PLANE_BASE_URL = os.getenv("CONTROL_PLANE_BASE_URL", "").strip().rstrip("/")
MEDIA_JOB_STORE_API_KEY = os.getenv("MEDIA_JOB_STORE_API_KEY", "").strip()
TIMEOUT_SECONDS = max(2.0, float(os.getenv("MEDIA_JOB_STORE_TIMEOUT_SECONDS", "8")))
RETRIES = max(1, min(4, int(os.getenv("MEDIA_JOB_STORE_RETRIES", "2"))))


class MediaJobStore:
    def _local_store(self):
        from infra.persistence import store as local_store
        return local_store

    def _remote_config(self) -> tuple[str, str]:
        if not CONTROL_PLANE_BASE_URL:
            raise RuntimeError("CONTROL_PLANE_BASE_URL is not configured")
        if not MEDIA_JOB_STORE_API_KEY:
            raise RuntimeError("MEDIA_JOB_STORE_API_KEY is not configured")
        return CONTROL_PLANE_BASE_URL, MEDIA_JOB_STORE_API_KEY

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        allow_404: bool = False,
    ) -> requests.Response | None:
        base, key = self._remote_config()
        url = f"{base}{path}"
        headers = {
            "X-MEDIA-JOB-KEY": key,
            "Accept": "application/json",
        }
        if json_body is not None:
            headers["Content-Type"] = "application/json"

        last_error: Exception | None = None
        for attempt in range(RETRIES):
            try:
                response = requests.request(
                    method,
                    url,
                    headers=headers,
                    json=json_body,
                    timeout=TIMEOUT_SECONDS,
                )
                if allow_404 and response.status_code == 404:
                    try:
                        detail = response.json().get("detail")
                    except Exception:
                        detail = None
                    if detail == "Unknown media job":
                        return None
                    raise RuntimeError(
                        "Media job store endpoint returned an unexpected 404: "
                        f"{response.text[:1000]}"
                    )
                if response.status_code < 500:
                    response.raise_for_status()
                    return response
                last_error = RuntimeError(
                    f"Media job store returned HTTP {response.status_code}: "
                    f"{response.text[:1000]}"
                )
            except requests.RequestException as exc:
                last_error = exc

            if attempt + 1 < RETRIES:
                time.sleep(0.4 * (attempt + 1))

        raise RuntimeError(f"Media job store request failed: {last_error}")

    def put_job(
        self,
        job_id: str,
        kind: str,
        status: str,
        payload: Any = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        if MODE == "local":
            self._local_store().put_job(
                job_id,
                kind,
                status,
                payload=payload,
                result=result,
                error=error,
            )
            return
        if MODE != "remote":
            raise RuntimeError(
                f"Unsupported MEDIA_JOB_STORE_MODE={MODE!r}; expected 'local' or 'remote'"
            )

        self._request(
            "PUT",
            f"/internal/media-jobs/{job_id}",
            json_body={
                "kind": kind,
                "status": status,
                "payload": payload,
                "result": result,
                "error": error,
            },
        )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if MODE == "local":
            return self._local_store().get_job(job_id)
        if MODE != "remote":
            raise RuntimeError(
                f"Unsupported MEDIA_JOB_STORE_MODE={MODE!r}; expected 'local' or 'remote'"
            )

        response = self._request(
            "GET",
            f"/internal/media-jobs/{job_id}",
            allow_404=True,
        )
        if response is None:
            return None
        body = response.json()
        job = body.get("job")
        return job if isinstance(job, dict) else None


store = MediaJobStore()
