from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import requests
from requests_oauthlib import OAuth1

from .config import Settings, credentials_for, normalize_account
from .db import Database


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class XApiError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"X API HTTP {status_code}: {message}")


class XPublishConflict(RuntimeError):
    pass


class XService:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def _oauth(self, account: str) -> OAuth1:
        c = credentials_for(account)
        return OAuth1(c.api_key, c.api_key_secret, c.access_token, c.access_token_secret)

    @staticmethod
    def _json_or_text(response: requests.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text[:4000]

    @classmethod
    def _raise_x(cls, response: requests.Response) -> None:
        if response.status_code < 400:
            return
        body = cls._json_or_text(response)
        raise XApiError(response.status_code, str(body)[:4000])

    async def validate_account(self, account: str) -> dict[str, Any]:
        key = normalize_account(account)
        oauth = self._oauth(key)

        def run():
            r = requests.get(
                f"{self.settings.api_base}/2/users/me",
                params={"user.fields": "username"},
                auth=oauth,
                timeout=30,
            )
            self._raise_x(r)
            return r.json()

        payload = await asyncio.to_thread(run)
        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        username = normalize_account(data.get("username", "")) if isinstance(data, dict) else ""
        if not username:
            raise RuntimeError(f"X /2/users/me returned no username: {payload}")
        if username != key:
            raise RuntimeError(f"X credential/account mismatch: requested @{key}, authenticated as @{username}")
        return data

    async def _download_media(self, url: str, media_type: str) -> tuple[Path, str]:
        suffix = ".mp4" if media_type == "VIDEO" else ".jpg"
        fd, raw_path = tempfile.mkstemp(prefix="sh01-x-", suffix=suffix)
        os.close(fd)
        path = Path(raw_path)
        total = 0
        content_type = ""

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").split(";")[0].strip()
                    with path.open("wb") as f:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > self.settings.media_max_bytes:
                                raise RuntimeError(
                                    f"X media exceeds configured X_MEDIA_MAX_BYTES={self.settings.media_max_bytes}"
                                )
                            f.write(chunk)

            if total <= 0:
                raise RuntimeError("Downloaded X media is empty")

            if not content_type:
                content_type = mimetypes.guess_type(str(path))[0] or (
                    "video/mp4" if media_type == "VIDEO" else "image/jpeg"
                )
            return path, content_type
        except Exception:
            path.unlink(missing_ok=True)
            raise

    async def _upload_image(self, account: str, path: Path, content_type: str) -> str:
        def run():
            with path.open("rb") as f:
                r = requests.post(
                    self.settings.upload_base,
                    auth=self._oauth(account),
                    files={"media": (path.name, f, content_type)},
                    data={"media_category": "tweet_image"},
                    timeout=180,
                )
            self._raise_x(r)
            payload = r.json()
            media_id = str(payload.get("media_id_string") or payload.get("media_id") or "").strip()
            if not media_id:
                raise RuntimeError(f"X image upload returned no media id: {payload}")
            return media_id

        return await asyncio.to_thread(run)

    async def _upload_video(self, account: str, path: Path, content_type: str) -> str:
        total_bytes = path.stat().st_size

        def init():
            r = requests.post(
                self.settings.upload_base,
                auth=self._oauth(account),
                data={
                    "command": "INIT",
                    "total_bytes": str(total_bytes),
                    "media_type": content_type or "video/mp4",
                    "media_category": "tweet_video",
                },
                timeout=60,
            )
            self._raise_x(r)
            payload = r.json()
            media_id = str(payload.get("media_id_string") or payload.get("media_id") or "").strip()
            if not media_id:
                raise RuntimeError(f"X video INIT returned no media id: {payload}")
            return media_id

        media_id = await asyncio.to_thread(init)

        segment = 0
        with path.open("rb") as f:
            while True:
                chunk = f.read(self.settings.video_chunk_bytes)
                if not chunk:
                    break

                def append(seg=segment, data=chunk):
                    r = requests.post(
                        self.settings.upload_base,
                        auth=self._oauth(account),
                        data={"command": "APPEND", "media_id": media_id, "segment_index": str(seg)},
                        files={"media": (f"segment-{seg}.bin", data, "application/octet-stream")},
                        timeout=180,
                    )
                    self._raise_x(r)

                await asyncio.to_thread(append)
                segment += 1

        def finalize():
            r = requests.post(
                self.settings.upload_base,
                auth=self._oauth(account),
                data={"command": "FINALIZE", "media_id": media_id},
                timeout=60,
            )
            self._raise_x(r)
            try:
                return r.json()
            except Exception:
                return {}

        final_payload = await asyncio.to_thread(finalize)
        processing = final_payload.get("processing_info") if isinstance(final_payload, dict) else None

        started = asyncio.get_running_loop().time()
        while processing:
            state = str(processing.get("state", "")).lower()
            if state == "succeeded":
                break
            if state == "failed":
                raise RuntimeError(f"X video processing failed: {processing}")
            if asyncio.get_running_loop().time() - started >= self.settings.video_poll_timeout_seconds:
                raise TimeoutError(f"X video processing timed out; last processing_info={processing}")

            wait_for = processing.get("check_after_secs")
            try:
                delay = max(self.settings.video_poll_interval_seconds, float(wait_for or 0))
            except (TypeError, ValueError):
                delay = self.settings.video_poll_interval_seconds
            await asyncio.sleep(delay)

            def status():
                r = requests.get(
                    self.settings.upload_base,
                    auth=self._oauth(account),
                    params={"command": "STATUS", "media_id": media_id},
                    timeout=60,
                )
                self._raise_x(r)
                return r.json()

            payload = await asyncio.to_thread(status)
            processing = payload.get("processing_info") if isinstance(payload, dict) else None

        return media_id

    async def _upload_media(self, account: str, media_type: str, media_url: str) -> str:
        path, content_type = await self._download_media(media_url, media_type)
        try:
            if media_type == "IMAGE":
                return await self._upload_image(account, path, content_type)
            return await self._upload_video(account, path, content_type)
        finally:
            path.unlink(missing_ok=True)

    async def _create_post(self, account: str, text: str, media_id: str) -> dict[str, Any]:
        def run():
            r = requests.post(
                f"{self.settings.api_base}/2/tweets",
                auth=self._oauth(account),
                json={"text": text, "media": {"media_ids": [media_id]}},
                timeout=120,
            )
            self._raise_x(r)
            return r.json()

        return await asyncio.to_thread(run)

    @staticmethod
    def _default_idempotency_key(account: str, media_type: str, media_url: str, text: str) -> str:
        digest = hashlib.sha256(
            f"{normalize_account(account)}\\n{media_type}\\n{media_url}\\n{text}".encode()
        ).hexdigest()
        return f"auto:{digest}"

    async def _job(self, account: str, key: str) -> dict[str, Any] | None:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "select * from public.x_publish_jobs where account_key=$1 and idempotency_key=$2",
                account,
                key,
            )
        return dict(row) if row else None

    async def publish_media(self, *, account: str, media_type: str, media_url: str, text: str,
                            idempotency_key: str | None) -> dict[str, Any]:
        account = normalize_account(account)
        media_type = media_type.upper()
        if media_type not in {"IMAGE", "VIDEO"}:
            raise ValueError("media_type must be IMAGE or VIDEO")

        await self.validate_account(account)

        key = (
            idempotency_key.strip()
            if idempotency_key and idempotency_key.strip()
            else self._default_idempotency_key(account, media_type, media_url, text)
        )

        async with self.db.acquire() as conn:
            inserted = await conn.fetchval(
                """
                insert into public.x_publish_jobs (
                    account_key,idempotency_key,media_type,media_url,text_body,status,
                    created_at,updated_at
                ) values ($1,$2,$3,$4,$5,'started',now(),now())
                on conflict (account_key,idempotency_key) do nothing
                returning id
                """,
                account, key, media_type, media_url, text,
            )

        job = await self._job(account, key)
        if not job:
            raise RuntimeError("X idempotency reservation failed unexpectedly")

        if not inserted:
            if job["status"] == "published":
                return {
                    "ok": True,
                    "reused": True,
                    "platform": "x",
                    "account": account,
                    "media_type": job["media_type"],
                    "idempotency_key": key,
                    "media_id": job["media_id"],
                    "post_id": job["post_id"],
                    "permalink": job["permalink"],
                }
            if job["status"] == "publish_ambiguous":
                raise XPublishConflict(
                    f"X job {key} has ambiguous publish state. Refusing to auto-retry because that "
                    "could create a duplicate. Reconcile the X account manually, then either mark "
                    "the DB job or submit a new idempotency key intentionally."
                )
            if job["status"] == "started" and not job.get("media_id"):
                updated = job["updated_at"]
                if updated.tzinfo is None:
                    updated = updated.replace(tzinfo=timezone.utc)
                age = (utcnow() - updated).total_seconds()
                if age < self.settings.stale_job_seconds:
                    raise XPublishConflict(f"X job {key} is already in progress ({age:.0f}s old); retry later")

        media_id = str(job.get("media_id") or "").strip()
        if not media_id:
            try:
                media_id = await self._upload_media(account, media_type, media_url)
                async with self.db.acquire() as conn:
                    await conn.execute(
                        """
                        update public.x_publish_jobs
                           set media_id=$3,status='media_uploaded',last_error=null,updated_at=now()
                         where account_key=$1 and idempotency_key=$2
                        """,
                        account, key, media_id,
                    )
            except Exception as exc:
                async with self.db.acquire() as conn:
                    await conn.execute(
                        """
                        update public.x_publish_jobs
                           set status='media_failed',last_error=$3,updated_at=now()
                         where account_key=$1 and idempotency_key=$2
                        """,
                        account, key, str(exc)[:4000],
                    )
                raise

        try:
            payload = await self._create_post(account, text, media_id)
        except XApiError as exc:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update public.x_publish_jobs
                       set status='publish_failed',last_error=$3,updated_at=now()
                     where account_key=$1 and idempotency_key=$2
                    """,
                    account, key, str(exc)[:4000],
                )
            raise
        except Exception as exc:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update public.x_publish_jobs
                       set status='publish_ambiguous',last_error=$3,updated_at=now()
                     where account_key=$1 and idempotency_key=$2
                    """,
                    account, key, str(exc)[:4000],
                )
            raise XPublishConflict(
                f"X create-post result is ambiguous for {key}; refusing automatic retry: {exc}"
            ) from exc

        data = payload.get("data", payload) if isinstance(payload, dict) else {}
        post_id = str(data.get("id", "")).strip() if isinstance(data, dict) else ""
        if not post_id:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update public.x_publish_jobs
                       set status='publish_ambiguous',last_error=$3,updated_at=now()
                     where account_key=$1 and idempotency_key=$2
                    """,
                    account, key, f"X returned success without post id: {payload}"[:4000],
                )
            raise XPublishConflict("X returned a successful create response without a post id")

        permalink = f"https://x.com/i/web/status/{post_id}"
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update public.x_publish_jobs
                   set status='published',post_id=$3,permalink=$4,last_error=null,
                       published_at=now(),updated_at=now()
                 where account_key=$1 and idempotency_key=$2
                """,
                account, key, post_id, permalink,
            )

        return {
            "ok": True,
            "reused": False,
            "platform": "x",
            "account": account,
            "media_type": media_type,
            "idempotency_key": key,
            "media_id": media_id,
            "post_id": post_id,
            "permalink": permalink,
            "x": payload,
        }
