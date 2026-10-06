from __future__ import annotations

import asyncio
import hashlib
import json
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx
import requests
from requests_oauthlib import OAuth1

from .config import Settings, credentials_for, normalize_account
from .db import Database


class TumblrApiError(RuntimeError):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Tumblr API HTTP {status_code}: {message}")


class TumblrPublishConflict(RuntimeError):
    pass


class TumblrService:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db

    def _oauth(self, account: str) -> OAuth1:
        c = credentials_for(account)
        return OAuth1(c.consumer_key, c.consumer_secret, c.token, c.token_secret)

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.settings.user_agent}

    @staticmethod
    def _payload(response: requests.Response) -> Any:
        try:
            return response.json()
        except Exception:
            return response.text[:4000]

    @classmethod
    def _raise_tumblr(cls, response: requests.Response) -> None:
        if response.status_code >= 400:
            raise TumblrApiError(response.status_code, str(cls._payload(response))[:4000])

    async def validate_account(self, account: str) -> dict[str, Any]:
        key = normalize_account(account)
        creds = credentials_for(key)
        def run():
            r = requests.get(
                f"{self.settings.api_base}/v2/user/info",
                auth=self._oauth(key), headers=self._headers(), timeout=30,
            )
            self._raise_tumblr(r)
            return r.json()
        payload = await asyncio.to_thread(run)
        response = payload.get("response", {}) if isinstance(payload, dict) else {}
        user = response.get("user", {}) if isinstance(response, dict) else {}
        blogs = user.get("blogs", []) if isinstance(user, dict) else []
        wanted = creds.blog_identifier.strip().lower()
        wanted_host = (urlparse(wanted).hostname or wanted).lower() if "://" in wanted else wanted
        matched = None
        for blog in blogs if isinstance(blogs, list) else []:
            if not isinstance(blog, dict):
                continue
            name = str(blog.get("name") or "").strip().lower()
            url = str(blog.get("url") or "").strip().lower()
            candidates = {name, str(blog.get("uuid") or "").strip().lower(), url}
            host = (urlparse(url).hostname or "").lower()
            if host: candidates.add(host)
            if name: candidates.add(f"{name}.tumblr.com")
            if wanted in candidates or wanted_host in candidates:
                matched = blog
                break
        if matched is None:
            raise RuntimeError(
                f"Tumblr credential/blog mismatch for @{key}: configured blog identifier "
                f"{creds.blog_identifier!r} was not found in the authenticated user's blogs"
            )
        return matched

    async def _download_media(self, url: str, media_type: str) -> tuple[Path, str]:
        suffix = ".mp4" if media_type == "VIDEO" else ".jpg"
        fd, raw = tempfile.mkstemp(prefix="sh01-tumblr-", suffix=suffix)
        os.close(fd)
        path = Path(raw)
        total = 0
        content_type = ""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=300), follow_redirects=True) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()
                    content_type = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
                    with path.open("wb") as f:
                        async for chunk in response.aiter_bytes(1024 * 1024):
                            total += len(chunk)
                            if total > self.settings.media_max_bytes:
                                raise RuntimeError(
                                    f"Tumblr media exceeds TUMBLR_MEDIA_MAX_BYTES={self.settings.media_max_bytes}"
                                )
                            f.write(chunk)
            if total <= 0:
                raise RuntimeError("Downloaded Tumblr media is empty")
            if not content_type:
                content_type = mimetypes.guess_type(str(path))[0] or (
                    "video/mp4" if media_type == "VIDEO" else "image/jpeg"
                )
            if media_type == "VIDEO" and content_type not in {"video/mp4", "video/quicktime"}:
                raise RuntimeError(
                    f"Tumblr native video requires MP4/MOV; downloaded content-type={content_type!r}"
                )
            return path, content_type
        except Exception:
            path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _default_key(account: str, media_type: str, media_url: str, text: str) -> str:
        digest = hashlib.sha256(
            f"{normalize_account(account)}\n{media_type}\n{media_url}\n{text}".encode()
        ).hexdigest()
        return f"auto:{digest}"

    async def _job(self, account: str, key: str) -> dict[str, Any] | None:
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "select * from public.tumblr_publish_jobs where account_key=$1 and idempotency_key=$2",
                account, key,
            )
        return dict(row) if row else None

    async def _create_post(self, *, account: str, media_type: str, path: Path,
                           content_type: str, text: str, tags: list[str], state: str) -> dict[str, Any]:
        creds = credentials_for(account)
        identifier = "media"
        if media_type == "IMAGE":
            media_block = {"type": "image", "media": [{"type": content_type, "identifier": identifier}]}
        else:
            media_block = {"type": "video", "media": {"type": content_type, "identifier": identifier}}
        content: list[dict[str, Any]] = [media_block]
        if text.strip():
            content.append({"type": "text", "text": text.strip()})
        body: dict[str, Any] = {"content": content, "state": state}
        if tags:
            body["tags"] = ",".join(tags)
        endpoint = f"{self.settings.api_base}/v2/blog/{quote(creds.blog_identifier, safe='')}/posts"
        def run():
            with path.open("rb") as f:
                files = {
                    "json": ("post.json", json.dumps(body), "application/json"),
                    identifier: (path.name, f, content_type),
                }
                r = requests.post(
                    endpoint, auth=self._oauth(account), headers=self._headers(), files=files, timeout=600,
                )
            self._raise_tumblr(r)
            return r.json()
        return await asyncio.to_thread(run)

    async def _best_effort_permalink(self, account: str, blog_identifier: str, post_id: str) -> str | None:
        def run():
            r = requests.get(
                f"{self.settings.api_base}/v2/blog/{quote(blog_identifier, safe='')}/posts/{quote(post_id, safe='')}",
                auth=self._oauth(account), headers=self._headers(), timeout=30,
            )
            if r.status_code >= 400:
                return None
            payload = r.json()
            response = payload.get("response", {}) if isinstance(payload, dict) else {}
            post = response.get("post", response) if isinstance(response, dict) else {}
            return (post.get("post_url") or post.get("url")) if isinstance(post, dict) else None
        try:
            return await asyncio.to_thread(run)
        except Exception:
            return None

    async def publish_media(self, *, account: str, media_type: str, media_url: str,
                            text: str, tags: list[str], state: str,
                            idempotency_key: str | None) -> dict[str, Any]:
        account = normalize_account(account)
        media_type = media_type.upper()
        await self.validate_account(account)
        creds = credentials_for(account)
        key = idempotency_key.strip() if idempotency_key and idempotency_key.strip() else self._default_key(account, media_type, media_url, text)

        async with self.db.acquire() as conn:
            inserted = await conn.fetchval(
                """
                insert into public.tumblr_publish_jobs (
                    account_key,idempotency_key,media_type,media_url,text_body,state,status,created_at,updated_at
                ) values ($1,$2,$3,$4,$5,$6,'started',now(),now())
                on conflict (account_key,idempotency_key) do nothing returning id
                """,
                account, key, media_type, media_url, text, state,
            )
        job = await self._job(account, key)
        if not job:
            raise RuntimeError("Tumblr idempotency reservation failed unexpectedly")
        if not inserted:
            if job["status"] == "published":
                return {
                    "ok": True, "reused": True, "platform": "tumblr", "account": account,
                    "media_type": job["media_type"], "idempotency_key": key,
                    "post_id": job["post_id"], "permalink": job["permalink"],
                }
            if job["status"] in {"started", "publish_ambiguous"}:
                raise TumblrPublishConflict(
                    f"Tumblr job {key} is in {job['status']!r} state; refusing automatic retry "
                    "because a previous create request may have succeeded."
                )

        try:
            path, content_type = await self._download_media(media_url, media_type)
        except Exception as exc:
            async with self.db.acquire() as conn:
                await conn.execute(
                    "update public.tumblr_publish_jobs set status='media_failed',last_error=$3,updated_at=now() where account_key=$1 and idempotency_key=$2",
                    account, key, str(exc)[:4000],
                )
            raise

        try:
            try:
                payload = await self._create_post(
                    account=account, media_type=media_type, path=path, content_type=content_type,
                    text=text, tags=tags, state=state,
                )
            except TumblrApiError as exc:
                async with self.db.acquire() as conn:
                    await conn.execute(
                        "update public.tumblr_publish_jobs set status='publish_failed',last_error=$3,updated_at=now() where account_key=$1 and idempotency_key=$2",
                        account, key, str(exc)[:4000],
                    )
                raise
            except Exception as exc:
                async with self.db.acquire() as conn:
                    await conn.execute(
                        "update public.tumblr_publish_jobs set status='publish_ambiguous',last_error=$3,updated_at=now() where account_key=$1 and idempotency_key=$2",
                        account, key, str(exc)[:4000],
                    )
                raise TumblrPublishConflict(
                    f"Tumblr create-post result is ambiguous for {key}; refusing automatic retry: {exc}"
                ) from exc
        finally:
            path.unlink(missing_ok=True)

        response = payload.get("response", {}) if isinstance(payload, dict) else {}
        post_id = str(response.get("id") or "").strip() if isinstance(response, dict) else ""
        if not post_id:
            async with self.db.acquire() as conn:
                await conn.execute(
                    "update public.tumblr_publish_jobs set status='publish_ambiguous',last_error=$3,updated_at=now() where account_key=$1 and idempotency_key=$2",
                    account, key, f"Tumblr returned success without post id: {payload}"[:4000],
                )
            raise TumblrPublishConflict("Tumblr returned a successful create response without a post id")

        permalink = await self._best_effort_permalink(account, creds.blog_identifier, post_id)
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update public.tumblr_publish_jobs
                   set status='published',post_id=$3,permalink=$4,last_error=null,published_at=now(),updated_at=now()
                 where account_key=$1 and idempotency_key=$2
                """,
                account, key, post_id, permalink,
            )
        return {
            "ok": True, "reused": False, "platform": "tumblr", "account": account,
            "media_type": media_type, "idempotency_key": key, "post_id": post_id,
            "permalink": permalink, "tumblr": payload,
        }
