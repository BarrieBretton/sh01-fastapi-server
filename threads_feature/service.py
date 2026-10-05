from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from cryptography.fernet import Fernet, InvalidToken

from .config import ACCOUNT_MAP, Settings
from .db import Database

TERMINAL_CONTAINER_FAILURES = {"ERROR", "EXPIRED", "FAILED"}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_account(account: str) -> str:
    return account.strip().lstrip("@").lower()


class ThreadsService:
    def __init__(self, settings: Settings, db: Database):
        self.settings = settings
        self.db = db
        self.fernet = Fernet(settings.encryption_key.encode())

    def _encrypt(self, token: str) -> str:
        return self.fernet.encrypt(token.encode()).decode()

    def _decrypt(self, encrypted: str) -> str:
        try:
            return self.fernet.decrypt(encrypted.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError(
                "Stored Threads token cannot be decrypted. Verify "
                "THREADS_TOKEN_ENCRYPTION_KEY is identical on every SH01 instance."
            ) from exc

    def user_id_for(self, account: str) -> str:
        key = normalize_account(account)
        user_id = ACCOUNT_MAP.get(key)
        if not user_id:
            raise ValueError(f"Unknown Threads account: @{key}")
        return user_id

    async def _token_row(self, account: str) -> dict[str, Any] | None:
        key = normalize_account(account)
        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                """
                select account_key, threads_user_id, access_token_encrypted,
                       expires_at, refreshed_at, verified_at, last_refresh_error,
                       created_at, updated_at
                  from threads_auth
                 where account_key = $1
                """,
                key,
            )
        return dict(row) if row else None

    async def list_token_status(self) -> list[dict[str, Any]]:
        async with self.db.acquire() as conn:
            rows = await conn.fetch(
                """
                select account_key, threads_user_id, expires_at, refreshed_at,
                       verified_at, last_refresh_error, created_at, updated_at
                  from threads_auth
                 order by account_key
                """
            )
        return [dict(r) for r in rows]

    async def validate_token(self, account: str, token: str) -> dict[str, Any]:
        expected_user_id = self.user_id_for(account)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.settings.api_base}/me",
                params={"fields": "id,username", "access_token": token},
            )
        response.raise_for_status()
        payload = response.json()
        actual_user_id = str(payload.get("id", "")).strip()
        if not actual_user_id:
            raise RuntimeError(f"Threads /me returned no user id: {payload}")
        if actual_user_id != expected_user_id:
            raise RuntimeError(
                f"Token/account mismatch for @{normalize_account(account)}: expected "
                f"Threads user id {expected_user_id}, got {actual_user_id}"
            )
        return payload

    async def bootstrap_token(self, account: str, access_token: str, expires_in: int | None = None) -> dict[str, Any]:
        key = normalize_account(account)
        user_id = self.user_id_for(key)
        identity = await self.validate_token(key, access_token)
        lifetime = expires_in or self.settings.assumed_long_lived_lifetime_seconds
        now = utcnow()
        expires_at = now + timedelta(seconds=lifetime)

        async with self.db.acquire() as conn:
            await conn.execute(
                """
                insert into threads_auth (
                    account_key, threads_user_id, access_token_encrypted,
                    expires_at, verified_at, refreshed_at,
                    last_refresh_error, created_at, updated_at
                )
                values ($1,$2,$3,$4,$5,$6,null,$5,$5)
                on conflict (account_key) do update set
                    threads_user_id = excluded.threads_user_id,
                    access_token_encrypted = excluded.access_token_encrypted,
                    expires_at = excluded.expires_at,
                    verified_at = excluded.verified_at,
                    refreshed_at = excluded.refreshed_at,
                    last_refresh_error = null,
                    updated_at = excluded.updated_at
                """,
                key, user_id, self._encrypt(access_token), expires_at, now, now,
            )

        return {
            "ok": True,
            "account": key,
            "threads_user_id": user_id,
            "username": identity.get("username"),
            "expires_at": expires_at.isoformat(),
        }

    async def refresh_token(self, account: str, force: bool = False) -> dict[str, Any]:
        key = normalize_account(account)
        row = await self._token_row(key)
        if not row:
            raise RuntimeError(f"No stored Threads token for @{key}")

        expires_at = row["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        remaining = expires_at - utcnow()
        threshold = timedelta(days=self.settings.refresh_threshold_days)

        if not force and remaining > threshold:
            return {
                "ok": True, "account": key, "refreshed": False,
                "reason": "outside_refresh_window",
                "expires_at": expires_at.isoformat(),
                "days_remaining": round(remaining.total_seconds() / 86400, 2),
            }

        old_token = self._decrypt(row["access_token_encrypted"])
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(
                    f"{self.settings.api_base}/refresh_access_token",
                    params={"grant_type": "th_refresh_token", "access_token": old_token},
                )
            response.raise_for_status()

            # Meta's current official collection documents an empty refresh response.
            # If a JSON token is returned, accept it; otherwise keep the same token.
            token = old_token
            lifetime = self.settings.assumed_long_lived_lifetime_seconds
            if response.content:
                try:
                    body = response.json()
                    token = str(body.get("access_token") or token)
                    lifetime = int(body.get("expires_in") or lifetime)
                except (ValueError, TypeError, json.JSONDecodeError):
                    pass

            identity = await self.validate_token(key, token)
            now = utcnow()
            new_expiry = now + timedelta(seconds=lifetime)
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update threads_auth
                       set access_token_encrypted = $2,
                           expires_at = $3,
                           refreshed_at = $4,
                           verified_at = $4,
                           last_refresh_error = null,
                           updated_at = $4
                     where account_key = $1
                    """,
                    key, self._encrypt(token), new_expiry, now,
                )
            return {
                "ok": True, "account": key, "refreshed": True,
                "threads_user_id": identity.get("id"),
                "username": identity.get("username"),
                "expires_at": new_expiry.isoformat(),
            }
        except Exception as exc:
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update threads_auth
                       set last_refresh_error = $2, updated_at = $3
                     where account_key = $1
                    """,
                    key, str(exc)[:4000], utcnow(),
                )
            raise

    async def refresh_all(self, force: bool = False) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for account in ACCOUNT_MAP:
            try:
                results.append(await self.refresh_token(account, force=force))
            except Exception as exc:
                results.append({"ok": False, "account": account, "error": str(exc)})
        return {"ok": all(x.get("ok") for x in results), "results": results}

    async def get_valid_token(self, account: str) -> str:
        key = normalize_account(account)
        row = await self._token_row(key)
        if not row:
            raise RuntimeError(f"No Threads token stored for @{key}. Bootstrap this account first.")
        expires_at = row["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at <= utcnow():
            raise RuntimeError(f"Threads token for @{key} is already expired; fresh OAuth is required.")
        if expires_at - utcnow() <= timedelta(days=self.settings.refresh_threshold_days):
            await self.refresh_token(key, force=True)
            row = await self._token_row(key)
            assert row is not None
        return self._decrypt(row["access_token_encrypted"])

    async def _request_container_status(self, account: str, container_id: str) -> dict[str, Any]:
        token = await self.get_valid_token(account)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.settings.api_base}/{container_id}",
                params={"fields": "id,status,error_message", "access_token": token},
            )
        response.raise_for_status()
        return response.json()

    async def wait_for_container(self, account: str, container_id: str) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        started = loop.time()
        last: dict[str, Any] = {}
        while True:
            last = await self._request_container_status(account, container_id)
            status = str(last.get("status", "")).upper()
            if status in {"FINISHED", "PUBLISHED"}:
                return last
            if status in TERMINAL_CONTAINER_FAILURES:
                raise RuntimeError(f"Threads container failed: {last}")
            if loop.time() - started >= self.settings.poll_timeout_seconds:
                raise TimeoutError(
                    f"Threads container {container_id} was not ready within "
                    f"{self.settings.poll_timeout_seconds}s; last state={last}"
                )
            await asyncio.sleep(self.settings.poll_interval_seconds)

    async def _create_container(self, *, account: str, media_type: str, media_url: str,
                                text: str, alt_text: str | None, reply_control: str | None) -> str:
        token = await self.get_valid_token(account)
        params: dict[str, Any] = {"media_type": media_type, "text": text, "access_token": token}
        if media_type == "IMAGE":
            params["image_url"] = media_url
        elif media_type == "VIDEO":
            params["video_url"] = media_url
        else:
            raise ValueError("media_type must be IMAGE or VIDEO")
        if alt_text:
            params["alt_text"] = alt_text
        if reply_control:
            params["reply_control"] = reply_control

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.settings.api_base}/me/threads", params=params)
        response.raise_for_status()
        payload = response.json()
        container_id = str(payload.get("id", "")).strip()
        if not container_id:
            raise RuntimeError(f"Threads container creation returned no id: {payload}")
        return container_id

    async def _publish_container(self, account: str, container_id: str) -> str:
        token = await self.get_valid_token(account)
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.settings.api_base}/me/threads_publish",
                params={"creation_id": container_id, "access_token": token},
            )
        response.raise_for_status()
        payload = response.json()
        post_id = str(payload.get("id", "")).strip()
        if not post_id:
            raise RuntimeError(f"Threads publish returned no post id: {payload}")
        return post_id

    async def _post_details(self, account: str, post_id: str) -> dict[str, Any]:
        token = await self.get_valid_token(account)
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{self.settings.api_base}/{post_id}",
                params={"fields": "id,permalink,text,timestamp,username", "access_token": token},
            )
        if response.is_error:
            return {"id": post_id, "permalink": None}
        return response.json()

    @staticmethod
    def _default_idempotency_key(account: str, media_type: str, media_url: str, text: str) -> str:
        digest = hashlib.sha256(
            f"{normalize_account(account)}\n{media_type}\n{media_url}\n{text}".encode()
        ).hexdigest()
        return f"auto:{digest}"

    async def publish_media(self, *, account: str, media_type: str, media_url: str,
                            text: str = "", alt_text: str | None = None,
                            reply_control: str | None = None,
                            idempotency_key: str | None = None) -> dict[str, Any]:
        account = normalize_account(account)
        self.user_id_for(account)
        media_type = media_type.upper()
        key = (idempotency_key.strip() if idempotency_key and idempotency_key.strip()
               else self._default_idempotency_key(account, media_type, media_url, text))

        async with self.db.acquire() as conn:
            row = await conn.fetchrow(
                "select * from threads_publish_jobs where account_key=$1 and idempotency_key=$2",
                account, key,
            )

        if row and row["status"] == "published":
            return {
                "ok": True, "reused": True, "account": account,
                "media_type": row["media_type"], "idempotency_key": key,
                "container_id": row["container_id"], "post_id": row["post_id"],
                "permalink": row["permalink"],
            }

        if row:
            container_id = row["container_id"]
            if not container_id:
                raise RuntimeError(f"Corrupt publish job for {key}: missing container_id")
        else:
            container_id = await self._create_container(
                account=account, media_type=media_type, media_url=media_url, text=text,
                alt_text=alt_text, reply_control=reply_control,
            )
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    insert into threads_publish_jobs (
                        account_key,idempotency_key,media_type,media_url,text_body,
                        container_id,status,created_at,updated_at
                    ) values ($1,$2,$3,$4,$5,$6,'container_created',now(),now())
                    on conflict (account_key,idempotency_key) do nothing
                    """,
                    account, key, media_type, media_url, text, container_id,
                )

        await self.wait_for_container(account, container_id)
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update threads_publish_jobs set status='container_ready', last_error=null,
                       updated_at=now() where account_key=$1 and idempotency_key=$2
                """,
                account, key,
            )

        try:
            post_id = await self._publish_container(account, container_id)
        except Exception as exc:
            try:
                after = await self._request_container_status(account, container_id)
            except Exception:
                after = {}
            status = str(after.get("status", "")).upper()
            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update threads_publish_jobs set status=$3,last_error=$4,updated_at=now()
                    where account_key=$1 and idempotency_key=$2
                    """,
                    account, key,
                    "publish_ambiguous" if status == "PUBLISHED" else "publish_failed",
                    str(exc)[:4000],
                )
            raise RuntimeError(
                f"Threads publish failed for existing container {container_id}; "
                f"container_status={status}. Retry with the same idempotency key; "
                f"no new container will be created. Original error: {exc}"
            ) from exc

        details = await self._post_details(account, post_id)
        permalink = details.get("permalink")
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update threads_publish_jobs set status='published',post_id=$3,permalink=$4,
                       last_error=null,published_at=now(),updated_at=now()
                where account_key=$1 and idempotency_key=$2
                """,
                account, key, post_id, permalink,
            )
        return {
            "ok": True, "reused": False, "account": account, "media_type": media_type,
            "idempotency_key": key, "container_id": container_id, "post_id": post_id,
            "permalink": permalink, "threads": details,
        }
