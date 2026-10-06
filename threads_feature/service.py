from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from cryptography.fernet import Fernet, InvalidToken

from .config import ACCOUNT_MAP, Settings
from .db import Database

TERMINAL_CONTAINER_FAILURES = {"ERROR", "EXPIRED", "FAILED"}
REFRESH_LOCK_NAME = "threads_token_refresh_scheduler_v1"
logger = logging.getLogger("threads.refresh")


class ThreadsRefreshError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str, *, transient: bool = False):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.transient = transient
        super().__init__(f"Threads refresh HTTP {status_code} [{code}]: {message}")


class ThreadsApiError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        operation: str,
        account: str,
        payload: object,
    ):
        self.status_code = status_code
        self.operation = operation
        self.account = account
        self.payload = payload
        super().__init__(
            f"Threads API HTTP {status_code} during {operation} for @{account}: {payload}"
        )


def _meta_error(response: httpx.Response) -> ThreadsRefreshError:
    code = str(response.status_code)
    message = (response.text or "").strip()[:2000] or f"HTTP {response.status_code}"
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error", payload)
            if isinstance(err, dict):
                code = str(err.get("code") or err.get("error_subcode") or code)
                message = str(err.get("message") or err.get("error_user_msg") or message)[:2000]
    except Exception:
        pass
    transient = response.status_code == 429 or response.status_code >= 500
    return ThreadsRefreshError(response.status_code, code, message, transient=transient)


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
                       last_refresh_attempt_at, last_refresh_success_at, next_refresh_after,
                       consecutive_refresh_failures, refresh_error_code, refresh_error_status,
                       created_at, updated_at
                  from public.threads_auth
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
                       verified_at, last_refresh_error, last_refresh_attempt_at,
                       last_refresh_success_at, next_refresh_after,
                       consecutive_refresh_failures, refresh_error_code, refresh_error_status,
                       created_at, updated_at
                  from public.threads_auth
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
        next_refresh_after = expires_at - timedelta(days=self.settings.refresh_threshold_days)

        async with self.db.acquire() as conn:
            await conn.execute(
                """
                insert into public.threads_auth (
                    account_key, threads_user_id, access_token_encrypted,
                    expires_at, verified_at, refreshed_at,
                    last_refresh_error, last_refresh_attempt_at, last_refresh_success_at,
                    next_refresh_after, consecutive_refresh_failures,
                    refresh_error_code, refresh_error_status, created_at, updated_at
                )
                values ($1,$2,$3,$4,$5,$6,null,null,$6,$7,0,null,null,$5,$5)
                on conflict (account_key) do update set
                    threads_user_id = excluded.threads_user_id,
                    access_token_encrypted = excluded.access_token_encrypted,
                    expires_at = excluded.expires_at,
                    verified_at = excluded.verified_at,
                    refreshed_at = excluded.refreshed_at,
                    last_refresh_error = null,
                    last_refresh_attempt_at = null,
                    last_refresh_success_at = excluded.last_refresh_success_at,
                    next_refresh_after = excluded.next_refresh_after,
                    consecutive_refresh_failures = 0,
                    refresh_error_code = null,
                    refresh_error_status = null,
                    updated_at = excluded.updated_at
                """,
                key, user_id, self._encrypt(access_token), expires_at, now, now, next_refresh_after,
            )

        return {
            "ok": True,
            "account": key,
            "threads_user_id": user_id,
            "username": identity.get("username"),
            "expires_at": expires_at.isoformat(),
            "next_refresh_after": next_refresh_after.isoformat(),
        }

    async def _record_refresh_failure(
        self, account: str, exc: Exception, when: datetime, remaining: timedelta
    ) -> None:
        status = exc.status_code if isinstance(exc, ThreadsRefreshError) else None
        code = exc.code if isinstance(exc, ThreadsRefreshError) else exc.__class__.__name__
        message = str(exc)[:4000]
        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update public.threads_auth
                   set last_refresh_error = $2,
                       refresh_error_code = $3,
                       refresh_error_status = $4,
                       consecutive_refresh_failures = consecutive_refresh_failures + 1,
                       updated_at = $5
                 where account_key = $1
                """,
                account, message, code, status, when,
            )
        if remaining <= timedelta(days=self.settings.refresh_alert_expiry_days):
            await self._send_refresh_alert(account, message, remaining)

    async def _send_refresh_alert(self, account: str, error: str, remaining: timedelta) -> None:
        chat_id = self.settings.refresh_alert_chat_id
        bot_token = self.settings.telegram_bot_token
        if not chat_id or not bot_token:
            logger.error(
                "Threads token refresh is inside alert window for @%s but Telegram alerting is not configured",
                account,
            )
            return
        days = max(0.0, remaining.total_seconds() / 86400)
        text = (
            f"THREADS TOKEN REFRESH ALERT\n"
            f"Account: @{account}\n"
            f"Days remaining: {days:.2f}\n"
            f"Error: {error[:1500]}"
        )
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{bot_token}/sendMessage",
                    json={"chat_id": chat_id, "text": text},
                )
            if response.status_code >= 300:
                logger.error("Telegram refresh alert failed: HTTP %s %s", response.status_code, response.text[:500])
        except Exception:
            logger.exception("Telegram refresh alert crashed")

    async def refresh_token(self, account: str, force: bool = False) -> dict[str, Any]:
        key = normalize_account(account)
        row = await self._token_row(key)
        if not row:
            raise RuntimeError(f"No stored Threads token for @{key}")

        now = utcnow()
        expires_at = row["expires_at"]
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        remaining = expires_at - now
        threshold = timedelta(days=self.settings.refresh_threshold_days)

        if not force and remaining > threshold:
            return {
                "ok": True, "account": key, "refreshed": False,
                "reason": "outside_refresh_window",
                "expires_at": expires_at.isoformat(),
                "days_remaining": round(remaining.total_seconds() / 86400, 2),
            }

        old_token = self._decrypt(row["access_token_encrypted"])
        reference = row.get("last_refresh_success_at") or row.get("refreshed_at") or row.get("created_at")
        if reference and reference.tzinfo is None:
            reference = reference.replace(tzinfo=timezone.utc)
        token_age = now - reference if reference else timedelta(days=999)

        async with self.db.acquire() as conn:
            await conn.execute(
                """
                update public.threads_auth
                   set last_refresh_attempt_at = $2, updated_at = $2
                 where account_key = $1
                """,
                key, now,
            )

        last_exc: Exception | None = None
        response: httpx.Response | None = None
        for attempt in range(1, self.settings.refresh_max_attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    response = await client.get(
                        f"{self.settings.api_base}/refresh_access_token",
                        params={"grant_type": "th_refresh_token", "access_token": old_token},
                    )
                if response.status_code >= 400:
                    exc = _meta_error(response)
                    # Fresh long-lived tokens may not yet be refreshable. A forced
                    # smoke test must not poison health/error counters for that case.
                    if response.status_code == 400 and token_age < timedelta(hours=24):
                        retry_at = reference + timedelta(hours=24) if reference else now + timedelta(hours=24)
                        async with self.db.acquire() as conn:
                            await conn.execute(
                                """
                                update public.threads_auth
                                   set last_refresh_error = null,
                                       refresh_error_code = 'token_too_young_to_refresh',
                                       refresh_error_status = 400,
                                       consecutive_refresh_failures = 0,
                                       next_refresh_after = greatest(coalesce(next_refresh_after, $2), $2),
                                       updated_at = $3
                                 where account_key = $1
                                """,
                                key, retry_at, now,
                            )
                        return {
                            "ok": True, "account": key, "refreshed": False,
                            "reason": "token_too_young_to_refresh",
                            "retry_after": retry_at.isoformat(),
                            "expires_at": expires_at.isoformat(),
                        }
                    raise exc
                last_exc = None
                break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                transient = True
            except ThreadsRefreshError as exc:
                last_exc = exc
                transient = exc.transient

            if not transient or attempt >= self.settings.refresh_max_attempts:
                break
            delay = self.settings.refresh_retry_base_seconds * (2 ** (attempt - 1))
            await asyncio.sleep(delay)

        if last_exc is not None or response is None:
            exc = last_exc or RuntimeError("Threads refresh produced no response")
            await self._record_refresh_failure(key, exc, utcnow(), remaining)
            raise exc

        # Meta's official collection currently documents a successful refresh
        # response with no body. If a token/expires_in is returned, use it; if
        # not, retain the existing token and renew our known lifetime.
        token = old_token
        lifetime = self.settings.assumed_long_lived_lifetime_seconds
        if response.content:
            try:
                body = response.json()
                if isinstance(body, dict):
                    token = str(body.get("access_token") or token)
                    lifetime = int(body.get("expires_in") or lifetime)
            except (ValueError, TypeError, json.JSONDecodeError):
                pass

        # Treat post-refresh verification/persistence as part of the refresh
        # transaction from an observability perspective. A 2xx from Meta is not
        # considered a completed refresh until the returned/retained token still
        # resolves to the expected Threads user and the new metadata is persisted.
        try:
            identity = await self.validate_token(key, token)
            success_at = utcnow()
            new_expiry = success_at + timedelta(seconds=lifetime)
            next_refresh_after = new_expiry - threshold

            async with self.db.acquire() as conn:
                await conn.execute(
                    """
                    update public.threads_auth
                       set access_token_encrypted = $2,
                           expires_at = $3,
                           refreshed_at = $4,
                           verified_at = $4,
                           last_refresh_success_at = $4,
                           next_refresh_after = $5,
                           last_refresh_error = null,
                           refresh_error_code = null,
                           refresh_error_status = null,
                           consecutive_refresh_failures = 0,
                           updated_at = $4
                     where account_key = $1
                    """,
                    key, self._encrypt(token), new_expiry, success_at, next_refresh_after,
                )
        except Exception as exc:
            # Preserve the original exception even if recording telemetry itself
            # fails (for example, during a simultaneous database outage).
            try:
                await self._record_refresh_failure(key, exc, utcnow(), remaining)
            except Exception:
                logger.exception(
                    "Failed to persist Threads post-refresh failure telemetry for @%s",
                    key,
                )
            raise

        return {
            "ok": True, "account": key, "refreshed": True,
            "threads_user_id": identity.get("id"),
            "username": identity.get("username"),
            "expires_at": new_expiry.isoformat(),
            "next_refresh_after": next_refresh_after.isoformat(),
        }

    async def refresh_all(self, force: bool = False) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        for account in ACCOUNT_MAP:
            try:
                results.append(await self.refresh_token(account, force=force))
            except Exception as exc:
                results.append({"ok": False, "account": account, "error": str(exc)[:2000]})
        return {"ok": all(x.get("ok") for x in results), "results": results}

    async def refresh_all_scheduled(self) -> dict[str, Any]:
        # Session-level advisory lock: released automatically if a worker dies.
        async with self.db.acquire() as lock_conn:
            locked = await lock_conn.fetchval(
                "select pg_try_advisory_lock(hashtext($1)::bigint)", REFRESH_LOCK_NAME
            )
            if not locked:
                return {"ok": True, "skipped": True, "reason": "refresh_lock_held_elsewhere"}
            try:
                return await self.refresh_all(force=False)
            finally:
                await lock_conn.fetchval(
                    "select pg_advisory_unlock(hashtext($1)::bigint)", REFRESH_LOCK_NAME
                )

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

        old_token = self._decrypt(row["access_token_encrypted"])
        if expires_at - utcnow() <= timedelta(days=self.settings.refresh_threshold_days):
            try:
                result = await self.refresh_token(key, force=False)
                if result.get("refreshed"):
                    row = await self._token_row(key)
                    assert row is not None
                    return self._decrypt(row["access_token_encrypted"])
            except Exception:
                # Refresh outages should not stop publishing while the existing
                # token is still valid. The failure is persisted/alerted separately.
                logger.exception("Threads refresh failed for @%s; using still-valid token", key)
        return old_token

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

        if response.is_error:
            try:
                error_payload: object = response.json()
            except Exception:
                error_payload = (response.text or "")[:4000]
            raise ThreadsApiError(
                status_code=response.status_code,
                operation="create_container",
                account=normalize_account(account),
                payload=error_payload,
            )

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
                "select * from public.threads_publish_jobs where account_key=$1 and idempotency_key=$2",
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
                    insert into public.threads_publish_jobs (
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
                update public.threads_publish_jobs set status='container_ready', last_error=null,
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
                    update public.threads_publish_jobs set status=$3,last_error=$4,updated_at=now()
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
                update public.threads_publish_jobs set status='published',post_id=$3,permalink=$4,
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
