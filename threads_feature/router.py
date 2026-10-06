from __future__ import annotations

import asyncio
import hmac
import logging


from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .config import ACCOUNT_MAP, Settings
from .db import Database
from .models import BootstrapTokenRequest, ImagePublishRequest, UnifiedPublishRequest, VideoPublishRequest
from .service import ThreadsApiError, ThreadsService

router = APIRouter(prefix="/threads", tags=["threads"])

_settings: Settings | None = None
_db: Database | None = None
_service: ThreadsService | None = None
_scheduler_task: asyncio.Task | None = None
logger = logging.getLogger("threads.scheduler")


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def get_service() -> ThreadsService:
    global _db, _service
    if _service is None:
        settings = get_settings()
        _db = Database(settings)
        _service = ThreadsService(settings, _db)
    return _service


def require_internal_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    try:
        expected = get_settings().internal_api_key
    except RuntimeError as exc:
        # The master/control-plane intentionally may not carry Threads runtime
        # secrets. Keep app import/startup healthy and fail only if a Threads
        # endpoint is actually invoked on such an instance.
        raise HTTPException(status_code=503, detail=f"Threads feature is not configured: {exc}") from exc
    if not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")


async def _refresh_scheduler_loop() -> None:
    try:
        settings = get_settings()
    except RuntimeError:
        # Master/control-plane intentionally may not carry Threads secrets.
        return

    await asyncio.sleep(settings.refresh_initial_delay_seconds)
    while True:
        try:
            result = await get_service().refresh_all_scheduled()
            if not result.get("ok"):
                logger.error("Scheduled Threads refresh sweep had failures: %s", result)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Scheduled Threads token refresh sweep crashed")
        await asyncio.sleep(settings.refresh_check_interval_seconds)


@router.on_event("startup")
async def start_refresh_scheduler() -> None:
    global _scheduler_task
    try:
        settings = get_settings()
    except RuntimeError:
        return
    if not settings.refresh_scheduler_enabled:
        logger.info("Threads refresh scheduler disabled")
        return
    if _scheduler_task is None or _scheduler_task.done():
        _scheduler_task = asyncio.create_task(
            _refresh_scheduler_loop(), name="threads-token-refresh"
        )


@router.on_event("shutdown")
async def stop_refresh_scheduler() -> None:
    global _scheduler_task
    if _scheduler_task is not None:
        _scheduler_task.cancel()
        try:
            await _scheduler_task
        except asyncio.CancelledError:
            pass
        _scheduler_task = None


@router.get("/health")
async def health(_: None = Depends(require_internal_key)):
    return {"ok": True, "accounts": ACCOUNT_MAP}


@router.get("/tokens/status")
async def token_status(_: None = Depends(require_internal_key)):
    rows = await get_service().list_token_status()
    configured = {row["account_key"] for row in rows}
    return {
        "ok": True,
        "accounts": [
            {"account": account, "threads_user_id": user_id, "configured": account in configured}
            for account, user_id in ACCOUNT_MAP.items()
        ],
        "tokens": rows,
    }


@router.post("/tokens/bootstrap")
async def bootstrap_token(payload: BootstrapTokenRequest, _: None = Depends(require_internal_key)):
    try:
        return await get_service().bootstrap_token(payload.account, payload.access_token, payload.expires_in)
    except ThreadsApiError as exc:
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        raise HTTPException(
            status_code=status,
            detail={
                "platform": "threads",
                "account": exc.account,
                "operation": exc.operation,
                "upstream_status": exc.status_code,
                "meta": exc.payload,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tokens/refresh/{account}")
async def refresh_token(account: str, force: bool = Query(default=False), _: None = Depends(require_internal_key)):
    try:
        return await get_service().refresh_token(account, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tokens/refresh-all")
async def refresh_all(force: bool = Query(default=False), _: None = Depends(require_internal_key)):
    return await get_service().refresh_all(force=force)


@router.post("/tokens/refresh-scheduled")
async def refresh_scheduled(_: None = Depends(require_internal_key)):
    """Run one production-style refresh sweep under the distributed DB lock."""
    return await get_service().refresh_all_scheduled()


@router.post("/publish/image")
async def publish_image(payload: ImagePublishRequest, _: None = Depends(require_internal_key)):
    try:
        return await get_service().publish_media(
            account=payload.account, media_type="IMAGE", media_url=str(payload.image_url),
            text=payload.text, alt_text=payload.alt_text, reply_control=payload.reply_control,
            idempotency_key=payload.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/publish/video")
async def publish_video(payload: VideoPublishRequest, _: None = Depends(require_internal_key)):
    try:
        return await get_service().publish_media(
            account=payload.account, media_type="VIDEO", media_url=str(payload.video_url),
            text=payload.text, alt_text=payload.alt_text, reply_control=payload.reply_control,
            idempotency_key=payload.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/publish")
async def publish_unified(payload: UnifiedPublishRequest, _: None = Depends(require_internal_key)):
    try:
        media_url = str(payload.image_url) if payload.media_type == "IMAGE" else str(payload.video_url)
        return await get_service().publish_media(
            account=payload.account, media_type=payload.media_type, media_url=media_url,
            text=payload.text, alt_text=payload.alt_text, reply_control=payload.reply_control,
            idempotency_key=payload.idempotency_key,
        )
    except ThreadsApiError as exc:
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        raise HTTPException(
            status_code=status,
            detail={
                "platform": "threads",
                "account": exc.account,
                "operation": exc.operation,
                "upstream_status": exc.status_code,
                "meta": exc.payload,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc