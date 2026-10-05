from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .config import ACCOUNT_MAP, Settings
from .db import Database
from .models import BootstrapTokenRequest, ImagePublishRequest, UnifiedPublishRequest, VideoPublishRequest
from .service import ThreadsService

router = APIRouter(prefix="/threads", tags=["threads"])

_settings: Settings | None = None
_db: Database | None = None
_service: ThreadsService | None = None


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
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
