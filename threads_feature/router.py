from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from .config import ACCOUNT_MAP, Settings
from .db import Database
from .models import BootstrapTokenRequest, ImagePublishRequest, UnifiedPublishRequest, VideoPublishRequest
from .service import ThreadsService

settings = Settings.from_env()
db = Database(settings)
service = ThreadsService(settings, db)
router = APIRouter(prefix="/threads", tags=["threads"])


def require_internal_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    if not hmac.compare_digest(x_api_key or "", settings.internal_api_key):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")


@router.get("/health")
async def health(_: None = Depends(require_internal_key)):
    return {"ok": True, "accounts": ACCOUNT_MAP}


@router.get("/tokens/status")
async def token_status(_: None = Depends(require_internal_key)):
    rows = await service.list_token_status()
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
        return await service.bootstrap_token(payload.account, payload.access_token, payload.expires_in)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tokens/refresh/{account}")
async def refresh_token(account: str, force: bool = Query(default=False), _: None = Depends(require_internal_key)):
    try:
        return await service.refresh_token(account, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tokens/refresh-all")
async def refresh_all(force: bool = Query(default=False), _: None = Depends(require_internal_key)):
    return await service.refresh_all(force=force)


@router.post("/publish/image")
async def publish_image(payload: ImagePublishRequest, _: None = Depends(require_internal_key)):
    try:
        return await service.publish_media(
            account=payload.account, media_type="IMAGE", media_url=str(payload.image_url),
            text=payload.text, alt_text=payload.alt_text, reply_control=payload.reply_control,
            idempotency_key=payload.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/publish/video")
async def publish_video(payload: VideoPublishRequest, _: None = Depends(require_internal_key)):
    try:
        return await service.publish_media(
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
        return await service.publish_media(
            account=payload.account, media_type=payload.media_type, media_url=media_url,
            text=payload.text, alt_text=payload.alt_text, reply_control=payload.reply_control,
            idempotency_key=payload.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
