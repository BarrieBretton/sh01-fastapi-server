from __future__ import annotations

import hmac

from fastapi import APIRouter, Depends, Header, HTTPException

from .config import ACCOUNT_PREFIXES, Settings
from .db import Database
from .models import XPublishRequest
from .service import XApiError, XPublishConflict, XService

router = APIRouter(prefix="/x", tags=["x"])

_settings: Settings | None = None
_db: Database | None = None
_service: XService | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def get_service() -> XService:
    global _db, _service
    if _service is None:
        settings = get_settings()
        _db = Database(settings)
        _service = XService(settings, _db)
    return _service


def require_internal_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
    try:
        expected = get_settings().internal_api_key
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=f"X feature is not configured: {exc}") from exc
    if not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Invalid X-API-Key")


@router.get("/health")
async def health(_: None = Depends(require_internal_key)):
    return {"ok": True, "accounts": sorted(ACCOUNT_PREFIXES)}


@router.post("/publish")
async def publish(payload: XPublishRequest, _: None = Depends(require_internal_key)):
    try:
        media_url = str(payload.image_url) if payload.media_type == "IMAGE" else str(payload.video_url)
        return await get_service().publish_media(
            account=payload.account,
            media_type=payload.media_type,
            media_url=media_url,
            text=payload.text,
            idempotency_key=payload.idempotency_key,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except XPublishConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except XApiError as exc:
        status = exc.status_code if 400 <= exc.status_code < 600 else 502
        raise HTTPException(status_code=status, detail=str(exc)) from exc
