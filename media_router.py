from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from b2_helper import get_b2_manager

logger = logging.getLogger("media_renderer")

router = APIRouter(prefix="/media", tags=["Telegram Audio Social Media Renderer"])

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_STORAGE_API_KEY = os.getenv("TELEGRAM_STORAGE_API_KEY", "").strip()
PUBLIC_API_BASE = os.getenv(
    "PUBLIC_API_BASE",
    "https://sh01.vivojaymail.workers.dev",
).strip().rstrip("/")
BACKBLAZE_BUCKET_NAME = os.getenv("BACKBLAZE_BUCKET_NAME", "").strip()

MAX_AUDIO_BYTES = int(os.getenv("MEDIA_RENDER_MAX_AUDIO_BYTES", str(250 * 1024 * 1024)))
MAX_IMAGE_BYTES = int(os.getenv("MEDIA_RENDER_MAX_IMAGE_BYTES", str(30 * 1024 * 1024)))
B2_PREFIX = os.getenv(
    "MEDIA_RENDER_B2_PREFIX",
    "renders/telegram-audio-social",
).strip().strip("/")
B2_AUTH_SECONDS = max(
    60,
    min(int(os.getenv("MEDIA_RENDER_B2_AUTH_SECONDS", "3600")), 604800),
)


class AudioImageRenderRequest(BaseModel):
    audio_file_id: str = Field(min_length=1)
    image_url: str = Field(min_length=1)
    queue_id: str = Field(default="", max_length=512)


def _require_media_key(x_api_key: str | None) -> None:
    """Reuse the existing Telegram Storage API key for media mutations."""
    if not TELEGRAM_STORAGE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_STORAGE_API_KEY is not configured",
        )
    if not x_api_key or not secrets.compare_digest(x_api_key, TELEGRAM_STORAGE_API_KEY):
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")


def _render_file_name(token: str) -> str:
    return f"{B2_PREFIX}/{token}.mp4"


def _validate_token(token: str) -> str:
    token = str(token or "").strip().lower()
    if len(token) != 32 or any(c not in "0123456789abcdef" for c in token):
        raise HTTPException(status_code=400, detail="Invalid render token")
    return token


def _is_public_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail=f"Image host cannot be resolved: {host}") from exc

    if not infos:
        return False

    for info in infos:
        raw = info[4][0]
        ip = ipaddress.ip_address(raw)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False
    return True


def _validate_public_http_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="image_url must use http or https")
    if not parsed.hostname or not _is_public_ip(parsed.hostname):
        raise HTTPException(status_code=400, detail="image_url must resolve to a public host")


async def _download_public_image(url: str, destination: Path) -> None:
    """Download an image while validating every redirect destination."""
    current = url
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        for _ in range(6):
            _validate_public_http_url(current)
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise HTTPException(status_code=424, detail="Image redirect had no Location header")
                    current = urljoin(current, location)
                    continue

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=424,
                        detail=f"Image download failed with HTTP {response.status_code}",
                    )

                content_type = (response.headers.get("content-type") or "").lower()
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"image_url did not return an image (content-type={content_type or 'unknown'})",
                    )

                total = 0
                with destination.open("wb") as handle:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_IMAGE_BYTES:
                            raise HTTPException(status_code=413, detail="Image is too large")
                        handle.write(chunk)
                return

    raise HTTPException(status_code=400, detail="Too many image redirects")


async def _telegram_file_path(file_id: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN is not configured")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile"
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(url, params={"file_id": file_id})

    if response.status_code >= 400:
        raise HTTPException(status_code=424, detail=f"Telegram getFile failed: HTTP {response.status_code}")

    payload = response.json()
    if not payload.get("ok") or not payload.get("result", {}).get("file_path"):
        raise HTTPException(status_code=424, detail=f"Telegram getFile failed: {payload}")
    return str(payload["result"]["file_path"])


async def _download_telegram_audio(file_id: str, destination: Path) -> None:
    file_path = await _telegram_file_path(file_id)
    url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"

    async with httpx.AsyncClient(timeout=300.0) as client:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=424,
                    detail=f"Telegram file download failed: HTTP {response.status_code}",
                )
            total = 0
            with destination.open("wb") as handle:
                async for chunk in response.aiter_bytes(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_AUDIO_BYTES:
                        raise HTTPException(status_code=413, detail="Audio file is too large")
                    handle.write(chunk)


def _run_ffmpeg(image_path: Path, audio_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=500, detail="ffmpeg is not installed on SH01")

    prepared_image = output_path.parent / f"{output_path.stem}_prepared.png"

    # Stage 1:
    # Resize/pad the source image ONCE.
    # PNG avoids introducing a lossy intermediate image.
    prepare_cmd = [
        ffmpeg,
        "-y",

        "-i", str(image_path),

        "-vf",
        (
            "scale=1080:1920:force_original_aspect_ratio=decrease,"
            "pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black"
        ),

        "-frames:v", "1",

        str(prepared_image),
    ]

    prepare_result = subprocess.run(
        prepare_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )

    if (
        prepare_result.returncode != 0
        or not prepared_image.exists()
        or prepared_image.stat().st_size == 0
    ):
        logger.error(
            "ffmpeg image preparation failed: %s",
            prepare_result.stderr.decode("utf-8", errors="replace")[-8000:],
        )
        raise HTTPException(
            status_code=500,
            detail="ffmpeg failed to prepare the source image",
        )

    # Stage 2:
    # The image is already 1080x1920, so scale/pad is no longer
    # performed for every encoded frame.
    cmd = [
        ffmpeg,
        "-y",

        # Prepared static image
        "-loop", "1",
        "-framerate", "30",
        "-i", str(prepared_image),

        # Audio
        "-i", str(audio_path),

        # Hard-limit encoder parallelism for the 512 MB container
        "-threads", "1",

        # Video
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-tune", "stillimage",
        "-crf", "20",

        # Reduce x264 frame buffering/lookahead memory.
        "-x264-params",
        "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0",

        "-r", "30",
        "-pix_fmt", "yuv420p",

        # Audio
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",

        "-movflags", "+faststart",
        "-shortest",

        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=1800,
            check=False,
        )

        if (
            result.returncode != 0
            or not output_path.exists()
            or output_path.stat().st_size == 0
        ):
            logger.error(
                "ffmpeg failed: %s",
                result.stderr.decode("utf-8", errors="replace")[-8000:],
            )
            raise HTTPException(
                status_code=500,
                detail="ffmpeg failed to render the MP4",
            )

    finally:
        try:
            prepared_image.unlink(missing_ok=True)
        except Exception:
            pass

@router.post("/audio-image-video")
async def build_audio_image_video(
    request: AudioImageRenderRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
):
    _require_media_key(x_api_key)
    logger.info(
        "Media render START queue_id=%s image_url=%s",
        request.queue_id,
        request.image_url,
    )

    if not BACKBLAZE_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="BACKBLAZE_BUCKET_NAME is not configured")

    token = secrets.token_hex(16)
    b2_file_name = _render_file_name(token)

    try:
        with tempfile.TemporaryDirectory(prefix="sh01-media-render-") as tmp:
            root = Path(tmp)
            image_path = root / "image.bin"
            audio_path = root / "audio.bin"
            output_path = root / f"{token}.mp4"

            logger.info("Media render DOWNLOAD START queue_id=%s", request.queue_id)
            await asyncio.gather(
                _download_public_image(request.image_url, image_path),
                _download_telegram_audio(request.audio_file_id, audio_path),
            )

            logger.info(
                "Media render DOWNLOAD COMPLETE queue_id=%s image_bytes=%s audio_bytes=%s",
                request.queue_id,
                image_path.stat().st_size,
                audio_path.stat().st_size,
            )

            logger.info("Media render FFMPEG START queue_id=%s", request.queue_id)

            await asyncio.to_thread(_run_ffmpeg, image_path, audio_path, output_path)

            logger.info(
                "Media render FFMPEG COMPLETE queue_id=%s output_bytes=%s",
                request.queue_id,
                output_path.stat().st_size,
            )

            logger.info(
                "Media render B2 UPLOAD START queue_id=%s b2_file=%s",
                request.queue_id,
                b2_file_name,
            )

            manager = get_b2_manager()
            await asyncio.to_thread(
                manager.upload_file,
                output_path,
                BACKBLAZE_BUCKET_NAME,
                b2_file_name,
                "video/mp4",
                1,
            )

            size_bytes = output_path.stat().st_size

            logger.info(
                "Media render B2 UPLOAD COMPLETE queue_id=%s b2_file=%s",
                request.queue_id,
                b2_file_name,
            )

        logger.info(
            "Rendered Telegram audio social video queue_id=%s token=%s b2=%s size=%s",
            request.queue_id,
            token,
            b2_file_name,
            size_bytes,
        )

        return {
            "ok": True,
            "token": token,
            "video_url": f"{PUBLIC_API_BASE}/media/rendered/{token}",
            "b2_file_name": b2_file_name,
            "size_bytes": size_bytes,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Media render failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/rendered/{token}")
async def get_rendered_video(token: str):
    """Public fetch endpoint used by Meta and n8n/YouTube. Redirects to a fresh B2 signed URL."""
    token = _validate_token(token)
    b2_file_name = _render_file_name(token)

    try:
        manager = get_b2_manager()
        auth_token = await asyncio.to_thread(
            manager.get_download_authorization_token,
            b2_file_name,
            BACKBLAZE_BUCKET_NAME,
            B2_AUTH_SECONDS,
        )
        signed_url = manager.build_authorized_url(
            bucket_name=BACKBLAZE_BUCKET_NAME,
            file_name=b2_file_name,
            token=auth_token,
        )
        return RedirectResponse(url=signed_url, status_code=307)
    except Exception as exc:
        logger.warning("Unable to mint rendered-media URL token=%s err=%s", token, exc)
        raise HTTPException(status_code=404, detail="Rendered media not found or unavailable") from exc


@router.delete("/rendered/{token}")
async def delete_rendered_video(
    token: str,
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
):
    _require_media_key(x_api_key)
    token = _validate_token(token)
    b2_file_name = _render_file_name(token)

    try:
        manager = get_b2_manager()
        deleted = await asyncio.to_thread(
            manager.delete_file,
            b2_file_name,
            BACKBLAZE_BUCKET_NAME,
        )
        return {"ok": True, "token": token, "deleted": bool(deleted)}
    except Exception as exc:
        logger.exception("Rendered-media delete failed token=%s", token)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

