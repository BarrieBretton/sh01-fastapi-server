from __future__ import annotations

import asyncio
import hashlib
import hmac
import ipaddress
import logging
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from b2_helper import get_b2_manager


logger = logging.getLogger("media_renderer")

router = APIRouter(
    prefix="/media",
    tags=["Telegram Audio Social Media Renderer"],
)


TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()

TELEGRAM_STORAGE_API_KEY = os.getenv(
    "TELEGRAM_STORAGE_API_KEY",
    "",
).strip()

PUBLIC_API_BASE = os.getenv(
    "PUBLIC_API_BASE",
    "https://sh01.vivojaymail.workers.dev",
).strip().rstrip("/")

BACKBLAZE_BUCKET_NAME = os.getenv(
    "BACKBLAZE_BUCKET_NAME",
    "",
).strip()

MAX_AUDIO_BYTES = int(
    os.getenv(
        "MEDIA_RENDER_MAX_AUDIO_BYTES",
        str(250 * 1024 * 1024),
    )
)

MAX_IMAGE_BYTES = int(
    os.getenv(
        "MEDIA_RENDER_MAX_IMAGE_BYTES",
        str(30 * 1024 * 1024),
    )
)

B2_PREFIX = os.getenv(
    "MEDIA_RENDER_B2_PREFIX",
    "renders/telegram-audio-social",
).strip().strip("/")

B2_AUTH_SECONDS = max(
    60,
    min(
        int(
            os.getenv(
                "MEDIA_RENDER_B2_AUTH_SECONDS",
                "3600",
            )
        ),
        604800,
    ),
)

RENDER_JOB_RETENTION_SECONDS = max(
    3600,
    int(
        os.getenv(
            "MEDIA_RENDER_JOB_RETENTION_SECONDS",
            "86400",
        )
    ),
)


# ---------------------------------------------------------------------
# Render queue state
#
# SH01 runs ONE uvicorn worker on this 512 MB instance.
# This therefore deliberately acts as a single-slot renderer.
#
# Google Sheets remains the durable queue.
# This in-memory state exists only to protect the local process and
# expose progress/status while one render is running.
# ---------------------------------------------------------------------

_render_state_lock = asyncio.Lock()
_render_execution_lock = asyncio.Lock()

_render_jobs: dict[str, dict] = {}
_queue_to_job: dict[str, str] = {}

_active_job_id: str | None = None

_background_tasks: set[asyncio.Task] = set()


class AudioImageRenderRequest(BaseModel):
    audio_file_id: str = Field(min_length=1)
    image_url: str = Field(min_length=1)
    queue_id: str = Field(min_length=1, max_length=512)


def _require_media_key(
    x_api_key: str | None,
) -> None:
    if not TELEGRAM_STORAGE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_STORAGE_API_KEY is not configured",
        )

    if (
        not x_api_key
        or not secrets.compare_digest(
            x_api_key,
            TELEGRAM_STORAGE_API_KEY,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Missing/invalid X-API-KEY",
        )


def _render_file_name(
    token: str,
) -> str:
    return f"{B2_PREFIX}/{token}.mp4"


def _validate_token(
    token: str,
) -> str:
    token = str(token or "").strip().lower()

    if (
        len(token) != 32
        or any(
            c not in "0123456789abcdef"
            for c in token
        )
    ):
        raise HTTPException(
            status_code=400,
            detail="Invalid render token",
        )

    return token


def _render_token_for_queue(
    queue_id: str,
) -> str:
    """
    Stable B2 object token for a queue item.

    This makes retries after an SH01 restart use the same B2 object
    name instead of producing arbitrary orphan object names.
    """
    queue_id = str(queue_id or "").strip()

    if (
        queue_id
        and TELEGRAM_STORAGE_API_KEY
    ):
        return hmac.new(
            TELEGRAM_STORAGE_API_KEY.encode("utf-8"),
            queue_id.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()[:32]

    return secrets.token_hex(16)


def _job_payload(
    job: dict,
) -> dict:
    return {
        key: value
        for key, value in job.items()
        if not key.startswith("_")
    }


def _prune_jobs_locked() -> None:
    """
    Must only be called while _render_state_lock is held.
    """
    now = time.time()

    removable: list[str] = []

    for job_id, job in _render_jobs.items():
        if job.get("status") not in {
            "complete",
            "failed",
        }:
            continue

        finished_epoch = job.get(
            "_finished_epoch"
        )

        if not finished_epoch:
            continue

        if (
            now - float(finished_epoch)
            >= RENDER_JOB_RETENTION_SECONDS
        ):
            removable.append(job_id)

    for job_id in removable:
        job = _render_jobs.pop(
            job_id,
            None,
        )

        if not job:
            continue

        queue_id = str(
            job.get("queue_id") or ""
        ).strip()

        if (
            queue_id
            and _queue_to_job.get(queue_id)
            == job_id
        ):
            _queue_to_job.pop(
                queue_id,
                None,
            )


def _is_public_ip(
    host: str,
) -> bool:
    try:
        infos = socket.getaddrinfo(
            host,
            None,
            proto=socket.IPPROTO_TCP,
        )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "Image host cannot be resolved: "
                f"{host}"
            ),
        ) from exc

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


def _validate_public_http_url(
    url: str,
) -> None:
    parsed = urlparse(url)

    if parsed.scheme not in {
        "http",
        "https",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "image_url must use http or https"
            ),
        )

    if (
        not parsed.hostname
        or not _is_public_ip(
            parsed.hostname
        )
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "image_url must resolve to "
                "a public host"
            ),
        )


async def _download_public_image(
    url: str,
    destination: Path,
) -> None:
    current = url

    async with httpx.AsyncClient(
        timeout=60.0,
        follow_redirects=False,
    ) as client:
        for _ in range(6):
            _validate_public_http_url(
                current
            )

            async with client.stream(
                "GET",
                current,
            ) as response:
                if response.status_code in {
                    301,
                    302,
                    303,
                    307,
                    308,
                }:
                    location = (
                        response.headers.get(
                            "location"
                        )
                    )

                    if not location:
                        raise HTTPException(
                            status_code=424,
                            detail=(
                                "Image redirect had "
                                "no Location header"
                            ),
                        )

                    current = urljoin(
                        current,
                        location,
                    )

                    continue

                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=424,
                        detail=(
                            "Image download failed "
                            "with HTTP "
                            f"{response.status_code}"
                        ),
                    )

                content_type = (
                    response.headers.get(
                        "content-type"
                    )
                    or ""
                ).lower()

                if not content_type.startswith(
                    "image/"
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "image_url did not "
                            "return an image "
                            f"(content-type="
                            f"{content_type or 'unknown'})"
                        ),
                    )

                total = 0

                with destination.open(
                    "wb"
                ) as handle:
                    async for chunk in (
                        response.aiter_bytes(
                            1024 * 1024
                        )
                    ):
                        total += len(chunk)

                        if (
                            total
                            > MAX_IMAGE_BYTES
                        ):
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    "Image is too large"
                                ),
                            )

                        handle.write(chunk)

                return

    raise HTTPException(
        status_code=400,
        detail="Too many image redirects",
    )


async def _telegram_file_path(
    file_id: str,
) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail=(
                "TELEGRAM_BOT_TOKEN "
                "is not configured"
            ),
        )

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/getFile"
    )

    async with httpx.AsyncClient(
        timeout=30.0
    ) as client:
        response = await client.get(
            url,
            params={
                "file_id": file_id,
            },
        )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=424,
            detail=(
                "Telegram getFile failed: HTTP "
                f"{response.status_code}"
            ),
        )

    payload = response.json()

    if (
        not payload.get("ok")
        or not payload.get(
            "result",
            {},
        ).get("file_path")
    ):
        raise HTTPException(
            status_code=424,
            detail=(
                "Telegram getFile failed: "
                f"{payload}"
            ),
        )

    return str(
        payload["result"]["file_path"]
    )


async def _download_telegram_audio(
    file_id: str,
    destination: Path,
) -> None:
    file_path = await _telegram_file_path(
        file_id
    )

    url = (
        "https://api.telegram.org/file/bot"
        f"{TELEGRAM_BOT_TOKEN}/"
        f"{file_path}"
    )

    async with httpx.AsyncClient(
        timeout=300.0
    ) as client:
        async with client.stream(
            "GET",
            url,
        ) as response:
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=424,
                    detail=(
                        "Telegram file download "
                        "failed: HTTP "
                        f"{response.status_code}"
                    ),
                )

            total = 0

            with destination.open(
                "wb"
            ) as handle:
                async for chunk in (
                    response.aiter_bytes(
                        1024 * 1024
                    )
                ):
                    total += len(chunk)

                    if (
                        total
                        > MAX_AUDIO_BYTES
                    ):
                        raise HTTPException(
                            status_code=413,
                            detail=(
                                "Audio file is too large"
                            ),
                        )

                    handle.write(chunk)


def _run_ffmpeg(
    image_path: Path,
    audio_path: Path,
    output_path: Path,
) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if not ffmpeg:
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg is not installed on SH01"
            ),
        )

    prepared_image = (
        output_path.parent
        / f"{output_path.stem}_prepared.png"
    )

    prepare_cmd = [
        ffmpeg,
        "-y",

        "-i",
        str(image_path),

        "-vf",
        (
            "scale=1080:1920:"
            "force_original_aspect_ratio=decrease,"
            "pad=1080:1920:"
            "(ow-iw)/2:(oh-ih)/2:black"
        ),

        "-frames:v",
        "1",

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
        or prepared_image.stat().st_size
        == 0
    ):
        logger.error(
            "ffmpeg image preparation "
            "failed: %s",
            prepare_result.stderr.decode(
                "utf-8",
                errors="replace",
            )[-8000:],
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg failed to prepare "
                "the source image"
            ),
        )

    cmd = [
        ffmpeg,
        "-y",

        "-loop",
        "1",

        "-framerate",
        "30",

        "-i",
        str(prepared_image),

        "-i",
        str(audio_path),

        "-threads",
        "1",

        "-c:v",
        "libx264",

        "-preset",
        "ultrafast",

        "-tune",
        "stillimage",

        "-crf",
        "20",

        "-x264-params",
        (
            "ref=1:"
            "bframes=0:"
            "rc-lookahead=0:"
            "sync-lookahead=0:"
            "mbtree=0"
        ),

        "-r",
        "30",

        "-pix_fmt",
        "yuv420p",

        "-c:a",
        "aac",

        "-b:a",
        "192k",

        "-ar",
        "48000",

        "-movflags",
        "+faststart",

        "-shortest",

        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=7200,
            check=False,
        )

        if (
            result.returncode != 0
            or not output_path.exists()
            or output_path.stat().st_size
            == 0
        ):
            logger.error(
                "ffmpeg failed: %s",
                result.stderr.decode(
                    "utf-8",
                    errors="replace",
                )[-8000:],
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "ffmpeg failed to "
                    "render the MP4"
                ),
            )

    finally:
        try:
            prepared_image.unlink(
                missing_ok=True
            )
        except Exception:
            pass


async def _run_render_job(
    job_id: str,
    request: AudioImageRenderRequest,
) -> None:
    global _active_job_id

    job = _render_jobs[job_id]

    async with _render_execution_lock:
        job["status"] = "running"
        job["started_at"] = (
            time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            )
        )

        logger.info(
            "Media render JOB START "
            "job_id=%s queue_id=%s",
            job_id,
            request.queue_id,
        )

        token = str(
            job["token"]
        )

        b2_file_name = (
            _render_file_name(token)
        )

        try:
            with tempfile.TemporaryDirectory(
                prefix="sh01-media-render-"
            ) as tmp:
                root = Path(tmp)

                image_path = (
                    root / "image.bin"
                )

                audio_path = (
                    root / "audio.bin"
                )

                output_path = (
                    root / f"{token}.mp4"
                )

                logger.info(
                    "Media render DOWNLOAD START "
                    "job_id=%s queue_id=%s",
                    job_id,
                    request.queue_id,
                )

                await asyncio.gather(
                    _download_public_image(
                        request.image_url,
                        image_path,
                    ),
                    _download_telegram_audio(
                        request.audio_file_id,
                        audio_path,
                    ),
                )

                logger.info(
                    "Media render DOWNLOAD COMPLETE "
                    "job_id=%s queue_id=%s "
                    "image_bytes=%s audio_bytes=%s",
                    job_id,
                    request.queue_id,
                    image_path.stat().st_size,
                    audio_path.stat().st_size,
                )

                logger.info(
                    "Media render FFMPEG START "
                    "job_id=%s queue_id=%s",
                    job_id,
                    request.queue_id,
                )

                await asyncio.to_thread(
                    _run_ffmpeg,
                    image_path,
                    audio_path,
                    output_path,
                )

                logger.info(
                    "Media render FFMPEG COMPLETE "
                    "job_id=%s queue_id=%s "
                    "output_bytes=%s",
                    job_id,
                    request.queue_id,
                    output_path.stat().st_size,
                )

                logger.info(
                    "Media render B2 UPLOAD START "
                    "job_id=%s queue_id=%s "
                    "b2_file=%s",
                    job_id,
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

                size_bytes = (
                    output_path.stat().st_size
                )

                logger.info(
                    "Media render B2 UPLOAD COMPLETE "
                    "job_id=%s queue_id=%s "
                    "b2_file=%s",
                    job_id,
                    request.queue_id,
                    b2_file_name,
                )

            job.update(
                {
                    "status": "complete",
                    "token": token,
                    "video_url": (
                        f"{PUBLIC_API_BASE}"
                        f"/media/rendered/{token}"
                    ),
                    "b2_file_name": (
                        b2_file_name
                    ),
                    "size_bytes": (
                        size_bytes
                    ),
                    "error": None,
                }
            )

            logger.info(
                "Media render JOB COMPLETE "
                "job_id=%s queue_id=%s "
                "token=%s size=%s",
                job_id,
                request.queue_id,
                token,
                size_bytes,
            )

        except HTTPException as exc:
            job.update(
                {
                    "status": "failed",
                    "error": str(
                        exc.detail
                    ),
                }
            )

            logger.exception(
                "Media render JOB FAILED "
                "job_id=%s queue_id=%s",
                job_id,
                request.queue_id,
            )

        except Exception as exc:
            job.update(
                {
                    "status": "failed",
                    "error": str(exc),
                }
            )

            logger.exception(
                "Media render JOB FAILED "
                "job_id=%s queue_id=%s",
                job_id,
                request.queue_id,
            )

        finally:
            job["finished_at"] = (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                )
            )

            job["_finished_epoch"] = (
                time.time()
            )

            async with _render_state_lock:
                if (
                    _active_job_id
                    == job_id
                ):
                    _active_job_id = None


@router.post(
    "/audio-image-video",
)
async def build_audio_image_video(
    request: AudioImageRenderRequest,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
):
    """
    Submit ONE asynchronous render job.

    Returns immediately.

    Exactly one different queue item may be active at once.

    Re-submitting the SAME queue_id is idempotent while its
    retained job exists: the existing job is returned.
    """
    global _active_job_id

    _require_media_key(
        x_api_key
    )

    if not BACKBLAZE_BUCKET_NAME:
        raise HTTPException(
            status_code=500,
            detail=(
                "BACKBLAZE_BUCKET_NAME "
                "is not configured"
            ),
        )

    queue_id = (
        request.queue_id.strip()
    )

    async with _render_state_lock:
        _prune_jobs_locked()

        existing_job_id = (
            _queue_to_job.get(
                queue_id
            )
        )

        if existing_job_id:
            existing = (
                _render_jobs.get(
                    existing_job_id
                )
            )

            if (
                existing
                and existing.get("status")
                != "failed"
            ):
                payload = (
                    _job_payload(existing)
                )

                payload["ok"] = True
                payload["existing"] = True
                payload["status_url"] = (
                    f"{PUBLIC_API_BASE}"
                    f"/media/render-jobs/"
                    f"{existing_job_id}"
                )

                return JSONResponse(
                    status_code=(
                        200
                        if existing.get(
                            "status"
                        )
                        == "complete"
                        else 202
                    ),
                    content=payload,
                )

        if _active_job_id:
            active = _render_jobs.get(
                _active_job_id,
                {},
            )

            raise HTTPException(
                status_code=409,
                detail={
                    "code": (
                        "renderer_busy"
                    ),
                    "message": (
                        "SH01 renderer is "
                        "already processing "
                        "another item"
                    ),
                    "active_job_id": (
                        _active_job_id
                    ),
                    "active_queue_id": (
                        active.get(
                            "queue_id"
                        )
                    ),
                },
            )

        job_id = secrets.token_hex(
            16
        )

        token = (
            _render_token_for_queue(
                queue_id
            )
        )

        job = {
            "job_id": job_id,
            "queue_id": queue_id,
            "status": "accepted",
            "created_at": (
                time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                )
            ),
            "started_at": None,
            "finished_at": None,
            "token": token,
            "video_url": None,
            "b2_file_name": (
                _render_file_name(token)
            ),
            "size_bytes": None,
            "error": None,
        }

        _render_jobs[job_id] = job

        _queue_to_job[
            queue_id
        ] = job_id

        # Reserve the renderer BEFORE scheduling the background task.
        # This closes the race where two simultaneous POSTs both see
        # an apparently free renderer.
        _active_job_id = job_id

        try:
            task = asyncio.create_task(
                _run_render_job(
                    job_id,
                    request,
                )
            )

            _background_tasks.add(
                task
            )

            task.add_done_callback(
                _background_tasks.discard
            )

        except Exception:
            _active_job_id = None

            _render_jobs.pop(
                job_id,
                None,
            )

            if (
                _queue_to_job.get(
                    queue_id
                )
                == job_id
            ):
                _queue_to_job.pop(
                    queue_id,
                    None,
                )

            raise

        payload = _job_payload(
            job
        )

        payload.update(
            {
                "ok": True,
                "existing": False,
                "status_url": (
                    f"{PUBLIC_API_BASE}"
                    f"/media/render-jobs/"
                    f"{job_id}"
                ),
            }
        )

        return JSONResponse(
            status_code=202,
            content=payload,
        )


@router.get(
    "/render-jobs/{job_id}",
)
async def get_render_job(
    job_id: str,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
):
    _require_media_key(
        x_api_key
    )

    async with _render_state_lock:
        _prune_jobs_locked()

        job = _render_jobs.get(
            job_id
        )

        if not job:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Render job not found. "
                    "SH01 may have restarted "
                    "or the retained job expired."
                ),
            )

        payload = _job_payload(
            job
        )

        payload["ok"] = True

        return payload


@router.get(
    "/render-state",
)
async def get_render_state(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
):
    _require_media_key(
        x_api_key
    )

    async with _render_state_lock:
        _prune_jobs_locked()

        active = (
            _render_jobs.get(
                _active_job_id,
                {},
            )
            if _active_job_id
            else {}
        )

        return {
            "ok": True,
            "busy": bool(
                _active_job_id
            ),
            "active_job_id": (
                _active_job_id
            ),
            "active_queue_id": (
                active.get(
                    "queue_id"
                )
            ),
            "active_status": (
                active.get(
                    "status"
                )
            ),
        }


@router.get(
    "/rendered/{token}",
)
async def get_rendered_video(
    token: str,
):
    """
    Public endpoint used by Meta/n8n/YouTube.

    No MP4 is permanently stored on SH01.
    This mints a fresh signed B2 URL and redirects to B2.
    """
    token = _validate_token(
        token
    )

    b2_file_name = (
        _render_file_name(token)
    )

    try:
        manager = get_b2_manager()

        auth_token = (
            await asyncio.to_thread(
                manager.get_download_authorization_token,
                b2_file_name,
                BACKBLAZE_BUCKET_NAME,
                B2_AUTH_SECONDS,
            )
        )

        signed_url = (
            manager.build_authorized_url(
                bucket_name=(
                    BACKBLAZE_BUCKET_NAME
                ),
                file_name=(
                    b2_file_name
                ),
                token=auth_token,
            )
        )

        return RedirectResponse(
            url=signed_url,
            status_code=307,
        )

    except Exception as exc:
        logger.warning(
            "Unable to mint rendered-media URL "
            "token=%s err=%s",
            token,
            exc,
        )

        raise HTTPException(
            status_code=404,
            detail=(
                "Rendered media not found "
                "or unavailable"
            ),
        ) from exc


@router.delete(
    "/rendered/{token}",
)
async def delete_rendered_video(
    token: str,
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
):
    """
    Delete the durable B2 render after publishing succeeds.
    """
    _require_media_key(
        x_api_key
    )

    token = _validate_token(
        token
    )

    b2_file_name = (
        _render_file_name(token)
    )

    try:
        manager = get_b2_manager()

        deleted = (
            await asyncio.to_thread(
                manager.delete_file,
                b2_file_name,
                BACKBLAZE_BUCKET_NAME,
            )
        )

        return {
            "ok": True,
            "token": token,
            "deleted": bool(
                deleted
            ),
        }

    except Exception as exc:
        logger.exception(
            "Rendered-media delete failed "
            "token=%s",
            token,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
