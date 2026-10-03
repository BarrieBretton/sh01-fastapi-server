from __future__ import annotations

import asyncio
import hashlib
import gc
import hmac
import ipaddress
import json
import logging
import os
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlparse

import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field, model_validator

from b2_helper import get_b2_manager
from media_capacity import (
    MEDIA_RENDER_LOCK,
    HEAVY_MEDIA_CAPACITY,
    reserve_heavy_media_or_raise,
)
from infra.persistence import store

logger = logging.getLogger("caption_clipper")

router = APIRouter(prefix="/media", tags=["Caption Clip Renderer"])

MEDIA_CLIP_API_KEY = os.getenv("MEDIA_CLIP_API_KEY", "").strip() or os.getenv(
    "TELEGRAM_STORAGE_API_KEY", ""
).strip()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
PUBLIC_API_BASE = os.getenv(
    "PUBLIC_API_BASE", "https://sh01.vivojaymail.workers.dev"
).strip().rstrip("/")
BACKBLAZE_BUCKET_NAME = os.getenv("BACKBLAZE_BUCKET_NAME", "").strip()
B2_PREFIX = os.getenv(
    "MEDIA_RENDER_B2_PREFIX", "renders/telegram-audio-social"
).strip().strip("/")

MAX_SOURCE_BYTES = int(os.getenv("MEDIA_CLIP_MAX_SOURCE_BYTES", str(600 * 1024 * 1024)))
MAX_CLIP_SECONDS = int(os.getenv("MEDIA_CLIP_MAX_SECONDS", "180"))
JOB_RETENTION_SECONDS = max(3600, int(os.getenv("MEDIA_CLIP_JOB_RETENTION_SECONDS", "86400")))
FFMPEG_TIMEOUT_SECONDS = int(os.getenv("MEDIA_CLIP_FFMPEG_TIMEOUT_SECONDS", "7200"))
TRANSCRIBE_TIMEOUT_SECONDS = int(os.getenv("MEDIA_CLIP_TRANSCRIBE_TIMEOUT_SECONDS", "600"))
WHISPER_MODEL = os.getenv("MEDIA_CLIP_WHISPER_MODEL", "tiny").strip() or "tiny"
WHISPER_DEVICE = os.getenv("MEDIA_CLIP_WHISPER_DEVICE", "cpu").strip() or "cpu"
WHISPER_COMPUTE_TYPE = os.getenv("MEDIA_CLIP_WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
WHISPER_CACHE_DIR = os.getenv("MEDIA_CLIP_WHISPER_CACHE_DIR", "/tmp/faster-whisper").strip() or "/tmp/faster-whisper"
WHISPER_CPU_THREADS = max(1, int(os.getenv("MEDIA_CLIP_WHISPER_CPU_THREADS", "1")))
WHISPER_VAD = os.getenv("MEDIA_CLIP_WHISPER_VAD", "true").strip().lower() in {"1", "true", "yes", "on"}
RENDER_WIDTH = max(360, int(os.getenv("MEDIA_CLIP_RENDER_WIDTH", "720")))
RENDER_HEIGHT = max(640, int(os.getenv("MEDIA_CLIP_RENDER_HEIGHT", "1280")))
DURABLE_TIMEOUT_SECONDS = max(2, int(os.getenv("MEDIA_CLIP_DURABLE_TIMEOUT_SECONDS", "8")))
UNLOAD_WHISPER_BEFORE_RENDER = os.getenv(
    "MEDIA_CLIP_UNLOAD_WHISPER_BEFORE_RENDER", "true"
).strip().lower() in {"1", "true", "yes", "on"}

_whisper_model = None
_whisper_model_lock = asyncio.Lock()

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()
_background_tasks: set[asyncio.Task] = set()


PRESETS: dict[str, dict[str, Any]] = {
    "viral_punch": {
        "caption": {
            "font": "DejaVu Sans",
            "font_size": 78,
            "primary": "&H00FFFFFF",
            "highlight": "&H0000E5FF",  # yellow-ish in ASS BGR
            "outline": "&H00101010",
            "outline_width": 7,
            "shadow": 1,
            "margin_v": 300,
            "bold": True,
            "uppercase": True,
            "max_words": 5,
            "max_chars": 30,
            "animation": "pop",
        },
        "video_effect": "punchy",
        "fit_mode": "blur",
    },
    "clean_minimal": {
        "caption": {
            "font": "DejaVu Sans",
            "font_size": 62,
            "primary": "&H00FFFFFF",
            "highlight": "&H00FFFFFF",
            "outline": "&H00101010",
            "outline_width": 4,
            "shadow": 0,
            "margin_v": 250,
            "bold": True,
            "uppercase": False,
            "max_words": 7,
            "max_chars": 38,
            "animation": "fade",
        },
        "video_effect": "crisp",
        "fit_mode": "blur",
    },
    "karaoke_glow": {
        "caption": {
            "font": "DejaVu Sans",
            "font_size": 72,
            "primary": "&H00FFFFFF",
            "highlight": "&H0000FFFF",  # yellow in ASS BGR
            "outline": "&H00101010",
            "outline_width": 6,
            "shadow": 2,
            "margin_v": 290,
            "bold": True,
            "uppercase": False,
            "max_words": 6,
            "max_chars": 34,
            "animation": "bounce",
        },
        "video_effect": "crisp",
        "fit_mode": "blur",
    },
    "podcast_bold": {
        "caption": {
            "font": "DejaVu Sans",
            "font_size": 68,
            "primary": "&H00FFFFFF",
            "highlight": "&H00FFAA33",
            "outline": "&H00000000",
            "outline_width": 8,
            "shadow": 1,
            "margin_v": 335,
            "bold": True,
            "uppercase": False,
            "max_words": 6,
            "max_chars": 36,
            "animation": "pop",
        },
        "video_effect": "punchy",
        "fit_mode": "blur",
    },
    "cinematic": {
        "caption": {
            "font": "DejaVu Sans",
            "font_size": 58,
            "primary": "&H00FFFFFF",
            "highlight": "&H00FFFFFF",
            "outline": "&H00101010",
            "outline_width": 3,
            "shadow": 1,
            "margin_v": 220,
            "bold": False,
            "uppercase": False,
            "max_words": 8,
            "max_chars": 42,
            "animation": "fade",
        },
        "video_effect": "cinematic",
        "fit_mode": "blur",
    },
}


class WordStamp(BaseModel):
    word: str = Field(min_length=1, max_length=200)
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_order(self):
        if self.end <= self.start:
            raise ValueError("word.end must be greater than word.start")
        return self


class CaptionOverrides(BaseModel):
    font: str | None = Field(default=None, max_length=120)
    font_size: int | None = Field(default=None, ge=32, le=120)
    primary: str | None = Field(default=None, max_length=20)
    highlight: str | None = Field(default=None, max_length=20)
    outline: str | None = Field(default=None, max_length=20)
    outline_width: int | None = Field(default=None, ge=0, le=14)
    shadow: int | None = Field(default=None, ge=0, le=6)
    margin_v: int | None = Field(default=None, ge=80, le=700)
    bold: bool | None = None
    uppercase: bool | None = None
    max_words: int | None = Field(default=None, ge=2, le=12)
    max_chars: int | None = Field(default=None, ge=10, le=60)
    animation: Literal["none", "fade", "pop", "bounce", "slide_up"] | None = None


class ClipCaptionRequest(BaseModel):
    source_url: str | None = None
    telegram_file_id: str | None = None
    request_id: str | None = Field(default=None, max_length=512)

    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)

    preset: Literal[
        "viral_punch", "clean_minimal", "karaoke_glow", "podcast_bold", "cinematic"
    ] = "viral_punch"
    caption_overrides: CaptionOverrides | None = None

    fit_mode: Literal["blur", "crop", "fit_black"] | None = None
    video_effect: Literal["none", "crisp", "punchy", "cinematic", "warm"] | None = None
    fps: Literal[24, 25, 30, 60] = 30
    crf: int = Field(default=20, ge=16, le=30)
    audio_bitrate: Literal[128, 160, 192, 256] = 192

    language: str | None = Field(default=None, min_length=2, max_length=10)
    # transcription_model: Literal["whisper-1"] = "whisper-1"
    word_timestamps: list[WordStamp] | None = None

    @model_validator(mode="after")
    def validate_input(self):
        if bool(self.source_url) == bool(self.telegram_file_id):
            raise ValueError("Provide exactly one of source_url or telegram_file_id")
        if self.end_seconds is not None:
            if self.end_seconds <= self.start_seconds:
                raise ValueError("end_seconds must be greater than start_seconds")
            if self.end_seconds - self.start_seconds > MAX_CLIP_SECONDS:
                raise ValueError(f"Requested clip exceeds MEDIA_CLIP_MAX_SECONDS={MAX_CLIP_SECONDS}")
        return self


def _require_key(x_api_key: str | None) -> None:
    if not MEDIA_CLIP_API_KEY:
        raise HTTPException(status_code=500, detail="MEDIA_CLIP_API_KEY/TELEGRAM_STORAGE_API_KEY is not configured")
    if not x_api_key or not secrets.compare_digest(x_api_key, MEDIA_CLIP_API_KEY):
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")


def _public_ip(host: str) -> bool:
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise HTTPException(status_code=400, detail=f"Source host cannot be resolved: {host}") from exc
    if not infos:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
            return False
    return True


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise HTTPException(status_code=400, detail="source_url must use http or https")
    if not parsed.hostname or not _public_ip(parsed.hostname):
        raise HTTPException(status_code=400, detail="source_url must resolve to a public host")


async def _download_url(url: str, dest: Path) -> None:
    current = url
    async with httpx.AsyncClient(timeout=120.0, follow_redirects=False) as client:
        for _ in range(6):
            _validate_public_url(current)
            async with client.stream("GET", current) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise HTTPException(status_code=424, detail="Source redirect missing Location")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise HTTPException(status_code=424, detail=f"Source download failed HTTP {response.status_code}")
                total = 0
                with dest.open("wb") as f:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_SOURCE_BYTES:
                            raise HTTPException(status_code=413, detail="Source is too large")
                        f.write(chunk)
                return
    raise HTTPException(status_code=400, detail="Too many redirects")


async def _telegram_file_path(file_id: str) -> str:
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN is not configured")
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getFile",
            params={"file_id": file_id},
        )
    if r.status_code >= 400:
        raise HTTPException(status_code=424, detail=f"Telegram getFile failed HTTP {r.status_code}")
    payload = r.json()
    path = payload.get("result", {}).get("file_path")
    if not payload.get("ok") or not path:
        raise HTTPException(status_code=424, detail="Telegram getFile returned no file_path")
    return str(path)


async def _download_telegram(file_id: str, dest: Path) -> None:
    path = await _telegram_file_path(file_id)
    await _download_url(f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{path}", dest)


def _run(cmd: list[str], timeout: int, label: str) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=timeout)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-10000:]
        logger.error("%s failed: %s", label, err)
        raise RuntimeError(f"{label} failed: {err[-1500:]}")
    return result


def _ffprobe_duration(path: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is not installed")
    r = _run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        60,
        "ffprobe duration",
    )
    try:
        return float(r.stdout.decode().strip())
    except ValueError as exc:
        raise RuntimeError("Could not determine source duration") from exc


def _extract_audio(source: Path, start: float, duration: float, dest: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    _run(
        [
            ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
        ],
        900,
        "audio extraction",
    )


def _load_whisper_model_sync():
    global _whisper_model
    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError(
            "faster-whisper is not installed; add faster-whisper to requirements.txt"
        ) from exc

    Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(
        "Loading faster-whisper model=%s device=%s compute_type=%s cpu_threads=%s",
        WHISPER_MODEL,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
        WHISPER_CPU_THREADS,
    )
    _whisper_model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        download_root=WHISPER_CACHE_DIR,
        cpu_threads=WHISPER_CPU_THREADS,
        num_workers=1,
    )
    return _whisper_model


def _transcribe_faster_whisper_sync(
    audio_path: Path,
    language: str | None,
) -> list[dict[str, Any]]:
    model = _load_whisper_model_sync()
    segments, _info = model.transcribe(
        str(audio_path),
        language=language or None,
        beam_size=1,
        word_timestamps=True,
        vad_filter=WHISPER_VAD,
        condition_on_previous_text=False,
        temperature=0.0,
    )

    out: list[dict[str, Any]] = []
    for segment in segments:
        for item in segment.words or []:
            text = str(item.word or "").strip()
            if not text or item.start is None or item.end is None:
                continue
            out.append(
                {
                    "word": text,
                    "start": float(item.start),
                    "end": float(item.end),
                    "probability": (
                        float(item.probability)
                        if item.probability is not None
                        else None
                    ),
                }
            )

    if not out:
        raise RuntimeError("faster-whisper returned no word timestamps")
    return out


async def _transcribe_faster_whisper(
    audio_path: Path,
    language: str | None,
) -> list[dict[str, Any]]:
    async with _whisper_model_lock:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    _transcribe_faster_whisper_sync,
                    audio_path,
                    language,
                ),
                timeout=TRANSCRIBE_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"faster-whisper transcription exceeded {TRANSCRIBE_TIMEOUT_SECONDS}s"
            ) from exc


async def _release_whisper_model() -> None:
    global _whisper_model
    if not UNLOAD_WHISPER_BEFORE_RENDER:
        return
    async with _whisper_model_lock:
        if _whisper_model is not None:
            logger.info("Releasing faster-whisper model before FFmpeg render")
            _whisper_model = None
    await asyncio.to_thread(gc.collect)


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    cs = int(round(seconds * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _ass_escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ")


def _chunks(words: list[dict[str, Any]], max_words: int, max_chars: int) -> list[list[dict[str, Any]]]:
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    chars = 0
    for w in words:
        token = str(w["word"]).strip()
        projected = chars + len(token) + (1 if current else 0)
        punct_break = bool(current and str(current[-1]["word"]).rstrip().endswith((".", "!", "?", ":", ";")))
        long_gap = bool(current and float(w["start"]) - float(current[-1]["end"]) > 0.7)
        if current and (len(current) >= max_words or projected > max_chars or punct_break or long_gap):
            chunks.append(current)
            current = []
            chars = 0
        current.append(w)
        chars += len(token) + (1 if len(current) > 1 else 0)
    if current:
        chunks.append(current)
    return chunks


def _animation_tag(name: str, margin_v: int) -> str:
    if name == "fade":
        return r"\fad(80,80)"
    if name == "pop":
        return r"\fscx82\fscy82\t(0,90,\fscx112\fscy112)\t(90,190,\fscx100\fscy100)\fad(30,60)"
    if name == "bounce":
        return r"\fscx90\fscy90\t(0,80,\fscx116\fscy116)\t(80,150,\fscx96\fscy96)\t(150,220,\fscx100\fscy100)\fad(20,50)"
    if name == "slide_up":
        y2 = RENDER_HEIGHT - margin_v
        y1 = y2 + max(45, int(RENDER_HEIGHT * 0.047))
        x = RENDER_WIDTH // 2
        return rf"\an2\move({x},{y1},{x},{y2},0,180)\fad(25,70)"
    return ""


def _make_ass(words: list[dict[str, Any]], cfg: dict[str, Any], dest: Path) -> None:
    max_words = int(cfg["max_words"])
    max_chars = int(cfg["max_chars"])
    groups = _chunks(words, max_words, max_chars)

    bold = -1 if cfg.get("bold") else 0
    header = f"""[Script Info]\nScriptType: v4.00+\nPlayResX: {RENDER_WIDTH}\nPlayResY: {RENDER_HEIGHT}\nScaledBorderAndShadow: yes\nWrapStyle: 2\n\n[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\nStyle: Caption,{cfg['font']},{cfg['font_size']},{cfg['primary']},{cfg['highlight']},{cfg['outline']},&H64000000,{bold},0,0,0,100,100,0,0,1,{cfg['outline_width']},{cfg['shadow']},2,70,70,{cfg['margin_v']},1\n\n[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"""

    lines = [header]
    for group in groups:
        for idx, active in enumerate(group):
            start = float(active["start"])
            end = float(active["end"])
            if idx + 1 < len(group):
                end = max(end, min(float(group[idx + 1]["start"]), end + 0.25))
            pieces = []
            for j, w in enumerate(group):
                token = _ass_escape(str(w["word"]).upper() if cfg.get("uppercase") else str(w["word"]))
                if j == idx:
                    pieces.append(r"{\c" + cfg["highlight"] + r"}" + token + r"{\c" + cfg["primary"] + r"}")
                else:
                    pieces.append(token)
            text = " ".join(pieces)
            tag = _animation_tag(str(cfg.get("animation") or "none"), int(cfg["margin_v"]))
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Caption,,0,0,0,,{{{tag}}}{text}\n"
            )
    dest.write_text("".join(lines), encoding="utf-8")


def _video_filter(fit_mode: str, effect: str, ass_path: Path) -> str:
    ass = str(ass_path).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    w = RENDER_WIDTH
    h = RENDER_HEIGHT

    if fit_mode == "crop":
        base = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    elif fit_mode == "fit_black":
        base = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    else:
        # Keep the blurred background branch deliberately small, then upscale it.
        # This avoids two full 1080x1920 working frames on low-memory Render instances.
        bg_w = max(180, w // 2)
        bg_h = max(320, h // 2)
        base = (
            "split=2[bg][fg];"
            f"[bg]scale={bg_w}:{bg_h}:force_original_aspect_ratio=increase,"
            f"crop={bg_w}:{bg_h},boxblur=12:1,scale={w}:{h}[bg2];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg2];"
            "[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
        )

    effects = {
        "none": "",
        "crisp": ",eq=contrast=1.04:saturation=1.04,unsharp=3:3:0.25:3:3:0",
        "punchy": ",eq=contrast=1.08:saturation=1.12:brightness=0.01,unsharp=3:3:0.30:3:3:0",
        "cinematic": ",eq=contrast=1.07:saturation=0.92:gamma=0.98,vignette=PI/7",
        "warm": ",eq=contrast=1.04:saturation=1.08:gamma_r=1.03:gamma_b=0.97",
    }
    return base + effects.get(effect, "") + f",ass='{ass}'"


def _render_video(
    source: Path,
    output: Path,
    ass_path: Path,
    start: float,
    duration: float,
    fit_mode: str,
    effect: str,
    fps: int,
    crf: int,
    audio_bitrate: int,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    vf = _video_filter(fit_mode, effect, ass_path)
    cmd = [
        ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-vf", vf,
        "-threads", "1",
        "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", str(crf),
        "-r", str(fps), "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", f"{audio_bitrate}k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest", str(output),
    ]
    _run(cmd, FFMPEG_TIMEOUT_SECONDS, "caption clip render")
    if not output.exists() or output.stat().st_size == 0:
        raise RuntimeError("Rendered output is empty")


def _request_token(request_id: str) -> str:
    key = MEDIA_CLIP_API_KEY or "caption-clips"
    return hmac.new(key.encode(), request_id.encode(), hashlib.sha256).hexdigest()[:32]


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


CAPTION_JOB_KIND = "caption_clip"


def _job_id_for_request(request_id: str) -> str:
    key = MEDIA_CLIP_API_KEY or "caption-clips"
    return hmac.new(
        key.encode("utf-8"),
        f"caption-job:{request_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()[:32]


def _durable_payload(req: ClipCaptionRequest, request_id: str, token: str) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "token": token,
        "request": req.model_dump(mode="json"),
    }


async def _persist_job(job: dict[str, Any], payload: dict[str, Any] | None = None) -> bool:
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                store.put_job,
                str(job["job_id"]),
                CAPTION_JOB_KIND,
                str(job.get("status") or "unknown"),
                payload,
                _job_view(job),
                job.get("error"),
            ),
            timeout=DURABLE_TIMEOUT_SECONDS,
        )
        return True
    except asyncio.TimeoutError:
        logger.warning(
            "Caption job persistence timed out job_id=%s after %ss",
            job.get("job_id"),
            DURABLE_TIMEOUT_SECONDS,
        )
        return False
    except Exception as exc:
        logger.warning("Caption job persistence failed job_id=%s: %s", job.get("job_id"), exc)
        return False


async def _load_durable_job(job_id: str) -> dict[str, Any] | None:
    try:
        row = await asyncio.wait_for(
            asyncio.to_thread(store.get_job, job_id),
            timeout=DURABLE_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Caption durable-store lookup timed out after {DURABLE_TIMEOUT_SECONDS}s",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Caption durable-store lookup failed: {type(exc).__name__}: {exc}",
        ) from exc
    if not row or row.get("kind") != CAPTION_JOB_KIND:
        return None

    result = row.get("result")
    if isinstance(result, dict):
        job = dict(result)
    else:
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        job = {
            "job_id": job_id,
            "request_id": payload.get("request_id"),
            "token": payload.get("token"),
            "status": row.get("status"),
            "stage": row.get("status"),
            "error": row.get("error"),
        }

    job.setdefault("job_id", job_id)
    job["durable"] = True
    job["durable_updated_at"] = row.get("updated_at")
    return job


async def _prune_jobs() -> None:
    now = time.time()
    async with _jobs_lock:
        stale = [
            job_id for job_id, job in _jobs.items()
            if job.get("status") in {"complete", "failed"}
            and now - float(job.get("_finished_epoch") or now) >= JOB_RETENTION_SECONDS
        ]
        for job_id in stale:
            _jobs.pop(job_id, None)


async def _execute(job_id: str, req: ClipCaptionRequest) -> None:
    job = _jobs[job_id]
    durable_payload = _durable_payload(
        req,
        str(job["request_id"]),
        str(job["token"]),
    )
    capacity_lease_id = str(job.get("_capacity_lease_id") or "")

    async with MEDIA_RENDER_LOCK:
        job.update(
            status="running",
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            stage="download",
        )
        await _persist_job(job, durable_payload)

        try:
            with tempfile.TemporaryDirectory(prefix="sh01-caption-clip-") as tmp:
                root = Path(tmp)
                source = root / "source.bin"
                audio = root / "clip.wav"
                ass = root / "captions.ass"
                output = root / "output.mp4"

                if req.source_url:
                    await _download_url(req.source_url, source)
                else:
                    await _download_telegram(str(req.telegram_file_id), source)

                total_duration = await asyncio.to_thread(_ffprobe_duration, source)
                start = min(req.start_seconds, max(0.0, total_duration - 0.05))
                end = (
                    req.end_seconds
                    if req.end_seconds is not None
                    else min(total_duration, start + MAX_CLIP_SECONDS)
                )
                end = min(end, total_duration)
                duration = end - start
                if duration <= 0.05:
                    raise RuntimeError("Selected clip has no usable duration")
                if duration > MAX_CLIP_SECONDS + 0.01:
                    raise RuntimeError(f"Clip exceeds {MAX_CLIP_SECONDS}s maximum")

                job.update(
                    stage="transcription",
                    clip_start=start,
                    clip_end=end,
                    clip_duration=duration,
                )
                await _persist_job(job, durable_payload)

                if req.word_timestamps:
                    words = [w.model_dump() for w in req.word_timestamps]
                else:
                    await asyncio.to_thread(_extract_audio, source, start, duration, audio)
                    words = await _transcribe_faster_whisper(audio, req.language)
                    await _release_whisper_model()

                preset = json.loads(json.dumps(PRESETS[req.preset]))
                caption_cfg = preset["caption"]
                if req.caption_overrides:
                    for key, value in req.caption_overrides.model_dump().items():
                        if value is not None:
                            caption_cfg[key] = value
                fit_mode = req.fit_mode or preset["fit_mode"]
                video_effect = req.video_effect or preset["video_effect"]

                await asyncio.to_thread(_make_ass, words, caption_cfg, ass)
                job.update(
                    stage="render",
                    word_count=len(words),
                    resolved_preset={
                        "preset": req.preset,
                        "caption": caption_cfg,
                        "fit_mode": fit_mode,
                        "video_effect": video_effect,
                    },
                )
                await _persist_job(job, durable_payload)

                await asyncio.to_thread(
                    _render_video,
                    source,
                    output,
                    ass,
                    start,
                    duration,
                    fit_mode,
                    video_effect,
                    req.fps,
                    req.crf,
                    req.audio_bitrate,
                )

                job.update(stage="upload")
                await _persist_job(job, durable_payload)

                token = str(job["token"])
                b2_name = f"{B2_PREFIX}/{token}.mp4"
                if not BACKBLAZE_BUCKET_NAME:
                    raise RuntimeError("BACKBLAZE_BUCKET_NAME is not configured")
                manager = get_b2_manager()
                await asyncio.to_thread(
                    manager.upload_file,
                    output,
                    BACKBLAZE_BUCKET_NAME,
                    b2_name,
                    "video/mp4",
                    1,
                )

                job.update(
                    status="complete",
                    stage="complete",
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    video_url=f"{PUBLIC_API_BASE}/media/rendered/{token}",
                    b2_file_name=b2_name,
                    size_bytes=output.stat().st_size,
                    error=None,
                    error_type=None,
                    failed_stage=None,
                )
                await _persist_job(job, durable_payload)

        except Exception as exc:
            failed_stage = job.get("stage")

            logger.exception(
                "Caption clip job failed job_id=%s stage=%s",
                job_id,
                failed_stage,
            )

            if isinstance(exc, HTTPException):
                error_message = str(exc.detail)
            else:
                error_message = str(exc) or repr(exc)

            job.update(
                status="failed",
                stage="failed",
                failed_stage=failed_stage,
                completed_at=time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                ),
                error=error_message[:3000],
                error_type=type(exc).__name__,
            )
            await _persist_job(job, durable_payload)

        finally:
            job["_finished_epoch"] = time.time()
            if capacity_lease_id:
                await HEAVY_MEDIA_CAPACITY.release(capacity_lease_id)


@router.post("/clip-caption")
async def create_caption_clip(
    req: ClipCaptionRequest,
    x_api_key: str | None = Header(None, alias="X-API-KEY"),
):
    _require_key(x_api_key)
    await _prune_jobs()

    request_id = (req.request_id or str(uuid.uuid4())).strip()
    token = _request_token(request_id)
    job_id = _job_id_for_request(request_id)
    durable_payload = _durable_payload(req, request_id, token)

    async with _jobs_lock:
        local = _jobs.get(job_id)
        if local and local.get("status") in {"queued", "running", "complete"}:
            from fastapi.responses import JSONResponse
            status_code = 200 if local.get("status") == "complete" else 202
            return JSONResponse(status_code=status_code, content=_job_view(local))

    durable = await _load_durable_job(job_id)
    if durable and durable.get("status") == "complete":
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=200, content=_job_view(durable))

    previous_attempt = int((durable or {}).get("attempt") or 0)
    recovered_from = (durable or {}).get("status")

    lease = await reserve_heavy_media_or_raise(
        kind="clip_caption",
        job_id=job_id,
        request_id=request_id,
        endpoint="/media/clip-caption",
    )

    job = {
        "job_id": job_id,
        "request_id": request_id,
        "status": "queued",
        "stage": "queued",
        "preset": req.preset,
        "token": token,
        "status_url": f"{PUBLIC_API_BASE}/media/clip-caption-jobs/{job_id}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "video_url": None,
        "error": None,
        "attempt": previous_attempt + 1,
        "recovered_from": recovered_from,
        "_capacity_lease_id": lease.lease_id,
    }

    async with _jobs_lock:
        _jobs[job_id] = job

    try:
        persisted = await _persist_job(job, durable_payload)
        if not persisted:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "durable_job_registration_failed",
                    "message": "Caption job was not accepted because durable job registration failed",
                    "retryable": True,
                },
                headers={"Retry-After": "5"},
            )
        task = asyncio.create_task(_execute(job_id, req))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        async with _jobs_lock:
            _jobs.pop(job_id, None)
        await HEAVY_MEDIA_CAPACITY.release(lease.lease_id)
        raise

    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=202, content=_job_view(job))


@router.get("/clip-caption-jobs/{job_id}")
async def caption_clip_job(
    job_id: str,
    x_api_key: str | None = Header(None, alias="X-API-KEY"),
):
    _require_key(x_api_key)
    await _prune_jobs()

    async with _jobs_lock:
        local = _jobs.get(job_id)
        if local:
            return _job_view(local)

    durable = await _load_durable_job(job_id)
    if not durable:
        raise HTTPException(
            status_code=404,
            detail="Unknown or expired clip-caption job",
        )

    # If this process does not own the task but the durable row says it was
    # queued/running, the worker that owned it disappeared (restart/failover).
    # Report it as recoverable instead of pretending work is still progressing.
    if durable.get("status") in {"queued", "running"}:
        durable["previous_status"] = durable.get("status")
        durable["status"] = "interrupted"
        durable["stage"] = "interrupted"
        durable["recoverable"] = True
        durable["retry"] = {
            "method": "POST",
            "path": "/media/clip-caption",
            "instruction": "Resubmit the same request_id and request body to resume as a new attempt.",
        }

    return _job_view(durable)


@router.get("/clip-caption-presets")
async def caption_clip_presets(x_api_key: str | None = Header(None, alias="X-API-KEY")):
    _require_key(x_api_key)
    return {
        "presets": PRESETS,
        "animations": ["none", "fade", "pop", "bounce", "slide_up"],
        "video_effects": ["none", "crisp", "punchy", "cinematic", "warm"],
        "fit_modes": ["blur", "crop", "fit_black"],
    }
