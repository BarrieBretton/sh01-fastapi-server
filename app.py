# Imports

# General
import os
import socket
import threading
import logging
import subprocess
import time
import shutil
import sys
import uuid
import hashlib
import hmac

import ssl
import certifi
# Set the SSL context to use certifi's certificates
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())

# import random
# import sqlite3
# import base64
# import uuid
# import json

from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dotenv import load_dotenv
from image_templating import router as image_templating_router
from infra.routes import router as infra_router
from infra.media_job_store_routes import router as media_job_store_router
from media_router import router as media_router
from caption_router import router as caption_router
from auto_editor import router as auto_editor_router
from threads_feature.router import router as threads_router
from x_feature.router import router as x_router, require_internal_key as require_x_internal_key

# Add this import at the top with other imports
from sheets_helper import (
    get_sheets_service,
    read_sheet_by_name,
    filter_status_rows,
    update_cell,
    get_column_letter,
    append_rows,
    # list_sheet_names,
    # SPREADSHEET_ID
)

# B2 imports
from b2_helper import get_b2_manager, upload_to_b2

# YT Handler Module
from youtube_handler import YouTubeHandler

# Basic HTTP Requests
import httpx
import redis
import requests
import asyncio

# FastApi Server
from requests_oauthlib import OAuth1
from fastapi import FastAPI, HTTPException, Body, Query, Header
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Kite Zerodha Connectivity
from kiteconnect import KiteConnect, KiteTicker

# Crons + Scheduling
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Google OAuth
# from google_auth_oauthlib.flow import InstalledAppFlow
# from googleapiclient.discovery import build
# from google.oauth2.credentials import Credentials

# =====================================================
# SANITY CHECKS   
# =====================================================

# if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
#     print("ERROR: FFmpeg or FFprobe not found! Exiting.")
#     sys.exit(1)

# =====================================================
# ENV & LOGGING SETUP   
# =====================================================

load_dotenv(".env")

# temp
ffmpeg_bin_path = r"C:\Users\Vivan.Jaiswal\Documents\ffmpeg-2025-12-18-git-78c75d546a-essentials_build\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

# Create logs dir
os.makedirs("logs", exist_ok=True)

# Proper logging config (only once!)
logger = logging.getLogger("python_server")
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

# File handler (rotating)
file_handler = RotatingFileHandler("logs/app.log", maxBytes=10_485_760, backupCount=5)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

DOWNLOADS_DIR = Path("downloads").absolute()
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

print(f"DOWNLOADS_DIR: {DOWNLOADS_DIR}")

# B2 Configuration
BACKBLAZE_KEY_ID = os.getenv("BACKBLAZE_KEY_ID")
BACKBLAZE_APPLICATION_KEY = os.getenv("BACKBLAZE_APPLICATION_KEY")
BACKBLAZE_BUCKET_NAME = os.getenv("BACKBLAZE_BUCKET_NAME")
B2_REFRESH_API_KEY = os.getenv("B2_REFRESH_API_KEY", "").strip()

if not BACKBLAZE_KEY_ID or not BACKBLAZE_APPLICATION_KEY:
    logger.warning("BACKBLAZE_KEY_ID or BACKBLAZE_APPLICATION_KEY not set - B2 uploads will fail")
if not BACKBLAZE_BUCKET_NAME:
    logger.warning("BACKBLAZE_BUCKET_NAME not set - bucket name must be provided per upload")

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Silence noisy libraries
for noisy in ("urllib3", "requests", "requests_oauthlib", "apscheduler"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger.info("Application starting up...")

# =====================================================
# TELEGRAM STORAGE CONFIGURATION
# =====================================================

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
).strip()

# This sheet acts as the Telegram storage database.
# It is intentionally NOT caller-configurable.
TELEGRAM_STORAGE_SHEET = "telegram-storage"

TELEGRAM_BOT_API_BASE = (
    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
    if TELEGRAM_BOT_TOKEN
    else ""
)

TELEGRAM_FILE_API_BASE = (
    f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}"
    if TELEGRAM_BOT_TOKEN
    else ""
)

TELEGRAM_STORAGE_API_KEY = os.getenv(
    "TELEGRAM_STORAGE_API_KEY",
    "",
).strip()

TELEGRAM_MEDIA_SIGNING_SECRET = os.getenv(
    "TELEGRAM_MEDIA_SIGNING_SECRET",
    "",
).strip()

PUBLIC_API_BASE = os.getenv(
    "PUBLIC_API_BASE",
    "https://sh01.vivojaymail.workers.dev",
).strip().rstrip("/")

if not TELEGRAM_BOT_TOKEN:
    logger.warning(
        "TELEGRAM_BOT_TOKEN not configured - Telegram storage endpoints disabled"
    )

if not TELEGRAM_STORAGE_API_KEY:
    logger.warning(
        "TELEGRAM_STORAGE_API_KEY not configured - Telegram storage API unavailable"
    )

# =====================================================
# CONFIG & AUTH
# =====================================================

# FB Page Connectivity
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
    logger.warning("FB_PAGE_ID or FB_PAGE_ACCESS_TOKEN missing – /video-to-fb will be disabled")

def create_oauth1(consumer_key, consumer_secret, token, token_secret, name):
    missing = []
    if not consumer_key: missing.append("consumer_key")
    if not consumer_secret: missing.append("consumer_secret")
    if not token: missing.append("oauth_token")
    if not token_secret: missing.append("oauth_token_secret")

    if missing:
        logger.error(f"Tumblr OAuth misconfigured for {name}: missing {', '.join(missing)}")
        return None
    return OAuth1(consumer_key, consumer_secret, token, token_secret)

# X/Twitter OAuth1
oauth_x = OAuth1(
    os.getenv("API_KEY"),
    os.getenv("API_KEY_SECRET"),
    os.getenv("ACCESS_TOKEN"),
    os.getenv("ACCESS_TOKEN_SECRET"),
)

# Tumblr accounts
oauth_erika = create_oauth1(
    os.getenv("TUMBLR_CONSUMER_KEY_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_CONSUMER_SECRET_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_TOKEN_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_TOKEN_SECRET_ERIKA_DEVEREUX"),
    "erika.devereux"
)
T_EK_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_ERIKA_DEVEREUX")

oauth_vlvt = create_oauth1(
    os.getenv("TUMBLR_CONSUMER_KEY_VLVT_AVE"),
    os.getenv("TUMBLR_CONSUMER_SECRET_VLVT_AVE"),
    os.getenv("TUMBLR_TOKEN_VLVT_AVE"),
    os.getenv("TUMBLR_TOKEN_SECRET_VLVT_AVE"),
    "vlvt.ave"
)
T_VL_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_VLVT_AVE")

oauth_cyootstuff = create_oauth1(
    os.getenv("TUMBLR_CONSUMER_KEY_CYOOTSTUFF"),
    os.getenv("TUMBLR_CONSUMER_SECRET_CYOOTSTUFF"),
    os.getenv("TUMBLR_TOKEN_CYOOTSTUFF"),
    os.getenv("TUMBLR_TOKEN_SECRET_CYOOTSTUFF"),
    "cyootstuff"
)
T_CY_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_CYOOTSTUFF")

# After loading env
required_blog_ids = {
    "T_EK_BID": T_EK_BID,
    "T_VL_BID": T_VL_BID,
    "T_CY_BID": T_CY_BID,
}

for var, val in required_blog_ids.items():
    if not val:
        logger.error(f"{var} is missing in .env")

# n8n
N8N_API_KEY = os.getenv("N8N_API_KEY")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "").rstrip("/") + "/api/v1"

# Redis for distributed cron lock
REDIS_URL = os.getenv("REDIS_URL", "redis://n8n-redis:6379/0")
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

LOCK_NAME = "cron_master_lock"
LOCK_TTL = 24 * 60 * 60  # 24 hours

# Default workflows to activate daily
DEFAULT_WORKFLOWS = [
    "ai-image-model-5 dev",
    "ig-sidehustle PINTEREST 3",
    "office-hours",
    "__backup",
    "readiness-probe",
    "report-render-cloud-stats",
    "ai-image-model-cyootstuff",
    "streamables-to-lilnubbns",
    "streamables-to-dave.commercial7 2",
    "streamables-to-aand.cut",
    "media-distribution-center 2",
]

# Models

class YouTubeVideoResponse(BaseModel):
    video_id: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    dislike_count: int
    comment_count: int
    thumbnail_url: str
    channel_title: str
    relevance_score: float
    engagement_score: float

class VideoToFBResponse(BaseModel):
    success: bool
    facebook: dict[str, str | Any] # Adjust later based on EXACTLY what `upload_to_facebook` actually returns
    local_file: str
    size_mb: float

# Start

app = FastAPI(title="Python Server - SideHustle-01", version="2.2")
app.include_router(image_templating_router, prefix="/img")
app.include_router(infra_router)
app.include_router(media_job_store_router)
app.include_router(media_router)
app.include_router(caption_router)
app.include_router(auto_editor_router)
app.include_router(threads_router)
app.include_router(x_router)

def pick_tumblr_account(name: str):
    name = (name or "").lower().strip()
    if name == "vlvt.ave":
        if oauth_vlvt is None:
            raise HTTPException(status_code=500, detail="Tumblr account 'vlvt.ave' not configured")
        return oauth_vlvt, T_VL_BID, "VLVT_AVE"
    if name == "erika.devereux":
        if oauth_erika is None:
            raise HTTPException(status_code=500, detail="Tumblr account 'erika.devereux' not configured")
        return oauth_erika, T_EK_BID, "ERIKA_DEVEREUX"
    if name == "cyootstuff":
        if oauth_cyootstuff is None:
            raise HTTPException(status_code=500, detail="Tumblr account 'cyootstuff' not configured")
        return oauth_cyootstuff, T_CY_BID, "CYOOTSTUFF"

    raise HTTPException(status_code=400, detail=f"Unknown tumblr_account: {name}")


# =====================================================
# FB HELPER
# =====================================================
def upload_to_facebook(video_path: Path, description: str = "") -> Dict[str, str]:
    """
    Upload a local MP4 to Facebook Page using direct upload (no resumable).
    Requires +faststart in MP4.
    """
    if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="Facebook credentials missing (FB_PAGE_ID / FB_PAGE_ACCESS_TOKEN)"
        )

    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")

    file_size = video_path.stat().st_size
    logger.info("Uploading to Facebook: %s (%.2f MB)", video_path.name, file_size / 1_048_576)

    url = f"https://graph-video.facebook.com/v20.0/{FB_PAGE_ID}/videos"

    # Open file in binary mode
    with open(video_path, "rb") as f:
        files = {"source": (video_path.name, f, "video/mp4")}
        data = {
            "access_token": FB_PAGE_ACCESS_TOKEN,
            "description": (description or "Uploaded via API").strip(),
            "title": (description.split("\n")[0][:100] if description else "Video Post").strip(),
        }

        try:
            resp = requests.post(url, files=files, data=data, timeout=900)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.HTTPError as e:
            error_body = resp.text
            logger.error("Facebook upload failed: %s %s | Response: %s", resp.status_code, resp.reason, error_body)
            try:
                err_json = resp.json()
                msg = err_json.get("error", {}).get("message", "Unknown error")
            except:
                msg = error_body[:200]
            raise HTTPException(status_code=500, detail=f"Facebook API error: {msg}")
        except Exception as e:
            logger.exception("Unexpected error during FB upload")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    video_id = result.get("id")
    if not video_id:
        raise RuntimeError("Facebook returned no video_id")

    fb_url = f"https://www.facebook.com/{FB_PAGE_ID}/videos/{video_id}"
    logger.info("Facebook upload SUCCESS: video_id=%s", video_id)
    return {"video_id": video_id, "url": fb_url}

# =====================================================
# YT HELPER
# =====================================================
async def fetch_videos():
    handler = YouTubeHandler()
    videos = await handler.get_videos_by_handle("@username", sort_by="engagement")
    for video in videos:
        print(video.title, video.engagement_score)

# =====================================================
# CLEANUP HELPER
# =====================================================

def cleanup_video_file(video_path: Path) -> None:
    """
    Safely removes a video file with proper error handling.
    
    Args:
        video_path: Path to the video file to remove
    """
    try:
        if video_path.exists():
            video_path.unlink()
            logger.info("Cleaned up downloaded file: %s", video_path)
        else:
            logger.debug("File already removed: %s", video_path)
    except PermissionError as e:
        logger.error("Permission denied when cleaning up %s: %s", video_path, e)
    except Exception as e:
        logger.warning("Failed to clean up file %s: %s", video_path, e)

# =====================================================
# B2 MODELS (replacing Cloudinary)
# =====================================================
class B2DeleteRequest(BaseModel):
    file_name: str
    bucket_name: str = None

# =====================================================
# KITE MANAGER
# =====================================================

class KiteManager:
    def __init__(self, api_key: str, api_secret: str, initial_token: str | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self._kite = KiteConnect(api_key=api_key)
        self._access_token = None
        self._lock = threading.RLock()
        if initial_token:
            self.set_access_token(initial_token)
            logger.info("KiteManager initialized with access token from env")

    def set_access_token(self, token: str):
        if not token:
            raise ValueError("Access token cannot be empty")
        with self._lock:
            self._access_token = token
            self._kite.set_access_token(token)
            logger.info("Kite access token updated in-memory")

    def has_token(self) -> bool:
        with self._lock:
            return bool(self._access_token and self._access_token.strip())

    def login_url(self) -> str:
        url = self._kite.login_url()
        logger.info("Generated Kite login URL")
        return url

    def generate_session(self, request_token: str) -> dict:
        if not request_token:
            raise ValueError("request_token is required")
        
        with self._lock:
            try:
                logger.info("Attempting to generate Kite session with request_token=%s...", request_token[:10])
                data = self._kite.generate_session(request_token, api_secret=self.api_secret)
                
                token = data.get("access_token")
                if not token:
                    logger.error("Kite generate_session succeeded but returned NO access_token! Response: %s", data)
                    raise RuntimeError("Kite returned empty access_token. Likely wrong API_SECRET or revoked app.")
                
                self.set_access_token(token)
                logger.info("Kite session generated SUCCESSFULLY. User: %s", data.get("user_id"))
                return data

            except Exception as e:
                logger.exception("Kite generate_session FAILED completely")
                if "invalid" in str(e).lower() or "secret" in str(e).lower():
                    raise RuntimeError(f"Kite session failed: {e} → Check KITE_API_SECRET is correct and matches your app on developers.kite.trade")
                raise RuntimeError(f"Kite session failed: {e}")

    def historical_data(self, instrument_token: int, from_date: datetime, to_date: datetime, interval: str):
        if not self.has_token():
            raise RuntimeError("Kite access token not set")
        return self._kite.historical_data(instrument_token, from_date, to_date, interval)

    def ltp(self, instrument_identifiers):
        if not self.has_token():
            raise RuntimeError("Kite access token not set")
        return self._kite.ltp(instrument_identifiers)

kite_manager = KiteManager(
    os.getenv("KITE_API_KEY"),
    os.getenv("KITE_API_SECRET"),
    os.getenv("KITE_ACCESS_TOKEN", "").strip() or None,
)

# =====================================================
# FASTAPI APP
# =====================================================

# Pydantic models
class TumblrPostRequest(BaseModel):
    image_url: str
    caption: str | None = None
    tumblr_account: str | None = None

class KiteGenerateSessionRequest(BaseModel):
    request_token: str

class KiteSetTokenRequest(BaseModel):
    access_token: str

class KiteCandlesRequest(BaseModel):
    instrument_token: int
    interval: str = "day"
    days: int = 30

class KiteLTPRequest(BaseModel):
    symbols: list[str] | str

class TelegramStoreRequest(BaseModel):
    source_url: str
    chat_id: int

    caption: str | None = None
    text: str | None = None
    file_name: str | None = None

    metadata: dict[str, Any] | None = None


class TelegramIndexMessageRequest(BaseModel):
    update: dict[str, Any]

    metadata: dict[str, Any] | None = None


class TelegramMarkUsedRequest(BaseModel):
    storage_id: str

# =====================================================
# B2 HELPERS
# =====================================================

def require_refresh_key(x_api_key: str | None):
    if not B2_REFRESH_API_KEY:
        # If you really want it open, leave env var empty.
        # But strongly recommended to set a key.
        return
    if not x_api_key or x_api_key != B2_REFRESH_API_KEY:
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")


# =====================================================
# HELPERS
# =====================================================

def safe_json(response):
    try:
        return response.json()
    except Exception:
        logger.warning("Non-JSON response from n8n (likely auth page): %s...", response.text[:200])
        return None

def get_all_workflows():
    url = f"{N8N_BASE_URL}/workflows"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}
    logger.debug("Fetching all n8n workflows from %s", url)
    r = requests.get(url, headers=headers, timeout=60)
    data = safe_json(r)
    if not data:
        raise RuntimeError("n8n API returned non-JSON. Check N8N_BASE_URL and API key.")
    return data.get("data", [])

def deactivate_workflow(wf):
    url = f"{N8N_BASE_URL}/workflows/{wf['id']}/deactivate"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}
    logger.info("Deactivating workflow: %s (%s)", wf["name"], wf["id"])
    return requests.post(url, headers=headers, timeout=60)

def activate_workflow(wf):
    url = f"{N8N_BASE_URL}/workflows/{wf['id']}/activate"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}
    logger.info("Activating workflow: %s (%s)", wf["name"], wf["id"])
    return requests.post(url, headers=headers, timeout=60)

def get_streamable_mp4(video_url: str, retries: int = 2) -> Path:
    """
    Downloads a YouTube video using yt-dlp.
    NO re-encoding, NO compatibility checks — just download the best mp4.
    """
    output_file = DOWNLOADS_DIR / f"video_{uuid.uuid4().hex}.mp4"
   
    # yt-dlp command templates (NO COOKIES)
    cmd_templates = [
        [sys.executable, "-m", "yt_dlp", "--force-ipv4", "--no-check-certificate",
         "-f", "bestvideo+bestaudio/best",
         "--merge-output-format", "mp4",
         "--socket-timeout", "30",
         "-o", str(output_file)],
    
        [sys.executable, "-m", "yt_dlp", "--force-ipv4", "--no-check-certificate",
         "-f", "best",
         "--merge-output-format", "mp4",
         "--socket-timeout", "30",
         "-o", str(output_file)]
    ]
   
    last_exception = None
   
    for attempt in range(retries):
        for cmd in cmd_templates:
            try:
                logger.info("Running yt-dlp attempt %d: %s", attempt + 1, " ".join(cmd + [video_url]))
                result = subprocess.run(cmd + [video_url], capture_output=True, text=True)
                logger.debug("yt-dlp stdout:\n%s", result.stdout)
                logger.debug("yt-dlp stderr:\n%s", result.stderr)
                if result.returncode != 0:
                    raise RuntimeError(f"yt-dlp exited with code {result.returncode}")
                if not output_file.exists() or output_file.stat().st_size == 0:
                    raise RuntimeError("MP4 file not created or empty")
                
                # Verify playable duration
                ffprobe_cmd = [
                    "ffprobe", "-v", "error", "-show_entries",
                    "format=duration", "-of",
                    "default=noprint_wrappers=1:nokey=1", str(output_file)
                ]
                ff_result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                duration_str = ff_result.stdout.strip()
                duration = float(duration_str) if duration_str else 0.0
                if duration <= 0:
                    raise RuntimeError("Downloaded file has 0 playback length")
                
                logger.info("Downloaded successfully: %s (%d bytes, duration %.2f s)",
                            output_file, output_file.stat().st_size, duration)
                
                # === RE-ENCODING COMPLETELY SKIPPED ===
                # We no longer check codec/pixel format or re-encode
                return output_file
                
            except Exception as e:
                last_exception = e
                logger.warning("yt-dlp attempt failed: %s", e)
                if output_file.exists():
                    try:
                        output_file.unlink()
                    except Exception:
                        pass
    logger.error("All yt-dlp download attempts failed for URL: %s", video_url)
    raise HTTPException(status_code=500, detail=f"Failed to download video: {last_exception}")

# =====================================================
# TELEGRAM STORAGE HELPERS
# =====================================================

def require_telegram_config() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_BOT_TOKEN is not configured",
        )


def telegram_api_call(
    method: str,
    data: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    """
    Call Telegram Bot API and return the result object.
    """
    require_telegram_config()

    url = f"{TELEGRAM_BOT_API_BASE}/{method}"

    try:
        response = requests.post(
            url,
            data=data or {},
            timeout=timeout,
        )

        payload = response.json()

    except Exception as exc:
        logger.exception(
            "Telegram API request failed: %s",
            method,
        )
        raise RuntimeError(
            f"Telegram API request failed: {exc}"
        ) from exc

    if not response.ok or not payload.get("ok"):
        description = payload.get(
            "description",
            response.text[:500],
        )

        raise RuntimeError(
            f"Telegram {method} failed: {description}"
        )

    return payload.get("result") or {}


def extract_telegram_file(
    message: dict[str, Any],
) -> dict[str, Any] | None:
    """
    Extract the primary Telegram file object from a message.

    Supports:
      document
      video
      audio
      voice
      animation
      video_note
      photo
    """

    if not message:
        return None

    file_object = None
    file_type = None

    if message.get("document"):
        file_type = "document"
        file_object = message["document"]

    elif message.get("video"):
        file_type = "video"
        file_object = message["video"]

    elif message.get("audio"):
        file_type = "audio"
        file_object = message["audio"]

    elif message.get("voice"):
        file_type = "voice"
        file_object = message["voice"]

    elif message.get("animation"):
        file_type = "animation"
        file_object = message["animation"]

    elif message.get("video_note"):
        file_type = "video_note"
        file_object = message["video_note"]

    elif message.get("photo"):
        photos = message.get("photo") or []

        if photos:
            # Highest-resolution photo is normally last.
            file_type = "photo"
            file_object = photos[-1]

    if not file_object:
        return None

    unix_date = message.get("date")

    datetime_utc = ""

    if unix_date:
        try:
            datetime_utc = datetime.fromtimestamp(
                int(unix_date),
                tz=timezone.utc,
            ).isoformat()
        except Exception:
            datetime_utc = ""

    chat = message.get("chat") or {}
    sender = message.get("from") or {}

    return {
        "datetime_utc": datetime_utc,
        "telegram_date": unix_date or "",
        "chat_id": chat.get("id", ""),
        "message_id": message.get("message_id", ""),
        "file_id": file_object.get("file_id", ""),
        "file_unique_id": file_object.get(
            "file_unique_id",
            "",
        ),
        "file_type": file_type,
        "file_name": file_object.get(
            "file_name",
            "",
        ),
        "mime_type": file_object.get(
            "mime_type",
            "",
        ),
        "file_size": file_object.get(
            "file_size",
            "",
        ),
        "caption": message.get(
            "caption",
            "",
        ),
        "text": message.get(
            "text",
            "",
        ),
        "from_id": sender.get(
            "id",
            "",
        ),
        "from_username": sender.get(
            "username",
            "",
        ),
    }


def append_telegram_metadata(
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Add one Telegram file record to telegram-storage.

    Idempotency:
        the same Telegram chat_id + message_id will not be inserted twice.

    Every genuinely new record gets its own storage_id.
    New records start with used blank.
    """

    metadata = dict(metadata)

    service = get_sheets_service()

    rows = read_sheet_by_name(
        service,
        TELEGRAM_STORAGE_SHEET,
    )

    if not rows:
        raise ValueError(
            f"{TELEGRAM_STORAGE_SHEET} sheet has no header row"
        )

    header = [
        str(column).strip()
        for column in rows[0]
    ]

    required_columns = {
        "storage_id",
        "chat_id",
        "message_id",
        "file_id",
        "used",
    }

    missing = required_columns - set(header)

    if missing:
        raise ValueError(
            f"{TELEGRAM_STORAGE_SHEET} sheet is missing required column(s): "
            + ", ".join(sorted(missing))
        )

    chat_idx = header.index("chat_id")
    message_idx = header.index("message_id")

    incoming_chat_id = str(
        metadata.get("chat_id") or ""
    ).strip()

    incoming_message_id = str(
        metadata.get("message_id") or ""
    ).strip()

    #
    # Prevent duplicate indexing of the same Telegram message.
    #
    for existing_row in rows[1:]:
        existing_row = list(existing_row)

        if len(existing_row) < len(header):
            existing_row.extend(
                [""] * (len(header) - len(existing_row))
            )

        existing_chat_id = str(
            existing_row[chat_idx] or ""
        ).strip()

        existing_message_id = str(
            existing_row[message_idx] or ""
        ).strip()

        if (
            existing_chat_id == incoming_chat_id
            and existing_message_id == incoming_message_id
        ):
            existing_metadata = {
                header[index]: existing_row[index]
                for index in range(len(header))
            }

            logger.info(
                "Telegram message already indexed: chat_id=%s message_id=%s",
                incoming_chat_id,
                incoming_message_id,
            )

            return existing_metadata

    #
    # Only genuinely new items receive a new storage_id.
    #
    # These fields are owned exclusively by the storage backend.
    metadata["storage_id"] = str(uuid.uuid4())

    # Every genuinely new record always starts unused.
    metadata["used"] = ""

    row = [
        metadata.get(column, "")
        for column in header
    ]

    append_rows(
        service,
        TELEGRAM_STORAGE_SHEET,
        [row],
    )

    return metadata


def extract_message_from_update(
    update: dict[str, Any],
) -> dict[str, Any] | None:
    """
    n8n Telegram Trigger can return either the Telegram update object
    or the message itself depending on configuration.
    """

    if not update:
        return None

    if update.get("message"):
        return update["message"]

    if update.get("channel_post"):
        return update["channel_post"]

    if update.get("edited_message"):
        return update["edited_message"]

    if update.get("edited_channel_post"):
        return update["edited_channel_post"]

    # Already looks like a Telegram message.
    if update.get("message_id") and update.get("chat"):
        return update

    return None

def telegram_item_is_used(
    value: Any,
) -> bool:
    """
    Interpret the telegram-storage 'used' column.

    Blank = unused.

    These values mean used:
        used
        true
        yes
        y
        1
    """

    normalized = str(
        value or ""
    ).strip().lower()

    return normalized in {
        "used",
        "true",
        "yes",
        "y",
        "1",
    }

def get_telegram_storage_rows() -> list[dict[str, Any]]:
    service = get_sheets_service()

    rows = read_sheet_by_name(
        service,
        TELEGRAM_STORAGE_SHEET,
    )

    if not rows:
        return []

    header = [
        str(column).strip()
        for column in rows[0]
    ]

    results = []

    for row_number, row in enumerate(
        rows[1:],
        start=2,
    ):
        row = list(row)

        if len(row) < len(header):
            row.extend(
                [""] * (len(header) - len(row))
            )

        record = {
            header[index]: row[index]
            for index in range(len(header))
        }

        record["_row_number"] = row_number

        results.append(record)

    return results

def require_telegram_storage_key(
    x_api_key: str | None,
) -> None:
    if not TELEGRAM_STORAGE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_STORAGE_API_KEY is not configured",
        )

    if not x_api_key or x_api_key != TELEGRAM_STORAGE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing/invalid X-API-KEY",
        )

def get_unused_telegram_item(
    storage_id: str,
) -> dict[str, Any]:
    storage_id = str(storage_id or "").strip()

    if not storage_id:
        raise HTTPException(
            status_code=400,
            detail="storage_id is required",
        )

    rows = get_telegram_storage_rows()

    item = next(
        (
            row
            for row in rows
            if str(row.get("storage_id") or "").strip() == storage_id
        ),
        None,
    )

    if item is None:
        raise HTTPException(
            status_code=404,
            detail="Telegram storage item not found",
        )

    if telegram_item_is_used(item.get("used")):
        raise HTTPException(
            status_code=409,
            detail="Telegram storage item has already been used",
        )

    file_id = str(
        item.get("file_id") or ""
    ).strip()

    if not file_id:
        raise HTTPException(
            status_code=500,
            detail="Telegram storage item has no file_id",
        )

    return item


def sign_telegram_media_url(
    storage_id: str,
    expires: int,
) -> str:
    if not TELEGRAM_MEDIA_SIGNING_SECRET:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_MEDIA_SIGNING_SECRET is not configured",
        )

    message = f"{storage_id}:{expires}".encode("utf-8")

    return hmac.new(
        TELEGRAM_MEDIA_SIGNING_SECRET.encode("utf-8"),
        message,
        hashlib.sha256,
    ).hexdigest()

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/")
async def home():
    return {"message": "Python server is up!", "time": datetime.now().isoformat()}

@app.get("/test/square")
async def square_endpoint(x: float = -12):
    return {"x": x, "square": x * x}

@app.get("/b2/refresh-auth")
async def b2_refresh_auth(
    file_name: str = Query(..., description="Exact B2 file name, e.g. videos/123_x.mp4"),
    bucket_name: str = Query(None, description="Optional; defaults to BACKBLAZE_BUCKET_NAME"),
    valid_seconds: int = Query(604800, ge=1, le=604800),
    x_api_key: str | None = Header(default=None, alias="X-API-KEY"),
):
    require_refresh_key(x_api_key)

    manager = get_b2_manager()
    loop = asyncio.get_running_loop()

    # mint token
    token = await loop.run_in_executor(
        None,
        manager.get_download_authorization_token,
        file_name,
        bucket_name,
        valid_seconds,
    )

    bucket = bucket_name or manager.default_bucket_name
    url = manager.build_authorized_url(bucket, file_name, token)
    return {
        "bucket": bucket,
        "file_name": file_name,
        "valid_seconds": valid_seconds,
        "token": token,
        "url": url,
    }

@app.get("/yt/videos", response_model=List[YouTubeVideoResponse])
async def get_youtube_videos(
    handle: str = Query(
        ...,
        description="YouTube handle (e.g. '@username')",
    ),
    account: Optional[str] = Query(
        None,
        description=(
            "Target account from streamables. Required when "
            "exclude_existing=true."
        ),
    ),
    creator: Optional[str] = Query(
        None,
        description=(
            "Creator value used in streamables. "
            "Defaults to the YouTube handle."
        ),
    ),
    sort_by: str = Query(
        "newest",
        description="Sort by 'newest', 'relevance', or 'engagement'",
    ),
    max_results: int = Query(
        50,
        ge=1,
        le=50,
        description="Maximum number of videos to return",
    ),
    only_shorts: Optional[bool] = Query(
        None,
        description=(
            "true: retrieve actual videos from the channel's Shorts tab; "
            "false: preserve existing non-short-form (>180 sec) behavior; "
            "omit: return recent channel videos"
        ),
    ),
    exclude_existing: bool = Query(
        True,
        description=(
            "When true, exclude videos whose YouTube video ID is already "
            "present in the selected sheet_name for this exact "
            "account + creator pair."
        ),
    ),
    sheet_name: Literal["streamables", "streamables-yt"] = Query(
        "streamables",
        description="Google Sheets tab used for dedupe and optional appending",
    ),
    append_to_sheet: bool = Query(
        True,
        description="Append the final returned videos to sheet_name",
    ),
):
    """
    Fetch recent YouTube videos.

    When only_shorts=true, actual membership of the channel's /shorts tab
    is used rather than inferring Shorts from video duration.

    When exclude_existing=true, videos already represented by yt_link in
    the streamables sheet for the exact account + creator pair are removed.
    """

    try:
        handler = YouTubeHandler()

        normalized_handle = (
            str(handle or "")
            .strip()
            .removeprefix("@")
            .lower()
        )

        normalized_creator = (
            str(creator or normalized_handle)
            .strip()
            .removeprefix("@")
            .lower()
        )

        normalized_account = (
            str(account or "")
            .strip()
            .removeprefix("@")
            .lower()
        )

        if exclude_existing and not normalized_account:
            raise ValueError(
                "account is required when exclude_existing=true"
            )

        #
        # If we're going to remove existing Shorts, inspect more than the
        # requested output count so max_results=10 can still return 10 unseen
        # Shorts even when some of the newest Shorts are already in the sheet.
        #
        fetch_count = max_results

        if only_shorts is True and exclude_existing:
            fetch_count = min(
                max(max_results * 5, 100),
                250,
            )

        videos = await handler.get_videos_by_handle(
            handle=handle,
            sort_by=sort_by,
            max_results=fetch_count,
            only_shorts=only_shorts,
        )

        service = None
        rows = []
        header = []

        #
        # We only need Sheets when:
        #   1. deduping against existing rows, or
        #   2. appending the final returned rows.
        #
        if exclude_existing or append_to_sheet:
            service = get_sheets_service()

            rows = read_sheet_by_name(
                service,
                sheet_name,
            )

            if rows:
                header = [
                    str(column).strip()
                    for column in rows[0]
                ]

        #
        # ---------------------------------------------------------
        # DEDUPE
        # ---------------------------------------------------------
        #
        # IMPORTANT:
        # Dedupe ONLY against the selected sheet_name.
        #
        if exclude_existing:
            required_columns = {
                "account",
                "creator_handle",
                "yt_link",
            }

            if not rows:
                raise ValueError(
                    f"{sheet_name} sheet has no header row"
                )

            missing_columns = required_columns - set(header)

            if missing_columns:
                raise ValueError(
                    f"{sheet_name} sheet is missing required column(s): "
                    + ", ".join(sorted(missing_columns))
                )

            account_idx = header.index("account")
            creator_idx = header.index("creator_handle")
            yt_link_idx = header.index("yt_link")

            existing_video_ids = set()

            for row in rows[1:]:
                row = list(row)

                # Safely accommodate short/incomplete sheet rows.
                if len(row) < len(header):
                    row.extend(
                        [""] * (len(header) - len(row))
                    )

                row_account = (
                    str(row[account_idx] or "")
                    .strip()
                    .removeprefix("@")
                    .lower()
                )

                row_creator = (
                    str(row[creator_idx] or "")
                    .strip()
                    .removeprefix("@")
                    .lower()
                )

                #
                # Dedupe only inside this exact:
                #
                #     sheet_name + account + creator_handle
                #
                if (
                    row_account != normalized_account
                    or row_creator != normalized_creator
                ):
                    continue

                yt_link = str(
                    row[yt_link_idx] or ""
                ).strip()

                video_id = handler.extract_video_id(
                    yt_link
                )

                if video_id:
                    existing_video_ids.add(video_id)

            final_videos = [
                video
                for video in videos
                if video.video_id not in existing_video_ids
            ][:max_results]

        else:
            final_videos = videos[:max_results]

        #
        # ---------------------------------------------------------
        # APPEND FINAL RETURNED VIDEOS
        # ---------------------------------------------------------
        #
        # Only the records that are actually being returned are appended.
        #
        if append_to_sheet and final_videos:
            if service is None:
                service = get_sheets_service()

            if not header:
                rows = read_sheet_by_name(
                    service,
                    sheet_name,
                )

                if not rows:
                    raise ValueError(
                        f"{sheet_name} sheet has no header row"
                    )

                header = [
                    str(column).strip()
                    for column in rows[0]
                ]

            append_supported_columns = {
                "account",
                "creator_handle",
                "yt_link",
                "caption",
                "video_id",
                "title",
                "published_at",
                "view_count",
                "like_count",
                "dislike_count",
                "comment_count",
                "thumbnail_url",
                "channel_title",
                "relevance_score",
                "engagement_score",
            }

            mapped_columns = append_supported_columns.intersection(header)

            if not mapped_columns:
                raise ValueError(
                    f"{sheet_name} sheet contains no columns "
                    "that /yt/videos knows how to populate"
                )

            append_values = []

            for video in final_videos:
                #
                # Canonical values available for mapping.
                #
                # Direct response-model fields are retained, while the
                # contextual/semantic sheet aliases make this compatible
                # with the existing streamables layout.
                #
                video_values = {
                    # Context
                    "account": normalized_account,
                    "creator_handle": normalized_creator,

                    # Existing streamables-compatible aliases
                    "yt_link": (
                        f"https://www.youtube.com/watch?v={video.video_id}"
                    ),
                    "caption": video.title,

                    # Raw /yt/videos response fields
                    "video_id": video.video_id,
                    "title": video.title,
                    "published_at": video.published_at,
                    "view_count": video.view_count,
                    "like_count": video.like_count,
                    "dislike_count": video.dislike_count,
                    "comment_count": video.comment_count,
                    "thumbnail_url": video.thumbnail_url,
                    "channel_title": video.channel_title,
                    "relevance_score": video.relevance_score,
                    "engagement_score": video.engagement_score,
                }

                #
                # Build the row in EXACT sheet-column order.
                #
                # Columns not represented by the endpoint are intentionally
                # left blank rather than guessed.
                #
                append_values.append([
                    video_values.get(column, "")
                    for column in header
                ])

            append_rows(
                service,
                sheet_name,
                append_values,
            )

            logger.info(
                "Appended %d /yt/videos result(s) to sheet '%s'",
                len(append_values),
                sheet_name,
            )

        return final_videos

    except ValueError as error:
        logger.error(
            "Error fetching YouTube videos: %s",
            error,
        )
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        logger.exception(
            "Unexpected error fetching YouTube videos"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch videos: {error}",
        )

@app.get("/b2/signed-url")
async def get_b2_signed_url(filename: str = "yt_video.mp4") -> Dict[str, str]:
    """
    Generate a signed upload URL for B2 (placeholder - B2 uses direct auth).
    For B2, you typically upload directly with credentials.
    This endpoint is kept for API compatibility but returns info message.
    """
    if not BACKBLAZE_KEY_ID or not BACKBLAZE_APPLICATION_KEY or not BACKBLAZE_BUCKET_NAME:
            raise HTTPException(500, "B2 not configured")

    return {
        "message": "B2 uses direct authentication. Use /sheets/process_streamables endpoint.",
        "bucket": BACKBLAZE_BUCKET_NAME,
        "filename": filename,
        "note": "B2 SDK handles authentication automatically with BACKBLAZE_KEY_ID and BACKBLAZE_APPLICATION_KEY"
    }

@app.get("/video-to-fb", response_model=VideoToFBResponse)
async def video_to_fb(
    video_url: str = Query(...),
    fb_description: str = Query("")
) -> Dict[str, Any]:  # FastAPI will convert to the response_model anyway
    mp4_path: Path | None = None
    try:
        loop = asyncio.get_running_loop()
        mp4_path = await loop.run_in_executor(None, get_streamable_mp4, video_url)

        # At this point, mp4_path is guaranteed to be Path (not None)
        # because get_streamable_mp4 either returns a Path or raises an exception

        fb_result = await loop.run_in_executor(
            None, upload_to_facebook, mp4_path, fb_description
        )

        return {
            "success": True,
            "facebook": fb_result,
            "local_file": str(mp4_path),
            "size_mb": round(mp4_path.stat().st_size / (1024 * 1024), 2),
        }
    finally:
        if mp4_path and mp4_path.exists():
            cleanup_video_file(mp4_path)

@app.post("/b2/delete")
async def delete_b2_resource(body: B2DeleteRequest):
    """
    Delete a file from Backblaze B2 by filename.
    
    Request body:
    {
        "file_name": "videos/123456_video.mp4",
        "bucket_name": "my-bucket"  // optional if BACKBLAZE_BUCKET_NAME is set
    }
    """
    try:
        loop = asyncio.get_running_loop()
        manager = get_b2_manager()
        
        result = await loop.run_in_executor(
            None,
            manager.delete_file,
            body.file_name,
            body.bucket_name
        )
        
        return {
            "success": result,
            "file_name": body.file_name,
            "message": "File deleted successfully" if result else "File not found"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /b2/delete")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/b2/delete")
async def delete_b2_resource_get(
    file_name: str = Query(..., description="B2 file name to delete"),
    bucket_name: str = Query(None, description="Bucket name (optional if env var set)")
):
    """
    Delete a file from B2 (GET method for convenience).
    
    Example: /b2/delete?file_name=videos/123456_video.mp4
    """
    try:
        loop = asyncio.get_running_loop()
        manager = get_b2_manager()
        
        result = await loop.run_in_executor(
            None,
            # manager.delete_file,
            manager.soft_delete_file,
            file_name,
            bucket_name
        )
        
        return {
            "success": result,
            "file_name": file_name,
            "message": "File deleted successfully" if result else "File not found"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /b2/delete")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.post("/n8n/local/deactivate_all")
async def deactivate_all():
    workflows = get_all_workflows()
    results = []
    for wf in workflows:
        r = deactivate_workflow(wf)
        results.append({"workflow": wf["name"], "id": wf["id"], "status": r.status_code})
    return {"action": "deactivate_all", "count": len(results), "results": results}

@app.post("/n8n/local/activate")
async def activate_selected(workflows: str = Query(None)):
    names_to_activate = [x.strip() for x in workflows.split(",")] if workflows else DEFAULT_WORKFLOWS
    all_wfs = get_all_workflows()

    matched = [wf for wf in all_wfs if wf["name"] in names_to_activate]
    unmatched = [n for n in names_to_activate if n not in [wf["name"] for wf in all_wfs]]

    for wf in all_wfs:
        deactivate_workflow(wf)

    activated = []
    for wf in matched:
        r = activate_workflow(wf)
        activated.append({"workflow": wf["name"], "id": wf["id"], "status": r.status_code})

    return {
        "action": "activate_selected",
        "requested": names_to_activate,
        "activated_count": len(activated),
        "activated": activated,
        "unmatched": unmatched,
    }

@app.post("/post_tumblr")
async def post_tumblr_image(body: TumblrPostRequest):
    oauth, blog_id, account = pick_tumblr_account(body.tumblr_account)
    caption = (body.caption or "").strip()
    try:
        logger.info("Using blog_id=%s, image_url=%s, caption=%s", blog_id, body.image_url, body.caption)
        resp = requests.post(
            f"https://api.tumblr.com/v2/blog/{blog_id}/post",
            data = {
                "type": "photo",
                "source": body.image_url,
                "caption": caption
            },
            auth=oauth,
            timeout=120,
        )
        resp.raise_for_status()
        logger.info("Posted to Tumblr (%s)", account)
        return {"message": "Posted to Tumblr", "account": account}
    except Exception as e:
        logger.exception("Tumblr post failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/post_image")
async def post_image_to_x(
    image_url: str = Body(...),
    text: str = Body(""),
    x_api_key: str | None = Header(None, alias="X-API-Key"),
):
    # Legacy compatibility route. New callers should use /x/publish.
    # It is intentionally protected now; the old route used to be public.
    require_x_internal_key(x_api_key)
    try:
        img_resp = requests.get(image_url, stream=True, timeout=90)
        img_resp.raise_for_status()

        files = {"media": ("image.jpg", img_resp.raw, img_resp.headers.get("Content-Type", "image/jpeg"))}
        upload = requests.post(
            "https://upload.twitter.com/1.1/media/upload.json",
            auth=oauth_x,
            files=files,
            data={"media_category": "tweet_image"},
            timeout=180,
        )
        upload.raise_for_status()
        media_id = upload.json()["media_id_string"]

        tweet = requests.post(
            "https://api.x.com/2/tweets",
            auth=oauth_x,
            json={"text": text, "media": {"media_ids": [media_id]}},
            timeout=180,
        )
        tweet.raise_for_status()
        logger.info("Posted to X successfully")
        return {"message": "Posted to X", "tweet": tweet.json()}
    except Exception as e:
        logger.exception("X post failed")
        raise HTTPException(status_code=500, detail=str(e))

# Google Sheets Endpoints
@app.post("/sheets/process_streamables")
async def process_streamables_endpoint(
    acc_name: str = Query("dave.commercial7"),
    sheet_name: str = Query("streamables"),
    k: int = Query(10, ge=1, le=100, description="Max rows to process"),
    max_concurrency: int = Query(2, ge=1, le=5, description="Parallel jobs limit"),
    dry_run: bool = Query(False, description="If true, no download/upload/sheet update"),
):
    # concurrency helper
    semaphore = asyncio.Semaphore(max_concurrency)
    event_loop = asyncio.get_running_loop()

    service = get_sheets_service()
    rows = read_sheet_by_name(service, sheet_name)
    if not rows:
        return {"success": False, "message": "No data"}

    header = rows[0]

    filtered = filter_status_rows(rows, ['staged', 'uploaded'], exclude_status=True)
    filtered = [r for r in filtered if r.get("account") == acc_name]

    total_candidates = len(filtered)
    filtered = filtered[:k]   # hard limit BEFORE any download/upload

    # Column letters
    status_col = get_column_letter(header, "status")
    url_col = get_column_letter(header, "url")

    # Caption column (safe handling in case your helper throws)
    try:
        caption_col = get_column_letter(header, "caption")
    except Exception:
        caption_col = None
        logger.warning("No 'caption' column found in sheet header; skipping caption writes.")

    # One handler instance shared across tasks
    yt_handler = YouTubeHandler()

    async def process_row(row):
        async with semaphore:
            row_num = row["row_number"]
            yt_link = (row.get("yt_link") or "").strip()

            if not yt_link:
                return {"row_number": row_num, "status": "error", "error": "No yt_link"}

            if dry_run:
                title_preview = None
                try:
                    title_preview = await yt_handler.get_video_title(yt_link)
                except Exception as e:
                    title_preview = None
                    logger.warning("Dry-run: could not fetch title for row %s: %s", row_num, e)

                return {
                    "row_number": row_num,
                    "status": "dry_run",
                    "yt_link": yt_link,
                    "caption_preview": title_preview,
                }

            mp4_path = None
            try:
                # 0) Fetch title first
                title = await yt_handler.get_video_title(yt_link)

                # 1) Download
                mp4_path = await event_loop.run_in_executor(None, get_streamable_mp4, yt_link)

                # 2) Upload
                b2_url = await event_loop.run_in_executor(None, upload_to_b2, mp4_path, yt_link)

                # 3) Sheet updates
                if caption_col is not None:
                    update_cell(service, sheet_name, row_num, caption_col, title)

                update_cell(service, sheet_name, row_num, status_col, "staged")
                update_cell(service, sheet_name, row_num, url_col, b2_url)

                return {
                    "row_number": row_num,
                    "status": "success",
                    "caption": title,
                    "url": b2_url,
                }

            except Exception as e:
                logger.exception("Failed on row %s", row_num)
                return {
                    "row_number": row_num,
                    "status": "error",
                    "error": str(e),
                }

            finally:
                if mp4_path:
                    cleanup_video_file(mp4_path)

    tasks = [process_row(row) for row in filtered]
    results = await asyncio.gather(*tasks)

    return {
        "success": True,
        "dry_run": dry_run,
        "requested_k": k,
        "max_concurrency": max_concurrency,
        "total_candidates": total_candidates,
        "processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results,
    }

# Kite Endpoints

@app.get("/kite/login_url")
async def kite_login_url():
    return {"login_url": kite_manager.login_url()}

@app.get("/kite/callback")
async def kite_callback(request_token: str = None):
    if not request_token:
        raise HTTPException(400, "request_token required")
    session = kite_manager.generate_session(request_token)
    return {"message": "Kite session created", "access_token": session.get("access_token")}

@app.post("/kite/generate_session")
async def kite_generate_session(body: KiteGenerateSessionRequest):
    session = kite_manager.generate_session(body.request_token)
    return {"message": "session_generated", "session": session}

@app.post("/kite/set_token")
async def kite_set_token(body: KiteSetTokenRequest):
    kite_manager.set_access_token(body.access_token)
    return {"message": "access_token_set"}

@app.post("/kite/ltp")
async def kite_ltp(body: KiteLTPRequest):
    if not kite_manager.has_token():
        raise HTTPException(401, "No valid Kite access token. Go to /kite/force_refresh and re-authenticate.")
    try:
        return kite_manager.ltp(body.symbols)
    except Exception as e:
        if "access_token" in str(e):
            raise HTTPException(401, "Invalid/expired Kite token. Use /kite/force_refresh to get new login URL.")
        raise

@app.post("/kite/candles")
async def kite_candles(body: KiteCandlesRequest):
    if not kite_manager.has_token():
        raise HTTPException(401, "No Kite access token. Use /kite/force_refresh")
    try:
        to_date = datetime.utcnow()
        from_date = to_date - timedelta(days=body.days)
        data = kite_manager.historical_data(body.instrument_token, from_date, to_date, body.interval)
        return {"count": len(data), "candles": data}
    except Exception as e:
        if "access_token" in str(e):
            raise HTTPException(401, "Kite token invalid. Use /kite/force_refresh")
        raise

@app.get("/kite/debug")
async def kite_debug():
    return {
        "has_token": kite_manager.has_token(),
        "token_preview": kite_manager._access_token[:20] + "..." if kite_manager._access_token else None,
        "api_key": os.getenv("KITE_API_KEY"),
        "api_secret_length": len(os.getenv("KITE_API_SECRET", "")),
        "api_secret_preview": os.getenv("KITE_API_SECRET", "")[:8] + "..." if os.getenv("KITE_API_SECRET") else None,
        "env_token_set": bool(os.getenv("KITE_ACCESS_TOKEN", "").strip())
    }

@app.post("/kite/force_refresh")
async def force_kite_refresh():
    with kite_manager._lock:
        kite_manager._access_token = None
        kite_manager._kite.set_access_token(None)
    logger.warning("Kite token cleared from memory. Ready for fresh login.")
    return {
        "status": "token_cleared",
        "message": "Old token removed. Use the login_url below to authenticate again.",
        "login_url": kite_manager.login_url()
    }

@app.post("/telegram/store")
async def telegram_store(
    body: TelegramStoreRequest,
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Store a remote file inside the requested Telegram chat/group
    and index it in the fixed telegram-storage Google Sheet.
    """
    require_telegram_storage_key(x_api_key)

    try:
        require_telegram_config()

        source_url = (
            body.source_url or ""
        ).strip()

        if not source_url:
            raise HTTPException(
                status_code=400,
                detail="source_url is required",
            )

        caption = (
            body.caption or ""
        ).strip()

        telegram_message = telegram_api_call(
            "sendDocument",
            {
                "chat_id": str(body.chat_id),
                "document": source_url,
                "caption": caption[:1024],
            },
            timeout=300,
        )

        metadata = extract_telegram_file(
            telegram_message
        )

        if not metadata:
            raise RuntimeError(
                "Telegram accepted the message but returned no file metadata"
            )

        metadata["source_url"] = source_url

        if body.file_name:
            metadata["source_file_name"] = (
                body.file_name
            )

        if body.text:
            metadata["text"] = body.text

        if body.metadata:
            for key, value in body.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        metadata = append_telegram_metadata(
            metadata
        )

        return {
            "success": True,
            "storage_id": metadata["storage_id"],
            "chat_id": metadata["chat_id"],
            "message_id": metadata["message_id"],
            "file_id": metadata["file_id"],
            "file_unique_id": metadata.get(
                "file_unique_id"
            ),
            "used": metadata.get("used", ""),
            "telegram": metadata,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Telegram storage failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Telegram storage failed: {exc}",
        )

@app.post("/telegram/index-message")
async def telegram_index_message(
    body: TelegramIndexMessageRequest,
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Index a Telegram Trigger/message in telegram-storage.

    The chat_id is taken directly from the Telegram message.
    """
    require_telegram_storage_key(x_api_key)

    try:
        message = extract_message_from_update(
            body.update
        )

        if not message:
            raise HTTPException(
                status_code=400,
                detail="No Telegram message found in update",
            )

        metadata = extract_telegram_file(
            message
        )

        if not metadata:
            return {
                "success": True,
                "indexed": False,
                "reason": "Message contains no supported file",
            }

        if body.metadata:
            for key, value in body.metadata.items():
                if key not in metadata:
                    metadata[key] = value

        metadata = append_telegram_metadata(
            metadata
        )

        return {
            "success": True,
            "indexed": True,
            "storage_id": metadata["storage_id"],
            "telegram": metadata,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Failed indexing Telegram message"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Telegram indexing failed: {exc}",
        )

@app.get("/telegram/file-info")
async def telegram_file_info(
    storage_id: str = Query(...),
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Return Telegram file information for one unused storage item.
    """
    require_telegram_storage_key(x_api_key)

    try:
        rows = get_telegram_storage_rows()

        record = next(
            (
                row
                for row in rows
                if str(
                    row.get("storage_id") or ""
                ).strip() == storage_id.strip()
            ),
            None,
        )

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram storage item not found",
            )

        if telegram_item_is_used(
            record.get("used")
        ):
            raise HTTPException(
                status_code=404,
                detail="Telegram storage item is already marked as used",
            )

        file_id = str(
            record.get("file_id") or ""
        ).strip()

        if not file_id:
            raise RuntimeError(
                "Stored record contains no Telegram file_id"
            )

        result = telegram_api_call(
            "getFile",
            {
                "file_id": file_id,
            },
        )

        return {
            "success": True,
            "storage_id": storage_id,
            "chat_id": record.get("chat_id"),
            "message_id": record.get("message_id"),
            "file_id": file_id,
            "file_unique_id": result.get(
                "file_unique_id"
            ),
            "file_size": result.get(
                "file_size"
            ),
            "file_path": result.get(
                "file_path"
            ),
            "file_name": record.get(
                "file_name"
            ),
            "mime_type": record.get(
                "mime_type"
            ),
            "used": False,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Telegram getFile failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

@app.get("/telegram/file")
async def telegram_get_file(
    storage_id: str = Query(...),
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Retrieve one unused Telegram file using our own storage_id.

    Used items are intentionally not retrievable through this endpoint.
    """
    require_telegram_storage_key(x_api_key)

    upstream = None

    try:
        rows = get_telegram_storage_rows()

        record = next(
            (
                row
                for row in rows
                if str(
                    row.get("storage_id") or ""
                ).strip() == storage_id.strip()
            ),
            None,
        )

        if not record:
            raise HTTPException(
                status_code=404,
                detail="Telegram storage item not found",
            )

        if telegram_item_is_used(
            record.get("used")
        ):
            raise HTTPException(
                status_code=404,
                detail="Telegram storage item is already marked as used",
            )

        file_id = str(
            record.get("file_id") or ""
        ).strip()

        if not file_id:
            raise RuntimeError(
                "Stored record contains no Telegram file_id"
            )

        file_info = telegram_api_call(
            "getFile",
            {
                "file_id": file_id,
            },
        )

        file_path = file_info.get(
            "file_path"
        )

        if not file_path:
            raise RuntimeError(
                "Telegram returned no file_path"
            )

        telegram_url = (
            f"{TELEGRAM_FILE_API_BASE}/{file_path}"
        )

        upstream = requests.get(
            telegram_url,
            stream=True,
            timeout=300,
        )

        upstream.raise_for_status()

        content_type = upstream.headers.get(
            "Content-Type",
            record.get("mime_type")
            or "application/octet-stream",
        )

        response_headers = {
            "X-Telegram-Storage-ID": storage_id,
        }

        content_length = upstream.headers.get(
            "Content-Length"
        )

        if content_length:
            response_headers[
                "Content-Length"
            ] = content_length

        file_name = str(
            record.get("file_name") or ""
        ).strip()

        if file_name:
            safe_name = file_name.replace(
                '"',
                "",
            )

            response_headers[
                "Content-Disposition"
            ] = f'inline; filename="{safe_name}"'

        def stream_file():
            try:
                for chunk in upstream.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        yield chunk
            finally:
                upstream.close()

        return StreamingResponse(
            stream_file(),
            media_type=content_type,
            headers=response_headers,
        )

    except HTTPException:
        raise

    except Exception as exc:
        if upstream is not None:
            upstream.close()

        logger.exception(
            "Telegram file retrieval failed"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Telegram retrieval failed: {exc}",
        )

@app.get("/telegram/items")
async def telegram_get_items(
    chat_id: int = Query(...),
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Return ALL indexed Telegram items belonging to chat_id
    except those marked as used.

    Blank 'used' cells are treated as unused.
    """

    require_telegram_storage_key(x_api_key)

    try:
        rows = get_telegram_storage_rows()

        items = []

        for row in rows:
            try:
                row_chat_id = int(
                    row.get("chat_id") or 0
                )
            except (TypeError, ValueError):
                continue

            if row_chat_id != chat_id:
                continue

            if telegram_item_is_used(
                row.get("used")
            ):
                continue

            # Internal sheet row shouldn't be exposed.
            clean_row = {
                key: value
                for key, value in row.items()
                if key != "_row_number"
            }

            items.append(clean_row)

        return {
            "success": True,
            "chat_id": chat_id,
            "count": len(items),
            "items": items,
        }

    except Exception as exc:
        logger.exception(
            "Failed retrieving Telegram storage index"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Telegram index retrieval failed: {exc}",
        )

@app.post("/telegram/mark-used")
async def telegram_mark_used(
    body: TelegramMarkUsedRequest,
    x_api_key: str | None = Header(
        None,
        alias="X-API-KEY",
    ),
):
    """
    Mark a Telegram storage item as used.
    """
    require_telegram_storage_key(x_api_key)

    try:
        service = get_sheets_service()

        rows = read_sheet_by_name(
            service,
            TELEGRAM_STORAGE_SHEET,
        )

        if not rows:
            raise HTTPException(
                status_code=404,
                detail="telegram-storage sheet is empty",
            )

        header = [
            str(column).strip()
            for column in rows[0]
        ]

        if "storage_id" not in header:
            raise HTTPException(
                status_code=500,
                detail="telegram-storage has no storage_id column",
            )

        if "used" not in header:
            raise HTTPException(
                status_code=500,
                detail="telegram-storage has no used column",
            )

        storage_idx = header.index(
            "storage_id"
        )

        target_row = None

        for row_number, row in enumerate(
            rows[1:],
            start=2,
        ):
            row = list(row)

            if len(row) <= storage_idx:
                continue

            if str(
                row[storage_idx] or ""
            ).strip() == body.storage_id.strip():
                target_row = row_number
                break

        if target_row is None:
            raise HTTPException(
                status_code=404,
                detail="storage_id not found",
            )

        used_column = get_column_letter(
            header,
            "used",
        )

        update_cell(
            service,
            TELEGRAM_STORAGE_SHEET,
            target_row,
            used_column,
            "yes",
        )

        return {
            "success": True,
            "storage_id": body.storage_id,
            "used": True,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Failed marking Telegram item as used"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Could not mark item used: {exc}",
        )

@app.get("/telegram/source-url")
async def telegram_source_url(
    storage_id: str = Query(...),
    valid_seconds: int = Query(
        600,
        ge=30,
        le=3600,
    ),
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
):
    """
    Mint a short-lived public URL that Cloudinary can fetch.

    No Telegram bot token is exposed.
    No file is downloaded by n8n.
    """
    require_telegram_storage_key(x_api_key)

    # Validate that the item exists and is still unused.
    get_unused_telegram_item(storage_id)

    expires = int(time.time()) + valid_seconds

    signature = sign_telegram_media_url(
        storage_id,
        expires,
    )

    url = (
        f"{PUBLIC_API_BASE}"
        f"/telegram/media-source/{storage_id}"
        f"?expires={expires}"
        f"&signature={signature}"
    )

    return {
        "storage_id": storage_id,
        "expires": expires,
        "valid_seconds": valid_seconds,
        "url": url,
    }

@app.get("/telegram/media-source/{storage_id}")
async def telegram_media_source(
    storage_id: str,
    expires: int = Query(...),
    signature: str = Query(...),
):
    """
    Public, short-lived streaming proxy:

        Telegram -> FastAPI streaming -> Cloudinary

    Nothing is written to disk and the whole file is never loaded into RAM.
    """

    if not TELEGRAM_MEDIA_SIGNING_SECRET:
        raise HTTPException(
            status_code=500,
            detail="TELEGRAM_MEDIA_SIGNING_SECRET is not configured",
        )

    now = int(time.time())

    if expires < now:
        raise HTTPException(
            status_code=403,
            detail="Media URL has expired",
        )

    expected_signature = sign_telegram_media_url(
        storage_id,
        expires,
    )

    if not hmac.compare_digest(
        signature,
        expected_signature,
    ):
        raise HTTPException(
            status_code=403,
            detail="Invalid media signature",
        )

    item = get_unused_telegram_item(
        storage_id,
    )

    try:
        file_info = telegram_api_call(
            "getFile",
            {
                "file_id": item["file_id"],
            },
        )

        file_path = str(
            file_info.get("file_path") or ""
        ).strip()

        if not file_path:
            raise RuntimeError(
                "Telegram getFile returned no file_path"
            )

        telegram_url = (
            f"{TELEGRAM_FILE_API_BASE}/{file_path}"
        )

        client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30,
                read=300,
                write=30,
                pool=30,
            )
        )

        request = client.build_request(
            "GET",
            telegram_url,
        )

        response = await client.send(
            request,
            stream=True,
        )

        if response.status_code != 200:
            body = await response.aread()

            await response.aclose()
            await client.aclose()

            raise RuntimeError(
                "Telegram file download failed: "
                f"{response.status_code} "
                f"{body[:300]!r}"
            )

        async def stream_telegram_file():
            try:
                async for chunk in response.aiter_bytes(
                    chunk_size=64 * 1024
                ):
                    yield chunk

            finally:
                await response.aclose()
                await client.aclose()

        media_type = (
            str(item.get("mime_type") or "").strip()
            or response.headers.get("content-type")
            or "application/octet-stream"
        )

        headers = {
            "Cache-Control": "private, no-store",
            "X-Telegram-Storage-ID": storage_id,
        }

        if response.headers.get("content-length"):
            headers["Content-Length"] = (
                response.headers["content-length"]
            )

        return StreamingResponse(
            stream_telegram_file(),
            media_type=media_type,
            headers=headers,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Telegram media-source failed for storage_id=%s",
            storage_id,
        )

        raise HTTPException(
            status_code=502,
            detail=f"Telegram media proxy failed: {exc}",
        )

# =====================================================
# CRON: Exactly one worker runs this daily at 9:50 AM IST
# =====================================================

def become_cron_master() -> bool:
    worker_id = f"{socket.gethostname()}-{os.getpid()}"
    acquired = redis_client.set(LOCK_NAME, worker_id, nx=True, ex=LOCK_TTL)
    if acquired:
        logger.info("This worker is the CRON MASTER: %s", worker_id)
        return True
    else:
        master = redis_client.get(LOCK_NAME) or "unknown"
        logger.info("Cron master already elected: %s (this worker skipped)", master)
        return False

async def run_daily_activation():
    url = "http://localhost:5000/n8n/local/activate"
    logger.info("Executing daily 9:50 AM n8n workflow activation...")
    try:
        async with httpx.AsyncClient(timeout=180) as client:
            r = await client.post(url)
            if r.status_code == 200:
                logger.info("Daily n8n activation succeeded")
            else:
                logger.error("Daily activation failed: %s %s", r.status_code, r.text)
    except Exception as e:
        logger.exception("Daily activation job crashed: %s", e)

def start_cron_scheduler():
    if not become_cron_master():
        return

    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    scheduler.add_job(
        func=lambda: httpx.AsyncClient().post("http://localhost:5000/n8n/local/activate"),
        trigger=CronTrigger(hour=9, minute=50),
        id="daily_n8n_activate",
        name="Daily n8n workflow activation",
        max_instances=1,
        coalesce=True,
        replace_existing=True,
    )
    scheduler.start()
    logger.info("APScheduler started — this worker will run daily cron at 9:50 AM IST")

# =====================================================
# Worker startup hook (runs in every uvicorn worker)
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
else:
    time.sleep(4)
    try:
        start_cron_scheduler()
    except Exception as e:
        logger.error("Failed to initialize cron master: %s", e)
