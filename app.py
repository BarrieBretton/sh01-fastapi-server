# Imports

# General
import os
import socket
import threading
import logging
import subprocess
import time

from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict

# Basic HTTP Requests
import httpx
import redis
import requests
import asyncio

# FastApi Server
from requests_oauthlib import OAuth1
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Cloudinary
import cloudinary
import cloudinary.uploader
import cloudinary.api

# Kite Zerodha Connectivity
from kiteconnect import KiteConnect, KiteTicker
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Telegram Client (Supports larger video file handling)
from telethon import TelegramClient
from telethon.tl.types import DocumentAttributeFilename, DocumentAttributeVideo
from telethon.errors import FloodWaitError

# =====================================================
# SANITY CHECKS   
# =====================================================

import shutil
import sys

if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
    print("ERROR: FFmpeg or FFprobe not found! Exiting.")
    sys.exit(1)

# =====================================================
# ENV & LOGGING SETUP   
# =====================================================

load_dotenv(".env")

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

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Silence noisy libraries
for noisy in ("urllib3", "requests", "requests_oauthlib", "apscheduler"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger.info("Application starting up...")

# =====================================================
# CONFIG & AUTH
# =====================================================

# Telegram Client's User Session
TELEGRAM_API_ID = os.getenv("TELEGRAM_API_ID")
TELEGRAM_API_HASH = os.getenv("TELEGRAM_API_HASH")
# TELEGRAM_SESSION = "tg_mtproto_session"          # file name

# if TELEGRAM_API_ID and TELEGRAM_API_HASH:
#     tg_client = TelegramClient(TELEGRAM_SESSION, int(TELEGRAM_API_ID), TELEGRAM_API_HASH)
#     tg_client.start()                     # will ask for phone/code only the first run
# else:
#     tg_client = None
#     logger.warning("TELEGRAM_API_ID/HASH missing – MTProto upload disabled")

# Telegram Bot
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = -5071573945

if not TELEGRAM_BOT_TOKEN:
    logger.warning("TELEGRAM_BOT_TOKEN not set in .env - Telegram features will not work")

# FB Page Connectivity
FB_PAGE_ID = os.getenv("FB_PAGE_ID")
FB_PAGE_ACCESS_TOKEN = os.getenv("FB_PAGE_ACCESS_TOKEN")

if not FB_PAGE_ID or not FB_PAGE_ACCESS_TOKEN:
    logger.warning("FB_PAGE_ID or FB_PAGE_ACCESS_TOKEN missing – /video-to-fb will be disabled")

# X/Twitter OAuth1
oauth_x = OAuth1(
    os.getenv("API_KEY"),
    os.getenv("API_KEY_SECRET"),
    os.getenv("ACCESS_TOKEN"),
    os.getenv("ACCESS_TOKEN_SECRET"),
)

# Tumblr accounts
oauth_erika = OAuth1(
    os.getenv("TUMBLR_CONSUMER_KEY_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_CONSUMER_SECRET_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_TOKEN_ERIKA_DEVEREUX"),
    os.getenv("TUMBLR_TOKEN_SECRET_ERIKA_DEVEREUX"),
)
T_EK_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_ERIKA_DEVEREUX")

oauth_vlvt = OAuth1(
    os.getenv("TUMBLR_CONSUMER_KEY_VLVT_AVE"),
    os.getenv("TUMBLR_CONSUMER_SECRET_VLVT_AVE"),
    os.getenv("TUMBLR_TOKEN_VLVT_AVE"),
    os.getenv("TUMBLR_TOKEN_SECRET_VLVT_AVE"),
)
T_VL_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_VLVT_AVE")

oauth_cyootstuff = OAuth1(
    os.getenv("TUMBLR_CONSUMER_KEY_CYOOTSTUFF_AVE"),
    os.getenv("TUMBLR_CONSUMER_SECRET_CYOOTSTUFF_AVE"),
    os.getenv("TUMBLR_TOKEN_CYOOTSTUFF_AVE"),
    os.getenv("TUMBLR_TOKEN_SECRET_CYOOTSTUFF_AVE"),
)
T_CY_BID = os.getenv("TUMBLR_BLOG_IDENTIFIER_CYOOTSTUFF_AVE")

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
    "ai-image-model-4",
    "ig-sidehustle PINTEREST 3",
    "office-hours",
    "__backup",
    "readiness-probe",
    "report-render-cloud-stats",
    "ai-image-model-cyootstuff",
]

# Cloudinary Setup
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET")
CLOUDINARY_UPLOAD_PRESET = os.getenv("CLOUDINARY_UPLOAD_PRESET")  # Optional

if CLOUDINARY_CLOUD_NAME and CLOUDINARY_API_KEY and CLOUDINARY_API_SECRET:
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True
    )
    logger.info("Cloudinary configured successfully")
    if CLOUDINARY_UPLOAD_PRESET:
        logger.info("Upload preset configured: %s", CLOUDINARY_UPLOAD_PRESET)
else:
    logger.warning("Cloudinary credentials not set - Cloudinary features disabled")

# Start

app = FastAPI(title="Python Server - X/Tumblr/Kite/n8n/Telegram", version="2.1")

def pick_tumblr_account(name: str):
    name = (name or "").lower().strip()
    if name == "vlvt.ave":
        return oauth_vlvt, T_VL_BID, "VLVT_AVE"
    if name == "erika.devereux":
        return oauth_erika, T_EK_BID, "ERIKA_DEVEREUX"
    if name == "cyootstuff":
        return oauth_cyootstuff, T_CY_BID, "CYOOTSTUFF"

# Global client
tg_client = None

# async def init_telegram_client():
#     global tg_client
#     if TELEGRAM_API_ID and TELEGRAM_API_HASH:
#         tg_client = TelegramClient(TELEGRAM_SESSION, int(TELEGRAM_API_ID), TELEGRAM_API_HASH)
#         await tg_client.start()  # ← Now properly awaited
#         logger.info("Telegram MTProto client started and authenticated")
#     else:
#         tg_client = None
#         logger.warning("TELEGRAM_API_ID/HASH missing – MTProto upload disabled")

@app.on_event("startup")
async def startup_bot_client():
    global tg_client
    if not TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN missing – MTProto upload disabled")
        return

    if not TELEGRAM_API_ID or not TELEGRAM_API_HASH:
        logger.error("TELEGRAM_API_ID and TELEGRAM_API_HASH required")
        return

    # ONE session file, shared safely
    session_file = Path("bot.session")
    tg_client = TelegramClient(
        str(session_file),  # Use file path
        int(TELEGRAM_API_ID),
        TELEGRAM_API_HASH
    )

    # First worker creates session, others reuse
    await tg_client.start(bot_token=TELEGRAM_BOT_TOKEN)

    me = await tg_client.get_me()
    logger.info(f"Telegram Bot logged in as @{me.username}")

# =====================================================
# CLOUDINARY HELPER
# =====================================================

def upload_to_cloudinary(video_path: Path, resource_type: str = "video") -> dict:
    """
    Upload a video to Cloudinary and return the URL and public_id.
    
    Args:
        video_path: Path to the video file
        resource_type: Type of resource (default: "video")
    
    Returns:
        dict with keys: public_id, url, secure_url, duration, format
    
    Raises:
        HTTPException: If upload fails
    """
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
        raise HTTPException(
            status_code=500, 
            detail="Cloudinary not configured. Set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET"
        )
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    
    try:
        logger.info("Uploading to Cloudinary: %s (%.2f MB)", video_path.name, file_size_mb)
        
        # Upload with optimization settings
        response = cloudinary.uploader.upload(
            str(video_path),
            resource_type=resource_type,
            folder="telegram_videos",  # Organize in a folder
            overwrite=True,
            invalidate=True,
            # Optional: Add transformations or settings
            # quality="auto",
            # fetch_format="auto"
        )
        
        logger.info(
            "Cloudinary upload successful. public_id: %s, url: %s",
            response.get("public_id"),
            response.get("secure_url")
        )
        
        return {
            "public_id": response.get("public_id"),
            "url": response.get("url"),
            "secure_url": response.get("secure_url"),
            "duration": response.get("duration"),
            "format": response.get("format"),
            "bytes": response.get("bytes"),
            "created_at": response.get("created_at")
        }
        
    except cloudinary.exceptions.Error as e:
        logger.error("Cloudinary upload failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Cloudinary upload failed: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during Cloudinary upload")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def delete_from_cloudinary(public_id: str, resource_type: str = "video") -> dict:
    """
    Delete a resource from Cloudinary by public_id.
    
    Args:
        public_id: The Cloudinary public_id of the resource
        resource_type: Type of resource (default: "video")
    
    Returns:
        dict with deletion status
    
    Raises:
        HTTPException: If deletion fails or resource not found
    """
    if not all([CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET]):
        raise HTTPException(
            status_code=500,
            detail="Cloudinary not configured"
        )
    
    try:
        logger.info("Attempting to delete from Cloudinary: public_id=%s", public_id)
        
        response = cloudinary.uploader.destroy(
            public_id,
            resource_type=resource_type,
            invalidate=True
        )
        
        result = response.get("result")
        
        if result == "ok":
            logger.info("Successfully deleted from Cloudinary: %s", public_id)
            return {
                "success": True,
                "message": f"Resource '{public_id}' deleted successfully",
                "public_id": public_id,
                "result": result
            }
        elif result == "not found":
            logger.warning("Resource not found on Cloudinary: %s", public_id)
            return {
                "success": False,
                "message": f"Resource '{public_id}' not found on Cloudinary",
                "public_id": public_id,
                "result": result
            }
        else:
            logger.error("Unexpected Cloudinary deletion result: %s", result)
            return {
                "success": False,
                "message": f"Unexpected result: {result}",
                "public_id": public_id,
                "result": result
            }
            
    except cloudinary.exceptions.Error as e:
        logger.error("Cloudinary deletion failed: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Cloudinary deletion failed: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error during Cloudinary deletion")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

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
# TELEGRAM HELPER
# =====================================================

def upload_video_to_telegram(
    video_path: Path, 
    chat_id: int = TELEGRAM_CHAT_ID,
    max_retries: int = 3,
    chunk_timeout: int = 600  # 10 minutes
) -> str:
    """
    Uploads a video file to Telegram with retry logic and proper error handling.
    
    Args:
        video_path: Path to the video file
        chat_id: Telegram chat ID (default: -5071573945)
        max_retries: Number of retry attempts (default: 3)
        chunk_timeout: Timeout in seconds for upload (default: 600)
    
    Returns:
        file_id of the uploaded video
    
    Raises:
        HTTPException: If upload fails after all retries
    """
    if not TELEGRAM_BOT_TOKEN:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN not configured")
    
    if not video_path.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {video_path}")
    
    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    
    # Telegram's file size limit is 50MB for bots (2GB for premium users via bot API)
    if file_size_mb > 50:
        logger.warning(
            "Video file is %.2f MB (>50MB). Upload may fail or take very long. "
            "Consider compressing the video.", 
            file_size_mb
        )
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendVideo"
    
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Uploading video to Telegram (attempt %d/%d): %s (%.2f MB)",
                attempt, max_retries, video_path.name, file_size_mb
            )
            
            with open(video_path, "rb") as video_file:
                files = {"video": (video_path.name, video_file, "video/mp4")}
                data = {
                    "chat_id": chat_id,
                    "supports_streaming": True  # Enable streaming for better playback
                }
                
                # Use a session with retry adapter for better connection handling
                session = requests.Session()
                
                # Increase timeout significantly for large files
                response = session.post(
                    url, 
                    files=files, 
                    data=data, 
                    timeout=chunk_timeout
                )
                
                # Check HTTP status
                response.raise_for_status()
                
                result = response.json()
                
                if not result.get("ok"):
                    error_msg = result.get("description", "Unknown Telegram API error")
                    logger.error("Telegram API returned ok=false: %s", error_msg)
                    raise HTTPException(
                        status_code=500, 
                        detail=f"Telegram API error: {error_msg}"
                    )
                
                file_id = result["result"]["video"]["file_id"]
                logger.info(
                    "Video uploaded successfully to Telegram. file_id: %s (attempt %d)", 
                    file_id, attempt
                )
                
                return file_id
        
        except requests.exceptions.Timeout as e:
            last_error = e
            logger.error(
                "Upload timeout on attempt %d/%d: %s", 
                attempt, max_retries, str(e)
            )
            
        except requests.exceptions.ConnectionError as e:
            last_error = e
            logger.error(
                "Connection error on attempt %d/%d: %s", 
                attempt, max_retries, str(e)
            )
            
        except requests.exceptions.HTTPError as e:
            last_error = e
            logger.error(
                "HTTP error on attempt %d/%d: %s - Response: %s", 
                attempt, max_retries, str(e), e.response.text if e.response else "N/A"
            )
            
            # Don't retry on 4xx errors (client errors)
            if e.response and 400 <= e.response.status_code < 500:
                logger.error("Client error (4xx) - not retrying")
                break
                
        except Exception as e:
            last_error = e
            logger.exception(
                "Unexpected error during upload attempt %d/%d", 
                attempt, max_retries
            )
        
        # Wait before retrying (exponential backoff)
        if attempt < max_retries:
            wait_time = 2 ** attempt  # 2, 4, 8 seconds
            logger.info("Waiting %d seconds before retry...", wait_time)
            time.sleep(wait_time)
    
    # All retries failed
    logger.error(
        "Failed to upload video to Telegram after %d attempts. Last error: %s",
        max_retries, str(last_error)
    )
    raise HTTPException(
        status_code=500, 
        detail=f"Telegram upload failed after {max_retries} attempts: {str(last_error)}"
    )

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
# CLOUDINARY MODEL
# =====================================================
class CloudinaryDeleteRequest(BaseModel):
    public_id: str
    resource_type: str = "video"  # Default to video, can be "image", "raw", etc.

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
    Downloads a YouTube video using yt-dlp and ensures the MP4 is IG/FB-compatible.
    NO COOKIES USED.
    
    Steps:
        1. Download best video+audio.
        2. Probe codecs with ffprobe.
        3. Re-encode to H.264 + AAC + yuv420p if needed.
    
    Args:
        video_url: URL of the YouTube video
        retries: Number of retry attempts on failure

    Returns:
        Path object of the final MP4 file ready for IG/FB upload
    """
    output_file = DOWNLOADS_DIR / f"video_{int(datetime.utcnow().timestamp())}.mp4"
    
    # yt-dlp command templates (NO COOKIES)
    cmd_templates = [
        ["yt-dlp", "--force-ipv4", "--no-check-certificate",
         "-f", "bestvideo+bestaudio/best",
         "--merge-output-format", "mp4",
         "--socket-timeout", "30",
         "-o", str(output_file)],
        
        ["yt-dlp", "--force-ipv4", "--no-check-certificate",
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

                # --- Check IG/FB compatibility ---
                def is_ig_fb_compatible(path: Path) -> bool:
                    try:
                        ffprobe_cmd = [
                            "ffprobe", "-v", "error",
                            "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name,pix_fmt",
                            "-of", "default=noprint_wrappers=1:nokey=1",
                            str(path)
                        ]
                        result = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
                        if result.returncode != 0:
                            return False
                        lines = result.stdout.strip().splitlines()
                        if len(lines) < 2:
                            return False
                        codec, pix_fmt = lines
                        return codec == "h264" and pix_fmt == "yuv420p"
                    except Exception:
                        return False

                if not is_ig_fb_compatible(output_file):
                    logger.info("Video not IG/FB compatible, re-encoding...")
                    safe_file = output_file.with_name(output_file.stem + "_igfb.mp4")
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(output_file),
                        "-c:v", "libx264",
                        "-preset", "medium",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:a", "aac",
                        "-b:a", "128k",
                        "-movflags", "+faststart",   # REQUIRED FOR FB DIRECT UPLOAD
                        "-f", "mp4",
                        str(safe_file)
                    ]
                    ff_result = subprocess.run(cmd, capture_output=True, text=True)
                    if ff_result.returncode != 0:
                        logger.error("FFmpeg re-encode failed:\nstdout: %s\nstderr: %s", ff_result.stdout, ff_result.stderr)
                        raise RuntimeError(f"FFmpeg re-encode failed with code {ff_result.returncode}")
                    try:
                        output_file.unlink()
                    except Exception:
                        pass
                    return safe_file

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

async def upload_via_mtproto(video_path: Path, chat_id: int = TELEGRAM_CHAT_ID) -> str:
    """
    Uploads a file via MTProto (Telethon) and returns a public t.me link.
    Works for any size (up to 2 GB for normal accounts, 4 GB for premium).
    """
    if not tg_client:
        raise RuntimeError("MTProto client not configured")

    file_name = video_path.name
    file_size = video_path.stat().st_size

    logger.info("Uploading via MTProto: %s (%.2f MB)", file_name, file_size / 1_048_576)

    is_video = file_name.lower().endswith(('.mp4', '.mov', '.avi', '.mkv'))
    try:
        if is_video:
            ffprobe_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)
            ]
            duration = float(subprocess.check_output(ffprobe_cmd).decode().strip() or 0)

            entity = await tg_client.send_file(
                chat_id,
                str(video_path),
                video_note=False,
                supports_streaming=True,
                attributes=[
                    DocumentAttributeFilename(file_name=file_name),
                    DocumentAttributeVideo(
                        duration=int(duration),
                        w=1280, h=720,
                        supports_streaming=True
                    )
                ]
            )
        else:
            entity = await tg_client.send_file(
                chat_id,
                str(video_path),
                caption="",
                attributes=[DocumentAttributeFilename(file_name=file_name)]
            )

        link_chat_id = str(chat_id).replace("-", "")
        public_url = f"https://t.me/c/{link_chat_id}/{entity.id}"

        logger.info("MTProto upload finished → %s", public_url)
        return public_url

    except FloodWaitError as e:
        logger.error("Flood wait %s seconds", e.seconds)
        raise HTTPException(status_code=429, detail=f"Telegram flood wait {e.seconds}s")
    except Exception as e:
        logger.exception("MTProto upload failed")
        raise

# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/")
async def home():
    return {"message": "Python server is up!", "time": datetime.utcnow().isoformat()}

@app.get("/test/square")
async def square_endpoint(x: float = -12):
    return {"x": x, "square": x * x}

@app.get("/get_streamable")
async def streamable(video_url: str):
    """
    Download YouTube video and upload to Cloudinary ONLY.
    Returns Cloudinary public_id and secure_url.
    """
    mp4_path = None

    try:
        logger.info("Starting download for %s", video_url)
        
        # Run blocking download in thread pool
        loop = asyncio.get_running_loop()
        mp4_path = await loop.run_in_executor(None, get_streamable_mp4, video_url)

        if not mp4_path.exists():
            raise HTTPException(status_code=500, detail="Download succeeded but file missing")

        file_size_mb = round(mp4_path.stat().st_size / (1024 * 1024), 2)

        # Upload to Cloudinary (blocking operation, run in thread pool)
        cloudinary_data = await loop.run_in_executor(None, upload_to_cloudinary, mp4_path)

        return {
            "success": True,
            "public_id": cloudinary_data["public_id"],
            "url": cloudinary_data["secure_url"],  # Use secure_url (HTTPS)
            "file_size_mb": file_size_mb,
            "duration": cloudinary_data.get("duration"),
            "format": cloudinary_data.get("format"),
            "message": "Video processed and uploaded to Cloudinary"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /get_streamable")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if mp4_path:
            cleanup_video_file(mp4_path)

@app.get("/video-to-fb")
async def video_to_fb(
    video_url: str = Query(..., description="YouTube URL"),
    fb_description: str = Query("", description="Optional caption for FB")
):
    """
    1. Download + re-encode (IG/FB compatible)  
    2. Upload **same file** to Facebook  
    3. Upload **same file** to Cloudinary  
    4. Return both public URLs  
    5. Delete local file
    """
    mp4_path: Path | None = None
    try:
        # ------------------------------------------------------------------
        # 1. Download & re-encode (blocking – run in thread-pool)
        # ------------------------------------------------------------------
        loop = asyncio.get_running_loop()
        mp4_path = await loop.run_in_executor(None, get_streamable_mp4, video_url)

        # ------------------------------------------------------------------
        # 2. Upload to Facebook (blocking)
        # ------------------------------------------------------------------
        fb_result = await loop.run_in_executor(
            None, upload_to_facebook, mp4_path, fb_description
        )

        # ------------------------------------------------------------------
        # 3. Upload to Cloudinary (blocking)
        # ------------------------------------------------------------------
        cloudinary_data = await loop.run_in_executor(
            None, upload_to_cloudinary, mp4_path
        )

        return {
            "success": True,
            "facebook": fb_result,                     # {video_id, url}
            "cloudinary": {
                "public_id": cloudinary_data["public_id"],
                "secure_url": cloudinary_data["secure_url"],
                "duration": cloudinary_data.get("duration"),
                "format": cloudinary_data.get("format"),
            },
            "local_file": str(mp4_path),
            "size_mb": round(mp4_path.stat().st_size / (1024 * 1024), 2),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /video-to-fb")
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")
    finally:
        if mp4_path:
            cleanup_video_file(mp4_path)

@app.post("/cloudinary/delete")
async def delete_cloudinary_resource(body: CloudinaryDeleteRequest):
    """
    Delete a resource from Cloudinary by public_id.
    
    Request body:
    {
        "public_id": "telegram_videos/xyz123",
        "resource_type": "video"  // optional, defaults to "video"
    }
    """
    try:
        # Run blocking Cloudinary operation in thread pool
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            delete_from_cloudinary, 
            body.public_id, 
            body.resource_type
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /cloudinary/delete")
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.get("/cloudinary/delete")
async def delete_cloudinary_resource_get(
    public_id: str = Query(..., description="Cloudinary public_id to delete"),
    resource_type: str = Query("video", description="Resource type (video, image, raw)")
):
    """
    Delete a resource from Cloudinary by public_id (GET method for convenience).
    
    Example: /cloudinary/delete?public_id=telegram_videos/xyz123&resource_type=video
    """
    try:
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, 
            delete_from_cloudinary, 
            public_id, 
            resource_type
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in /cloudinary/delete")
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
    try:
        resp = requests.post(
            f"https://api.tumblr.com/v2/blog/{blog_id}/post",
            data={"type": "photo", "source": body.image_url, "caption": body.caption or ""},
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
async def post_image_to_x(image_url: str = Body(...), text: str = Body("")):
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


