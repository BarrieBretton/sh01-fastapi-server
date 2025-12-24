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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Add this import at the top with other imports
from sheets_helper import (
    get_sheets_service,
    read_sheet_by_name,
    filter_status_rows,
    update_cell,
    get_column_letter,
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
from fastapi import FastAPI, HTTPException, Body, Query
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
    "ai-image-model-4",
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
# ENDPOINTS
# =====================================================

@app.get("/")
async def home():
    return {"message": "Python server is up!", "time": datetime.now().isoformat()}

@app.get("/test/square")
async def square_endpoint(x: float = -12):
    return {"x": x, "square": x * x}

@app.get("/yt/videos", response_model=List[YouTubeVideoResponse])
async def get_youtube_videos(
    handle: str = Query(..., description="YouTube handle (e.g., '@username')"),
    sort_by: str = Query("newest", description="Sort by 'newest', 'relevance', or 'engagement'"),
    max_results: int = Query(50, description="Maximum number of videos to return"),
):
    """
    Fetch videos for a YouTube handle and sort them by relevance, engagement, or newest uploads.
    """
    try:
        handler = YouTubeHandler()
        videos = await handler.get_videos_by_handle(handle, sort_by, max_results)
        return videos
    except ValueError as e:
        logger.error(f"Error fetching YouTube videos: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error fetching YouTube videos")
        raise HTTPException(status_code=500, detail=f"Failed to fetch videos: {e}")

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
            manager.delete_file,
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

# Google Sheets Endpoints
@app.post("/sheets/process_streamables")
async def process_streamables_endpoint(
    acc_name: str = Query("dave.commercial7"),
    sheet_name: str = Query("streamables"),
):
    service = get_sheets_service()
    rows = read_sheet_by_name(service, sheet_name)
    if not rows:
        return {"success": False, "message": "No data"}
    header = rows[0]
    filtered = filter_status_rows(rows, ['staged', 'uploaded'], exclude_status=True)
    filtered = [r for r in filtered if r.get("account") == acc_name]
    status_col = get_column_letter(header, "status")
    url_col = get_column_letter(header, "url")
    results = []
    event_loop = asyncio.get_running_loop()
    for row in filtered:
        row_num = row["row_number"]
        yt_link = row.get("yt_link", "").strip()
        if not yt_link:
            results.append({"row_number": row_num, "status": "error", "error": "No yt_link"})
            continue
        try:
            # 1. Download + re-encode
            mp4_path = await event_loop.run_in_executor(None, get_streamable_mp4, yt_link)
            # 2. Upload to B2
            b2_url = await event_loop.run_in_executor(None, upload_to_b2, mp4_path, yt_link)
            # 3. Update sheet
            update_cell(service, sheet_name, row_num, status_col, "staged")
            update_cell(service, sheet_name, row_num, url_col, b2_url)
            results.append({
                "row_number": row_num,
                "status": "success",
                "url": b2_url
            })
            cleanup_video_file(mp4_path)
        except Exception as e:
            logger.exception(f"Failed on row {row_num}")
            results.append({"row_number": row_num, "status": "error", "error": str(e)})
    return {
        "success": True,
        "total_processed": len(results),
        "successful": sum(1 for r in results if r["status"] == "success"),
        "results": results
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
