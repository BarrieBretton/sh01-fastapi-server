# Imports

import os
import socket
import threading
import logging
import subprocess

from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta

import httpx
import redis
import requests
from requests_oauthlib import OAuth1
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from kiteconnect import KiteConnect, KiteTicker
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

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

]

def pick_tumblr_account(name: str):
    name = (name or "").lower().strip()
    if name == "vlvt.ave":
        return oauth_vlvt, T_VL_BID, "VLVT_AVE"
    return oauth_erika, T_EK_BID, "ERIKA_DEVEREUX"

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
                # Re-raise with clear message
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

app = FastAPI(title="Python Server - X/Tumblr/Kite/n8n", version="2.0")

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
    r = requests.get(url, headers=headers, timeout=10)
    data = safe_json(r)
    if not data:
        raise RuntimeError("n8n API returned non-JSON. Check N8N_BASE_URL and API key.")
    return data.get("data", [])

def deactivate_workflow(wf):
    url = f"{N8N_BASE_URL}/workflows/{wf['id']}/deactivate"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}
    logger.info("Deactivating workflow: %s (%s)", wf["name"], wf["id"])
    return requests.post(url, headers=headers, timeout=10)

def activate_workflow(wf):
    url = f"{N8N_BASE_URL}/workflows/{wf['id']}/activate"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}
    logger.info("Activating workflow: %s (%s)", wf["name"], wf["id"])
    return requests.post(url, headers=headers, timeout=10)

def get_youtube_stream_url(video_url: str) -> str:
    try:
        # Run yt-dlp command to get the direct stream URL
        result = subprocess.run(
            ["yt-dlp", "-f", "best", "-g", video_url],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {e.stderr.strip()}")

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
def streamable(video_url: str = Query(..., description="YouTube video URL")):
    """
    Returns the direct streamable URL of a YouTube video.
    """
    stream_url = get_youtube_stream_url(video_url)
    return {"video_url": video_url, "stream_url": stream_url}

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

    # Deactivate all first
    for wf in all_wfs:
        deactivate_workflow(wf)

    # Activate selected
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
            timeout=20,
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
        img_resp = requests.get(image_url, stream=True, timeout=15)
        img_resp.raise_for_status()

        files = {"media": ("image.jpg", img_resp.raw, img_resp.headers.get("Content-Type", "image/jpeg"))}
        upload = requests.post(
            "https://upload.twitter.com/1.1/media/upload.json",
            auth=oauth_x,
            files=files,
            data={"media_category": "tweet_image"},
            timeout=30,
        )
        upload.raise_for_status()
        media_id = upload.json()["media_id_string"]

        tweet = requests.post(
            "https://api.x.com/2/tweets",
            auth=oauth_x,
            json={"text": text, "media": {"media_ids": [media_id]}},
            timeout=30,
        )
        tweet.raise_for_status()
        logger.info("Posted to X successfully")
        return {"message": "Posted to X", "tweet": tweet.json()}
    except Exception as e:
        logger.exception("X post failed")
        raise HTTPException(status_code=500, detail=str(e))

# Kite endpoints (unchanged logic, just better logging)
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
    # Don't wipe with empty string — just clear the variable safely
    with kite_manager._lock:
        kite_manager._access_token = None
        kite_manager._kite.set_access_token(None)  # This is allowed
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
        async with httpx.AsyncClient(timeout=30.0) as client:
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
    # Only for manual testing: python app.py
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
else:
    # This runs in each uvicorn worker process
    import time
    time.sleep(4)  # Wait for Redis + network
    try:
        start_cron_scheduler()
    except Exception as e:
        logger.error("Failed to initialize cron master: %s", e)

