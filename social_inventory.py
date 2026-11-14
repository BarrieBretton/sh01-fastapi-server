"""Bounded, opt-in owner-media inventory router. No publishing or deletion routes.

Install with: from social_inventory import router; app.include_router(router)
All network content stays server-side; only metadata and fingerprints go to the dashboard storage.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import http.client
import io
import ipaddress
import json
import math
import os
import re
import shutil
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, urlencode, urljoin, urlsplit

from fastapi import APIRouter, Header, HTTPException
from PIL import Image, ImageOps, ImageStat
from pydantic import BaseModel, Field

ALGORITHM = "sh01-social-perceptual"
VERSION = 1
MAX_CACHE = 500
MAX_PAIRS = 15000
MAX_GROUP_MEMBERS = 100
MAX_GROUPS = 200
CDNS = {
    "instagram": ("cdninstagram.com", "fbcdn.net"),
    "facebook": ("fbcdn.net", "fbsbx.com"),
    "tumblr": ("media.tumblr.com", "vt.tumblr.com"),
}
API_HOSTS = ("graph.facebook.com", "api.tumblr.com")
_SCAN_LOCK = threading.Lock()
router = APIRouter(prefix="/social/inventory", tags=["social inventory"])


class SafeError(Exception):
    """Only static codes, never upstream response text or URLs."""


class IngestError(SafeError):
    def __init__(self, code: str, accepted: int, kind: str):
        super().__init__(code)
        self.accepted, self.kind = accepted, kind


@dataclass
class Budget:
    seconds: float = 120
    max_bytes: int = 200 * 1024 * 1024
    started: float = field(default_factory=time.monotonic)
    used_bytes: int = 0

    def remaining(self) -> float:
        left = self.seconds - (time.monotonic() - self.started)
        if left <= 0:
            raise SafeError("scan_time_budget")
        return left

    def consume(self, size: int) -> None:
        self.used_bytes += size
        if self.used_bytes > self.max_bytes:
            raise SafeError("scan_byte_budget")


def host_allowed(host: str, allowed: tuple[str, ...], suffix: bool = False) -> bool:
    return any(host == item or (suffix and host.endswith("." + item)) for item in allowed)


def validate_url(url: str, allowed: tuple[str, ...], *, suffix: bool = False,
                 resolver=socket.getaddrinfo) -> tuple[str, str, str]:
    """Validate and pin one public address; the connection never resolves it again."""
    try:
        p = urlsplit(url)
        host = (p.hostname or "").lower()
        if (p.scheme != "https" or p.username or p.password or p.fragment or
                p.port not in (None, 443) or len(url) > 16384 or
                not host_allowed(host, allowed, suffix)):
            raise SafeError("url_not_allowed")
        records = resolver(host, 443, type=socket.SOCK_STREAM)
        addresses = sorted({r[4][0] for r in records})
        if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
            raise SafeError("nonpublic_address")
        return host, addresses[0], p.path + ("?" + p.query if p.query else "") or "/"
    except SafeError:
        raise
    except Exception:
        raise SafeError("url_resolution_failed") from None


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host: str, address: str, timeout: float):
        super().__init__(host, timeout=timeout, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except Exception:
            raw.close()
            raise


class SafeHTTP:
    """No proxy environment, no response-body errors, no automatic redirects."""
    def __init__(self, budget: Budget):
        self.budget = budget

    def request(self, url: str, allowed: tuple[str, ...], *, method="GET", headers=None,
                payload=None, limit=2 * 1024 * 1024, suffix=False, redirects=0) -> bytes:
        headers = dict(headers or {})
        headers.setdefault("User-Agent", "SH01-Social-Inventory/1")
        body = None if payload is None else json.dumps(payload, separators=(",", ":")).encode()
        if body is not None:
            headers["Content-Type"] = "application/json"
        for hop in range(redirects + 1):
            self.budget.remaining()
            host, address, path = validate_url(url, allowed, suffix=suffix)
            conn = PinnedHTTPS(host, address, min(12, self.budget.remaining()))
            try:
                conn.request(method, path, body=body, headers=headers)
                response = conn.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    location = response.getheader("Location")
                    if hop == redirects or not location or headers.get("Authorization") or method != "GET":
                        raise SafeError("redirect_refused")
                    url = urljoin(url, location)
                    continue  # Next iteration revalidates DNS, host and scheme.
                if response.status != 200:
                    if response.status == 429:
                        raise SafeError("upstream_rate_limited")
                    if response.status in (401, 403):
                        raise SafeError("upstream_permission_denied")
                    raise SafeError("upstream_http_error")
                declared = response.getheader("Content-Length")
                if declared and int(declared) > limit:
                    raise SafeError("media_too_large")
                chunks, size = [], 0
                while True:
                    self.budget.remaining()
                    if conn.sock:
                        conn.sock.settimeout(min(12, self.budget.remaining()))
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        return b"".join(chunks)
                    size += len(chunk)
                    self.budget.consume(len(chunk))
                    if size > limit:
                        raise SafeError("media_too_large")
                    chunks.append(chunk)
            except SafeError:
                raise
            except Exception:
                raise SafeError("upstream_transport_error") from None
            finally:
                conn.close()
        raise SafeError("redirect_refused")

    def json(self, url, allowed, **kwargs):
        try:
            data = json.loads(self.request(url, allowed, **kwargs))
            if not isinstance(data, dict):
                raise ValueError()
            return data
        except SafeError:
            raise
        except Exception:
            raise SafeError("upstream_invalid_json") from None


def config_for(account_id: str, platform: str) -> dict:
    try:
        accounts = json.loads(os.environ.get("SOCIAL_ACCOUNTS_JSON", "{}"))
        cfg = dict(accounts[account_id])
        if cfg["platform"] != platform or not cfg.get("handle"):
            raise ValueError()
        provider_id = str(cfg.get("provider_id", account_id))
        pattern = r"[0-9]+" if platform in ("facebook", "instagram") else r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}"
        if not re.fullmatch(pattern, provider_id):
            raise ValueError()
        if cfg.get("workflow_id") and not re.fullmatch(r"[A-Za-z0-9_.:@-]{1,160}", str(cfg["workflow_id"])):
            raise ValueError()
        cfg.update(account_id=account_id, provider_id=provider_id)
        return cfg
    except Exception:
        raise SafeError("account_not_configured") from None


def env_secret(name: str) -> str:
    if not isinstance(name, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{0,100}", name):
        raise SafeError("credential_not_configured")
    value = os.environ.get(name)
    if not value:
        raise SafeError("credential_not_configured")
    return value


def safe_text(value: Any, length=3000) -> str:
    # Caption remains text. Remove URLs and credential-shaped assignments, not just query values.
    value = re.sub(r"https?://\S+", "[link omitted]", str(value or ""))
    value = re.sub(r"(?i)(access_token|api_key|authorization|password|secret)\s*[:=]\s*\S+", r"\1=[redacted]", value)
    return value[:length]


def permalink(value: str, platform: str) -> str | None:
    """Match the dashboard canonical post URL contract; never return tracking fields."""
    try:
        from urllib.parse import parse_qsl
        p = urlsplit(str(value))
        host = (p.hostname or "").lower().removeprefix("www.")
        if p.scheme != "https" or p.username or p.password or p.port not in (None, 443):
            return None
        if platform == "instagram":
            if host == "instagram.com" and re.fullmatch(r"/(?:p|reel|tv)/[A-Za-z0-9_-]+/?", p.path):
                return "https://www.instagram.com" + p.path.rstrip("/") + "/"
        elif platform == "facebook":
            if host not in ("facebook.com", "m.facebook.com", "web.facebook.com"):
                return None
            path = p.path.rstrip("/") or "/"
            query = urlencode(sorted((k, v) for k, v in parse_qsl(p.query) if k in ("story_fbid", "id", "v") and re.fullmatch(r"[A-Za-z0-9_]+", v)))
            if re.fullmatch(r"/[A-Za-z0-9_./-]+", path) and ("/posts/" in path or "/videos/" in path or path.startswith("/reel/") or query):
                return "https://www.facebook.com" + path + ("?" + query if query else "")
        elif platform == "tumblr":
            old = re.match(r"^/post/(\d+)", p.path)
            new = re.match(r"^/([A-Za-z0-9_-]+)/(\d+)", p.path)
            if old and host.endswith(".tumblr.com"):
                return "https://www.tumblr.com/" + host[:-11] + "/" + old[1]
            if new and host == "tumblr.com":
                return "https://www.tumblr.com/" + new[1] + "/" + new[2]
    except Exception:
        pass
    return None


def iso_time(value) -> str | None:
    try:
        dt = datetime.fromtimestamp(value, timezone.utc) if isinstance(value, (int, float)) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


@dataclass
class Post:
    metadata: dict
    media_url: str | None = None
    video_id: str | None = None


def normalize_post(raw: dict, cfg: dict) -> Post | None:
    platform = cfg["platform"]
    post_id = str(raw.get("id_string", raw.get("id", "")))
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,150}", post_id):
        raise SafeError("invalid_post_metadata")
    kind, url, video_id = None, None, None
    metrics = {}
    if platform == "instagram":
        kind = {"IMAGE": "IMAGE", "VIDEO": "VIDEO", "CAROUSEL_ALBUM": "CAROUSEL"}.get(raw.get("media_type"))
        url = raw.get("media_url") if kind != "CAROUSEL" else None
        link, date, caption = raw.get("permalink"), raw.get("timestamp"), raw.get("caption")
        metrics = {"likes": number(raw.get("like_count")), "comments": number(raw.get("comments_count"))}
    elif platform == "facebook":
        attachments = raw.get("attachments", {}).get("data", [])
        a = attachments[0] if attachments else {}
        children = a.get("subattachments", {}).get("data", [])
        atype = str(a.get("media_type", a.get("type", ""))).lower()
        if len(attachments) > 1 or children or atype == "album":
            kind = "CAROUSEL"
        elif atype in ("photo", "image"):
            kind, url = "IMAGE", a.get("media", {}).get("image", {}).get("src")
        elif atype == "video" or str(a.get("type", "")).startswith("video"):
            kind, url = "VIDEO", a.get("media", {}).get("source")
            video_id = str(a.get("target", {}).get("id", "")) or None
        link, date, caption = raw.get("permalink_url"), raw.get("created_time"), raw.get("message")
        metrics = {"likes": number(raw.get("likes", {}).get("summary", {}).get("total_count")),
                   "comments": number(raw.get("comments", {}).get("summary", {}).get("total_count")),
                   "shares": number(raw.get("shares", {}).get("count"))}
    else:
        # NPF media in the post body only. Reblog trail is not the owner's own media body.
        blocks = [b for b in raw.get("content", []) if b.get("type") in ("image", "video")]
        if len(blocks) > 1:
            kind = "CAROUSEL"
        elif blocks:
            b = blocks[0]
            kind = "IMAGE" if b["type"] == "image" else "VIDEO"
            media = b.get("media", [])
            if isinstance(media, dict):
                media = [media]
            choices = [m for m in media if isinstance(m, dict) and m.get("url")]
            if choices:
                url = max(choices, key=lambda m: int(m.get("width", 0))) ["url"]
            elif kind == "VIDEO":
                url = b.get("url")
        elif raw.get("photos"):
            kind = "CAROUSEL" if len(raw["photos"]) > 1 else "IMAGE"
            url = raw["photos"][0].get("original_size", {}).get("url") if kind == "IMAGE" else None
        elif raw.get("type") == "video":
            kind, url = "VIDEO", raw.get("video_url")
        link, date = raw.get("post_url"), raw.get("timestamp")
        caption = "\n".join(b.get("text", "") for b in raw.get("content", []) if b.get("type") == "text") or raw.get("summary", raw.get("caption", ""))
        # note_count combines likes/reblogs/replies and is preserved as notes.
        metrics = {"notes": number(raw.get("note_count"))}
    if not kind:
        return None
    metadata = {"platform": platform, "account_id": cfg["account_id"], "account_handle": safe_text(cfg["handle"], 120),
                "post_id": post_id, "permalink": permalink(link, platform), "media_type": kind,
                "published_at": iso_time(date), "caption": safe_text(caption),
                "metrics": {k: v for k, v in metrics.items() if v is not None}}
    if metadata["permalink"] is None:
        raise SafeError("missing_or_unsupported_permalink")
    if cfg.get("workflow_id"):
        metadata["workflow_id"] = str(cfg["workflow_id"])
    return Post(metadata, url, video_id)


class Provider:
    def __init__(self, cfg: dict, http: SafeHTTP):
        self.cfg, self.http = cfg, http
        self.version = os.environ.get("META_GRAPH_VERSION", "v24.0")
        if not re.fullmatch(r"v\d+\.\d+", self.version):
            raise SafeError("invalid_graph_version")

    def get(self, path: str, params: dict):
        platform = self.cfg["platform"]
        host = "api.tumblr.com" if platform == "tumblr" else "graph.facebook.com"
        url = f"https://{host}/{path}?{urlencode(params)}"
        if platform != "tumblr" or self.cfg.get("token_env"):
            headers = {"Authorization": "Bearer " + env_secret(self.cfg.get("token_env", ""))}
        else:
            try:
                from oauthlib.oauth1 import Client, SIGNATURE_TYPE_AUTH_HEADER
            except ImportError:
                raise SafeError("tumblr_oauth_dependency_missing") from None
            mapping = self.cfg.get("oauth_env", {})
            client = Client(env_secret(mapping.get("consumer_key", "")),
                            client_secret=env_secret(mapping.get("consumer_secret", "")),
                            resource_owner_key=env_secret(mapping.get("token", "")),
                            resource_owner_secret=env_secret(mapping.get("token_secret", "")),
                            signature_type=SIGNATURE_TYPE_AUTH_HEADER)
            _, headers, _ = client.sign(url)
        data = self.http.json(url, (host,), headers=headers)
        if data.get("error") or data.get("meta", {}).get("status", 200) != 200:
            raise SafeError("provider_api_error")
        return data

    def page(self, cursor: str | int | None, limit: int) -> tuple[list[dict], str | int | None]:
        cfg = self.cfg
        if cfg["platform"] == "tumblr":
            offset = int(cursor or 0)
            response = self.get(f"v2/blog/{quote(cfg['provider_id'], safe='')}/posts", {"npf": "true", "limit": limit, "offset": offset})["response"]
            posts = response.get("posts", [])
            total = response.get("total_posts")
            more = bool(posts) and (offset + len(posts) < total if isinstance(total, int) else len(posts) == limit)
            return posts, offset + len(posts) if more else None
        fields = ("id,caption,media_type,media_url,permalink,timestamp,like_count,comments_count" if cfg["platform"] == "instagram" else
                  "id,message,created_time,permalink_url,attachments{media_type,type,media,target,subattachments},likes.limit(0).summary(true),comments.limit(0).summary(true),shares")
        params = {"fields": fields, "limit": limit}
        if cursor:
            params["after"] = cursor
        edge = "media" if cfg["platform"] == "instagram" else "posts"
        response = self.get(f"{self.version}/{cfg['provider_id']}/{edge}", params)
        paging = response.get("paging", {})
        after = paging.get("cursors", {}).get("after") if paging.get("next") else None
        if paging.get("next") and not after:
            raise SafeError("provider_cursor_missing")
        # Never follow paging.next: it may contain credentials.
        return response.get("data", []), after

    def media_url(self, post: Post) -> str:
        if post.media_url:
            return post.media_url
        if self.cfg["platform"] == "facebook" and post.video_id and re.fullmatch(r"\d+", post.video_id):
            source = self.get(f"{self.version}/{post.video_id}", {"fields": "source"}).get("source")
            if source:
                return source
        raise SafeError("media_source_unavailable")


def dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    px = list(small.tobytes())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | int(px[y * 9 + x] > px[y * 9 + x + 1])
    return f"{value:016x}"


def crop_hashes(image: Image.Image) -> list[str]:
    hashes = [dhash(image)]
    w, h = image.size
    for scale in (0.8, 0.6):
        cw, ch = int(w * scale), int(h * scale)
        for x in (0, (w - cw) // 2, w - cw):
            for y in (0, (h - ch) // 2, h - ch):
                hashes.append(dhash(image.crop((x, y, x + cw, y + ch))))
    return list(dict.fromkeys(hashes))


def image_fingerprint(image: Image.Image, *, use_orb=False) -> dict:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((1024, 1024))
    gray = image.convert("L")
    fp = {"algorithm": ALGORITHM, "version": VERSION, "media_type": "IMAGE", "hashes": crop_hashes(image),
          "gray_stddev": round(ImageStat.Stat(gray.resize((64, 64))).stddev[0], 2)}
    if use_orb:
        try:
            import cv2
            import numpy as np
            keypoints, descriptors = cv2.ORB_create(nfeatures=96).detectAndCompute(np.asarray(gray), None)
            if descriptors is not None and len(keypoints) >= 12:
                fp["orb"] = {"points": [[round(k.pt[0] / image.width, 4), round(k.pt[1] / image.height, 4)] for k in keypoints[:96]],
                             "descriptors": base64.b64encode(descriptors[:96].tobytes()).decode()}
        except ImportError:
            pass
    return fp


def fingerprint_bytes(data: bytes, kind: str, budget: Budget, use_orb=False) -> dict:
    budget.remaining()
    if kind == "IMAGE":
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.width * im.height > 20_000_000 or min(im.size) < 16:
                    raise SafeError("image_dimensions_unsupported")
                if getattr(im, "n_frames", 1) > 1:
                    raise SafeError("animated_image_unsupported")
                im.load()
                fp = image_fingerprint(im, use_orb=use_orb)
        except SafeError:
            raise
        except Exception:
            raise SafeError("image_decode_failed") from None
    else:
        fp = video_fingerprint(data, budget)
    fp["sha256"] = hashlib.sha256(data).hexdigest()
    return fp


def run_media(command: list[str], budget: Budget) -> bytes:
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                timeout=min(15, budget.remaining()), check=True)
        return result.stdout
    except FileNotFoundError:
        raise SafeError("ffmpeg_not_installed") from None
    except subprocess.TimeoutExpired:
        raise SafeError("video_decode_timeout") from None
    except Exception:
        raise SafeError("video_decode_failed") from None


def video_fingerprint(data: bytes, budget: Budget) -> dict:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise SafeError("ffmpeg_not_installed")
    with tempfile.TemporaryDirectory(prefix="sh01-inventory-") as tmp:
        path = Path(tmp) / "input.media"
        path.write_bytes(data)
        # Restrict demuxer protocol access: ffmpeg never receives a remote URL.
        info = json.loads(run_media(["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "mov,matroska,webm", "-show_entries",
                                     "format=duration:stream=codec_type,width,height", "-of", "json", str(path)], budget))
        duration = float(info.get("format", {}).get("duration", 0))
        streams = [s for s in info.get("streams", []) if s.get("codec_type") == "video"]
        if not streams or not math.isfinite(duration) or not 0.5 <= duration <= 900:
            raise SafeError("video_duration_unsupported")
        if int(streams[0].get("width", 0)) * int(streams[0].get("height", 0)) > 4096 * 2160:
            raise SafeError("video_dimensions_unsupported")
        frames = []
        # Five ordered samples. The same relative positions are compared, never frame bags.
        for fraction in (0.1, 0.3, 0.5, 0.7, 0.9):
            raw = run_media(["ffmpeg", "-v", "error", "-nostdin", "-threads", "1", "-protocol_whitelist", "file,pipe",
                             "-format_whitelist", "mov,matroska,webm", "-ss", str(round(duration * fraction, 3)), "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                             "-vf", "scale=320:320:force_original_aspect_ratio=decrease", "-threads", "1",
                             "-f", "image2pipe", "-vcodec", "png", "pipe:1"], budget)
            with Image.open(io.BytesIO(raw)) as image:
                frames.append(image_fingerprint(image))
        return {"algorithm": ALGORITHM, "version": VERSION, "media_type": "VIDEO", "duration": round(duration, 3),
                "sample_positions": [0.1, 0.3, 0.5, 0.7, 0.9], "frames": frames}


def valid_fingerprint(fp, kind=None) -> bool:
    if not isinstance(fp, dict) or fp.get("algorithm") != ALGORITHM or fp.get("version") != VERSION:
        return False
    media_type = fp.get("media_type")
    if kind and media_type != kind:
        return False
    if media_type == "IMAGE":
        hashes = fp.get("hashes")
        return isinstance(hashes, list) and 1 <= len(hashes) <= 19 and all(isinstance(h, str) and re.fullmatch(r"[a-f0-9]{16}", h) for h in hashes) and number(fp.get("gray_stddev")) is not None
    if media_type == "VIDEO":
        frames, duration = fp.get("frames"), fp.get("duration")
        return (number(duration) is not None and 0.5 <= duration <= 900 and fp.get("sample_positions") == [0.1, 0.3, 0.5, 0.7, 0.9]
                and isinstance(frames, list) and len(frames) == 5 and all(valid_fingerprint(f, "IMAGE") for f in frames))
    return False


def hash_distance(a: dict, b: dict) -> int:
    return min((int(x, 16) ^ int(y, 16)).bit_count() for x in a["hashes"] for y in b["hashes"])


def orb_evidence(a: dict, b: dict) -> dict | None:
    try:
        import cv2
        import numpy as np
        aa, bb = a["orb"], b["orb"]
        pts_a, pts_b = np.array(aa["points"], dtype=np.float32), np.array(bb["points"], dtype=np.float32)
        if not (12 <= len(pts_a) <= 96 and 12 <= len(pts_b) <= 96):
            return None
        da = np.frombuffer(base64.b64decode(aa["descriptors"], validate=True), dtype=np.uint8).reshape((-1, 32))
        db = np.frombuffer(base64.b64decode(bb["descriptors"], validate=True), dtype=np.uint8).reshape((-1, 32))
        if len(da) != len(pts_a) or len(db) != len(pts_b):
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.72 * p[1].distance]
        # Distinct destination points avoid repeated textures supplying fake support.
        if len(good) < 12 or len({m.trainIdx for m in good}) < 10:
            return None
        src = np.float32([pts_a[m.queryIdx] for m in good])
        dst = np.float32([pts_b[m.trainIdx] for m in good])
        _, mask = cv2.findHomography(src, dst, cv2.RANSAC, 0.015, maxIters=500)
        count = int(mask.sum()) if mask is not None else 0
        if count >= 10 and count / len(good) >= 0.65:
            return {"method": "orb_geometric_candidate", "inliers": count, "matches": len(good)}
    except (ImportError, KeyError, ValueError, TypeError):
        pass
    return None


def compare_fingerprints(a: dict, b: dict) -> dict | None:
    if not valid_fingerprint(a) or not valid_fingerprint(b) or a["media_type"] != b["media_type"]:
        return None
    digest = a.get("sha256")
    if isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) and digest == b.get("sha256"):
        return {"method": "equal_file_sha256"}
    if a["media_type"] == "IMAGE":
        if min(a["gray_stddev"], b["gray_stddev"]) < 8:
            return None
        distance = hash_distance(a, b)
        if distance <= 4:
            evidence = orb_evidence(a, b)
            return dict(evidence or {"method": "multicrop_dhash_candidate"}, min_hamming=distance)
        # The geometric fallback is bounded and only run after a looser perceptual prefilter.
        if distance <= 14:
            return orb_evidence(a, b)
        return None
    delta = abs(a["duration"] - b["duration"])
    if delta > max(1.0, min(a["duration"], b["duration"]) * 0.04):
        return None
    distances = [hash_distance(x, y) for x, y in zip(a["frames"], b["frames"])]
    informative = sum(min(x["gray_stddev"], y["gray_stddev"]) >= 8 for x, y in zip(a["frames"], b["frames"]))
    if informative >= 4 and max(distances) <= 6 and sum(distances) / 5 <= 4:
        return {"method": "ordered_video_frames_candidate", "frame_hamming": distances, "duration_delta_seconds": round(delta, 3)}
    return None


def cluster_evidence(pairs: list[dict]) -> dict:
    """Aggregate only verified pair evidence; metrics are not confidence scores."""
    methods: dict[str, int] = {}
    values: dict[str, list[float]] = {}
    for evidence in pairs:
        method = evidence["method"]
        methods[method] = methods.get(method, 0) + 1
        if "min_hamming" in evidence:
            values.setdefault("hamming", []).append(evidence["min_hamming"])
        if "frame_hamming" in evidence:
            values.setdefault("hamming", []).extend(evidence["frame_hamming"])
        if "duration_delta_seconds" in evidence:
            values.setdefault("duration_delta_seconds", []).append(evidence["duration_delta_seconds"])
        if evidence.get("matches"):
            values.setdefault("orb_inlier_ratio", []).append(round(evidence["inliers"] / evidence["matches"], 4))
        if method == "equal_file_sha256":
            values.setdefault("sha256_equal", []).append(1)
    return {"method": "complete_link_candidate_cluster", "version": VERSION, "pair_count": len(pairs),
            "method_counts": dict(sorted(methods.items())),
            "score_bounds": {key: {"min": min(items), "max": max(items)} for key, items in sorted(values.items())}}


def build_groups(records: dict[str, dict], scanned_ids: set[str], cfg: dict, budget: Budget) -> tuple[list[dict], int, bool]:
    """Deterministic disjoint greedy complete-link clusters seeded by scanned posts.

    Each emitted cluster is a verified clique: every member pair was accepted.
    Cached-to-cached checks needed to prove this also consume MAX_PAIRS. A budget
    interruption retains only already verified members, with incomplete coverage.
    """
    groups, comparisons, truncated = [], 0, False
    ids = sorted(post_id for post_id, fp in records.items() if valid_fingerprint(fp))
    seeds = sorted(set(ids) & scanned_ids)
    assigned: set[str] = set()
    checked: dict[tuple[str, str], dict | None] = {}

    def pair(a: str, b: str) -> dict | None:
        nonlocal comparisons
        key = tuple(sorted((a, b)))
        if key not in checked:
            if comparisons >= MAX_PAIRS:
                raise SafeError("comparison_pair_budget")
            budget.remaining()
            comparisons += 1
            # Canonical direction also makes optional geometric verification repeatable.
            checked[key] = compare_fingerprints(records[key[0]], records[key[1]])
        return checked[key]

    def emit(members: list[str], evidence: list[dict]) -> None:
        if len(members) < 2:
            return
        members = sorted(members)
        kind = records[members[0]]["media_type"]
        identity = json.dumps([cfg["platform"], cfg["account_id"], kind, ALGORITHM, VERSION, "complete_link_v1", members], separators=(",", ":"))
        groups.append({"group_id": "dup_" + hashlib.sha256(identity.encode()).hexdigest()[:32], "platform": cfg["platform"],
                       "account_id": cfg["account_id"], "media_type": kind, "algorithm": ALGORITHM,
                       "confidence": "candidate", "members": members, "evidence": cluster_evidence(evidence)})
        assigned.update(members)

    for seed in seeds:
        if seed in assigned:
            continue
        if len(groups) >= MAX_GROUPS:
            return groups, comparisons, True
        members, evidence = [seed], []
        try:
            for candidate in ids:
                budget.remaining()
                if candidate == seed or candidate in assigned or records[candidate]["media_type"] != records[seed]["media_type"]:
                    continue
                if len(members) >= MAX_GROUP_MEMBERS:
                    truncated = True
                    break
                supporting = []
                for member in members:
                    match = pair(member, candidate)
                    if match is None:
                        break
                    supporting.append(match)
                else:
                    # Add a candidate only once ALL pair checks completed successfully.
                    members.append(candidate)
                    evidence.extend(supporting)
        except SafeError:
            emit(members, evidence)
            return groups, comparisons, True
        emit(members, evidence)
    return groups, comparisons, truncated


class Dashboard:
    def __init__(self, http: SafeHTTP):
        self.http = http
        self.base = os.environ.get("OPS_DASHBOARD_URL", "").rstrip("/")
        p = urlsplit(self.base)
        if p.scheme != "https" or not p.hostname or p.path or p.query or p.fragment or p.username or p.password or p.port not in (None, 443):
            raise SafeError("dashboard_not_configured")
        self.hosts = (p.hostname,)
        self.headers = {"X-INGEST-KEY": env_secret("INGEST_TOKEN")}

    def cache(self, cfg: dict) -> tuple[dict, bool]:
        records, offset = {}, 0
        while offset < MAX_CACHE:
            q = urlencode({"platform": cfg["platform"], "account_id": cfg["account_id"], "offset": offset, "limit": min(100, MAX_CACHE - offset)})
            data = self.http.json(self.base + "/api/ingest/fingerprints?" + q, self.hosts, headers=self.headers, limit=4 * 1024 * 1024)
            rows = data.get("fingerprints")
            if not isinstance(rows, list) or not isinstance(data.get("has_more"), bool):
                raise SafeError("dashboard_cache_invalid")
            for row in rows[:MAX_CACHE - offset]:
                fp = row.get("fingerprint")
                if valid_fingerprint(fp):
                    records[str(row["post_id"])] = fp
            offset += len(rows)
            if not data.get("has_more"):
                return records, False
            if not rows:
                raise SafeError("dashboard_cache_invalid")
        return records, True

    def ingest(self, kind: str, rows: list[dict]) -> int:
        count = 0
        batch_size = 10
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            try:
                ack = self.http.json(self.base + "/api/ingest/" + ("posts" if kind == "posts" else "duplicate-groups"), self.hosts,
                                     method="POST", headers=self.headers, payload={kind: batch})
                if ack.get("accepted") != len(batch):
                    raise SafeError("dashboard_ingest_unacknowledged")
            except SafeError as exc:
                raise IngestError(str(exc), count, kind) from None
            count += len(batch)
        return count


def encode_cursor(state: dict, account_id: str, platform: str) -> str:
    payload = dict(state, account=account_id, platform=platform, expires=int(time.time()) + 86400)
    raw = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(env_secret("SOCIAL_INVENTORY_API_KEY").encode(), raw.encode(), hashlib.sha256).hexdigest()
    return raw + "." + signature


def decode_cursor(cursor: str | None, account_id: str, platform: str) -> dict:
    if not cursor:
        return {"after": None, "skip": 0}
    try:
        raw, signature = cursor.rsplit(".", 1)
        expected = hmac.new(env_secret("SOCIAL_INVENTORY_API_KEY").encode(), raw.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            raise ValueError()
        data = json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))
        if data["account"] != account_id or data["platform"] != platform or data["expires"] < time.time():
            raise ValueError()
        if not isinstance(data.get("skip"), int) or not 0 <= data["skip"] <= 20:
            raise ValueError()
        return {"after": data.get("after"), "skip": data["skip"]}
    except Exception:
        raise SafeError("invalid_or_expired_cursor") from None


class ScanRequest(BaseModel):
    platform: Literal["facebook", "instagram", "tumblr"]
    account_id: str = Field(min_length=1, max_length=150, pattern=r"^[A-Za-z0-9_.-]+$")
    max_posts: int = Field(default=20, ge=1, le=100)
    cursor: str | None = Field(default=None, max_length=12000)

    class Config:
        extra = "forbid"


def scan(request: ScanRequest, *, cfg=None, provider=None, dashboard=None, http=None, budget=None) -> dict:
    cfg = cfg or config_for(request.account_id, request.platform)
    budget = budget or Budget(seconds=max(15, min(300, int(os.environ.get("SOCIAL_SCAN_MAX_SECONDS", "120")))))
    http = http or SafeHTTP(budget)
    provider = provider or Provider(cfg, http)
    dashboard = dashboard or Dashboard(http)
    state = decode_cursor(request.cursor, request.account_id, request.platform)
    errors, known, cache_truncated = [], {}, False
    try:
        known, cache_truncated = dashboard.cache(cfg)
    except SafeError as exc:
        errors.append({"stage": "fingerprint_cache", "code": str(exc)})
        cache_truncated = True
    posts, scanned_ids = [], set()
    counts = {"visited": 0, "media_posts": 0, "unsupported_posts": 0, "fingerprinted": 0, "reused": 0,
              "carousels_metadata_only": 0, "failed_items": 0, "missing_permalink_skipped": 0, "posts_ingested": 0, "duplicate_groups_ingested": 0}
    exhausted, stop_reason = False, "max_posts"
    checkpoint = dict(state)
    use_orb = os.environ.get("SOCIAL_ENABLE_ORB", "0") == "1"
    try:
        while counts["visited"] < request.max_posts:
            budget.remaining()
            # Stable page size makes within-page resume deterministic even when max_posts changes.
            try:
                raw_posts, next_after = provider.page(state["after"], 20)
            except SafeError:
                raise
            except Exception:
                raise SafeError("provider_response_invalid") from None
            if not isinstance(raw_posts, list) or len(raw_posts) > 20:
                raise SafeError("provider_page_invalid")
            if not raw_posts:
                exhausted = next_after is None
                if not exhausted:
                    raise SafeError("provider_empty_page")
                stop_reason = "provider_exhausted"
                break
            for index in range(state["skip"], len(raw_posts)):
                if counts["visited"] >= request.max_posts:
                    break
                budget.remaining()
                checkpoint = {"after": state["after"], "skip": index}
                post = None
                try:
                    post = normalize_post(raw_posts[index], cfg)
                    if post is None:
                        counts["unsupported_posts"] += 1
                    else:
                        counts["media_posts"] += 1
                        metadata = post.metadata
                        kind, post_id = metadata["media_type"], metadata["post_id"]
                        if kind == "CAROUSEL":
                            counts["carousels_metadata_only"] += 1
                        elif valid_fingerprint(known.get(post_id), kind):
                            metadata["fingerprint"] = known[post_id]
                            counts["reused"] += 1
                            scanned_ids.add(post_id)
                        else:
                            url = provider.media_url(post)
                            data = http.request(url, CDNS[request.platform], suffix=True, redirects=3,
                                                limit=(10 if kind == "IMAGE" else 40) * 1024 * 1024)
                            fp = fingerprint_bytes(data, kind, budget, use_orb=use_orb)
                            metadata["fingerprint"] = known[post_id] = fp
                            counts["fingerprinted"] += 1
                            scanned_ids.add(post_id)
                        posts.append(metadata)
                except Exception as exc:
                    code = str(exc) if isinstance(exc, SafeError) else "item_processing_failed"
                    failed_id = post.metadata["post_id"] if post else str(raw_posts[index].get("id_string", raw_posts[index].get("id", ""))) if isinstance(raw_posts[index], dict) else ""
                    errors.append({"stage": "item", "code": code, **({"post_id": failed_id} if re.fullmatch(r"[A-Za-z0-9_-]{1,150}", failed_id) else {})})
                    counts["failed_items"] += 1
                    if code == "missing_or_unsupported_permalink":
                        counts["missing_permalink_skipped"] += 1
                    if post:
                        posts.append(post.metadata)  # Persist metadata even if media is inaccessible.
                    if code in ("scan_time_budget", "scan_byte_budget"):
                        stop_reason = code
                        raise SafeError(code)
                counts["visited"] += 1
                checkpoint = {"after": state["after"], "skip": index + 1}
            if checkpoint["skip"] >= len(raw_posts):
                checkpoint = {"after": next_after, "skip": 0}
                if next_after is None:
                    exhausted, stop_reason = True, "provider_exhausted"
                    break
            if counts["visited"] >= request.max_posts:
                break
            state = dict(checkpoint)
    except SafeError as exc:
        stop_reason = str(exc)
        errors.append({"stage": "scan", "code": str(exc)})
    groups, comparisons, comparison_truncated = [], 0, False
    try:
        groups, comparisons, comparison_truncated = build_groups(known, scanned_ids, cfg, budget)
    except SafeError:
        comparison_truncated = True
        errors.append({"stage": "comparison", "code": "comparison_budget_exhausted"})
    # Reserve an independent 30-second, 8 MiB delivery budget even when media budget ran out.
    if isinstance(dashboard, Dashboard):
        dashboard.http = SafeHTTP(Budget(seconds=30, max_bytes=8 * 1024 * 1024))
    try:
        counts["posts_ingested"] = dashboard.ingest("posts", posts)
        counts["duplicate_groups_ingested"] = dashboard.ingest("groups", groups)
    except SafeError as exc:
        if isinstance(exc, IngestError):
            counts["posts_ingested" if exc.kind == "posts" else "duplicate_groups_ingested"] = exc.accepted
        errors.append({"stage": "dashboard_ingest", "code": str(exc)})
        # Replaying from request cursor recovers unacknowledged/partial batches idempotently.
        checkpoint = decode_cursor(request.cursor, request.account_id, request.platform)
        exhausted, stop_reason = False, "dashboard_ingest_failed_replay"
    return {"ok": not errors, "platform": request.platform, "account_id": request.account_id, "counts": counts,
            "coverage": {"provider_exhausted": exhausted, "full_history_verified": False, "pagination_truncated": not exhausted,
                         "stop_reason": stop_reason, "cache_truncated": cache_truncated, "comparisons": comparisons,
                         "comparison_truncated": comparison_truncated, "cache_limit": MAX_CACHE, "pair_limit": MAX_PAIRS,
                         "group_member_limit": MAX_GROUP_MEMBERS, "group_limit": MAX_GROUPS,
                         "candidate_suggestions_only": True, "orb_requested": use_orb},
            "resume_cursor": None if exhausted else encode_cursor(checkpoint, request.account_id, request.platform),
            "errors": errors[:110]}


@router.post("/scan")
async def scan_route(request: ScanRequest, x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    expected = os.environ.get("SOCIAL_INVENTORY_API_KEY", "")
    if not expected:
        raise HTTPException(status_code=503, detail="social_inventory_not_configured")
    if not x_api_key or not hmac.compare_digest(expected, x_api_key):
        raise HTTPException(status_code=401, detail="invalid_api_key")
    if not _SCAN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="scan_already_running")
    def run_and_release():
        try:
            return scan(request)
        finally:
            _SCAN_LOCK.release()
    # Shield prevents client cancellation releasing the lock while its thread is still running.
    try:
        return await asyncio.shield(asyncio.to_thread(run_and_release))
    except SafeError as exc:
        code = str(exc)
        status = 400 if code in ("account_not_configured", "invalid_or_expired_cursor") else 503
        raise HTTPException(status_code=status, detail=code) from None
    except Exception:
        raise HTTPException(status_code=503, detail="social_inventory_failed") from None


@router.get("/status")
async def inventory_status(x_api_key: str | None = Header(default=None, alias="X-API-KEY")):
    expected = os.environ.get("SOCIAL_INVENTORY_API_KEY", "")
    if not expected:raise HTTPException(status_code=503, detail="social_inventory_not_configured")
    if not x_api_key or not hmac.compare_digest(expected,x_api_key):raise HTTPException(status_code=401, detail="invalid_api_key")
    import shutil
    try:
        configs=json.loads(os.environ.get("SOCIAL_ACCOUNTS_JSON","{}"))
        if not isinstance(configs,dict):raise ValueError()
        configured=len(configs)
    except Exception:raise HTTPException(status_code=503,detail="social_accounts_config_invalid") from None
    return {"configured_accounts":configured,"scan_running":_SCAN_LOCK.locked(),
            "ffmpeg_available":bool(shutil.which("ffmpeg")),"ffprobe_available":bool(shutil.which("ffprobe")),
            "dashboard_configured":bool(os.environ.get("OPS_DASHBOARD_URL") and os.environ.get("OPS_INGEST_TOKEN")),
            "delete_enabled":os.environ.get("SOCIAL_DELETE_ENABLED")=="1",
            "live_provider_access_verified":False,"limits":{"cache":MAX_CACHE,"pairs":MAX_PAIRS,"groups":MAX_GROUPS}}
