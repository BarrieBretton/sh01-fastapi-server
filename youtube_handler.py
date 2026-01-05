import os
import certifi
import logging
import re
from urllib.parse import urlparse, parse_qs
from typing import List, Dict
from dotenv import load_dotenv
import httpx
from pydantic import BaseModel

# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_handler")

# YouTube Data API v3
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not set in .env")


class YouTubeVideo(BaseModel):
    video_id: str
    title: str
    published_at: str
    view_count: int
    like_count: int
    dislike_count: int = 0
    comment_count: int
    thumbnail_url: str
    channel_title: str
    relevance_score: float = 0.0
    engagement_score: float = 0.0


class YouTubeHandler:
    def __init__(self, api_key: str = YOUTUBE_API_KEY):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"

    # ----------------------------
    # NEW: Extract video ID
    # ----------------------------
    def extract_video_id(self, url_or_id: str) -> str:
        """
        Accepts:
          - full youtube URL (watch?v=...), short youtu.be/..., shorts/..., embed/...
          - bare video id (11 chars)
        Returns: video_id or "".
        """
        s = (url_or_id or "").strip()
        if not s:
            return ""

        # If it already looks like a YouTube video id (usually 11 chars)
        if re.fullmatch(r"[0-9A-Za-z_-]{11}", s):
            return s

        try:
            u = urlparse(s)
            host = (u.netloc or "").lower()

            # youtu.be/<id>
            if "youtu.be" in host:
                vid = u.path.strip("/").split("/")[0]
                vid = vid.split("?", 1)[0].split("#", 1)[0] # (This safely strips ?si=... / fragments if they sneak in.)
                return vid if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid or "") else ""

            # youtube.com variants
            if "youtube.com" in host or "m.youtube.com" in host:
                qs = parse_qs(u.query or "")

                # youtube.com/watch?v=<id>
                if "v" in qs and qs["v"]:
                    vid = qs["v"][0]
                    return vid if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid or "") else ""

                parts = [p for p in u.path.split("/") if p]

                # youtube.com/shorts/<id>
                if len(parts) >= 2 and parts[0] == "shorts":
                    vid = parts[1]
                    return vid if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid or "") else ""

                # youtube.com/embed/<id>
                if len(parts) >= 2 and parts[0] == "embed":
                    vid = parts[1]
                    return vid if re.fullmatch(r"[0-9A-Za-z_-]{11}", vid or "") else ""

        except Exception:
            return ""

        return ""

    # ----------------------------
    # NEW: Fetch title from URL/ID
    # ----------------------------
    async def get_video_title(self, url_or_id: str) -> str:
        """
        Returns the video's title using videos.list(part=snippet).
        """
        video_id = self.extract_video_id(url_or_id)
        if not video_id:
            raise ValueError(f"Could not extract video_id from: {url_or_id}")

        url = f"{self.base_url}/videos"
        params = {
            "part": "snippet",
            "id": video_id,
            "key": self.api_key,
        }

        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

        items = data.get("items", [])
        if not items:
            raise ValueError(f"Video not found / unavailable for id: {video_id}")

        snippet = items[0].get("snippet", {})
        title = (snippet.get("title") or "").strip()
        return title

    async def _fetch_channel_id(self, handle: str) -> str:
        # Accept handle with or without @, normalize to without @
        if handle.startswith("@"):
            handle = handle[1:]

        url = f"{self.base_url}/channels"
        params = {
            "part": "id",
            "forHandle": handle,
            "key": self.api_key,
        }

        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            # FIX: don't log as error on success
            logger.debug("YouTube channel lookup response: %s", data)

            if response.status_code == 401:
                raise ValueError("Invalid or unauthorized API key (401)")

            if not data.get("items"):
                raise ValueError(f"Channel not found for handle: @{handle}")

            channel_id = data["items"][0]["id"]
            logger.info(f"Successfully resolved @{handle} → Channel ID: {channel_id}")
            return channel_id

    async def _fetch_video_stats(self, video_ids: List[str]) -> Dict[str, Dict]:
        """Fetch stats + snippet for multiple video IDs in one call (quota-efficient)."""
        url = f"{self.base_url}/videos"
        params = {
            "part": "snippet,statistics",
            "id": ",".join(video_ids),
            "key": self.api_key,
        }
        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        stats_by_id = {}
        for item in data.get("items", []):
            vid = item["id"]
            stats_by_id[vid] = {
                "statistics": item.get("statistics", {}),
                "snippet": item.get("snippet", {}),
            }
        return stats_by_id

    def _calculate_relevance_score(self, video: Dict) -> float:
        """Calculate a relevance score based on engagement metrics."""
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        return (view_count * 0.4) + (like_count * 0.3) + (comment_count * 0.3)

    def _calculate_engagement_score(self, video: Dict) -> float:
        """Calculate an engagement score."""
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 1))
        like_count = int(stats.get("likeCount", 0))
        dislike_count = int(stats.get("dislikeCount", 0))
        comment_count = int(stats.get("commentCount", 0))

        like_ratio = like_count / (like_count + dislike_count + 1) if (like_count + dislike_count) > 0 else 0
        ctr = (like_count + comment_count) / view_count if view_count > 0 else 0
        return (like_ratio * 0.5) + (ctr * 0.5)

    async def get_videos_by_handle(
        self,
        handle: str,
        sort_by: str = "newest",
        max_results: int = 50,
    ) -> List[YouTubeVideo]:
        channel_id = await self._fetch_channel_id(handle)
        if not channel_id:
            raise ValueError(f"Channel not found for handle: {handle}")

        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": min(max_results, 50),
            "order": "date",
            "type": "video",
            "key": self.api_key,
        }
        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            search_data = response.json()

        items = search_data.get("items", [])
        if not items:
            return []

        video_ids = [item["id"]["videoId"] for item in items]
        stats_by_id = await self._fetch_video_stats(video_ids)

        video_details = []
        for item in items:
            video_id = item["id"]["videoId"]
            snippet = item["snippet"]
            stats_dict = stats_by_id.get(video_id, {})
            video_snippet = stats_dict.get("snippet", snippet)
            video_stats = stats_dict.get("statistics", {})

            video_data = {
                "video_id": video_id,
                "title": snippet["title"],
                "published_at": snippet["publishedAt"],
                "channel_title": video_snippet.get("channelTitle", snippet.get("channelTitle", "")),
                "thumbnail_url": snippet["thumbnails"]["high"]["url"],
                "view_count": int(video_stats.get("viewCount", 0)),
                "like_count": int(video_stats.get("likeCount", 0)),
                "dislike_count": 0,
                "comment_count": int(video_stats.get("commentCount", 0)),
                "relevance_score": 0.0,
                "engagement_score": 0.0,
            }

            video_data["relevance_score"] = self._calculate_relevance_score(video_data)
            video_data["engagement_score"] = self._calculate_engagement_score(video_data)

            video_details.append(YouTubeVideo(**video_data))

        if sort_by == "newest":
            video_details.sort(key=lambda x: x.published_at, reverse=True)
        elif sort_by == "relevance":
            video_details.sort(key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "engagement":
            video_details.sort(key=lambda x: x.engagement_score, reverse=True)

        return video_details[:max_results]
