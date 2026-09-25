import os
import certifi
import logging
import re
import asyncio
import yt_dlp

from urllib.parse import urlparse, parse_qs
from typing import List, Dict

from dotenv import load_dotenv
import httpx
from pydantic import BaseModel

load_dotenv(".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_handler")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not set in .env")

# Explicit API-side definition used by only_shorts:
# short-form means a video whose contentDetails.duration is <= 180 seconds.
SHORT_FORM_MAX_SECONDS = 180


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

    def extract_video_id(self, url_or_id: str) -> str:
        """
        Accepts:
          - full YouTube URL (watch?v=..., youtu.be, shorts, embed)
          - bare video ID
        Returns video ID or an empty string.
        """
        value = (url_or_id or "").strip()
        if not value:
            return ""

        if re.fullmatch(r"[0-9A-Za-z_-]{11}", value):
            return value

        try:
            parsed_url = urlparse(value)
            host = (parsed_url.netloc or "").lower()

            if "youtu.be" in host:
                video_id = parsed_url.path.strip("/").split("/")[0]
                video_id = video_id.split("?", 1)[0].split("#", 1)[0]
                return (
                    video_id
                    if re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id or "")
                    else ""
                )

            if "youtube.com" in host or "m.youtube.com" in host:
                query_params = parse_qs(parsed_url.query or "")

                if "v" in query_params and query_params["v"]:
                    video_id = query_params["v"][0]
                    return (
                        video_id
                        if re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id or "")
                        else ""
                    )

                path_parts = [part for part in parsed_url.path.split("/") if part]

                if len(path_parts) >= 2 and path_parts[0] == "shorts":
                    video_id = path_parts[1]
                    return (
                        video_id
                        if re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id or "")
                        else ""
                    )

                if len(path_parts) >= 2 and path_parts[0] == "embed":
                    video_id = path_parts[1]
                    return (
                        video_id
                        if re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id or "")
                        else ""
                    )
        except Exception:
            return ""

        return ""

    async def get_video_title(self, url_or_id: str) -> str:
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
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        items = data.get("items", [])
        if not items:
            raise ValueError(f"Video not found / unavailable for id: {video_id}")

        return (items[0].get("snippet", {}).get("title") or "").strip()

    async def _fetch_channel_id(self, handle: str) -> str:
        normalized_handle = handle.removeprefix("@")

        url = f"{self.base_url}/channels"
        params = {
            "part": "id",
            "forHandle": normalized_handle,
            "key": self.api_key,
        }

        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        logger.debug("YouTube channel lookup response: %s", data)

        if not data.get("items"):
            raise ValueError(f"Channel not found for handle: @{normalized_handle}")

        channel_id = data["items"][0]["id"]
        logger.info(
            "Successfully resolved @%s → Channel ID: %s",
            normalized_handle,
            channel_id,
        )
        return channel_id

    async def _fetch_short_video_ids(
        self,
        handle: str,
        max_results: int = 50,
    ) -> List[str]:
        """
        Fetch actual videos from the channel's YouTube Shorts tab using yt-dlp.

        This does NOT infer Shorts from duration. It reads:
            https://www.youtube.com/@<handle>/shorts
        """
        normalized_handle = (handle or "").strip().removeprefix("@")

        if not normalized_handle:
            raise ValueError("YouTube handle cannot be empty")

        shorts_url = f"https://www.youtube.com/@{normalized_handle}/shorts"

        def fetch_with_ytdlp():
            options = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": "in_playlist",
                "playlistend": max_results,
                "force_ipv4": True,
                "nocheckcertificate": True,
            }

            with yt_dlp.YoutubeDL(options) as ydl:
                return ydl.extract_info(
                    shorts_url,
                    download=False,
                )

        try:
            payload = await asyncio.wait_for(
                asyncio.to_thread(fetch_with_ytdlp),
                timeout=90,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Timed out retrieving Shorts for @{normalized_handle}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"yt-dlp failed retrieving Shorts for "
                f"@{normalized_handle}: {exc}"
            ) from exc

        video_ids = []

        for entry in (payload or {}).get("entries", []):
            if not entry:
                continue

            video_id = str(entry.get("id") or "").strip()

            if re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id):
                video_ids.append(video_id)

        # Preserve Shorts-tab order while removing accidental duplicates.
        return list(dict.fromkeys(video_ids))

    @staticmethod
    def _duration_to_seconds(duration: str) -> int | None:
        """
        Convert YouTube ISO 8601 durations such as PT59S, PT2M30S,
        PT1H2M3S, or P1DT2H into seconds.
        """
        match = re.fullmatch(
            r"P(?:(?P<days>\d+)D)?"
            r"(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?"
            r"(?:(?P<seconds>\d+)S)?)?",
            duration or "",
        )
        if not match:
            logger.warning("Could not parse YouTube duration: %r", duration)
            return None

        parts = match.groupdict(default="0")
        return (
            int(parts["days"]) * 86_400
            + int(parts["hours"]) * 3_600
            + int(parts["minutes"]) * 60
            + int(parts["seconds"])
        )

    @staticmethod
    def _matches_short_form_filter(
        duration_seconds: int | None,
        only_shorts: bool | None,
    ) -> bool:
        if only_shorts is None:
            return True

        # Never claim a video fits an explicit filter if the API gave no
        # usable duration to classify it.
        if duration_seconds is None:
            return False

        is_short_form = duration_seconds <= SHORT_FORM_MAX_SECONDS
        return is_short_form if only_shorts else not is_short_form

    async def _fetch_video_stats(self, video_ids: List[str]) -> Dict[str, Dict]:
        """
        Fetch snippets, statistics, and durations for video IDs.

        YouTube videos.list accepts at most 50 IDs per request, so larger
        collections are fetched in batches.
        """
        if not video_ids:
            return {}

        stats_by_id = {}

        async with httpx.AsyncClient(
            verify=certifi.where(),
            timeout=30,
        ) as client:
            for start in range(0, len(video_ids), 50):
                batch_ids = video_ids[start:start + 50]

                url = f"{self.base_url}/videos"
                params = {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(batch_ids),
                    "key": self.api_key,
                }

                response = await client.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                for item in data.get("items", []):
                    video_id = item["id"]

                    stats_by_id[video_id] = {
                        "statistics": item.get("statistics", {}),
                        "snippet": item.get("snippet", {}),
                        "content_details": item.get("contentDetails", {}),
                    }

        return stats_by_id

    def _calculate_relevance_score(self, video: Dict) -> float:
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        return (view_count * 0.4) + (like_count * 0.3) + (comment_count * 0.3)

    def _calculate_engagement_score(self, video: Dict) -> float:
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 1))
        like_count = int(stats.get("likeCount", 0))
        dislike_count = int(stats.get("dislikeCount", 0))
        comment_count = int(stats.get("commentCount", 0))

        like_ratio = (
            like_count / (like_count + dislike_count + 1)
            if (like_count + dislike_count) > 0
            else 0
        )
        ctr = (like_count + comment_count) / view_count if view_count > 0 else 0
        return (like_ratio * 0.5) + (ctr * 0.5)

    async def get_videos_by_handle(
        self,
        handle: str,
        sort_by: str = "newest",
        max_results: int = 50,
        only_shorts: bool | None = None,
    ) -> List[YouTubeVideo]:
        if sort_by not in {"newest", "relevance", "engagement"}:
            raise ValueError(
                "sort_by must be one of: newest, relevance, engagement"
            )

        #
        # IMPORTANT:
        # only_shorts=True now uses the ACTUAL YouTube Shorts tab.
        #
        if only_shorts is True:
            video_ids = await self._fetch_short_video_ids(
                handle=handle,
                max_results=max_results,
            )

            if not video_ids:
                return []

            stats_by_id = await self._fetch_video_stats(video_ids)

            video_details = []

            for video_id in video_ids:
                stats_dict = stats_by_id.get(video_id)

                # A video may disappear/become private between yt-dlp and
                # videos.list. Simply skip it.
                if not stats_dict:
                    continue

                snippet = stats_dict.get("snippet", {})
                video_stats = stats_dict.get("statistics", {})

                thumbnails = snippet.get("thumbnails", {})

                thumbnail_url = (
                    thumbnails.get("high", {}).get("url")
                    or thumbnails.get("medium", {}).get("url")
                    or thumbnails.get("default", {}).get("url")
                    or ""
                )

                video_data = {
                    "video_id": video_id,
                    "title": snippet.get("title", ""),
                    "published_at": snippet.get("publishedAt", ""),
                    "channel_title": snippet.get("channelTitle", ""),
                    "thumbnail_url": thumbnail_url,
                    "view_count": int(video_stats.get("viewCount", 0)),
                    "like_count": int(video_stats.get("likeCount", 0)),
                    "dislike_count": 0,
                    "comment_count": int(video_stats.get("commentCount", 0)),
                    "relevance_score": self._calculate_relevance_score(
                        stats_dict
                    ),
                    "engagement_score": self._calculate_engagement_score(
                        stats_dict
                    ),
                }

                video_details.append(YouTubeVideo(**video_data))

            if sort_by == "newest":
                video_details.sort(
                    key=lambda video: video.published_at,
                    reverse=True,
                )
            elif sort_by == "relevance":
                video_details.sort(
                    key=lambda video: video.relevance_score,
                    reverse=True,
                )
            else:
                video_details.sort(
                    key=lambda video: video.engagement_score,
                    reverse=True,
                )

            return video_details[:max_results]

        #
        # Existing behaviour for only_shorts=False / omitted.
        #
        # Keeping this path essentially unchanged minimizes the blast radius.
        #
        channel_id = await self._fetch_channel_id(handle)

        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": min(max_results, 50),
            "order": "date",
            "type": "video",
            "key": self.api_key,
        }

        async with httpx.AsyncClient(
            verify=certifi.where(),
            timeout=30,
        ) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            search_data = response.json()

        items = search_data.get("items", [])

        if not items:
            return []

        video_ids = [
            item["id"]["videoId"]
            for item in items
            if item.get("id", {}).get("videoId")
        ]

        stats_by_id = await self._fetch_video_stats(video_ids)

        video_details = []

        for item in items:
            video_id = item.get("id", {}).get("videoId")

            if not video_id:
                continue

            snippet = item.get("snippet", {})
            stats_dict = stats_by_id.get(video_id, {})
            video_snippet = stats_dict.get("snippet", snippet)
            video_stats = stats_dict.get("statistics", {})
            content_details = stats_dict.get("content_details", {})

            #
            # Preserve your old only_shorts=False semantics:
            # false means > 180 seconds.
            #
            if only_shorts is False:
                duration_seconds = self._duration_to_seconds(
                    content_details.get("duration", "")
                )

                if duration_seconds is None:
                    continue

                if duration_seconds <= SHORT_FORM_MAX_SECONDS:
                    continue

            thumbnails = snippet.get("thumbnails", {})

            thumbnail_url = (
                thumbnails.get("high", {}).get("url")
                or thumbnails.get("medium", {}).get("url")
                or thumbnails.get("default", {}).get("url")
                or ""
            )

            video_data = {
                "video_id": video_id,
                "title": snippet.get("title", ""),
                "published_at": snippet.get("publishedAt", ""),
                "channel_title": video_snippet.get(
                    "channelTitle",
                    snippet.get("channelTitle", ""),
                ),
                "thumbnail_url": thumbnail_url,
                "view_count": int(video_stats.get("viewCount", 0)),
                "like_count": int(video_stats.get("likeCount", 0)),
                "dislike_count": 0,
                "comment_count": int(video_stats.get("commentCount", 0)),
                "relevance_score": self._calculate_relevance_score(
                    stats_dict
                ),
                "engagement_score": self._calculate_engagement_score(
                    stats_dict
                ),
            }

            video_details.append(YouTubeVideo(**video_data))

        if sort_by == "newest":
            video_details.sort(
                key=lambda video: video.published_at,
                reverse=True,
            )
        elif sort_by == "relevance":
            video_details.sort(
                key=lambda video: video.relevance_score,
                reverse=True,
            )
        else:
            video_details.sort(
                key=lambda video: video.engagement_score,
                reverse=True,
            )

        return video_details[:max_results]
