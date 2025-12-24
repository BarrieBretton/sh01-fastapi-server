import os
import certifi
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
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
    published_at: datetime
    view_count: int
    like_count: int
    dislike_count: int
    comment_count: int
    thumbnail_url: str
    channel_title: str
    relevance_score: float = 0.0
    engagement_score: float = 0.0

class YouTubeHandler:
    def __init__(self, api_key: str = YOUTUBE_API_KEY):
        self.api_key = api_key
        self.base_url = "https://www.googleapis.com/youtube/v3"

    async def _fetch_channel_id(self, handle: str) -> str:
        # Accept handle with or without @, normalize to without @
        if handle.startswith("@"):
            handle = handle[1:]

        url = f"{self.base_url}/channels"
        params = {
            "part": "id",                # 'id' is enough; use "snippet,id" if you want title too
            "forHandle": handle,         # Works with or without @ — Google accepts both
            "key": self.api_key,
        }

        async with httpx.AsyncClient(verify=certifi.where(), timeout=30) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()  # Will raise on 4xx/5xx

            data = response.json()

            if not data.get("items"):
                raise ValueError(f"Channel not found for handle: @{handle}")

            channel_id = data["items"][0]["id"]
            logger.info(f"Successfully resolved @{handle} → Channel ID: {channel_id}")
            return channel_id

    async def _fetch_channel_videos(self, channel_id: str, max_results: int = 50) -> List[Dict]:
        """Fetch all videos for a channel, sorted by newest first."""
        url = f"{self.base_url}/search"
        params = {
            "part": "snippet",
            "channelId": channel_id,
            "maxResults": max_results,
            "order": "date",
            "type": "video",
            "key": self.api_key,
        }
        async with httpx.AsyncClient(verify=certifi.where()) as client:
            response = await client.get(url, params=params)
            return response.json().get("items", [])

    async def _fetch_video_stats(self, video_id: str) -> Dict:
        """Fetch detailed stats for a video (views, likes, dislikes, etc.)."""
        url = f"{self.base_url}/videos"
        params = {
            "part": "statistics,snippet",
            "id": video_id,
            "key": self.api_key,
        }
        async with httpx.AsyncClient(verify=certifi.where()) as client:
            response = await client.get(url, params=params)
            return response.json().get("items", [{}])[0]

    def _calculate_relevance_score(self, video: Dict) -> float:
        """Calculate a relevance score based on engagement metrics."""
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 0))
        like_count = int(stats.get("likeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        # Simple weighted score (customize weights as needed)
        return (view_count * 0.4) + (like_count * 0.3) + (comment_count * 0.3)

    def _calculate_engagement_score(self, video: Dict) -> float:
        """Calculate an engagement score (CTR, like-to-dislike ratio, etc.)."""
        stats = video.get("statistics", {})
        view_count = int(stats.get("viewCount", 1))
        like_count = int(stats.get("likeCount", 0))
        dislike_count = int(stats.get("dislikeCount", 0))
        comment_count = int(stats.get("commentCount", 0))
        # Avoid division by zero
        like_ratio = like_count / (like_count + dislike_count + 1) if (like_count + dislike_count) > 0 else 0
        ctr = (like_count + comment_count) / view_count if view_count > 0 else 0
        return (like_ratio * 0.5) + (ctr * 0.5)

    async def get_videos_by_handle(
        self,
        handle: str,
        sort_by: str = "newest",
        max_results: int = 50,
    ) -> List[YouTubeVideo]:
        """
        Fetch videos for a YouTube handle and sort them.

        Args:
            handle: YouTube handle (e.g., "@username").
            sort_by: "newest", "relevance", or "engagement".
            max_results: Maximum number of videos to return.

        Returns:
            List of YouTubeVideo objects, sorted by the specified criteria.
        """
        channel_id = await self._fetch_channel_id(handle)
        if not channel_id:
            raise ValueError(f"Channel not found for handle: {handle}")

        videos = await self._fetch_channel_videos(channel_id, max_results)
        video_details = []
        for video in videos:
            video_id = video["id"]["videoId"]
            stats = await self._fetch_video_stats(video_id)
            if not stats:
                continue
            video_data = {
                **video["snippet"],
                **stats["statistics"],
                "published_at": video["snippet"]["publishedAt"],
                "video_id": video_id,
            }
            video_data["relevance_score"] = self._calculate_relevance_score(video_data)
            video_data["engagement_score"] = self._calculate_engagement_score(video_data)
            video_details.append(YouTubeVideo(**video_data))

        # Sort videos
        if sort_by == "newest":
            video_details.sort(key=lambda x: x.published_at, reverse=True)
        elif sort_by == "relevance":
            video_details.sort(key=lambda x: x.relevance_score, reverse=True)
        elif sort_by == "engagement":
            video_details.sort(key=lambda x: x.engagement_score, reverse=True)
        else:
            raise ValueError(f"Invalid sort_by: {sort_by}")

        return video_details[:max_results]

# Example usage
async def example_usage():
    handler = YouTubeHandler()
    videos = await handler.get_videos_by_handle("@username", sort_by="engagement")
    for video in videos:
        print(f"{video.title} (Score: {video.engagement_score:.2f})")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
