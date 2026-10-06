from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, HttpUrl, model_validator

class TumblrPublishRequest(BaseModel):
    account: str
    media_type: Literal["IMAGE", "VIDEO", "image", "video"]
    image_url: HttpUrl | None = None
    video_url: HttpUrl | None = None
    text: str = ""
    caption: str = ""
    tags: list[str] = Field(default_factory=list)
    state: Literal["published", "draft", "queue", "private"] = "published"
    idempotency_key: str | None = Field(default=None, max_length=255)

    @model_validator(mode="after")
    def normalize(self):
        self.media_type = self.media_type.upper()
        if self.media_type == "IMAGE" and self.image_url is None:
            raise ValueError("image_url is required for IMAGE")
        if self.media_type == "VIDEO" and self.video_url is None:
            raise ValueError("video_url is required for VIDEO")
        if not self.text:
            self.text = self.caption
        self.tags = [str(x).strip().lstrip("#") for x in self.tags if str(x).strip()]
        return self
