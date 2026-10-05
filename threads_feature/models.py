from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, model_validator


class BootstrapTokenRequest(BaseModel):
    account: str
    access_token: str = Field(min_length=10)
    expires_in: int = Field(default=5_184_000, ge=60)


class ImagePublishRequest(BaseModel):
    account: str
    image_url: HttpUrl
    text: str = ""
    caption: str = ""
    title: str = ""
    alt_text: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)
    reply_control: Literal["everyone", "accounts_you_follow", "mentioned_only"] | None = None

    @model_validator(mode="after")
    def normalize_text(self):
        if not self.text and self.caption:
            self.text = self.caption
        return self


class VideoPublishRequest(BaseModel):
    account: str
    video_url: HttpUrl
    title: str = ""
    caption: str = ""
    text: str = ""
    alt_text: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)
    reply_control: Literal["everyone", "accounts_you_follow", "mentioned_only"] | None = None

    @model_validator(mode="after")
    def normalize_text(self):
        if not self.text:
            parts = [self.title.strip(), self.caption.strip()]
            self.text = "\n\n".join(x for x in parts if x)
        return self


class UnifiedPublishRequest(BaseModel):
    account: str
    media_type: Literal["IMAGE", "VIDEO", "image", "video"]
    image_url: HttpUrl | None = None
    video_url: HttpUrl | None = None
    title: str = ""
    caption: str = ""
    text: str = ""
    alt_text: str | None = None
    idempotency_key: str | None = Field(default=None, max_length=255)
    reply_control: Literal["everyone", "accounts_you_follow", "mentioned_only"] | None = None

    @model_validator(mode="after")
    def validate_media(self):
        mt = self.media_type.upper()
        self.media_type = mt
        if mt == "IMAGE" and self.image_url is None:
            raise ValueError("image_url is required for IMAGE")
        if mt == "VIDEO" and self.video_url is None:
            raise ValueError("video_url is required for VIDEO")
        if not self.text:
            if mt == "VIDEO":
                self.text = "\n\n".join(x for x in (self.title.strip(), self.caption.strip()) if x)
            else:
                self.text = self.caption
        return self
