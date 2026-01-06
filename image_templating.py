# image_templating.py
from __future__ import annotations

import io
import os
import math
import httpx
import random

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, ImageOps


router = APIRouter(tags=["image-templating"])

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FONT_REL = "assets/FunnelSans-VariableFont_wght.ttf"

def _resolve_font_path(p: str) -> str:
    if not p:
        return ""
    # allow env var or passed paths with \ or /
    path = Path(p)
    if path.is_absolute():
        return str(path)
    # resolve relative to the module/app directory
    return str((PROJECT_ROOT / path).resolve())

# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class TemplateConfig:
    width: int = int(os.getenv("TEMPLATE_W", "1080"))
    height: int = int(os.getenv("TEMPLATE_H", "1920"))
    font_weight: int = int(os.getenv("FONT_WGHT", "600"))  # 100..900 typically

    # Flourishes sizing relative to width
    flourish_size_min_ratio: float = float(os.getenv("FLOURISH_MIN_RATIO", "0.10"))  # 10% of width
    flourish_size_max_ratio: float = float(os.getenv("FLOURISH_MAX_RATIO", "0.16"))  # 16% of width

    # Text margins
    text_side_margin: int = int(os.getenv("TEXT_SIDE_MARGIN", "40"))
    text_overlay_gap: int = int(os.getenv("TEXT_OVERLAY_GAP", "20"))  # above overlay top

    # Overlay
    overlay_height_ratio: float = float(os.getenv("OVERLAY_HEIGHT_RATIO", "0.3333333"))  # 1/3
    overlay_max_alpha: int = int(os.getenv("OVERLAY_MAX_ALPHA", "220"))  # slightly less than 255 looks nicer

    # Font
    # Put a real condensed serif font file path here. Examples:
    # - Windows: C:\\Windows\\Fonts\\georgiab.ttf (not condensed, but serif)
    # - Ship your own: assets/fonts/YourCondensedSerif.ttf
    font_path: str = os.getenv("FONT_TTF_PATH", DEFAULT_FONT_REL).strip()

    # Shadow
    shadow_offset: Tuple[int, int] = (2, 2)
    shadow_alpha: int = 160


CFG = TemplateConfig()


# ----------------------------
# Request / Response models
# ----------------------------

class RenderTemplateRequest(BaseModel):
    background_url: str = Field(..., description="BG image URL (can be ImageKit URL)")
    flourish_url_1: str = Field(..., description="Flourish image URL #1")
    flourish_url_2: str = Field(..., description="Flourish image URL #2")
    text: str = Field(..., description="Caption text")

    keep_subject_in_frame: bool = Field(
        False,
        description="Only applies to background: if true, never crop subject; pad with black bars if needed.",
    )

    seed: Optional[int] = Field(
        None,
        description="Optional RNG seed to make flourish placement reproducible.",
    )

    # Output
    fmt: str = Field("png", description="png or jpeg")
    jpeg_quality: int = Field(90, ge=50, le=95)

    # Optional: upload the result to ImageKit and return JSON instead of image bytes
    upload_to_imagekit: bool = Field(False)
    upload_file_name: str = Field("render.png")


class RenderTemplateUploadResponse(BaseModel):
    uploaded: bool
    url: Optional[str] = None
    fileId: Optional[str] = None
    name: Optional[str] = None


# ----------------------------
# ImageKit upload (optional)
# ----------------------------

def _imagekit_upload(file_bytes: bytes, file_name: str) -> dict:
    """
    Optional helper: uploads rendered bytes to ImageKit.

    Needs env vars:
      IMAGEKIT_PRIVATE_KEY
      IMAGEKIT_PUBLIC_KEY
      IMAGEKIT_URL_ENDPOINT (optional; not required for upload)
    """
    private_key = os.getenv("IMAGEKIT_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError("IMAGEKIT_PRIVATE_KEY not set")

    # ImageKit upload endpoint uses basic auth with private key.
    # Username: private_key, Password: blank
    upload_url = "https://upload.imagekit.io/api/v1/files/upload"

    import requests  # ok to use requests here (already in your project)

    files = {
        "file": (file_name, file_bytes),
    }
    data = {
        "fileName": file_name,
        # You can set "folder": "/renders" etc if you want:
        # "folder": "/renders",
        # "useUniqueFileName": "true",
    }

    resp = requests.post(upload_url, auth=(private_key, ""), files=files, data=data, timeout=60)
    if resp.status_code >= 400:
        raise RuntimeError(f"ImageKit upload failed: {resp.status_code} {resp.text[:400]}")
    return resp.json()


# ----------------------------
# Core helpers
# ----------------------------

async def _fetch_image(url: str) -> Image.Image:
    try:
        async with httpx.AsyncClient(timeout=25) as client:
            r = await client.get(url, follow_redirects=True)
            r.raise_for_status()
            content_type = (r.headers.get("content-type") or "").lower()
            if "image" not in content_type and not url.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                # still allow if server doesn't set content-type properly, but attempt open
                pass
            return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {url} ({e})")


def _fit_or_fill_background(bg: Image.Image, *, keep_subject_in_frame: bool) -> Image.Image:
    """
    Returns a 9:16 background in CFG.width x CFG.height.

    - keep_subject_in_frame=False -> fill canvas (center crop)
    - keep_subject_in_frame=True  -> contain inside canvas (no crop) + black padding
    """
    target = (CFG.width, CFG.height)
    if not keep_subject_in_frame:
        # Fill/crop to target (center)
        out = ImageOps.fit(bg, target, method=Image.LANCZOS, centering=(0.5, 0.5))
        return out.convert("RGBA")

    # Keep whole image: contain then pad with black
    contained = ImageOps.contain(bg, target, method=Image.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", target, (0, 0, 0, 255))
    x = (CFG.width - contained.width) // 2
    y = (CFG.height - contained.height) // 2
    canvas.alpha_composite(contained, (x, y))
    return canvas


def _circle_crop(img: Image.Image, size: int) -> Image.Image:
    img_sq = ImageOps.fit(img, (size, size), method=Image.LANCZOS, centering=(0.5, 0.5)).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.ellipse((0, 0, size - 1, size - 1), fill=255)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img_sq, (0, 0), mask=mask)
    return out


def _add_bottom_fade_overlay(base: Image.Image) -> None:
    """
    Adds black overlay from bottom to 1/3rd height, fading to transparent at overlay top.
    Mutates base in-place.
    """
    w, h = base.size
    overlay_h = int(h * CFG.overlay_height_ratio)

    overlay = Image.new("RGBA", (w, overlay_h), (0, 0, 0, 0))
    px = overlay.load()

    # y=overlay_h-1 is bottom, y=0 is top
    for y in range(overlay_h):
        # alpha: 0 at top -> max at bottom
        t = y / max(1, overlay_h - 1)
        alpha = int(CFG.overlay_max_alpha * t)
        for x in range(w):
            px[x, y] = (0, 0, 0, alpha)

    base.alpha_composite(overlay, (0, h - overlay_h))


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        fp = _resolve_font_path(CFG.font_path)
        if fp and Path(fp).exists():
            font = ImageFont.truetype(fp, size=size)

            # ---- variable font axis control (wght) ----
            try:
                axes = font.get_variation_axes()  # raises if not variable font
                values = [ax.default for ax in axes]

                for i, ax in enumerate(axes):
                    if getattr(ax, "tag", None) == "wght":
                        w = float(CFG.font_weight)
                        w = max(float(ax.minimum), min(float(ax.maximum), w))
                        values[i] = w
                        break

                font.set_variation_by_axes(values)
            except Exception:
                pass

            return font
    except Exception:
        pass

    return ImageFont.load_default()



def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow-compatible bbox measurement
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h


def _fit_font_size_for_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    *,
    start_size: int = 96,
    min_size: int = 18,
) -> Tuple[ImageFont.ImageFont, int, int]:
    """
    Binary search font size so text fits within max_width.
    Returns (font, text_w, text_h).
    """
    lo, hi = min_size, start_size
    best = None

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(mid)
        tw, th = _measure_text(draw, text, font)
        if tw <= max_width:
            best = (font, tw, th)
            lo = mid + 1
        else:
            hi = mid - 1

    if best is None:
        # Force min size even if it overflows
        font = _load_font(min_size)
        tw, th = _measure_text(draw, text, font)
        return font, tw, th

    return best


def _draw_text_with_shadow(
    base: Image.Image,
    text: str,
    *,
    y: int,
) -> None:
    """
    Centered text, white with slight black shadow.
    """
    draw = ImageDraw.Draw(base)
    max_text_w = CFG.width - (CFG.text_side_margin * 2)

    font, tw, th = _fit_font_size_for_width(draw, text, max_text_w)

    x = (CFG.width - tw) // 2

    # Shadow
    sx, sy = CFG.shadow_offset
    shadow = (0, 0, 0, CFG.shadow_alpha)
    draw.text((x + sx, y + sy), text, font=font, fill=shadow)

    # Main
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))


def _place_flourishes(
    base: Image.Image,
    f1: Image.Image,
    f2: Image.Image,
    *,
    rng: random.Random,
) -> None:
    """
    Random positions over background; keeps them out of bottom overlay area (top 2/3 region).
    """
    w, h = base.size
    overlay_h = int(h * CFG.overlay_height_ratio)
    safe_bottom = h - overlay_h  # overlay starts here
    max_y = max(0, safe_bottom - int(0.10 * h))  # keep a bit above overlay start

    def place_one(fimg: Image.Image):
        size = int(rng.uniform(CFG.flourish_size_min_ratio, CFG.flourish_size_max_ratio) * w)
        size = max(32, min(size, w // 3))
        blob = _circle_crop(fimg, size)

        # random position
        x = rng.randint(0, max(0, w - size))
        y = rng.randint(0, max(0, max_y - size))

        base.alpha_composite(blob, (x, y))

    place_one(f1)
    place_one(f2)


# ----------------------------
# Main render function
# ----------------------------

async def render_template(req: RenderTemplateRequest) -> Tuple[bytes, str]:
    rng = random.Random(req.seed)

    bg = await _fetch_image(req.background_url)
    fl1 = await _fetch_image(req.flourish_url_1)
    fl2 = await _fetch_image(req.flourish_url_2)

    # 1) Background to 9:16
    canvas = _fit_or_fill_background(bg, keep_subject_in_frame=req.keep_subject_in_frame)

    # 2) Flourishes
    _place_flourishes(canvas, fl1, fl2, rng=rng)

    # 3) Bottom overlay gradient
    _add_bottom_fade_overlay(canvas)

    # 4) Text position: overlay top is (h - overlay_h)
    overlay_h = int(CFG.height * CFG.overlay_height_ratio)
    overlay_top_y = CFG.height - overlay_h

    # "just above the black overlay with margin 20px from top edge of overlay"
    # => place text baseline at (overlay_top_y - 20px - text_height).
    # We'll compute y after measuring at chosen font size by letting the fitter return th.
    draw = ImageDraw.Draw(canvas)
    max_text_w = CFG.width - (CFG.text_side_margin * 2)
    font, tw, th = _fit_font_size_for_width(draw, req.text, max_text_w)

    text_y = overlay_top_y - CFG.text_overlay_gap - th
    text_y = max(20, text_y)  # keep on-canvas

    # redraw with same fitted font size:
    # easiest: temporarily override loader by drawing directly
    # We'll just call _draw_text_with_shadow which refits; acceptable since deterministic for same text.
    _draw_text_with_shadow(canvas, req.text, y=text_y)

    # Encode
    out = io.BytesIO()
    fmt = (req.fmt or "png").lower().strip()
    if fmt not in ("png", "jpeg", "jpg"):
        raise HTTPException(status_code=400, detail="fmt must be png or jpeg")

    if fmt in ("jpeg", "jpg"):
        # JPEG doesn't support alpha
        rgb = canvas.convert("RGB")
        rgb.save(out, format="JPEG", quality=req.jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        canvas.save(out, format="PNG", optimize=True)
        mime = "image/png"

    return out.getvalue(), mime


# ----------------------------
# FastAPI endpoint
# ----------------------------

@router.post(
    "/render",
    response_class=StreamingResponse,
    summary="Render 9:16 template from background + 2 flourishes + text",
)
async def render_endpoint(body: RenderTemplateRequest):
    img_bytes, mime = await render_template(body)

    if body.upload_to_imagekit:
        try:
            uploaded = _imagekit_upload(img_bytes, body.upload_file_name)
            # Return JSON instead of image
            return RenderTemplateUploadResponse(
                uploaded=True,
                url=uploaded.get("url"),
                fileId=uploaded.get("fileId"),
                name=uploaded.get("name"),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rendered, but upload failed: {e}")

    return StreamingResponse(io.BytesIO(img_bytes), media_type=mime)

@router.get("/debug/font")
def debug_font():
    from pathlib import Path
    fp = _resolve_font_path(CFG.font_path)
    return {"configured": CFG.font_path, "resolved": fp, "exists": Path(fp).exists()}
