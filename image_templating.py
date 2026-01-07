# image_templating.py
from __future__ import annotations

import io
import re
import os
import httpx
import random

from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont, ImageOps


router = APIRouter(tags=["image-templating"])

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_FONT_REL = "assets/Inconsolata-VariableFont_wdth,wght.ttf"

NAMED_COLORS = {
    "white": (255, 255, 255, 255),
    "black": (0, 0, 0, 255),
    "yellow": (255, 235, 59, 255),
    "red": (244, 67, 54, 255),
    "orange": (255, 152, 0, 255),
    "green": (76, 175, 80, 255),
    "pink": (233, 30, 99, 255),
    "fuchsia": (255, 0, 255, 255),
    "magenta": (255, 0, 255, 255),
    "purple": (156, 39, 176, 255),
    "blue": (33, 150, 243, 255),
    "cyan": (0, 188, 212, 255),
    "gray": (158, 158, 158, 255),
    "grey": (158, 158, 158, 255),
}

HEX_RE = re.compile(r"^#([0-9a-fA-F]{6})$")

def _resolve_font_path(p: str) -> str:
    if not p:
        return ""
    # allow env var or passed paths with \ or /
    path = Path(p)
    if path.is_absolute():
        return str(path)
    # resolve relative to the module/app directory
    return str((PROJECT_ROOT / path).resolve())

def _parse_color_token(tok: str) -> Optional[Tuple[int, int, int, int]]:
    if not tok:
        return None
    t = tok.strip().lower()

    if t in NAMED_COLORS:
        return NAMED_COLORS[t]

    m = HEX_RE.match(t)
    if m:
        hexv = m.group(1)
        r = int(hexv[0:2], 16)
        g = int(hexv[2:4], 16)
        b = int(hexv[4:6], 16)
        return (r, g, b, 255)

    return None


# Returns list of spans: [(text, rgba), ...]
def _parse_highlight_spans(text: str, *, default_rgba=(255, 255, 255, 255)) -> list[tuple[str, Tuple[int, int, int, int]]]:
    # Supports: [yellow]...[/yellow] and [#FFAA00]...[/#FFAA00]
    # Non-nested, simple, forgiving.
    spans: list[tuple[str, Tuple[int, int, int, int]]] = []
    i = 0
    cur_color = default_rgba

    tag_open = re.compile(r"\[([#a-zA-Z0-9]+)\]")
    tag_close = re.compile(r"\[/([#a-zA-Z0-9]+)\]")

    while i < len(text):
        m_open = tag_open.search(text, i)
        m_close = tag_close.search(text, i)

        # pick earliest tag
        m = None
        is_open = False
        if m_open and m_close:
            if m_open.start() < m_close.start():
                m = m_open
                is_open = True
            else:
                m = m_close
                is_open = False
        elif m_open:
            m = m_open
            is_open = True
        elif m_close:
            m = m_close
            is_open = False

        if not m:
            # no more tags
            if i < len(text):
                spans.append((text[i:], cur_color))
            break

        # emit text before tag
        if m.start() > i:
            spans.append((text[i:m.start()], cur_color))

        tok = m.group(1)

        if is_open:
            newc = _parse_color_token(tok)
            if newc is not None:
                cur_color = newc
            else:
                # unknown tag: keep it as literal text
                spans.append((m.group(0), cur_color))
        else:
            # closing tag: if it matches any known color/hex, just reset to default
            # (we’re not doing nested stack here)
            closec = _parse_color_token(tok)
            if closec is not None:
                cur_color = default_rgba
            else:
                spans.append((m.group(0), cur_color))

        i = m.end()

    # merge adjacent spans of same color
    merged: list[tuple[str, Tuple[int, int, int, int]]] = []
    for t, c in spans:
        if not t:
            continue
        if merged and merged[-1][1] == c:
            merged[-1] = (merged[-1][0] + t, c)
        else:
            merged.append((t, c))

    return merged

def _wrap_spans(draw: ImageDraw.ImageDraw, spans, font, max_width: int):
    """
    Wrap colored spans into lines, respecting HARD line breaks on '\n'.
    Returns: list[list[(text, rgba)]]
    """
    lines = []
    cur_line = []
    cur_w = 0.0

    def flush_line():
        nonlocal cur_line, cur_w
        # clean leading whitespace token
        while cur_line and cur_line[0][0].isspace():
            cur_line.pop(0)
        if cur_line:
            t0, c0 = cur_line[0]
            cur_line[0] = (t0.lstrip(), c0)
        lines.append(cur_line)
        cur_line = []
        cur_w = 0.0

    def push_token(tok: str, color):
        nonlocal cur_line, cur_w
        if tok == "":
            return
        pw = draw.textlength(tok, font=font)
        if cur_w + pw <= max_width or not cur_line:
            cur_line.append((tok, color))
            cur_w += pw
        else:
            flush_line()
            cur_line.append((tok.lstrip(), color))
            cur_w = draw.textlength(cur_line[0][0], font=font)

    for text, color in spans:
        if not text:
            continue

        # split by newline but keep the newline markers
        parts = re.split(r"(\n)", text)

        for part in parts:
            if part == "":
                continue

            if part == "\n":
                # hard break
                flush_line()
                continue

            # normal wrapping within this part
            tokens = re.split(r"(\s+)", part)
            for tok in tokens:
                if tok == "":
                    continue
                push_token(tok, color)

    if cur_line:
        flush_line()

    # Merge adjacent spans of same color within each line
    merged_lines = []
    for ln in lines:
        merged = []
        for t, c in ln:
            if not t:
                continue
            if merged and merged[-1][1] == c:
                merged[-1] = (merged[-1][0] + t, c)
            else:
                merged.append((t, c))
        merged_lines.append(merged)

    # Remove any empty lines produced by consecutive flushes (optional)
    # If you WANT blank lines, keep them (return empty lists).
    return merged_lines

def _line_width_spans(draw: ImageDraw.ImageDraw, line_spans, font, tracking_px: int) -> int:
    # width of a line of spans, char-tracking aware
    w = 0.0
    for seg_text, _seg_color in line_spans:
        if not seg_text:
            continue
        for idx, ch in enumerate(seg_text):
            w += draw.textlength(ch, font=font)
            if idx != len(seg_text) - 1:
                w += tracking_px
        # tracking between segments should also count if next segment exists and last char exists
        if seg_text and line_spans is not None:
            pass
    # add tracking between segments where appropriate:
    # easiest: rebuild full string and compute char-tracked width once
    full = "".join(t for t, _ in line_spans)
    if not full:
        return 0
    return int(sum(draw.textlength(ch, font=font) for ch in full) + tracking_px * (len(full) - 1))

def _draw_spans_tracked(draw, x, y, spans, font, tracking_px: int, shadow_rgba=None):
    cx = x
    for seg_text, seg_color in spans:
        for ch in seg_text:
            if shadow_rgba is not None:
                draw.text((cx, y), ch, font=font, fill=shadow_rgba)
            draw.text((cx, y), ch, font=font, fill=seg_color)
            cw = draw.textlength(ch, font=font)
            cx += int(cw) + tracking_px


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class TemplateConfig:
    width: int = int(os.getenv("TEMPLATE_W", "1080"))
    height: int = int(os.getenv("TEMPLATE_H", "1920"))
    font_weight: int = int(os.getenv("FONT_WGHT", "400"))  # default 400 now

    font_wdth: int = int(os.getenv("FONT_WDTH", "75"))  # <100 = condensed
    dial_radius_ratio: float = float(os.getenv("DIAL_RADIUS_RATIO", "0.1666667"))  # 1/6
    flourish_scale: float = float(os.getenv("FLOURISH_SCALE", "2.4"))  # 240%

    # Typography tweaks
    font_start_size: int = int(os.getenv("FONT_START_SIZE", "84"))      # was ~96
    font_min_size: int = int(os.getenv("FONT_MIN_SIZE", "18"))
    line_spacing_ratio: float = float(os.getenv("LINE_SPACING_RATIO", "1.02"))  # tighter (was 1.12)
    tracking_px: int = int(os.getenv("TRACKING_PX", "-2"))              # negative = tighter tracking

    # Flourishes sizing relative to width
    flourish_size_min_ratio: float = float(os.getenv("FLOURISH_MIN_RATIO", "0.10"))  # 10% of width
    flourish_size_max_ratio: float = float(os.getenv("FLOURISH_MAX_RATIO", "0.16"))  # 16% of width

    # Text margins
    text_side_margin: int = int(os.getenv("TEXT_SIDE_MARGIN", "40"))

    # Overlay
    overlay_height_ratio: float = float(os.getenv("OVERLAY_HEIGHT_RATIO", "0.6"))  # 3/5
    overlay_max_alpha: int = int(os.getenv("OVERLAY_MAX_ALPHA", "220"))  # slightly less than 255 looks nicer
    overlay_curve_gamma: float = float(os.getenv("OVERLAY_CURVE_GAMMA", "0.45"))

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
    flourish_url_1: Optional[str] = None
    flourish_url_2: Optional[str] = None
    text: str = Field(..., description="Caption text")
    uppercase: bool = Field(True, description="If true, force ALL CAPS. If false, keep input casing.")
    font_width: Optional[int] = Field(None, description="Overrides FONT_WDTH (50..200).")
    font_weight: Optional[int] = Field(None, description="Overrides FONT_WGHT (200..900).")
    debug: bool = Field(False, description="If true, return JSON debug info instead of rendering/uploading image.")
    enable_highlights: bool = Field(False, description="Enable inline color tags like [yellow]...[/yellow] or [#FF5733]...[ /#FF5733 ]")

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

class DeleteImageRequest(BaseModel):
    fileId: str = Field(..., description="ImageKit fileId to delete")

class DeleteImageResponse(BaseModel):
    deleted: bool
    fileId: str

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
            img = Image.open(io.BytesIO(r.content))
            img.load()
            return img.convert("RGBA")
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
    Adds black overlay from bottom up to overlay_height_ratio of image height,
    fading to transparent at overlay top. Uses a non-linear curve so it stays
    darker higher up (more "eccentric" toward black).
    """
    w, h = base.size
    overlay_h = int(h * CFG.overlay_height_ratio)
    overlay_h = max(1, min(h, overlay_h))

    overlay = Image.new("RGBA", (w, overlay_h), (0, 0, 0, 0))
    px = overlay.load()

    gamma = getattr(CFG, "overlay_curve_gamma", 0.45)  # < 1 => darker earlier/higher

    # y=0 top of overlay, y=overlay_h-1 bottom
    for y in range(overlay_h):
        # t: 0 at top, 1 at bottom
        t = y / max(1, overlay_h - 1)
        # non-linear curve: keeps it darker higher up
        a = int(CFG.overlay_max_alpha * (t ** gamma))
        for x in range(w):
            px[x, y] = (0, 0, 0, a)

    base.alpha_composite(overlay, (0, h - overlay_h))

def _load_font(size: int, *, weight: Optional[int] = None, width: Optional[int] = None):
    fp = _resolve_font_path(CFG.font_path)
    if not (fp and Path(fp).exists()):
        return ImageFont.load_default()

    font = ImageFont.truetype(fp, size=size)

    try:
        axes = font.get_variation_axes()

        # axes can be dicts or objects depending on Pillow build/font backend
        def ax_tag(ax):
            t = ax.get("tag") if isinstance(ax, dict) else getattr(ax, "tag", None)
            if isinstance(t, (bytes, bytearray)):
                t = t.decode("ascii", errors="ignore")
            return t

        def ax_min(ax):
            return float(ax["minimum"] if isinstance(ax, dict) else ax.minimum)

        def ax_max(ax):
            return float(ax["maximum"] if isinstance(ax, dict) else ax.maximum)

        def ax_def(ax):
            return float(ax["default"] if isinstance(ax, dict) else ax.default)

        values = [ax_def(ax) for ax in axes]

        w_req  = float(weight if weight is not None else CFG.font_weight)
        wd_req = float(width  if width  is not None else CFG.font_wdth)

        # First try tag-based (if tags exist)
        tag_map = {}
        for i, ax in enumerate(axes):
            t = ax_tag(ax)
            if t:
                tag_map[t] = i

        if "wght" in tag_map:
            i = tag_map["wght"]
            values[i] = max(ax_min(axes[i]), min(ax_max(axes[i]), w_req))
        if "wdth" in tag_map:
            i = tag_map["wdth"]
            values[i] = max(ax_min(axes[i]), min(ax_max(axes[i]), wd_req))

        # Fallback: tags are None -> assume [0]=wght, [1]=wdth
        if not tag_map:
            if len(axes) >= 1:
                values[0] = max(ax_min(axes[0]), min(ax_max(axes[0]), w_req))
            if len(axes) >= 2:
                values[1] = max(ax_min(axes[1]), min(ax_max(axes[1]), wd_req))

        font.set_variation_by_axes(values)
    except Exception:
        pass


    return font

def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> Tuple[int, int]:
    # Pillow-compatible bbox measurement
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    return w, h

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """
    Greedy wrap with HARD line breaks on '\n'.
    """
    text = (text or "")
    if text == "":
        return [""]

    out_lines: list[str] = []

    # Split into paragraphs by explicit newlines
    paragraphs = text.split("\n")

    for p_i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            # keep blank line
            out_lines.append("")
            continue

        words = para.split()  # spaces within the paragraph
        cur = ""

        for w in words:
            candidate = w if not cur else (cur + " " + w)
            tw, _ = _measure_text(draw, candidate, font)
            if tw <= max_width:
                cur = candidate
            else:
                if cur:
                    out_lines.append(cur)
                cur = w

                # hard-break long single words
                tw2, _ = _measure_text(draw, cur, font)
                if tw2 > max_width:
                    chunk = ""
                    for ch in w:
                        cand2 = chunk + ch
                        t3, _ = _measure_text(draw, cand2, font)
                        if t3 <= max_width:
                            chunk = cand2
                        else:
                            if chunk:
                                out_lines.append(chunk)
                            chunk = ch
                    cur = chunk

        if cur:
            out_lines.append(cur)

        # NOTE: don't add extra line here; split("\n") already enforces it
        # but we DO want to preserve empty lines (handled above).

    return out_lines

def _draw_dial(base: Image.Image, *, label="PGP", width: Optional[int] = None, weight: Optional[int] = None):
    draw = ImageDraw.Draw(base)
    w, h = base.size
    r = int(CFG.width * CFG.dial_radius_ratio)  # 1/6 width

    # Quarter circle: bounding box whose center is bottom-right corner
    bbox = (w - 2*r, h - 2*r, w, h)

    # Fill dial (slightly transparent black)
    draw.pieslice(bbox, start=180, end=270, fill=(0, 0, 0, 170))

    # Small "PGP" text inside dial
    font = _load_font(26, weight=weight, width=width)
    tx, ty = w - int(r * 0.78), h - int(r * 0.55)
    draw.text((tx, ty), label, font=font, fill=(255, 255, 255, 220))

    # Simple thin arrow under PGP (line + small arrowhead)
    ax1, ay = tx, ty + 34
    ax2 = ax1 + 48
    draw.line((ax1, ay, ax2, ay), fill=(255, 255, 255, 220), width=2)
    draw.line((ax2, ay, ax2 - 10, ay - 6), fill=(255, 255, 255, 220), width=2)
    draw.line((ax2, ay, ax2 - 10, ay + 6), fill=(255, 255, 255, 220), width=2)


def _fit_font_for_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    max_width: int,
    *,
    start_size: int = CFG.font_start_size,
    min_size: int = CFG.font_min_size,
    line_spacing_ratio: float = CFG.line_spacing_ratio,
    weight: Optional[int] = None,
    width: Optional[int] = None,
) -> tuple[ImageFont.ImageFont, list[str], int, int, int]:
    """
    Finds the largest font size where wrapped text fits within max_width.
    Returns (font, lines, block_w, block_h, line_h).
    """
    for size in range(start_size, min_size - 1, -1):
        font = _load_font(size, weight=weight, width=width)
        lines = _wrap_text(draw, text, font, max_width)

        # line height from font metrics
        ascent, descent = font.getmetrics()
        line_h = int((ascent + descent) * line_spacing_ratio)

        widths = [_measure_text(draw, ln, font)[0] for ln in lines] if lines else [0]
        block_w = max(widths) if widths else 0
        block_h = (len(lines) * line_h) if lines else line_h

        if block_w <= max_width:
            return font, lines, block_w, block_h, line_h

    # fallback
    font = _load_font(min_size, weight=weight, width=width)
    lines = _wrap_text(draw, text, font, max_width)
    ascent, descent = font.getmetrics()
    line_h = int((ascent + descent) * line_spacing_ratio)
    widths = [_measure_text(draw, ln, font)[0] for ln in lines] if lines else [0]
    block_w = max(widths) if widths else 0
    block_h = (len(lines) * line_h) if lines else line_h
    return font, lines, block_w, block_h, line_h

def _draw_text_tracked(draw, x, y, text, font, fill, tracking_px: int):
    if tracking_px == 0:
        draw.text((x, y), text, font=font, fill=fill)
        return

    cx = x
    for ch in text:
        draw.text((cx, y), ch, font=font, fill=fill)
        cw = draw.textlength(ch, font=font)
        cx += int(cw) + tracking_px


def _draw_text_with_shadow_bottom_anchored(
    base: Image.Image,
    text: str,
    *,
    bottom_padding: int = 30,
    tracking_px: int = 0,
    weight: Optional[int] = None,
    width: Optional[int] = None,
    enable_highlights: bool = False,
) -> None:
    draw = ImageDraw.Draw(base)
    max_text_w = CFG.width - (CFG.text_side_margin * 2)

    # choose font size using plain text (strip tags) so sizing is stable
    plain = re.sub(r"\[/?[#a-zA-Z0-9]+\]", "", text) if enable_highlights else text

    font, _lines_plain, _bw, block_h, line_h = _fit_font_for_wrapped_text(
        draw, plain, max_text_w, weight=weight, width=width
    )

    # build colored lines
    if enable_highlights:
        spans = _parse_highlight_spans(text, default_rgba=(255, 255, 255, 255))
        lines = _wrap_spans(draw, spans, font, max_text_w)  # list of span-lines
    else:
        # fallback: single-color spans
        spans = [(text, (255, 255, 255, 255))]
        lines = _wrap_spans(draw, spans, font, max_text_w)

    # recompute block height based on actual wrapped lines
    block_h = (len(lines) * line_h) if lines else line_h

    y0 = CFG.height - bottom_padding - block_h
    y0 = max(0, y0)

    sx, sy = CFG.shadow_offset
    shadow_rgba = (0, 0, 0, CFG.shadow_alpha)

    y = y0
    for line_spans in lines:
        tracked_w = _line_width_spans(draw, line_spans, font, tracking_px)
        x = (CFG.width - tracked_w) // 2

        # shadow
        _draw_spans_tracked(draw, x + sx, y + sy, line_spans, font, tracking_px, shadow_rgba=shadow_rgba)
        # main
        _draw_spans_tracked(draw, x, y, line_spans, font, tracking_px, shadow_rgba=None)

        y += line_h

def _place_flourishes(
    base: Image.Image,
    f1: Image.Image,
    f2: Image.Image,
    *,
    rng: random.Random,
) -> None:
    w, h = base.size

    # Keep blobs out of the bottom overlay/text region:
    # with overlay at 0.6h, overlay starts at 0.4h, so allow blobs only above ~0.4h
    overlay_h = int(h * CFG.overlay_height_ratio)
    overlay_top = h - overlay_h

    # give a little extra safety margin above overlay start
    max_y = max(0, overlay_top - int(0.06 * h))

    def place_one(fimg: Image.Image):
        size = int(rng.uniform(CFG.flourish_size_min_ratio, CFG.flourish_size_max_ratio) * w)
        size = int(size * CFG.flourish_scale)  # 2.4x
        size = max(32, min(size, w))
        blob = _circle_crop(fimg, size)

        x = rng.randint(0, max(0, w - size))
        y = rng.randint(0, max(0, max_y - size))
        base.alpha_composite(blob, (x, y))

    place_one(f1)
    place_one(f2)

def _font_debug_info(eff_wght: int, eff_wdth: int):
    fp = _resolve_font_path(CFG.font_path)
    info = {
        "resolved": fp,
        "exists": Path(fp).exists(),
        "config_defaults": {"wght": CFG.font_weight, "wdth": CFG.font_wdth},
        "effective": {"wght": eff_wght, "wdth": eff_wdth},
    }
    if not info["exists"]:
        info["used_default_font"] = True
        return info

    try:
        f = ImageFont.truetype(fp, size=40)
        axes = f.get_variation_axes() or []
        metas = [_axis_meta(a) for a in axes]
        info["axes"] = metas

        # try applying
        g = _load_font(40, weight=eff_wght, width=eff_wdth)
        # if no exception, we consider it applicable
        info["can_set_variations"] = True
    except Exception as e:
        info["can_set_variations"] = False
        info["error"] = str(e)

    return info


def _clamp(v: Optional[int], lo: int, hi: int, default: int) -> int:
    if v is None:
        return default
    try:
        v = int(v)
    except Exception:
        return default
    return max(lo, min(hi, v))


def _axis_meta(ax):
    # Pillow may return dicts OR objects depending on version/build.
    if isinstance(ax, dict):
        return {
            "tag": ax.get("tag"),
            "min": ax.get("minValue") or ax.get("minimum") or ax.get("min"),
            "max": ax.get("maxValue") or ax.get("maximum") or ax.get("max"),
            "default": ax.get("defaultValue") or ax.get("default"),
        }
    # object-like
    return {
        "tag": getattr(ax, "tag", None),
        "min": getattr(ax, "minimum", None),
        "max": getattr(ax, "maximum", None),
        "default": getattr(ax, "default", None),
    }


def _load_font(size: int, *, weight: Optional[int] = None, width: Optional[int] = None):
    fp = _resolve_font_path(CFG.font_path)
    if not (fp and Path(fp).exists()):
        return ImageFont.load_default()

    font = ImageFont.truetype(fp, size=size)

    # Decide requested values (use CFG defaults if not provided)
    w_req = float(weight if weight is not None else CFG.font_weight)
    wd_req = float(width if width is not None else CFG.font_wdth)

    try:
        axes = font.get_variation_axes() or []
        if not axes:
            return font

        metas = [_axis_meta(a) for a in axes]
        values = [m["default"] for m in metas]

        # If tags exist, use them; otherwise infer by ranges (your axes have tag=None)
        for i, m in enumerate(metas):
            tag = m["tag"]
            if isinstance(tag, (bytes, bytearray)):
                tag = tag.decode("ascii", errors="ignore")

            mn, mx = float(m["min"]), float(m["max"])

            if tag == "wght" or (tag is None and mn <= 200 <= mx and mn <= 900 <= mx):
                values[i] = max(mn, min(mx, w_req))
            elif tag == "wdth" or (tag is None and mn <= 50 <= mx and mn <= 200 <= mx):
                values[i] = max(mn, min(mx, wd_req))

        font.set_variation_by_axes(values)
    except Exception:
        # If variation fails, you still get the base font
        pass

    return font

def _imagekit_delete(file_id: str) -> dict:
    private_key = os.getenv("IMAGEKIT_PRIVATE_KEY", "").strip()
    if not private_key:
        raise RuntimeError("IMAGEKIT_PRIVATE_KEY not set")

    import requests

    url = f"https://api.imagekit.io/v1/files/{file_id}"
    resp = requests.delete(url, auth=(private_key, ""), timeout=30)

    if resp.status_code >= 400:
        raise RuntimeError(f"ImageKit delete failed: {resp.status_code} {resp.text[:400]}")

    # ImageKit typically returns JSON or empty; normalize
    try:
        return resp.json()
    except Exception:
        return {"deleted": True, "fileId": file_id}

# ----------------------------
# Main render function
# ----------------------------

async def render_template(req: RenderTemplateRequest) -> Tuple[bytes, str]:
    caption = (req.text or "")
    if req.uppercase:
        caption = caption.upper()

    rng = random.Random(req.seed)

    # Compute effective (clamped) variation values ONCE
    eff_wght = _clamp(req.font_weight, 200, 900, CFG.font_weight)
    eff_wdth = _clamp(req.font_width, 50, 200, CFG.font_wdth)

    bg = await _fetch_image(req.background_url)
    canvas = _fit_or_fill_background(bg, keep_subject_in_frame=req.keep_subject_in_frame)

    fl1 = await _fetch_image(req.flourish_url_1) if req.flourish_url_1 else None
    fl2 = await _fetch_image(req.flourish_url_2) if req.flourish_url_2 else None

    if fl1 and fl2:
        _place_flourishes(canvas, fl1, fl2, rng=rng)
    elif fl1:
        _place_flourishes(canvas, fl1, fl1, rng=rng)
    elif fl2:
        _place_flourishes(canvas, fl2, fl2, rng=rng)

    _add_bottom_fade_overlay(canvas)

    # Use SAME effective values for dial + text
    _draw_dial(canvas, label="PGP", width=eff_wdth, weight=eff_wght)

    _draw_text_with_shadow_bottom_anchored(
        canvas,
        caption,
        bottom_padding=60,
        tracking_px=CFG.tracking_px,
        weight=eff_wght,
        width=eff_wdth,
        enable_highlights=req.enable_highlights,
    )

    out = io.BytesIO()
    fmt = (req.fmt or "png").lower().strip()
    if fmt not in ("png", "jpeg", "jpg"):
        raise HTTPException(status_code=400, detail="fmt must be png or jpeg")

    if fmt in ("jpeg", "jpg"):
        canvas.convert("RGB").save(out, format="JPEG", quality=req.jpeg_quality, optimize=True)
        mime = "image/jpeg"
    else:
        canvas.save(out, format="PNG", optimize=True)
        mime = "image/png"

    return out.getvalue(), mime

# ----------------------------
# FastAPI endpoint
# ----------------------------

@router.post("/render", summary="Render 9:16 template from background + 2 flourishes + text")
async def render_endpoint(body: RenderTemplateRequest):
    # Debug mode: return JSON ONLY (no render/upload)
    if body.debug:
        eff_wght = _clamp(body.font_weight, 200, 900, CFG.font_weight)
        eff_wdth = _clamp(body.font_width, 50, 200, CFG.font_wdth)
        return JSONResponse(_font_debug_info(eff_wght, eff_wdth))

    img_bytes, mime = await render_template(body)

    if body.upload_to_imagekit:
        try:
            uploaded = _imagekit_upload(img_bytes, body.upload_file_name)
            return {
                "uploaded": True,
                "url": uploaded.get("url"),
                "fileId": uploaded.get("fileId"),
                "name": uploaded.get("name"),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Rendered, but upload failed: {e}")

    return StreamingResponse(io.BytesIO(img_bytes), media_type=mime)

@router.post("/delete", summary="Delete an uploaded image from ImageKit by fileId")
def delete_endpoint(body: DeleteImageRequest):
    try:
        _imagekit_delete(body.fileId)
        return DeleteImageResponse(deleted=True, fileId=body.fileId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/debug/font_render")
def debug_font_render(wght: int = 900, wdth: int = 50):
    fp = _resolve_font_path(CFG.font_path)
    img = Image.new("RGBA", (1400, 300), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)

    f = _load_font(140, weight=wght, width=wdth)
    d.text((20, 40), f"W:{wght} D:{wdth} Inconsolata", font=f, fill=(0, 0, 0, 255))

    out = io.BytesIO()
    img.save(out, format="PNG")
    out.seek(0)
    return StreamingResponse(out, media_type="image/png")


@router.get("/debug/font")
def debug_font():
    from pathlib import Path
    fp = _resolve_font_path(CFG.font_path)
    return {"configured": CFG.font_path, "resolved": fp, "exists": Path(fp).exists()}

@router.get("/debug/font_axes")
def debug_font_axes():
    fp = _resolve_font_path(CFG.font_path)
    if not Path(fp).exists():
        return {"error": "font file not found", "resolved": fp}

    try:
        font = ImageFont.truetype(fp, size=40)
        axes = font.get_variation_axes()

        out = []
        for a in axes:
            # Pillow may return dicts OR objects depending on version/build
            if isinstance(a, dict):
                tag = a.get("tag")
                mn = a.get("minimum")
                mx = a.get("maximum")
                d  = a.get("default")
            else:
                tag = getattr(a, "tag", None)
                mn = getattr(a, "minimum", None)
                mx = getattr(a, "maximum", None)
                d  = getattr(a, "default", None)

            if isinstance(tag, (bytes, bytearray)):
                tag = tag.decode("ascii", errors="ignore")

            out.append({"tag": tag, "min": mn, "max": mx, "default": d})

        return {"resolved": fp, "axes": out, "can_set_variations": hasattr(font, "set_variation_by_axes")}
    except Exception as e:
        return {"resolved": fp, "axes": None, "error": str(e)}

