"""Bounded, reusable edge branding for SH01 images and rendered videos.

Coordinates are final output pixels. Gradients and corner logos are rasterized
once; moving text is rendered by libass during the existing video final pass.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, model_validator

Edge = Literal["top", "right", "bottom", "left"]
Corner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]
HEX = r"^#[0-9a-fA-F]{6}$"
ORDER = ("top", "right", "bottom", "left")


class ColorStop(BaseModel):
    """One RGB color and its opacity (0 transparent, 1 fully opaque)."""

    color: str = Field(default="#000000", pattern=HEX)
    opacity: float = Field(default=0.0, ge=0, le=1)


class EdgeGradient(BaseModel):
    """Two-stop gradient spanning one full edge; outer is on the actual border."""

    edge: Edge
    thickness_px: int = Field(default=140, ge=2, le=600)
    outer: ColorStop = Field(default_factory=lambda: ColorStop(opacity=0.6))
    inner: ColorStop = Field(default_factory=ColorStop)


class CornerLogo(BaseModel):
    """Public image URL; rendered once at the final resolution, above gradients."""

    url: str = Field(min_length=1, max_length=2048)
    corner: Corner
    size_px: int = Field(default=48, ge=16, le=360, description="Longest image side in output pixels")
    opacity: float = Field(default=0.75, ge=0, le=1)
    margin_px: int = Field(default=28, ge=0, le=160)


class EdgeText(BaseModel):
    """Small static label or character marquee on adjacent final-frame edges.

    `edges` uses clockwise order top->right->bottom->left by default; reverse
    reverses travel. Examples: [left,top] is an L, [left,top,right] an upside-down
    U; `loop` repeatedly traverses all four edges. Left text bases face right;
    right text bases face left. Top/bottom characters stay upright.
    """

    text: str = Field(min_length=1, max_length=120)
    mode: Literal["static", "marquee", "path", "loop"] = "static"
    edge: Edge = "bottom"
    edges: list[Edge] = Field(default_factory=list)
    direction: Literal["forward", "reverse"] = "forward"
    color: str = Field(default="#FFFFFF", pattern=HEX)
    opacity: float = Field(default=0.85, ge=0, le=1)
    font_size_px: int = Field(default=22, ge=12, le=48)
    margin_px: int = Field(default=25, ge=8, le=120)
    speed_px_per_second: float = Field(default=95, ge=15, le=240)

    @model_validator(mode="after")
    def connected_path(self):
        if any(ord(char) < 32 for char in self.text):
            raise ValueError("edge text must be a single line without control characters")
        if self.mode == "path":
            if not 2 <= len(self.edges) <= 4 or len(set(self.edges)) != len(self.edges):
                raise ValueError("path requires two to four distinct connected edges")
        elif self.mode == "loop" and self.edges and len(self.edges) != 4:
            raise ValueError("loop requires all four edges, or omit edges for the default loop")
        elif self.mode in {"static", "marquee"} and self.edges:
            raise ValueError("static and marquee select a single edge with edge=...")
        default_loop = ["left", "top", "right", "bottom"] if self.direction == "forward" else ["left", "bottom", "right", "top"]
        edges = self.edges if self.edges else (default_loop if self.mode == "loop" else [])
        if edges and (len(set(edges)) != len(edges) or
                      any((ORDER.index(b) - ORDER.index(a)) % 4 != (1 if self.direction == "forward" else 3)
                          for a, b in zip(edges, edges[1:]))):
            raise ValueError("edges must be adjacent in the selected direction (no jumps)")
        if self.mode == "loop" and set(edges) != set(ORDER):
            raise ValueError("loop requires top, right, bottom and left")
        return self


class Branding(BaseModel):
    """Optional reusable edge layer; enabled=false suppresses preset defaults."""

    enabled: bool = True
    gradients: list[EdgeGradient] = Field(default_factory=list, max_length=4)
    logos: list[CornerLogo] = Field(default_factory=list, max_length=4)
    edge_text: list[EdgeText] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def unique_positions(self):
        if len({x.corner for x in self.logos}) != len(self.logos):
            raise ValueError("Use at most one logo per corner")
        if len({x.edge for x in self.gradients}) != len(self.gradients):
            raise ValueError("Use at most one gradient per edge")
        return self


def preset_branding(preset: str) -> Branding:
    """Tasteful baseline for every caption preset; caller overrides entire layer."""
    opacity = {"viral_punch": 0.57, "karaoke_glow": 0.53,
               "podcast_bold": 0.48, "cinematic": 0.44, "clean_minimal": 0.30}.get(preset, 0.30)
    return Branding(gradients=[EdgeGradient(edge="bottom", thickness_px=140,
                    outer=ColorStop(opacity=opacity), inner=ColorStop())])


def _color(value: str) -> tuple[int, int, int]:
    return tuple(bytes.fromhex(value[1:]))


def _gradient(canvas: Image.Image, spec: EdgeGradient) -> None:
    w, h = canvas.size
    width = min(spec.thickness_px, w) if spec.edge in {"left", "right"} else w
    height = min(spec.thickness_px, h) if spec.edge in {"top", "bottom"} else h
    layer = Image.new("RGBA", (width, height))
    painter = ImageDraw.Draw(layer)
    outer, inner = _color(spec.outer.color), _color(spec.inner.color)
    steps = height if spec.edge in {"top", "bottom"} else width
    for i in range(steps):
        t = i / max(1, steps - 1)
        if spec.edge in {"bottom", "right"}:
            t = 1 - t
        rgba = tuple(round(outer[c] * (1 - t) + inner[c] * t) for c in range(3)) + (
            round(255 * (spec.outer.opacity * (1 - t) + spec.inner.opacity * t)),)
        if spec.edge in {"top", "bottom"}:
            painter.line((0, i, width - 1, i), fill=rgba)
        else:
            painter.line((i, 0, i, height - 1), fill=rgba)
    canvas.alpha_composite(layer, (w - width if spec.edge == "right" else 0,
                                   h - height if spec.edge == "bottom" else 0))


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(name).is_file():
            return ImageFont.truetype(name, size)
    return ImageFont.load_default(size=size)


def _static_text(canvas: Image.Image, spec: EdgeText) -> None:
    font = _font(spec.font_size_px)
    bbox = ImageDraw.Draw(canvas).textbbox((0, 0), spec.text, font=font)
    label = Image.new("RGBA", (bbox[2] - bbox[0] + 4, bbox[3] - bbox[1] + 4))
    ImageDraw.Draw(label).text((2 - bbox[0], 2 - bbox[1]), spec.text, font=font,
                               fill=(*_color(spec.color), round(spec.opacity * 255)))
    if spec.edge == "left":
        label = label.rotate(90, expand=True)
    elif spec.edge == "right":
        label = label.rotate(-90, expand=True)
    w, h = canvas.size
    x = spec.margin_px if spec.edge == "left" else (w - label.width - spec.margin_px if spec.edge == "right" else (w - label.width) // 2)
    y = spec.margin_px if spec.edge == "top" else (h - label.height - spec.margin_px if spec.edge == "bottom" else (h - label.height) // 2)
    canvas.alpha_composite(label, (max(0, x), max(0, y)))


def composite_static(canvas: Image.Image, branding: Branding,
                     logos: list[tuple[CornerLogo, Image.Image]]) -> None:
    """Composite two-stop edge gradients, static text and up to four transparent logos."""
    if not branding.enabled:
        return
    for spec in branding.gradients:
        _gradient(canvas, spec)
    for spec in branding.edge_text:
        if spec.mode == "static":
            _static_text(canvas, spec)
    w, h = canvas.size
    for spec, original in logos:
        logo = original.convert("RGBA")
        if logo.width * logo.height > 4_000_000:
            raise ValueError("Logo exceeds four million decoded pixels")
        logo.thumbnail((spec.size_px, spec.size_px), Image.Resampling.LANCZOS)
        alpha = logo.getchannel("A").point(lambda a: round(a * spec.opacity))
        logo.putalpha(alpha)
        x = spec.margin_px if spec.corner.endswith("left") else w - logo.width - spec.margin_px
        y = spec.margin_px if spec.corner.startswith("top") else h - logo.height - spec.margin_px
        canvas.alpha_composite(logo, (max(0, x), max(0, y)))


def _edge_points(edge: Edge, direction: str, margin: int, width: int, height: int):
    p = {"tl": (margin, margin), "tr": (width - margin, margin),
         "br": (width - margin, height - margin), "bl": (margin, height - margin)}
    a, b, angle = {
        "top": (p["tl"], p["tr"], 0), "right": (p["tr"], p["br"], -90),
        "bottom": (p["br"], p["bl"], 0), "left": (p["bl"], p["tl"], 90),
    }[edge]
    return (b, a, angle) if direction == "reverse" else (a, b, angle)


def _ass_time(seconds: float) -> str:
    ticks = round(seconds * 100)
    return f"{ticks // 360000}:{ticks // 6000 % 60:02d}:{ticks // 100 % 60:02d}.{ticks % 100:02d}"


def animated_ass(branding: Branding, path: Path, width: int, height: int,
                 duration: float) -> bool:
    """Generate time-bounded ASS character events for connected edge paths.

    Characters move at a constant speed along each selected edge, switch their
    reading orientation at corners, and enter/leave at the endpoint. A loop
    restarts seamlessly at the same physical corner, with a bounded event count.
    """
    specs = [s for s in branding.edge_text if s.mode != "static"] if branding.enabled else []
    if not specs:
        return False
    lines = [f"[Script Info]\nScriptType: v4.00+\nPlayResX: {width}\nPlayResY: {height}\nScaledBorderAndShadow: yes\n\n"
             "[V4+ Styles]\nFormat: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
             "Style: Edge,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H00202020,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1\n\n"
             "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"]
    events = 0
    for spec in specs:
        default_loop = ["left", "top", "right", "bottom"] if spec.direction == "forward" else ["left", "bottom", "right", "top"]
        edges = spec.edges or (default_loop if spec.mode == "loop" else [spec.edge])
        legs = []
        cursor = 0.0
        for edge in edges:
            a, b, angle = _edge_points(edge, spec.direction, spec.margin_px, width, height)
            length = math.dist(a, b)
            legs.append((cursor, cursor + length, a, b, angle))
            cursor += length
        advance = spec.font_size_px * 0.64
        speed = spec.speed_px_per_second
        repeats = math.ceil(duration * speed / cursor) + 1 if spec.mode == "loop" else 1
        # ASS emits separate glyph events. On left/top/right in the forward
        # direction (and on bottom in reverse), the direction of travel is
        # also the direction in which upright/inward-facing text is read.
        # Start the last glyph first so the populated marquee reads naturally.
        # The opposite directions already read naturally in input order.
        reads_with_travel = (spec.direction == "forward") != (edges[0] == "bottom")
        alpha = round((1 - spec.opacity) * 255)
        color = spec.color[5:7] + spec.color[3:5] + spec.color[1:3]
        for cycle in range(repeats):
            for index, char in enumerate(spec.text):
                if char.isspace():
                    continue
                glyph_order = (len(spec.text) - 1 - index) if reads_with_travel else index
                offset = cycle * cursor + glyph_order * advance
                for leg_index, (begin, end, a, b, angle) in enumerate(legs):
                    first, last = (offset + begin) / speed, (offset + end) / speed
                    if first >= duration:
                        break
                    if last <= 0:
                        continue
                    if events >= 5000:
                        raise ValueError("Animated edge text exceeds the 5000-event budget")
                    span = max(0.001, last - first)
                    x0, y0 = a
                    x1, y1 = b
                    if last > duration:
                        fraction = max(0, (duration - first) / span)
                        x1 = round(x0 + (x1 - x0) * fraction)
                        y1 = round(y0 + (y1 - y0) * fraction)
                    fade = "" if spec.mode == "loop" else (
                        r"\fad(120,0)" if leg_index == 0 else r"\fad(0,120)" if leg_index == len(legs) - 1 else "")
                    glyph = char.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ")
                    tag = (f"\\move({x0},{y0},{round(x1)},{round(y1)})\\frz{angle}"
                           f"\\fs{spec.font_size_px}\\1c&H{color}&\\alpha&H{alpha:02X}&" + fade)
                    lines.append(f"Dialogue: 0,{_ass_time(first)},{_ass_time(min(duration,last))},Edge,,0,0,0,,{{{tag}}}{glyph}\n")
                    events += 1
    path.write_text("".join(lines), encoding="utf-8")
    return events > 0
