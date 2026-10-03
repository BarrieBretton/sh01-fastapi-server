from __future__ import annotations

import asyncio
import ctypes
import gc
import hashlib
import hmac
import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, model_validator

from b2_helper import get_b2_manager
from media_job_store import store
from media_capacity import MEDIA_RENDER_LOCK, HEAVY_MEDIA_CAPACITY, reserve_heavy_media_or_raise
from caption_router import (
    BACKBLAZE_BUCKET_NAME,
    B2_PREFIX,
    DURABLE_TIMEOUT_SECONDS,
    FFMPEG_TIMEOUT_SECONDS,
    MAX_CLIP_SECONDS,
    MEDIA_CLIP_API_KEY,
    PRESETS,
    PUBLIC_API_BASE,
    RENDER_HEIGHT,
    RENDER_WIDTH,
    CaptionOverrides,
    _layout_caption_tokens,
    _download_telegram,
    _download_url,
    _extract_audio,
    _ffprobe_duration,
    _make_ass,
    _release_whisper_model,
    _run,
    _transcribe_faster_whisper,
)

logger = logging.getLogger("auto_editor")
router = APIRouter(prefix="/media", tags=["Automatic Video Editor"])

AUTO_EDIT_JOB_KIND = "auto_edit"
AUTO_EDIT_MAX_OUTPUTS = max(1, min(3, int(os.getenv("AUTO_EDIT_MAX_OUTPUTS", "3"))))
AUTO_EDIT_MAX_SEGMENTS = max(2, min(20, int(os.getenv("AUTO_EDIT_MAX_SEGMENTS", "12"))))
AUTO_EDIT_SCENE_THRESHOLD = float(os.getenv("AUTO_EDIT_SCENE_THRESHOLD", "0.38"))
AUTO_EDIT_MIN_SEGMENT = float(os.getenv("AUTO_EDIT_MIN_SEGMENT", "0.75"))
AUTO_EDIT_DEFAULT_TARGET = float(os.getenv("AUTO_EDIT_DEFAULT_TARGET_SECONDS", "45"))
AUTO_EDIT_ASSET_MAX = max(1, min(12, int(os.getenv("AUTO_EDIT_ASSET_MAX", "8"))))
AUTO_EDIT_JOB_RETENTION_SECONDS = max(3600, int(os.getenv("AUTO_EDIT_JOB_RETENTION_SECONDS", "86400")))
AUTO_EDIT_MIN_WORD_PROBABILITY = float(os.getenv("AUTO_EDIT_MIN_WORD_PROBABILITY", "0.18"))
AUTO_EDIT_FILTER_LOW_CONFIDENCE_WORDS = os.getenv(
    "AUTO_EDIT_FILTER_LOW_CONFIDENCE_WORDS", "true"
).strip().lower() in {"1", "true", "yes", "on"}
AUTO_EDIT_ASR_DEBUG_TAIL_WORDS = max(
    0, min(50, int(os.getenv("AUTO_EDIT_ASR_DEBUG_TAIL_WORDS", "12")))
)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = asyncio.Lock()
_background_tasks: set[asyncio.Task] = set()

FILLERS = {
    "um", "uh", "erm", "hmm", "like", "actually", "basically", "literally",
    "you know", "i mean", "sort of", "kind of",
}
HOOK_WORDS = {
    "why", "how", "secret", "mistake", "never", "best", "worst", "truth",
    "problem", "important", "crazy", "surprising", "here's", "watch", "stop",
    "don't", "do not", "must", "need", "warning", "hack", "trick",
}


class OverlayAsset(BaseModel):
    url: str
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    kind: Literal["broll", "meme", "logo"] = "meme"
    position: Literal["center", "top", "bottom", "top_left", "top_right", "bottom_left", "bottom_right"] = "center"
    width: int | None = Field(default=None, ge=96, le=1080)
    opacity: float = Field(default=1.0, ge=0.05, le=1.0)

    @model_validator(mode="after")
    def validate_times(self):
        if self.end_seconds <= self.start_seconds:
            raise ValueError("overlay end_seconds must be greater than start_seconds")
        return self


class SfxAsset(BaseModel):
    url: str
    at_seconds: float = Field(ge=0)
    volume: float = Field(default=0.75, ge=0, le=2.0)


class AutoEditRequest(BaseModel):
    source_url: str | None = None
    telegram_file_id: str | None = None
    request_id: str | None = Field(default=None, max_length=512)

    start_seconds: float = Field(default=0.0, ge=0)
    end_seconds: float | None = Field(default=None, gt=0)
    target_seconds: float = Field(default=AUTO_EDIT_DEFAULT_TARGET, ge=5, le=180)
    output_count: int = Field(default=1, ge=1, le=AUTO_EDIT_MAX_OUTPUTS)

    preset: Literal["viral_punch", "clean_minimal", "karaoke_glow", "podcast_bold", "cinematic"] = "viral_punch"
    language: str | None = Field(default=None, min_length=2, max_length=10)
    caption_overrides: CaptionOverrides | None = None

    smart_cut: bool = True
    remove_silence: bool = True
    remove_fillers: bool = True
    silence_gap_seconds: float = Field(default=0.62, ge=0.25, le=2.0)
    cut_padding_seconds: float = Field(default=0.08, ge=0, le=0.35)
    scene_detection: bool = True
    scene_threshold: float = Field(default=AUTO_EDIT_SCENE_THRESHOLD, ge=0.1, le=0.9)
    highlight_mode: Literal["chronological", "best_moments", "hook_first"] = "hook_first"
    hook_text: Literal["off", "auto"] = "auto"

    fit_mode: Literal["blur", "crop", "fit_black"] | None = None
    video_effect: Literal["none", "crisp", "punchy", "cinematic", "warm"] | None = None
    dynamic_zoom: Literal["off", "subtle", "punchy"] = "subtle"
    reframe: Literal["center", "left", "right", "auto"] = "auto"
    transition: Literal["none", "fade", "wipeleft", "slideleft", "smoothleft"] = "fade"
    transition_seconds: float = Field(default=0.18, ge=0, le=0.6)

    music_url: str | None = None
    music_volume: float = Field(default=0.10, ge=0, le=1.0)
    music_bpm: float | None = Field(default=None, ge=40, le=240)
    beat_sync: bool = False
    duck_music: bool = True
    sound_effects: list[SfxAsset] = Field(default_factory=list)
    overlays: list[OverlayAsset] = Field(default_factory=list)

    fps: Literal[24, 25, 30, 60] = 30
    crf: int = Field(default=21, ge=16, le=30)
    audio_bitrate: Literal[128, 160, 192, 256] = 192

    @model_validator(mode="after")
    def validate_request(self):
        if bool(self.source_url) == bool(self.telegram_file_id):
            raise ValueError("Provide exactly one of source_url or telegram_file_id")
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if len(self.overlays) > AUTO_EDIT_ASSET_MAX:
            raise ValueError(f"At most {AUTO_EDIT_ASSET_MAX} overlay assets are allowed")
        if len(self.sound_effects) > AUTO_EDIT_ASSET_MAX:
            raise ValueError(f"At most {AUTO_EDIT_ASSET_MAX} sound effects are allowed")
        return self


def _require_key(key: str | None) -> None:
    if not MEDIA_CLIP_API_KEY:
        raise HTTPException(status_code=500, detail="MEDIA_CLIP_API_KEY is not configured")
    if not key or not secrets.compare_digest(key, MEDIA_CLIP_API_KEY):
        raise HTTPException(status_code=401, detail="Missing/invalid X-API-KEY")



def _trim_process_memory() -> None:
    """Best-effort release of freed Python/C-extension heap pages before FFmpeg."""
    gc.collect()
    try:
        libc = ctypes.CDLL("libc.so.6")
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except Exception:
        pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _job_view(job: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in job.items() if not k.startswith("_")}


async def _prune_jobs() -> None:
    now = time.time()
    async with _jobs_lock:
        stale = [
            job_id
            for job_id, job in _jobs.items()
            if job.get("status") in {"complete", "failed"}
            and now - float(job.get("_finished_epoch") or now)
            >= AUTO_EDIT_JOB_RETENTION_SECONDS
        ]
        for job_id in stale:
            _jobs.pop(job_id, None)


def _job_id(request_id: str) -> str:
    key = MEDIA_CLIP_API_KEY or "auto-edit"
    return hmac.new(key.encode(), f"auto-edit:{request_id}".encode(), hashlib.sha256).hexdigest()[:32]


def _token(request_id: str, variant: int) -> str:
    key = MEDIA_CLIP_API_KEY or "auto-edit"
    return hmac.new(key.encode(), f"auto-edit-output:{request_id}:{variant}".encode(), hashlib.sha256).hexdigest()[:32]


async def _persist(job: dict[str, Any], payload: dict[str, Any] | None = None) -> bool:
    try:
        await asyncio.wait_for(
            asyncio.to_thread(
                store.put_job,
                str(job["job_id"]),
                AUTO_EDIT_JOB_KIND,
                str(job.get("status") or "unknown"),
                payload,
                _job_view(job),
                job.get("error"),
            ),
            timeout=DURABLE_TIMEOUT_SECONDS,
        )
        return True
    except Exception as exc:
        logger.warning("auto-edit persist failed job=%s err=%s", job.get("job_id"), exc)
        return False


async def _load(job_id: str) -> dict[str, Any] | None:
    try:
        row = await asyncio.wait_for(asyncio.to_thread(store.get_job, job_id), timeout=DURABLE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=503, detail="Auto-edit durable job lookup timed out") from exc
    if not row or row.get("kind") != AUTO_EDIT_JOB_KIND:
        return None
    result = row.get("result")
    if isinstance(result, dict):
        out = dict(result)
    else:
        out = {"job_id": job_id, "status": row.get("status"), "error": row.get("error")}
    out["durable"] = True
    out["durable_updated_at"] = row.get("updated_at")
    return out



def _filter_transcript_words(words: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop only clearly low-confidence ASR words without token blacklists."""
    if not AUTO_EDIT_FILTER_LOW_CONFIDENCE_WORDS:
        return words, []

    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for w in words:
        p = w.get("probability")
        if isinstance(p, (int, float)) and float(p) < AUTO_EDIT_MIN_WORD_PROBABILITY:
            dropped.append(w)
        else:
            kept.append(w)

    if words and not kept:
        logger.warning(
            "auto-edit ASR filter would drop all %s words; keeping original transcript",
            len(words),
        )
        return words, []

    return kept, dropped



def _detect_scenes(source: Path, absolute_start: float, duration: float, threshold: float) -> list[float]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return []
    filt = f"select='gt(scene,{threshold})',showinfo"
    result = _run(
        [ffmpeg, "-hide_banner", "-ss", f"{absolute_start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), "-an", "-vf", filt, "-f", "null", "-"],
        min(900, FFMPEG_TIMEOUT_SECONDS),
        "scene detection",
    )
    text = result.stderr.decode("utf-8", errors="replace")
    times: list[float] = []
    for match in re.finditer(r"pts_time:([0-9.]+)", text):
        t = float(match.group(1))
        if 0.2 < t < duration - 0.2:
            times.append(t)
    return sorted(set(round(t, 3) for t in times))


def _clean_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower()).strip()


def _segment_score(words: list[dict[str, Any]], start: float, end: float) -> float:
    duration = max(0.25, end - start)
    cleaned = [_clean_word(str(w.get("word", ""))) for w in words]
    cleaned = [w for w in cleaned if w]
    if not cleaned:
        return 0.0
    non_fillers = [w for w in cleaned if w not in FILLERS]
    pace = min(4.5, len(non_fillers) / duration)
    hook_hits = sum(1 for w in cleaned[:10] if w in HOOK_WORDS)
    punctuation = sum(1 for w in words if str(w.get("word", "")).rstrip().endswith(("!", "?", ":")))
    confidence = [w.get("probability") for w in words if isinstance(w.get("probability"), (int, float))]
    confidence_score = (sum(confidence) / len(confidence)) if confidence else 0.75
    filler_ratio = 1 - (len(non_fillers) / max(1, len(cleaned)))
    return round((pace * 1.3) + (hook_hits * 1.4) + (punctuation * 0.5) + confidence_score - (filler_ratio * 2.0), 4)


def _build_segments(words: list[dict[str, Any]], duration: float, req: AutoEditRequest, scene_points: list[float]) -> list[dict[str, Any]]:
    if not words:
        return [{"start": 0.0, "end": duration, "word_indexes": [], "reason": "full_clip", "score": 0.0}]

    segments: list[dict[str, Any]] = []
    current: list[int] = []
    for i, word in enumerate(words):
        cleaned = _clean_word(str(word.get("word", "")))
        filler_break = req.remove_fillers and cleaned in FILLERS and len(current) >= 2
        gap_break = False
        if current:
            prev = words[current[-1]]
            gap_break = float(word["start"]) - float(prev["end"]) >= req.silence_gap_seconds
        if current and (gap_break or filler_break):
            first = words[current[0]]
            last = words[current[-1]]
            start = max(0.0, float(first["start"]) - req.cut_padding_seconds)
            end = min(duration, float(last["end"]) + req.cut_padding_seconds)
            if end - start >= AUTO_EDIT_MIN_SEGMENT:
                segments.append({"start": start, "end": end, "word_indexes": list(current), "reason": "speech", "score": 0.0})
            current = []
        if not (req.remove_fillers and cleaned in FILLERS):
            current.append(i)

    if current:
        first = words[current[0]]
        last = words[current[-1]]
        start = max(0.0, float(first["start"]) - req.cut_padding_seconds)
        end = min(duration, float(last["end"]) + req.cut_padding_seconds)
        if end - start >= AUTO_EDIT_MIN_SEGMENT:
            segments.append({"start": start, "end": end, "word_indexes": list(current), "reason": "speech", "score": 0.0})

    if not req.remove_silence or not segments:
        segments = [{"start": 0.0, "end": duration, "word_indexes": list(range(len(words))), "reason": "full_clip", "score": 0.0}]

    if req.scene_detection and scene_points:
        split: list[dict[str, Any]] = []
        for seg in segments:
            points = [p for p in scene_points if seg["start"] + 0.8 < p < seg["end"] - 0.8]
            bounds = [seg["start"], *points, seg["end"]]
            for a, b in zip(bounds, bounds[1:]):
                idxs = [i for i in seg["word_indexes"] if a <= (float(words[i]["start"]) + float(words[i]["end"])) / 2 <= b]
                if b - a >= AUTO_EDIT_MIN_SEGMENT and idxs:
                    split.append({"start": a, "end": b, "word_indexes": idxs, "reason": "scene+speech", "score": 0.0})
        if split:
            segments = split

    for seg in segments:
        seg_words = [words[i] for i in seg["word_indexes"]]
        seg["score"] = _segment_score(seg_words, seg["start"], seg["end"])

    # Merge extremely close segments to avoid hyperactive edits.
    merged: list[dict[str, Any]] = []
    for seg in sorted(segments, key=lambda x: x["start"]):
        if merged and seg["start"] - merged[-1]["end"] < 0.16:
            merged[-1]["end"] = max(merged[-1]["end"], seg["end"])
            merged[-1]["word_indexes"].extend(seg["word_indexes"])
            merged[-1]["word_indexes"] = sorted(set(merged[-1]["word_indexes"]))
            merged[-1]["score"] = max(merged[-1]["score"], seg["score"])
        else:
            merged.append(seg)
    return merged[: max(AUTO_EDIT_MAX_SEGMENTS * 2, AUTO_EDIT_MAX_SEGMENTS)]


def _snap_to_beat(t: float, bpm: float | None, max_shift: float = 0.18) -> float:
    if not bpm:
        return t
    beat = 60.0 / bpm
    snapped = round(t / beat) * beat
    return snapped if abs(snapped - t) <= max_shift else t


def _select_segments(base: list[dict[str, Any]], req: AutoEditRequest, variant: int) -> list[dict[str, Any]]:
    if not base:
        return []
    target = req.target_seconds
    if req.highlight_mode == "chronological":
        ranked = list(base)
    else:
        ranked = sorted(base, key=lambda s: (-float(s["score"]), float(s["start"])))
        if variant:
            # Deterministic diversity for multi-output generation.
            shift = variant % len(ranked)
            ranked = ranked[shift:] + ranked[:shift]

    chosen: list[dict[str, Any]] = []
    total = 0.0
    for seg in ranked:
        segdur = float(seg["end"]) - float(seg["start"])
        if chosen and total + segdur > target * 1.15:
            continue
        chosen.append(dict(seg))
        total += segdur
        if total >= target or len(chosen) >= AUTO_EDIT_MAX_SEGMENTS:
            break

    if not chosen:
        chosen = [dict(max(base, key=lambda s: float(s["score"])))]

    if req.highlight_mode == "hook_first":
        strongest = max(chosen, key=lambda s: float(s["score"]))
        rest = sorted((s for s in chosen if s is not strongest), key=lambda s: float(s["start"]))
        chosen = [strongest] + rest
    else:
        chosen = sorted(chosen, key=lambda s: float(s["start"]))

    if req.beat_sync and req.music_bpm:
        for seg in chosen:
            seg["start"] = max(0.0, _snap_to_beat(float(seg["start"]), req.music_bpm))
            seg["end"] = max(float(seg["start"]) + AUTO_EDIT_MIN_SEGMENT, _snap_to_beat(float(seg["end"]), req.music_bpm))
    return chosen


def _hook_text(words: list[dict[str, Any]], seg: dict[str, Any]) -> str:
    idxs = seg.get("word_indexes", [])
    toks = [str(words[i].get("word", "")).strip() for i in idxs[:8]]
    text = " ".join(t for t in toks if t).strip()
    if not text:
        return ""
    return text[:72].upper()


def _base_filter(fit_mode: str, effect: str, zoom: str, reframe: str, index: int, fps: int, duration: float) -> str:
    w, h = RENDER_WIDTH, RENDER_HEIGHT
    if fit_mode == "crop":
        base = f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}"
    elif fit_mode == "fit_black":
        base = f"scale={w}:{h}:force_original_aspect_ratio=decrease,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:black"
    else:
        bw, bh = max(180, w // 2), max(320, h // 2)
        base = (
            "split=2[bg][fg];"
            f"[bg]scale={bw}:{bh}:force_original_aspect_ratio=increase,crop={bw}:{bh},boxblur=12:1,scale={w}:{h}[bg2];"
            f"[fg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg2];"
            "[bg2][fg2]overlay=(W-w)/2:(H-h)/2"
        )
    effects = {
        "none": "",
        "crisp": ",eq=contrast=1.04:saturation=1.04,unsharp=3:3:0.25:3:3:0",
        "punchy": ",eq=contrast=1.08:saturation=1.12:brightness=0.01,unsharp=3:3:0.30:3:3:0",
        "cinematic": ",eq=contrast=1.07:saturation=0.92:gamma=0.98,vignette=PI/7",
        "warm": ",eq=contrast=1.04:saturation=1.08:gamma_r=1.03:gamma_b=0.97",
    }
    out = base + effects.get(effect, "")
    if zoom != "off":
        amount = 0.035 if zoom == "subtle" else 0.07
        direction = -1 if index % 2 else 1
        z0 = 1.0 + (amount if direction < 0 else 0.0)
        dz = amount / max(1.0, duration * fps)
        if direction > 0:
            zexpr = f"min(zoom+{dz:.7f},{1.0+amount:.4f})"
        else:
            zexpr = f"max(zoom-{dz:.7f},1.0)"
        if reframe == "left":
            xexpr = "0"
        elif reframe == "right":
            xexpr = "iw-iw/zoom"
        elif reframe == "auto" and index % 3 == 1:
            xexpr = "0"
        elif reframe == "auto" and index % 3 == 2:
            xexpr = "iw-iw/zoom"
        else:
            xexpr = "iw/2-(iw/zoom/2)"
        out += f",zoompan=z='{zexpr}':x='{xexpr}':y='ih/2-(ih/zoom/2)':d=1:s={w}x{h}:fps={fps}"
    return out


def _render_segment(source: Path, dest: Path, absolute_start: float, duration: float, req: AutoEditRequest, idx: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is not installed")
    preset = PRESETS[req.preset]
    fit_mode = req.fit_mode or str(preset["fit_mode"])
    effect = req.video_effect or str(preset["video_effect"])
    vf = _base_filter(fit_mode, effect, req.dynamic_zoom, req.reframe, idx, req.fps, duration)

    cmd = [
        ffmpeg, "-y",
        "-filter_threads", "1",
        "-filter_complex_threads", "1",
        "-ss", f"{absolute_start:.3f}",
        "-t", f"{duration:.3f}",
        "-i", str(source),
    ]

    # blur mode uses split/overlay and is therefore a complex filter graph.
    # crop/fit_black are simple single-chain filters and can stay on -vf.
    if fit_mode == "blur":
        cmd += [
            "-filter_complex", f"[0:v]{vf}[v]",
            "-map", "[v]",
            "-map", "0:a?",
        ]
    else:
        cmd += ["-vf", vf]

    cmd += [
        "-threads", "1",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0",
        "-tune", "zerolatency",
        "-crf", str(req.crf),
        "-r", str(req.fps),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", f"{req.audio_bitrate}k",
        "-ar", "48000",
        "-movflags", "+faststart",
        "-shortest",
        str(dest),
    ]

    _run(cmd, FFMPEG_TIMEOUT_SECONDS, f"auto-edit segment {idx}")


def _concat_no_transition(paths: list[Path], dest: Path, root: Path) -> float:
    ffmpeg = shutil.which("ffmpeg")
    concat = root / "concat.txt"
    concat.write_text("".join(f"file '{p.as_posix()}'\n" for p in paths), encoding="utf-8")
    _run([ffmpeg, "-y", "-filter_threads", "1", "-filter_complex_threads", "1", "-f", "concat", "-safe", "0", "-i", str(concat), "-c", "copy", str(dest)], FFMPEG_TIMEOUT_SECONDS, "auto-edit concat")
    return _ffprobe_duration(dest)


def _merge_transition(left: Path, right: Path, dest: Path, transition: str, seconds: float) -> float:
    ffmpeg = shutil.which("ffmpeg")
    dl = _ffprobe_duration(left)
    dr = _ffprobe_duration(right)
    d = min(seconds, max(0.03, dl / 4), max(0.03, dr / 4))
    offset = max(0.0, dl - d)
    fc = (
        f"[0:v][1:v]xfade=transition={transition}:duration={d:.3f}:offset={offset:.3f}[v];"
        f"[0:a][1:a]acrossfade=d={d:.3f}:c1=tri:c2=tri[a]"
    )
    _run([
        ffmpeg, "-y", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(left), "-i", str(right), "-filter_complex", fc,
        "-map", "[v]", "-map", "[a]", "-threads", "1", "-c:v", "libx264", "-preset", "ultrafast", "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0", "-crf", "22",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(dest)
    ], FFMPEG_TIMEOUT_SECONDS, "auto-edit transition")
    return dl + dr - d


def _combine_segments(paths: list[Path], root: Path, req: AutoEditRequest) -> tuple[Path, list[float], float]:
    if len(paths) == 1:
        return paths[0], [0.0], _ffprobe_duration(paths[0])
    starts = [0.0]
    if req.transition == "none" or req.transition_seconds <= 0:
        out = root / "combined.mp4"
        duration = _concat_no_transition(paths, out, root)
        cursor = 0.0
        starts = []
        for p in paths:
            starts.append(cursor)
            cursor += _ffprobe_duration(p)
        return out, starts, duration

    current = paths[0]
    current_duration = _ffprobe_duration(current)
    starts = [0.0]
    for i, nxt in enumerate(paths[1:], 1):
        nd = _ffprobe_duration(nxt)
        d = min(req.transition_seconds, max(0.03, current_duration / 4), max(0.03, nd / 4))
        starts.append(max(0.0, current_duration - d))
        merged = root / f"merged_{i}.mp4"
        current_duration = _merge_transition(current, nxt, merged, req.transition, req.transition_seconds)
        if current != paths[0] and current.exists() and current.name.startswith("merged_"):
            current.unlink(missing_ok=True)
        current = merged
    return current, starts, current_duration


def _remap_words(words: list[dict[str, Any]], segments: list[dict[str, Any]], output_starts: list[float]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for seg, out_start in zip(segments, output_starts):
        for idx in seg.get("word_indexes", []):
            w = words[idx]
            midpoint = (float(w["start"]) + float(w["end"])) / 2
            if not (float(seg["start"]) <= midpoint <= float(seg["end"])):
                continue
            out.append({
                **w,
                "start": max(0.0, out_start + float(w["start"]) - float(seg["start"])),
                "end": max(0.01, out_start + float(w["end"]) - float(seg["start"])),
            })
    out.sort(key=lambda x: float(x["start"]))
    return out


def _overlay_position(position: str) -> tuple[str, str]:
    margin = 28
    mapping = {
        "center": ("(W-w)/2", "(H-h)/2"),
        "top": ("(W-w)/2", str(margin)),
        "bottom": ("(W-w)/2", f"H-h-{margin}"),
        "top_left": (str(margin), str(margin)),
        "top_right": (f"W-w-{margin}", str(margin)),
        "bottom_left": (str(margin), f"H-h-{margin}"),
        "bottom_right": (f"W-w-{margin}", f"H-h-{margin}"),
    }
    return mapping[position]


def _ffprobe_has_video(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    r = _run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)], 30, "asset probe")
    return bool(r.stdout.decode().strip())


async def _download_assets(req: AutoEditRequest, root: Path) -> tuple[list[tuple[OverlayAsset, Path]], Path | None, list[tuple[SfxAsset, Path]]]:
    overlays: list[tuple[OverlayAsset, Path]] = []
    for i, spec in enumerate(req.overlays):
        p = root / f"overlay_{i}.bin"
        await _download_url(spec.url, p)
        overlays.append((spec, p))
    music = None
    if req.music_url:
        music = root / "music.bin"
        await _download_url(req.music_url, music)
    sfx: list[tuple[SfxAsset, Path]] = []
    for i, spec in enumerate(req.sound_effects):
        p = root / f"sfx_{i}.bin"
        await _download_url(spec.url, p)
        sfx.append((spec, p))
    return overlays, music, sfx


def _finalize_video(base: Path, ass: Path, output: Path, duration: float, req: AutoEditRequest, overlays: list[tuple[OverlayAsset, Path]], music: Path | None, sfx: list[tuple[SfxAsset, Path]], hook: str) -> None:
    ffmpeg = shutil.which("ffmpeg")
    cmd = [ffmpeg, "-y", "-filter_threads", "1", "-filter_complex_threads", "1", "-i", str(base)]
    input_index = 1
    overlay_indices: list[tuple[OverlayAsset, int]] = []
    for spec, path in overlays:
        if _ffprobe_has_video(path):
            cmd += ["-stream_loop", "-1", "-i", str(path)]
        else:
            cmd += ["-loop", "1", "-i", str(path)]
        overlay_indices.append((spec, input_index))
        input_index += 1
    music_index = None
    if music:
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        music_index = input_index
        input_index += 1
    sfx_indices: list[tuple[SfxAsset, int]] = []
    for spec, path in sfx:
        cmd += ["-i", str(path)]
        sfx_indices.append((spec, input_index))
        input_index += 1

    ass_escaped = str(ass).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    filters: list[str] = [f"[0:v]ass='{ass_escaped}'[v0]"]
    vlabel = "v0"
    for j, (spec, idx) in enumerate(overlay_indices, 1):
        width = spec.width or (RENDER_WIDTH if spec.kind == "broll" else min(420, int(RENDER_WIDTH * 0.55)))
        x, y = _overlay_position(spec.position)
        filters.append(f"[{idx}:v]scale={width}:-2,format=rgba,colorchannelmixer=aa={spec.opacity:.3f}[ov{j}]")
        filters.append(f"[{vlabel}][ov{j}]overlay=x={x}:y={y}:enable='between(t,{spec.start_seconds:.3f},{spec.end_seconds:.3f})'[v{j}]")
        vlabel = f"v{j}"

    audio_labels: list[str]
    if music_index is not None and req.duck_music:
        filters.append("[0:a]asplit=2[voice_main][voice_sc]")
        filters.append(
            f"[{music_index}:a]volume={req.music_volume:.3f},"
            f"atrim=0:{duration:.3f}[music]"
        )
        filters.append(
            "[music][voice_sc]"
            "sidechaincompress=threshold=0.04:ratio=8:attack=10:release=220"
            "[ducked]"
        )
        audio_labels = ["[voice_main]", "[ducked]"]
    else:
        audio_labels = ["[0:a]"]
        if music_index is not None:
            filters.append(
                f"[{music_index}:a]volume={req.music_volume:.3f},"
                f"atrim=0:{duration:.3f}[music]"
            )
            audio_labels.append("[music]")
    for j, (spec, idx) in enumerate(sfx_indices, 1):
        delay = int(spec.at_seconds * 1000)
        filters.append(f"[{idx}:a]volume={spec.volume:.3f},adelay={delay}|{delay}[sfx{j}]")
        audio_labels.append(f"[sfx{j}]")

    if len(audio_labels) > 1:
        filters.append("".join(audio_labels) + f"amix=inputs={len(audio_labels)}:duration=first:dropout_transition=0[aout]")
        amap = "[aout]"
    else:
        amap = "0:a"

    cmd += ["-filter_complex", ";".join(filters), "-map", f"[{vlabel}]", "-map", amap,
            "-threads", "1", "-c:v", "libx264", "-preset", "ultrafast", "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0", "-tune", "zerolatency",
            "-crf", str(req.crf), "-r", str(req.fps), "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", f"{req.audio_bitrate}k", "-ar", "48000",
            "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output)]
    _run(cmd, FFMPEG_TIMEOUT_SECONDS, "auto-edit final render")


def _inject_hook_ass(path: Path, hook: str) -> None:
    if not hook:
        return

    raw_tokens = hook.split()
    preferred = max(34, int(RENDER_HEIGHT * 0.035))
    ranges, font_size = _layout_caption_tokens(
        raw_tokens,
        preferred,
        auto_size=True,
        min_font_size=max(28, int(preferred * 0.72)),
        safe_width_ratio=0.84,
        max_lines=2,
        wrap_mode="balanced",
    )

    wrapped_lines: list[str] = []
    for a, b in ranges:
        line = " ".join(raw_tokens[a:b])
        wrapped_lines.append(
            line.replace("\\", r"\\")
                .replace("{", r"\{")
                .replace("}", r"\}")
                .replace("\n", " ")
        )
    text = r"\N".join(wrapped_lines)

    raw = path.read_text(encoding="utf-8")
    hook_style = (
        f"Style: Hook,DejaVu Sans,{preferred},&H00FFFFFF,&H00FFFFFF,"
        "&H00101010,&H70000000,-1,0,0,0,100,100,0,0,1,5,1,8,50,50,90,1\n"
    )
    marker = "\n[Events]\n"
    if marker in raw:
        raw = raw.replace(marker, "\n" + hook_style + marker, 1)

    raw += (
        f"Dialogue: 5,0:00:00.00,0:00:01.35,Hook,,0,0,0,,"
        f"{{\\fs{font_size}\\fad(80,160)}}{text}\n"
    )
    path.write_text(raw, encoding="utf-8")


async def _render_variant(source: Path, base_start: float, words: list[dict[str, Any]], segments: list[dict[str, Any]], req: AutoEditRequest, root: Path, variant: int, request_id: str, overlays: list[tuple[OverlayAsset, Path]], music: Path | None, sfx: list[tuple[SfxAsset, Path]]) -> dict[str, Any]:
    variant_root = root / f"variant_{variant}"
    variant_root.mkdir(parents=True, exist_ok=True)
    segment_paths: list[Path] = []
    for i, seg in enumerate(segments):
        p = variant_root / f"segment_{i}.mp4"
        logger.info("auto-edit variant=%s segment=%s render start", variant + 1, i + 1)
        await asyncio.to_thread(_render_segment, source, p, base_start + float(seg["start"]), float(seg["end"]) - float(seg["start"]), req, i)
        logger.info("auto-edit variant=%s segment=%s render complete", variant + 1, i + 1)
        segment_paths.append(p)

    logger.info("auto-edit variant=%s combine start segments=%s", variant + 1, len(segment_paths))
    combined, output_starts, duration = await asyncio.to_thread(_combine_segments, segment_paths, variant_root, req)
    logger.info("auto-edit variant=%s combine complete duration=%.3f", variant + 1, duration)
    mapped = _remap_words(words, segments, output_starts)
    preset = json.loads(json.dumps(PRESETS[req.preset]))
    caption_cfg = preset["caption"]
    if req.caption_overrides:
        for key, value in req.caption_overrides.model_dump().items():
            if value is not None:
                caption_cfg[key] = value

    ass = variant_root / "captions.ass"
    await asyncio.to_thread(_make_ass, mapped, caption_cfg, ass)

    strongest = max(segments, key=lambda s: float(s.get("score", 0)))
    hook = _hook_text(words, strongest) if req.hook_text == "auto" else ""
    await asyncio.to_thread(_inject_hook_ass, ass, hook)

    final = variant_root / "final.mp4"
    logger.info("auto-edit variant=%s final render start", variant + 1)
    await asyncio.to_thread(_finalize_video, combined, ass, final, duration, req, overlays, music, sfx, hook)
    logger.info("auto-edit variant=%s final render complete bytes=%s", variant + 1, final.stat().st_size if final.exists() else 0)

    token = _token(request_id, variant)
    b2_name = f"{B2_PREFIX}/{token}.mp4"
    manager = get_b2_manager()
    await asyncio.to_thread(manager.upload_file, final, BACKBLAZE_BUCKET_NAME, b2_name, "video/mp4", 1)

    edl_segments = []
    for seg, out_start in zip(segments, output_starts):
        edl_segments.append({
            "source_start": round(base_start + float(seg["start"]), 3),
            "source_end": round(base_start + float(seg["end"]), 3),
            "output_start": round(out_start, 3),
            "duration": round(float(seg["end"]) - float(seg["start"]), 3),
            "score": seg.get("score"),
            "reason": seg.get("reason"),
        })
    return {
        "variant": variant + 1,
        "token": token,
        "video_url": f"{PUBLIC_API_BASE}/media/rendered/{token}",
        "b2_file_name": b2_name,
        "size_bytes": final.stat().st_size,
        "duration": round(duration, 3),
        "hook_text": hook or None,
        "word_count": len(mapped),
        "edit_decision_list": edl_segments,
    }


async def _execute(job_id: str, req: AutoEditRequest) -> None:
    job = _jobs[job_id]
    request_id = str(job["request_id"])
    durable_payload = {"request_id": request_id, "request": req.model_dump(mode="json")}
    capacity_lease_id = str(job.get("_capacity_lease_id") or "")
    async with MEDIA_RENDER_LOCK:
        job.update(status="running", stage="download", started_at=_now())
        await _persist(job, durable_payload)
        try:
            with tempfile.TemporaryDirectory(prefix="sh01-auto-edit-") as tmp:
                root = Path(tmp)
                source = root / "source.bin"
                audio = root / "audio.wav"
                if req.source_url:
                    await _download_url(req.source_url, source)
                else:
                    await _download_telegram(str(req.telegram_file_id), source)

                total = await asyncio.to_thread(_ffprobe_duration, source)
                start = min(req.start_seconds, max(0.0, total - 0.05))
                end = min(total, req.end_seconds if req.end_seconds is not None else total)
                duration = end - start
                if duration <= 0.1:
                    raise RuntimeError("Selected source window has no usable duration")
                if duration > max(MAX_CLIP_SECONDS, 180):
                    raise RuntimeError("Auto-edit source window is too long for this deployment")

                job.update(stage="transcription", source_start=start, source_end=end, source_duration=duration)
                await _persist(job, durable_payload)
                await asyncio.to_thread(_extract_audio, source, start, duration, audio)
                words = await _transcribe_faster_whisper(audio, req.language)

                if AUTO_EDIT_ASR_DEBUG_TAIL_WORDS:
                    logger.info(
                        "auto-edit transcript tail=%s",
                        [
                            {
                                "word": w.get("word"),
                                "start": w.get("start"),
                                "end": w.get("end"),
                                "probability": w.get("probability"),
                            }
                            for w in words[-AUTO_EDIT_ASR_DEBUG_TAIL_WORDS:]
                        ],
                    )

                words, dropped_words = _filter_transcript_words(words)
                if dropped_words:
                    logger.info(
                        "auto-edit dropped low-confidence transcript words threshold=%.3f words=%s",
                        AUTO_EDIT_MIN_WORD_PROBABILITY,
                        [
                            {
                                "word": w.get("word"),
                                "start": w.get("start"),
                                "end": w.get("end"),
                                "probability": w.get("probability"),
                            }
                            for w in dropped_words
                        ],
                    )

                await _release_whisper_model()
                await asyncio.to_thread(_trim_process_memory)

                job.update(
                    stage="analysis",
                    transcript_word_count=len(words),
                    transcript_words_dropped=len(dropped_words),
                    transcript_min_probability=(
                        AUTO_EDIT_MIN_WORD_PROBABILITY
                        if AUTO_EDIT_FILTER_LOW_CONFIDENCE_WORDS
                        else None
                    ),
                )
                await _persist(job, durable_payload)
                scene_points = []
                if req.smart_cut:
                    if req.scene_detection:
                        scene_points = await asyncio.to_thread(
                            _detect_scenes,
                            source,
                            start,
                            duration,
                            req.scene_threshold,
                        )
                    base_segments = _build_segments(words, duration, req, scene_points)
                else:
                    base_segments = [{
                        "start": 0.0,
                        "end": duration,
                        "word_indexes": list(range(len(words))),
                        "reason": "smart_cut_disabled",
                        "score": _segment_score(words, 0.0, duration),
                    }]

                if not base_segments:
                    raise RuntimeError("Auto-editor could not produce any usable edit segments")

                overlays, music, sfx = await _download_assets(req, root)
                results = []
                for variant in range(req.output_count):
                    job.update(stage=f"render_variant_{variant + 1}", current_variant=variant + 1, planned_segments=len(base_segments))
                    await _persist(job, durable_payload)
                    selected = _select_segments(base_segments, req, variant)
                    result = await _render_variant(source, start, words, selected, req, root, variant, request_id, overlays, music, sfx)
                    results.append(result)

                job.update(
                    status="complete",
                    stage="complete",
                    completed_at=_now(),
                    outputs=results,
                    output_count=len(results),
                    scene_points=scene_points,
                    analysis={
                        "transcript_word_count": len(words),
                        "candidate_segments": len(base_segments),
                        "scene_count": len(scene_points),
                        "silence_removal": req.remove_silence,
                        "filler_removal": req.remove_fillers,
                        "highlight_mode": req.highlight_mode,
                        "transition": req.transition,
                        "dynamic_zoom": req.dynamic_zoom,
                        "beat_sync": bool(req.beat_sync and req.music_bpm),
                    },
                    error=None,
                )
                await _persist(job, durable_payload)
        except Exception as exc:
            failed_stage = job.get("stage")
            logger.exception("Auto-edit job failed job=%s stage=%s", job_id, failed_stage)
            detail = str(exc.detail) if isinstance(exc, HTTPException) else (str(exc) or repr(exc))
            job.update(status="failed", stage="failed", failed_stage=failed_stage, error=detail[:4000], error_type=type(exc).__name__, completed_at=_now())
            await _persist(job, durable_payload)
        finally:
            job["_finished_epoch"] = time.time()
            if capacity_lease_id:
                await HEAVY_MEDIA_CAPACITY.release(capacity_lease_id)


@router.post("/auto-edit")
async def create_auto_edit(req: AutoEditRequest, x_api_key: str | None = Header(None, alias="X-API-KEY")):
    _require_key(x_api_key)
    await _prune_jobs()
    if not BACKBLAZE_BUCKET_NAME:
        raise HTTPException(status_code=500, detail="BACKBLAZE_BUCKET_NAME is not configured")
    request_id = (req.request_id or str(uuid.uuid4())).strip()
    job_id = _job_id(request_id)

    async with _jobs_lock:
        local = _jobs.get(job_id)
        if local and local.get("status") in {"queued", "running", "complete"}:
            return JSONResponse(status_code=200 if local.get("status") == "complete" else 202, content=_job_view(local))

    durable = await _load(job_id)
    if durable and durable.get("status") == "complete":
        return JSONResponse(status_code=200, content=_job_view(durable))

    attempt = int((durable or {}).get("attempt") or 0) + 1
    lease = await reserve_heavy_media_or_raise(
        kind="auto_edit",
        job_id=job_id,
        request_id=request_id,
        endpoint="/media/auto-edit",
    )
    job = {
        "job_id": job_id,
        "request_id": request_id,
        "status": "queued",
        "stage": "queued",
        "attempt": attempt,
        "created_at": _now(),
        "status_url": f"{PUBLIC_API_BASE}/media/auto-edit-jobs/{job_id}",
        "outputs": [],
        "error": None,
        "_capacity_lease_id": lease.lease_id,
    }
    async with _jobs_lock:
        _jobs[job_id] = job
    try:
        persisted = await _persist(
            job,
            {"request_id": request_id, "request": req.model_dump(mode="json")},
        )
        if not persisted:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "durable_job_registration_failed",
                    "message": "Auto-edit job was not accepted because durable job registration failed",
                    "retryable": True,
                },
                headers={"Retry-After": "5"},
            )
        task = asyncio.create_task(_execute(job_id, req))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    except Exception:
        async with _jobs_lock:
            _jobs.pop(job_id, None)
        await HEAVY_MEDIA_CAPACITY.release(lease.lease_id)
        raise
    return JSONResponse(status_code=202, content=_job_view(job))


@router.get("/auto-edit-jobs/{job_id}")
async def get_auto_edit(job_id: str, x_api_key: str | None = Header(None, alias="X-API-KEY")):
    _require_key(x_api_key)
    await _prune_jobs()
    async with _jobs_lock:
        local = _jobs.get(job_id)
        if local:
            return _job_view(local)
    durable = await _load(job_id)
    if not durable:
        raise HTTPException(status_code=404, detail="Unknown or expired auto-edit job")
    if durable.get("status") in {"queued", "running"}:
        durable["previous_status"] = durable.get("status")
        durable["status"] = "interrupted"
        durable["stage"] = "interrupted"
        durable["recoverable"] = True
        durable["retry"] = "Resubmit the same request_id and body."
    return _job_view(durable)


@router.get("/auto-edit-capabilities")
async def auto_edit_capabilities(x_api_key: str | None = Header(None, alias="X-API-KEY")):
    _require_key(x_api_key)
    return {
        "smart_cuts": True,
        "silence_removal": True,
        "filler_removal": True,
        "scene_detection": True,
        "highlight_selection": ["chronological", "best_moments", "hook_first"],
        "hook_selection": True,
        "dynamic_zoom": ["off", "subtle", "punchy"],
        "reframe": ["center", "left", "right", "auto"],
        "transitions": ["none", "fade", "wipeleft", "slideleft", "smoothleft"],
        "captions": list(PRESETS.keys()),
        "overlays": ["broll", "meme", "logo"],
        "music_bed": True,
        "music_ducking": True,
        "beat_sync_with_supplied_bpm": True,
        "sound_effects": True,
        "multi_output": AUTO_EDIT_MAX_OUTPUTS,
        "face_tracking": False,
        "face_tracking_note": "True face tracking is intentionally not enabled on the 512 MB deployment; auto reframe uses deterministic center/left/right motion framing.",
        "render_size": [RENDER_WIDTH, RENDER_HEIGHT],
    }
