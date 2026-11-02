"""Local-first AI composition engine.

This module deliberately separates three concerns:

* observation: ffprobe, sampled frames, and word-level transcription;
* direction: a small local model proposes a typed CompositionPlan;
* execution: this module compiles the validated plan to an allow-listed
  FFmpeg argv list.

The LLM is never given an arbitrary shell tool and never receives permission
to write an FFmpeg filter graph directly.
"""
from __future__ import annotations

import asyncio
import gc
import hashlib
import ipaddress
import json
import logging
import math
import mimetypes
import os
import re
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

from ai_compose_schema import (
    AIComposeRequest,
    AIComposePlan,
    AssetRequest,
    BaseSegment,
    CompositionPlan,
    GraphicPlan,
    ResolvedLayer,
    SfxEvent,
    VoiceoverPlan,
)
from ai_compose_http import USER_AGENT, check_cooldown, observe_response, failure_details
from ai_compose_ir import EditPlan, camera_for_clip, compile_edit_plan
from ai_compose_policy import policy_prompt_context, resolve_editorial_policy
from ai_compose_quality import inspect_edit_plan, inspect_render

logger = logging.getLogger("ai_compose")

VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".m4v", ".avi", ".ts", ".m2ts"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac"}
STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "been", "being",
    "between", "could", "from", "have", "into", "more", "most", "only", "that",
    "their", "them", "there", "these", "this", "those", "through", "under", "very",
    "what", "when", "where", "which", "while", "with", "would", "your", "the", "for",
    "you", "how", "why", "will", "its", "it's", "than", "then", "they", "our", "out",
}


class ComposeError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _max_source_bytes() -> int:
    return max(32 * 1024 * 1024, int(os.getenv("AI_COMPOSE_MAX_SOURCE_BYTES", str(16 * 1024**3))))


def _run(command: list[str], timeout: int, label: str) -> subprocess.CompletedProcess:
    try:
        from media_resources import run_media_command

        result = run_media_command(command, timeout, label)
    except ImportError:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[-4000:]
        raise ComposeError(f"{label} failed: {detail[-1600:]}")
    return result


def _validate_public_url(url: str) -> None:
    parsed = urlparse(str(url).strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ComposeError("Only public HTTP(S) URLs are accepted")
    host = parsed.hostname.lower().rstrip(".")
    if host in {"localhost", "localhost.localdomain"}:
        raise ComposeError("Localhost URLs are not accepted")
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)}
    except socket.gaierror as exc:
        raise ComposeError(f"Could not resolve remote asset host: {host}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address)
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise ComposeError("Private or non-routable asset URLs are not accepted")


def _safe_filename(url: str, default_suffix: str = ".bin") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS | AUDIO_EXTENSIONS:
        suffix = default_suffix
    return suffix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def _download_async(url: str, destination: Path, *, max_bytes: int | None = None) -> None:
    try:
        import httpx
    except ImportError as exc:
        raise ComposeError("httpx is required for remote asset downloads") from exc
    current = url
    limit = max_bytes or _max_source_bytes()
    async with httpx.AsyncClient(timeout=180.0, follow_redirects=False) as client:
        for _ in range(6):
            check_cooldown(current)
            _validate_public_url(current)
            async with client.stream("GET", current, headers={"User-Agent": USER_AGENT}) as response:
                observe_response(current, response)
                if response.status_code in {301, 302, 303, 307, 308}:
                    location = response.headers.get("location")
                    if not location:
                        raise ComposeError("Remote asset redirect has no Location header")
                    current = urljoin(current, location)
                    continue
                if response.status_code >= 400:
                    raise ComposeError(f"Remote asset returned HTTP {response.status_code}")
                declared = response.headers.get("content-length")
                if declared and int(declared) > limit:
                    raise ComposeError("Remote asset exceeds the configured byte limit")
                total = 0
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    async for chunk in response.aiter_bytes(1024 * 1024):
                        total += len(chunk)
                        if total > limit:
                            raise ComposeError("Remote asset exceeds the configured byte limit")
                        output.write(chunk)
                return
    raise ComposeError("Too many redirects while downloading asset")


def _download_sync(url: str, destination: Path, *, max_bytes: int = 512 * 1024 * 1024) -> None:
    try:
        import requests
    except ImportError as exc:
        raise ComposeError("requests is required for remote asset downloads") from exc
    current = url
    for _ in range(6):
        check_cooldown(current)
        _validate_public_url(current)
        response = requests.get(
            current,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            allow_redirects=False,
            timeout=(20, 180),
        )
        try:
            observe_response(current, response)
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise ComposeError("Remote asset redirect has no Location header")
                current = urljoin(current, location)
                continue
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared and int(declared) > max_bytes:
                raise ComposeError("Remote asset exceeds the configured byte limit")
            total = 0
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as output:
                for chunk in response.iter_content(1024 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > max_bytes:
                        raise ComposeError("Remote asset exceeds the configured byte limit")
                    output.write(chunk)
            return
        finally:
            response.close()
    raise ComposeError("Too many redirects while downloading asset")


def _validated_local_source(source_path: str) -> Path:
    root_setting = os.getenv("MEDIA_LOCAL_SOURCE_DIR", "").strip()
    if not root_setting:
        raise ComposeError("Local media input is not enabled; set MEDIA_LOCAL_SOURCE_DIR")
    root = Path(root_setting).expanduser().resolve(strict=True)
    source = Path(source_path).expanduser().resolve(strict=True)
    if not root.is_dir() or not source.is_relative_to(root) or not source.is_file():
        raise ComposeError("source_path must be a file inside MEDIA_LOCAL_SOURCE_DIR")
    if source.stat().st_size > _max_source_bytes():
        raise ComposeError("Local source exceeds AI_COMPOSE_MAX_SOURCE_BYTES")
    return source


async def _telegram_download(file_id: str, destination: Path) -> None:
    import httpx
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise ComposeError("TELEGRAM_BOT_TOKEN is not configured")
    async with httpx.AsyncClient(timeout=40.0) as client:
        response = await client.get(f"https://api.telegram.org/bot{token}/getFile", params={"file_id": file_id})
        response.raise_for_status()
        payload = response.json()
        file_path = payload.get("result", {}).get("file_path")
        if not payload.get("ok") or not file_path:
            raise ComposeError("Telegram did not return a file path")
    await _download_async(f"https://api.telegram.org/file/bot{token}/{file_path}", destination)


async def acquire_source(request: AIComposeRequest, root: Path) -> Path:
    if request.source_path:
        return _validated_local_source(request.source_path)
    destination = root / "source.bin"
    if request.source_url:
        await _download_async(request.source_url, destination)
    elif request.telegram_file_id:
        await _telegram_download(request.telegram_file_id, destination)
    else:
        raise ComposeError("No media source supplied")
    return destination


def probe_media(source: Path) -> dict[str, Any]:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ComposeError("ffprobe is required")
    result = _run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration,size:stream=index,codec_type,width,height,r_frame_rate,codec_name,sample_rate,channels", "-of", "json", str(source)],
        120,
        "media probe",
    )
    try:
        data = json.loads(result.stdout.decode("utf-8"))
        duration = float(data.get("format", {}).get("duration") or 0.0)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ComposeError("ffprobe returned invalid media metadata") from exc
    streams = data.get("streams") or []
    video = next((item for item in streams if item.get("codec_type") == "video"), {})
    audio = next((item for item in streams if item.get("codec_type") == "audio"), {})
    if duration <= 0 or not video:
        raise ComposeError("Source has no usable video stream")
    return {
        "duration": duration,
        "size_bytes": int(data.get("format", {}).get("size") or 0),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name"),
        "fps": video.get("r_frame_rate"),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name"),
    }


def extract_audio(source: Path, start: float, duration: float, destination: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ComposeError("ffmpeg is required")
    _run(
        [ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(destination)],
        max(600, int(duration * 8) + 120),
        "audio extraction",
    )


def transcribe_chunked(source: Path, source_start: float, duration: float, language: str | None, root: Path) -> list[dict[str, Any]]:
    """Use the repository's isolated faster-whisper worker in bounded chunks."""
    words: list[dict[str, Any]] = []
    chunk_seconds = max(30, min(180, int(os.getenv("AI_COMPOSE_TRANSCRIPTION_CHUNK_SECONDS", "120"))))
    audio = root / "transcription.wav"
    output = root / "transcription.words.json"
    for offset in range(0, max(1, int(math.ceil(duration))), chunk_seconds):
        chunk_duration = min(chunk_seconds, duration - offset)
        if chunk_duration <= 0.1:
            break
        extract_audio(source, source_start + float(offset), chunk_duration, audio)
        output.unlink(missing_ok=True)
        result = _run(
            [sys.executable, "-m", "media_transcribe", str(audio), str(output), language or "", "--allow-empty"],
            max(900, int(chunk_duration * 12) + 180),
            "isolated Whisper transcription",
        )
        del result
        chunk_words = json.loads(output.read_text(encoding="utf-8"))
        for word in chunk_words:
            item = dict(word)
            item["start"] = float(item.get("start", 0.0)) + source_start + offset
            item["end"] = float(item.get("end", 0.0)) + source_start + offset
            words.append(item)
    return words


def extract_frames(source: Path, duration: float, root: Path, maximum: int = 48, source_start: float = 0.0) -> list[dict[str, Any]]:
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    count = max(4, min(maximum, int(math.ceil(duration / max(1.0, duration / maximum)))))
    times = [0.0 if count == 1 else (duration - 0.05) * i / (count - 1) for i in range(count)]
    frames: list[dict[str, Any]] = []
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return frames
    for index, timestamp in enumerate(times):
        path = frame_dir / f"frame_{index:04d}.jpg"
        try:
            _run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-ss", f"{source_start + timestamp:.3f}", "-i", str(source), "-frames:v", "1", "-vf", "scale=480:-2", "-q:v", "4", str(path)],
                90,
                "representative frame extraction",
            )
            if path.exists() and path.stat().st_size:
                frames.append({"time": round(timestamp, 3), "path": str(path)})
        except Exception as exc:
            logger.warning("Could not extract frame at %.2fs: %s", timestamp, exc)
    return frames


def _clean_token(value: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", value.lower()).strip("'")


def transcript_text(words: list[dict[str, Any]], maximum: int = 18000) -> str:
    text = " ".join(str(item.get("word", "")).strip() for item in words if str(item.get("word", "")).strip())
    return text[:maximum]


def _keywords(text: str, maximum: int = 8) -> list[str]:
    values: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower()):
        token = _clean_token(raw)
        if token and token not in STOPWORDS and token not in values:
            values.append(token)
        if len(values) >= maximum:
            break
    return values


def _words_in_range(words: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    return [word for word in words if float(word.get("end", 0.0)) > start and float(word.get("start", 0.0)) < end]


def _window_score(words: list[dict[str, Any]], start: float, end: float) -> float:
    selected = _words_in_range(words, start, end)
    if not selected:
        return 0.0
    tokens = [_clean_token(str(word.get("word", ""))) for word in selected]
    tokens = [token for token in tokens if token]
    density = len(tokens) / max(1.0, end - start)
    punctuation = sum(str(word.get("word", "")).rstrip().endswith(("?", "!", ":")) for word in selected)
    confidence_values = [float(word["probability"]) for word in selected if isinstance(word.get("probability"), (int, float))]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.75
    return round(min(5.0, density) + punctuation * 0.45 + confidence, 4)


def fallback_plan(request: AIComposeRequest, media: dict[str, Any], words: list[dict[str, Any]]) -> CompositionPlan:
    policy = resolve_editorial_policy(request, media, words)
    start = min(request.start_seconds, max(0.0, media["duration"] - 0.05))
    end = min(media["duration"], request.end_seconds or media["duration"])
    available = max(0.1, end - start)
    target = min(request.target_seconds, available)
    if available <= target + 0.2:
        segments = [BaseSegment(source_start=start, source_end=end, reason="source_window")]
    else:
        shot = 7.0 if request.preset == "viral_short" else 11.0
        candidates: list[tuple[float, float, float]] = []
        cursor = start
        while cursor < end - 0.25:
            candidate_end = min(end, cursor + min(shot, target))
            candidates.append((cursor, candidate_end, _window_score(words, cursor, candidate_end)))
            cursor += max(2.0, shot * 0.55)
        needed = max(1, int(math.ceil(target / shot)))
        ranked = sorted(candidates, key=lambda row: row[2], reverse=True)
        selected: list[tuple[float, float, float]] = []
        for item in ranked:
            if all(item[1] <= other[0] or item[0] >= other[1] for other in selected):
                selected.append(item)
            if len(selected) >= needed:
                break
        if not selected:
            selected = candidates[:needed]
        selected.sort(key=lambda row: row[0])
        running = 0.0
        segments = []
        for seg_start, seg_end, score in selected:
            remaining = target - running
            if remaining <= 0.1:
                break
            actual_end = min(seg_end, seg_start + remaining)
            segments.append(BaseSegment(source_start=seg_start, source_end=actual_end, score=score, reason="content_density"))
            running += actual_end - seg_start
        if not segments:
            segments = [BaseSegment(source_start=start, source_end=min(end, start + target), reason="fallback_window")]

    output_duration = sum(item.source_end - item.source_start for item in segments)
    full_text = transcript_text(words, 8000)
    keywords = _keywords(request.brief + " " + full_text, 6)
    asset_requests: list[AssetRequest] = []
    if policy.supplementary_visuals != "off" and request.max_asset_layers and keywords:
        query = " ".join(keywords[:3])
        asset_requests.append(
            AssetRequest(
                query=query,
                start=min(2.5, max(0.0, output_duration - 1.0)),
                end=min(output_duration, max(4.5, output_duration * 0.30)),
                placement="picture_in_picture",
                reason="support_the_opening_topic",
                benefit="Make the opening concept concrete without replacing the primary narrative.",
                confidence=0.62,
                evidence_refs=["opening_transcript_window"],
                must_not_cover=["speaker_face", "captions"],
            )
        )
        if output_duration > 12 and len(keywords) > 3:
            asset_requests.append(
                AssetRequest(
                    query=" ".join(keywords[3:6]),
                    start=max(6.0, output_duration * 0.48),
                    end=min(output_duration, max(9.0, output_duration * 0.68)),
                    placement="bottom",
                    x=0.0,
                    y=0.53,
                    width=1.0,
                    height=0.47,
                    reason="support_the_explanation",
                    benefit="Add a lower-frame explanatory cutaway while keeping the speaker and captions visible.",
                    confidence=0.56,
                    evidence_refs=["midpoint_transcript_window"],
                    must_not_cover=["captions"],
                )
            )

    hook = None
    graphics: list[GraphicPlan] = []
    if full_text:
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", full_text) if item.strip()]
        first = sentences[0] if sentences else full_text.strip()
        hook = first[:150] if first else None
        if hook and policy.motion_graphics != "off":
            graphics.append(
                GraphicPlan(
                    kind="hook",
                    start=0.0,
                    end=min(output_duration, 1.8),
                    text=hook,
                    placement="top",
                    animation="slide_down",
                    style="accent",
                    reason="fallback_transcript_hook",
                    evidence_refs=["transcript_opening"],
                )
            )
        if len(sentences) > 1 and output_duration >= 8.0 and policy.motion_graphics in {"hero", "selective"}:
            bridge = sentences[1][:180]
            bridge_start = min(output_duration - 1.0, max(2.0, output_duration * 0.42))
            graphics.append(
                GraphicPlan(
                    kind="callout",
                    start=bridge_start,
                    end=min(output_duration, bridge_start + 2.2),
                    text=bridge,
                    placement="lower_left",
                    animation="reveal",
                    style="default",
                    reason="fallback_transcript_bridge",
                    evidence_refs=["transcript_followup"],
                )
            )
    elif policy.motion_graphics == "hero":
        # A missing transcription must not silently turn a requested motion-
        # graphics edit into a plain crop. Use the user's own brief as a
        # non-factual title cue; never invent source claims.
        brief_title = re.split(r"(?<=[.!?])\s+", request.brief.strip())[0][:150]
        if brief_title:
            hook = brief_title
            graphics.append(
                GraphicPlan(
                    kind="hook",
                    start=0.0,
                    end=min(output_duration, 1.8),
                    text=brief_title,
                    placement="top",
                    animation="slide_down",
                    style="accent",
                    reason="fallback_brief_title",
                    evidence_refs=["user_brief"],
                )
            )
    sfx_events: list[SfxEvent] = []
    if policy.sfx != "off":
        for asset in asset_requests:
            sfx_events.append(SfxEvent(kind="whoosh", at=asset.start, volume=request.sfx.volume, reason="asset_entered"))
    return CompositionPlan(
        title=(request.brief[:120] or "AI composed short"),
        hook_text=hook if request.preset == "viral_short" else None,
        graphics=graphics,
        base_segments=segments,
        asset_requests=asset_requests,
        sfx_events=sfx_events,
        captions_enabled=request.captions_enabled,
        director_notes=[
            "Fallback plan used because no local director endpoint was available.",
            f"Selected output duration: {output_duration:.2f}s",
            f"Editorial policy: {policy.content_type}/{policy.editing_intensity}/{policy.supplementary_visuals}",
        ],
    )


def asset_search_key(query: str, kind: str) -> tuple[str, str]:
    """Ignore punctuation/case when detecting repeated failed searches."""
    return (" ".join(re.findall(r"\w+", query.casefold())), kind)


class LocalDirector:
    def __init__(self, request: AIComposeRequest, *, narrative_plan: CompositionPlan | None = None, failed_searches=None):
        self.request = request
        self._failed_searches = set(failed_searches or ())
        # Visual design owns graphics/assets/SFX, not already approved narration
        # or source cuts. Snapshot validated values so later mutations cannot
        # change this stage's input or silently replace the speaker's narrative.
        self._narrative_plan = (
            CompositionPlan.model_validate(narrative_plan.model_dump(mode="json"))
            if narrative_plan is not None else None
        )

    def _endpoint(self) -> str:
        return (self.request.llm_url or os.getenv("AI_COMPOSE_LLM_URL", "") or os.getenv("LLAMA_SERVER_URL", "")).strip().rstrip("/")

    def _model(self) -> str:
        return (self.request.llm_model or os.getenv("AI_COMPOSE_LLM_MODEL", "local-director")).strip()

    @staticmethod
    def _allow_deterministic_recovery() -> bool:
        """Allow a valid deterministic plan when a configured model emits unusable output."""
        return os.getenv("AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY", "false").lower() in {
            "1", "true", "yes"
        }

    def _plan_transformers(
        self,
        prompt: str,
        compact_prompt: str | None = None,
    ) -> CompositionPlan:
        """Optional in-process GPU backend for Kaggle.

        It is intentionally opt-in through AI_COMPOSE_HF_MODEL.  Kaggle
        images already contain torch in many configurations, while keeping
        this import lazy prevents the normal FFmpeg-only server from pulling
        a large ML dependency into memory.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ComposeError("Transformers backend requested but torch/transformers is unavailable") from exc
        model_id = os.getenv("AI_COMPOSE_HF_MODEL", "").strip()
        if not model_id:
            raise ComposeError("AI_COMPOSE_HF_MODEL is empty")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
        load_kwargs: dict[str, Any] = {"device_map": "auto"}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
        else:
            load_kwargs["torch_dtype"] = torch.float32
        if os.getenv("AI_COMPOSE_HF_LOAD_4BIT", "true").lower() in {"1", "true", "yes"}:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs.pop("torch_dtype", None)
            except Exception as exc:
                logger.info("4-bit loading unavailable; using native precision: %s", exc)
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)

        def generate_content(user_prompt: str) -> str:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a precise video editor. Return one concrete composition "
                        "plan instance as JSON, never a JSON Schema."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ]
            max_new_tokens = max(512, int(os.getenv("AI_COMPOSE_HF_MAX_NEW_TOKENS", "1800")))
            if hasattr(tokenizer, "apply_chat_template"):
                encoded = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(encoded, "input_ids"):
                    encoded = encoded.to(model.device)
                    input_ids = encoded["input_ids"]
                    output = model.generate(
                        **encoded,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                else:
                    input_ids = encoded.to(model.device)
                    output = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                    )
                generated = output[0][input_ids.shape[-1]:]
                return tokenizer.decode(generated, skip_special_tokens=True)

            encoded = tokenizer(user_prompt, return_tensors="pt").to(model.device)
            output = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False)
            return tokenizer.decode(output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True)

        try:
            return self._generate_valid_plan(generate_content, prompt, compact_prompt)
        finally:
            # Release captured objects without deleting the generator's bindings.
            model = None
            tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_valid_plan(
        self,
        generate_content: Callable[[str], str],
        prompt: str,
        compact_prompt: str | None = None,
    ) -> CompositionPlan:
        """Ask for bounded corrections; never invent missing editorial timings.

        Retry even when compact/full prompts are identical (common for short
        videos). Send the failed response AND field-level validation feedback,
        rather than repeating the same deterministic generation unchanged.
        """
        candidate_prompt = prompt
        errors: list[str] = []
        for attempt in range(1, 4):
            # Transport, model-loading, and GPU failures are not JSON failures.
            content = generate_content(candidate_prompt)
            try:
                payload = self._normalise_plan_payload(self._parse_json(content))
                if "base_segments" not in payload:
                    raise ComposeError("base_segments is required; return a plan instance, not a schema")
                if self._narrative_plan is not None:
                    approved = self._narrative_plan.model_dump(mode="json")
                    for field in ("base_segments", "narration_sections", "voiceovers"):
                        payload[field] = approved[field]
                plan = CompositionPlan.model_validate(payload)
                if self._failed_searches:
                    repeated = [a.query for a in plan.asset_requests
                                if asset_search_key(a.query, a.kind) in self._failed_searches]
                    if repeated:
                        raise ComposeError("Previously failed asset searches repeated: " + repr(repeated)
                                           + "; choose different concrete subjects grounded in the selected transcript, "
                                           "with benefit and evidence_refs. Changing confidence is not a correction.")
                    if not plan.asset_requests and self.request.editor.min_visual_layers > 0:
                        raise ComposeError("Required supporting visuals need new grounded asset_requests")
                expected = {
                    "intro": {"intro"}, "intro_outro": {"intro", "outro"},
                    "intro_bridge_outro": {"intro", "bridge", "outro"},
                }.get(self.request.editor.narration, set())
                missing = expected - {section.position for section in plan.narration_sections}
                if missing:
                    raise ComposeError("Missing requested narration_sections: " + ", ".join(sorted(missing))
                                       + "; each section needs position and non-empty text")
                if any(not section.text.strip() for section in plan.narration_sections):
                    raise ComposeError("Every narration_sections text must contain spoken words")
                if self.request.editor.enabled:
                    from ai_compose_editorial import narrative_errors
                    defects = narrative_errors(self.request, plan, getattr(self, '_planning_words', []))
                    if defects:
                        raise ComposeError('; '.join(defects))
                return plan
            except (ComposeError, ValueError, TypeError) as exc:
                if hasattr(exc, "errors"):
                    feedback = json.dumps(exc.errors(include_input=False, include_url=False),
                                          ensure_ascii=False, default=str)
                else:
                    feedback = str(exc)
                errors.append(f"attempt {attempt}: {feedback}")
                logger.warning("Director validation failed, attempt %s/3: %s", attempt, feedback)
                candidate_prompt = (compact_prompt or prompt) + (
                    "\nCORRECTION REQUIRED. Your previous response failed validation. "
                    "Return a complete corrected plan JSON, not a patch or JSON Schema. "
                    "Preserve the selected narrative and requested supporting assets; do not remove "
                    "features merely to avoid validation. For every asset_requests item provide "
                    "query, numeric start and numeric end. Choose relevant timing from the selected "
                    "narrative: 0 <= start < end <= the sum of base segment durations. "
                    "These are BODY output seconds before narration inserts, not source seconds. "
                    "Correct all listed field errors.\nValidation errors: " + feedback[:6000] +
                    "\nPrevious response (may be truncated):\n" + content[:14000]
                )
        raise ComposeError("Director failed after 3 validation attempts: " + "; ".join(errors))

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        value = content.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
            value = re.sub(r"\s*```$", "", value)
        candidates = [value]
        match = re.search(r"\{.*\}", value, flags=re.S)
        if match and match.group(0) != value:
            candidates.append(match.group(0))

        parse_error: Exception | None = None
        parsed: Any = None
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                parse_error = exc

        if parsed is None:
            try:
                from json_repair import repair_json

                repaired = repair_json(value)
                parsed = json.loads(repaired)
            except Exception as exc:
                raise ComposeError(
                    "Local director did not return valid JSON: "
                    f"{str(parse_error or exc)[:500]}"
                ) from exc

        if not isinstance(parsed, dict):
            raise ComposeError("Local director returned a non-object plan")
        return parsed

    @staticmethod
    def _normalise_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Coerce harmless small-model formatting differences before validation."""
        value = dict(payload)

        # The public contract has one current version. Small models commonly emit
        # `1` or `"1"` after seeing the version suffix in the prompt.
        value["schema_version"] = "sh01/ai-compose@1"

        # Qwen frequently emits a scalar note or object where the schema asks for
        # a bounded list. Preserve the content without weakening the typed model.
        if isinstance(value.get("director_notes"), str):
            note = value["director_notes"].strip()
            value["director_notes"] = [note] if note else []
        elif isinstance(value.get("director_notes"), dict):
            value["director_notes"] = [json.dumps(value["director_notes"], ensure_ascii=False)]

        # Apply the same safe singleton-to-list coercion to plan collections.
        for field in ("base_segments", "asset_requests", "sfx_events", "voiceovers", "graphics", "narration_sections"):
            if isinstance(value.get(field), dict):
                value[field] = [value[field]]
            elif isinstance(value.get(field), str):
                value[field] = []
            elif value.get(field,[]) is None:
                value[field] = []

        for item in value.get("asset_requests", []):
            if not isinstance(item, dict):
                continue
            if isinstance(item.get("evidence_refs"), str):
                item["evidence_refs"] = [item["evidence_refs"]]
            if isinstance(item.get("must_not_cover"), str):
                item["must_not_cover"] = [item["must_not_cover"]]
            try:
                if "confidence" in item:
                    item["confidence"] = float(item["confidence"])
            except (TypeError, ValueError):
                item["confidence"] = 0.0

        graphic_enums = {
            "kind": {"hook", "outro", "lower_third", "quote_card", "callout", "cta", "checklist", "stat", "timeline", "badge"},
            "placement": {"top", "upper_left", "center", "lower_left", "bottom"},
            "animation": {"none", "fade", "slide_up", "slide_down", "pop", "reveal"},
            "style": {"default", "accent", "warning", "quote", "cta"},
        }
        for item in value.get("graphics", []):
            if not isinstance(item, dict):
                continue
            for field, allowed in graphic_enums.items():
                if item.get(field) not in allowed:
                    item[field] = {
                        "kind": "callout",
                        "placement": "center",
                        "animation": "fade",
                        "style": "default",
                    }[field]
            if isinstance(item.get("evidence_refs"), str):
                item["evidence_refs"] = [item["evidence_refs"]]

        return value

    def _prompt(
        self,
        request: AIComposeRequest,
        media: dict[str, Any],
        words: list[dict[str, Any]],
        frames: list[dict[str, Any]],
        *,
        compact: bool = False,
    ) -> str:
        policy = resolve_editorial_policy(self.request, media, words)
        from ai_compose_editorial import speech_units
        units = [{'start':u[0]['start'],'end':u[-1]['end'],'text':' '.join(w['word'] for w in u)}
                 for u in speech_units(words)]
        scene_summary = [{"time": frame["time"], "frame_available": True} for frame in frames[:24 if compact else 48]]
        output_contract = (
            "Return one concrete JSON object, not a JSON Schema. Top-level keys are "
            "schema_version, title, hook_text, outro_text, base_segments, asset_requests, "
            "sfx_events, voiceovers, narration_sections, transition_style, graphics, captions_enabled, and director_notes. "
            "base_segments is REQUIRED and must be a non-empty array. Each item must "
            "contain source_start and source_end. Use empty arrays when optional features "
            "are not needed. Never output $defs, properties, required, definitions, "
            "markdown fences, or explanatory text."
        )
        return (
            "You are a short-form video editor. Return ONLY the requested concrete plan JSON. "
            "You are planning, not executing shell commands. Never invent source timestamps outside the media. "
            "base_segments use absolute source seconds. asset_requests use output-timeline seconds. "
            "graphics use output-timeline seconds. "
            "Use supplementary assets only when they clarify the spoken idea. Do not invent factual claims. "
            "When supporting visuals are allowed and a concrete visual opportunity exists, emit one or more "
            "asset_requests with REQUIRED query, start, and end; start/end must be numeric BODY "
            "output seconds satisfying 0 <= start < end <= sum(source_end-source_start). "
            "For example, an asset item has this shape (choose your own grounded query/timing): "
            '{"query":"relevant visual subject","start":2.0,"end":5.0,"kind":"video",'
            '"placement":"picture_in_picture"}. '
            "Also include benefit, confidence, evidence_refs, and must_not_cover. "
            "When motion graphics are allowed, emit concise graphics entries with kind, start, end, text, "
            "kind is hook/outro/lower_third/quote_card/callout/cta/checklist/stat/timeline/badge. "
            "For checklist/timeline use | separated short labels. For stat use only numbers present in evidence. "
            "placement, animation, and style. Keep graphics clear of speaker faces and captions; do not use "
            "graphics as a substitute for source dialogue. "
            "Use adaptive SFX sparingly: only for meaningful visual entrances, transitions, or payoffs. "
            "If a voiceover request is present, emit voiceovers with exact output-timeline start times; otherwise leave voiceovers empty. "
            f"{output_contract}\n"
            f"{policy_prompt_context(policy)}\n"
            f"Brief: {request.brief}\nPreset: {request.preset}\nTarget output seconds: {request.target_seconds}\n"
            f"Voiceover mode: {request.voiceover.mode}\nSFX mode: {request.sfx.mode}\n"
            f"Inserted narration mode: {request.editor.narration}; total speech plus pauses budget: {request.editor.narration_budget_seconds}s. "
            "When inserted narration is enabled, leave voiceovers empty and provide narration_sections: "
            '[{"position":"intro","text":"short evidence-grounded hook","evidence_refs":["source:0-3"]}, '
            '{"position":"bridge","after_segment":0,"text":"short grounded connection"}, '
            '{"position":"outro","text":"short grounded takeaway"}]. '
            "Include only the positions requested by mode; auto may omit unnecessary narration. "
            "Use at most 5-8 words per section, no invented facts. These are inserted BETWEEN source excerpts, "
            "never spoken over source dialogue. For a bridge select at least two meaningful base_segments. "
            "All other cue times refer to the BODY before these inserts. Base segments must have kind=source. "
            f"Required actual features: {request.editor.required_features}. "
            "Choose transition_style cut/dissolve/swipe/match appropriate to the brief.\n"
            f"Media metadata: {json.dumps(media, separators=(',', ':'))}\n"
            f"Timestamped transcript: {json.dumps([{'start':round(float(w['start']),2),'end':round(float(w['end']),2),'text':w.get('word','')} for w in words], ensure_ascii=False)[:14000 if compact else 26000]}\n"
            f"Representative frames: {json.dumps(scene_summary, separators=(',', ':'))}\n"
            + ("VISUAL DESIGN PASS: base_segments, narration_sections and voiceovers are locked "
               "to the validated narrative plan supplied in the brief. Do not rewrite them. "
               "Design graphics, supplementary assets and SFX for that existing narrative.\n"
               if self._narrative_plan is not None else "") +
            "Choose a small number of high-confidence segments and keep all arrays concise. "
            "Optimize for a clear opening promise, context, evidence and a complete payoff; no invented stakes or claims. "
            "Never select evenly spaced snippets. A source excerpt must begin/end at a sentence or pause boundary, "
            "unless it directly continues the preceding/following source excerpt. Prefer one coherent passage to disconnected highlights. "
            "Narration must add distinct context, connection and takeaway; never repeat the title as the script. "
            "Use 4-8 words per inserted section with source:start-end evidence_refs overlapping selected clips. "
            "Avoid long quotes on screen; keep graphic copy under 90 characters. "
            f"Available speech units (absolute source times): {json.dumps(units,ensure_ascii=False)[:14000]}"

        )

    def plan(self, media: dict[str, Any], words: list[dict[str, Any]], frames: list[dict[str, Any]]) -> tuple[CompositionPlan, list[str]]:
        self._planning_words = words
        endpoint = self._endpoint()
        warnings: list[str] = []
        hf_model = os.getenv("AI_COMPOSE_HF_MODEL", "").strip()
        if hf_model:
            try:
                prompt = self._prompt(self.request, media, words, frames)
                compact_prompt = self._prompt(self.request, media, words, frames, compact=True)
                return self._plan_transformers(prompt, compact_prompt), warnings
            except Exception as exc:
                if self.request.llm_required and not self._allow_deterministic_recovery():
                    raise ComposeError(f"Local Transformers director failed: {exc}") from exc
                warnings.append(
                    "Local Transformers director failed; deterministic recovery was used: "
                    f"{str(exc)[:240]}"
                )
                return fallback_plan(self.request, media, words), warnings
        if not endpoint:
            if self.request.llm_required and not self._allow_deterministic_recovery():
                raise ComposeError("AI_COMPOSE_LLM_URL is required when llm_required=true")
            warnings.append("No local director endpoint configured; deterministic recovery was used.")
            return fallback_plan(self.request, media, words), warnings
        prompt = self._prompt(self.request, media, words, frames)
        try:
            import requests

            def generate_remote(candidate_prompt: str) -> str:
                if os.getenv("AI_COMPOSE_LLM_BACKEND", "openai").strip().lower() == "ollama" or endpoint.endswith("/api/chat"):
                    response = requests.post(
                        endpoint if endpoint.endswith("/api/chat") else f"{endpoint}/api/chat",
                        json={"model": self._model(), "stream": False, "format": CompositionPlan.model_json_schema(), "messages": [{"role": "user", "content": candidate_prompt}]},
                        timeout=(30, 600),
                    )
                    response.raise_for_status()
                    content = response.json().get("message", {}).get("content", "")
                else:
                    response = requests.post(
                        endpoint if endpoint.endswith("/chat/completions") else f"{endpoint}/v1/chat/completions",
                        json={
                            "model": self._model(),
                            "temperature": 0.15,
                            "messages": [{"role": "system", "content": "Produce only valid JSON."}, {"role": "user", "content": candidate_prompt}],
                            "response_format": {"type": "json_schema", "json_schema": {"name": "composition_plan", "schema": CompositionPlan.model_json_schema(), "strict": True}},
                        },
                        timeout=(30, 600),
                    )
                    response.raise_for_status()
                    content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
                return str(content)

            plan = self._generate_valid_plan(
                generate_remote, prompt, self._prompt(self.request, media, words, frames, compact=True)
            )
            return plan, warnings
        except Exception as exc:
            if self.request.llm_required and not self._allow_deterministic_recovery():
                raise ComposeError(f"Local director failed: {exc}") from exc
            warnings.append(
                "Local director failed; deterministic recovery was used: "
                f"{str(exc)[:240]}"
            )
            return fallback_plan(self.request, media, words), warnings


def _candidate_tokens(candidate: dict[str, Any]) -> set[str]:
    text = " ".join(str(candidate.get(key, "")) for key in ("title", "description", "tags", "filename"))
    return set(_keywords(text, 80))


def _score_candidate(query: str, candidate: dict[str, Any]) -> float:
    wanted = set(_keywords(query, 16))
    available = _candidate_tokens(candidate)
    if not wanted:
        return 0.0
    overlap = len(wanted & available) / len(wanted)
    if candidate.get("source") == "local":
        overlap += 0.08
    return round(min(1.0, overlap), 4)


def _read_sidecar(path: Path) -> dict[str, Any]:
    for sidecar in (path.with_suffix(path.suffix + ".json"), path.with_suffix(".json")):
        if sidecar.exists():
            try:
                value = json.loads(sidecar.read_text(encoding="utf-8"))
                if isinstance(value, dict):
                    return value
            except (OSError, json.JSONDecodeError):
                pass
    return {}


def _search_local(library: Path, query: str, kind: str) -> list[dict[str, Any]]:
    if not library.exists() or not library.is_dir():
        return []
    extensions = VIDEO_EXTENSIONS if kind == "video" else IMAGE_EXTENSIONS
    results: list[dict[str, Any]] = []
    for path in library.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        metadata = _read_sidecar(path)
        result = {
            "asset_id": "local_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:20],
            "source": "local",
            "local_path": str(path.resolve()),
            "download_url": None,
            "filename": path.name,
            "title": metadata.get("title", path.stem),
            "description": metadata.get("description", ""),
            "tags": metadata.get("tags", []),
            "license": metadata.get("license", "local_library"),
            "attribution": metadata.get("attribution"),
            "kind": kind,
        }
        result["score"] = _score_candidate(query, result)
        results.append(result)
        if len(results) >= 500:
            break
    return sorted(results, key=lambda item: item["score"], reverse=True)


def _search_remote(query: str, kind: str, request: AIComposeRequest, *, diagnostics=None) -> list[dict[str, Any]]:
    diagnostics = diagnostics if diagnostics is not None else []
    cfg = request.asset_sources
    configured = (
        (cfg.pexels and bool(os.getenv("PEXELS_API_KEY", "").strip()))
        or (cfg.pixabay and bool(os.getenv("PIXABAY_API_KEY", "").strip()))
        or (cfg.unsplash and kind == "image" and bool(os.getenv("UNSPLASH_ACCESS_KEY", "").strip()))
        or cfg.wikimedia
    )
    if not configured:
        return []
    try:
        import requests
    except ImportError as exc:
        raise ComposeError("requests is required for remote asset search") from exc
    results: list[dict[str, Any]] = []
    limit = cfg.max_candidates_per_query
    if cfg.wikimedia:
        try:
            from ai_compose_commons import search
            queries = list(dict.fromkeys([query, ' '.join(_keywords(query, 3))]))
            for candidate_query in queries:
                matches = search(candidate_query, kind, limit)
                for candidate in matches:
                    candidate['score'] = _score_candidate(query, candidate)
                results.extend(matches)
                if matches:
                    break
        except Exception as exc:
            diagnostics.append({"provider": "wikimedia", **failure_details(exc, "https://commons.wikimedia.org/w/api.php")})
            logger.warning('Wikimedia search failed: %s', exc)
    if cfg.pexels and os.getenv("PEXELS_API_KEY", "").strip():
        try:
            endpoint = "https://api.pexels.com/videos/search" if kind == "video" else "https://api.pexels.com/v1/search"
            response = requests.get(endpoint, headers={"Authorization": os.getenv("PEXELS_API_KEY", "").strip()}, params={"query": query, "per_page": limit, "orientation": "portrait"}, timeout=(20, 60))
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("videos" if kind == "video" else "photos", []):
                if kind == "video":
                    files = sorted(item.get("video_files", []), key=lambda value: (value.get("width", 0) < 1080, -value.get("width", 0)))
                    download_url = files[0].get("link") if files else None
                    title = urlparse(item.get('url', '')).path.replace('-', ' ')
                    attribution = f"Video by {item.get('user', {}).get('name', 'unknown')} on Pexels"
                else:
                    download_url = item.get("src", {}).get("portrait") or item.get("src", {}).get("large2x")
                    title = item.get("alt") or f"Pexels photo {item.get('id')}"
                    attribution = f"Photo by {item.get('photographer', 'unknown')} on Pexels"
                if download_url:
                    results.append({"asset_id": f"pexels_{item.get('id')}", "source": "pexels", "download_url": download_url, "title": title, "description": item.get("alt", ""), "tags": [], "license": "Pexels", "attribution": attribution, "kind": kind, "score": _score_candidate(query, {"title": title, "description": item.get("alt", ""), "tags": []})})
        except Exception as exc:
            diagnostics.append({"provider": "pexels", "error_type": type(exc).__name__, "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            logger.warning("Pexels search failed: %s", exc)
    if cfg.pixabay and os.getenv("PIXABAY_API_KEY", "").strip():
        try:
            endpoint = "https://pixabay.com/api/videos/" if kind == "video" else "https://pixabay.com/api/"
            params = {"key": os.getenv("PIXABAY_API_KEY", "").strip(), "q": query, "per_page": limit, "safesearch": "true"}
            if kind != "video":
                params["image_type"] = "photo"
            response = requests.get(endpoint, params=params, timeout=(20, 60))
            response.raise_for_status()
            for item in response.json().get("hits", []):
                if kind == "video":
                    video = item.get("videos", {}).get("medium") or item.get("videos", {}).get("small") or {}
                    download_url = video.get("url")
                else:
                    download_url = item.get("largeImageURL") or item.get("webformatURL")
                if download_url:
                    title = f"Pixabay {kind} {item.get('id')}"
                    results.append({"asset_id": f"pixabay_{item.get('id')}", "source": "pixabay", "download_url": download_url, "title": title, "description": item.get("tags", ""), "tags": item.get("tags", ""), "license": "Pixabay Content License", "attribution": "Source: Pixabay", "kind": kind, "score": _score_candidate(query, {"title": title, "description": item.get("tags", ""), "tags": item.get("tags", "")})})
        except Exception as exc:
            diagnostics.append({"provider": "pixabay", "error_type": type(exc).__name__, "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            logger.warning("Pixabay search failed: %s", exc)
    if cfg.unsplash and kind == "image" and os.getenv("UNSPLASH_ACCESS_KEY", "").strip():
        try:
            response = requests.get("https://api.unsplash.com/search/photos", headers={"Authorization": f"Client-ID {os.getenv('UNSPLASH_ACCESS_KEY', '').strip()}"}, params={"query": query, "per_page": limit, "orientation": "portrait"}, timeout=(20, 60))
            response.raise_for_status()
            for item in response.json().get("results", []):
                image_url = item.get("urls", {}).get("regular")
                if image_url:
                    photographer = item.get("user", {}).get("name", "unknown")
                    results.append({"asset_id": f"unsplash_{item.get('id')}", "source": "unsplash", "download_url": image_url, "title": item.get("alt_description") or item.get("description") or "Unsplash photo", "description": item.get("alt_description") or "", "tags": [], "license": "Unsplash", "attribution": f"Photo by {photographer} on Unsplash", "kind": kind, "score": _score_candidate(query, {"title": item.get("alt_description", ""), "description": item.get("description", ""), "tags": []})})
        except Exception as exc:
            diagnostics.append({"provider": "unsplash", "error_type": type(exc).__name__, "http_status": getattr(getattr(exc, "response", None), "status_code", None)})
            logger.warning("Unsplash search failed: %s", exc)
    return sorted(results, key=lambda item: item.get("score", 0.0), reverse=True)


def _placement(asset: AssetRequest) -> tuple[float, float, float, float | None]:
    if asset.placement == "full":
        return 0.0, 0.0, 1.0, 1.0
    if asset.placement == "top":
        return 0.0, 0.0, 1.0, 0.47
    if asset.placement == "bottom":
        return 0.0, 0.53, 1.0, 0.47
    if asset.placement == "left":
        return 0.0, 0.0, 0.48, 1.0
    if asset.placement == "right":
        return 0.52, 0.0, 0.48, 1.0
    if asset.placement == "split":
        return 0.5, 0.0, 0.5, 1.0
    return asset.x, asset.y, asset.width, asset.height


def resolve_assets(request: AIComposeRequest, plan: CompositionPlan, root: Path, *, failed_searches=None, words=None) -> tuple[list[ResolvedLayer], list[dict[str, Any]], list[str]]:
    if not request.fetch_assets:
        return [], [], []
    asset_root = root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    library_setting = request.local_asset_library or os.getenv("AI_COMPOSE_ASSET_LIBRARY", "").strip()
    used: set[str] = set()
    layers: list[ResolvedLayer] = []
    manifest: list[dict[str, Any]] = []
    warnings: list[str] = []
    minimum_score = min(1.0, max(0.0, float(os.getenv("AI_COMPOSE_MIN_ASSET_SCORE", "0.18"))))
    failed_searches = failed_searches if failed_searches is not None else set()
    reports = []
    for index, asset_request in enumerate(plan.asset_requests[: request.max_asset_layers]):
        key = asset_search_key(asset_request.query, asset_request.kind)
        report = {"query": asset_request.query, "kind": asset_request.kind,
                  "minimum_score": minimum_score, "candidates": [], "outcome": "pending"}
        reports.append(report)
        if key in failed_searches:
            report["outcome"] = "previously_failed_search_skipped"
            continue
        candidates: list[dict[str, Any]] = []
        if request.asset_sources.local_library and library_setting:
            candidates.extend(_search_local(Path(library_setting), asset_request.query, asset_request.kind))
        if request.fetch_assets:
            candidates.extend(_search_remote(asset_request.query, asset_request.kind, request, diagnostics=report.setdefault("provider_errors", [])))
        candidates.sort(key=lambda item: (item.get("asset_id") in used, -float(item.get("score", 0.0))))
        eligible = [c for c in candidates if c.get('asset_id') not in used
                    and float(c.get('score',0) or 0) >= minimum_score][:3]
        report["candidate_count"] = len(candidates)
        report["eligible_count"] = len(eligible)
        for c in candidates:
            report["candidates"].append({"asset_id": c.get("asset_id"), "title": c.get("title"),
                "score": c.get("score", 0), "source": c.get("source"),
                "reason": "already_used" if c.get("asset_id") in used else
                    ("below_score_floor" if float(c.get("score", 0) or 0) < minimum_score else "eligible")})
        candidate = None
        source_offset = 0.0
        for attempt, choice in enumerate(eligible):
            try:
                local_path = Path(str(choice['local_path'])) if choice.get('local_path') else asset_root / (
                    f'asset_{index:02d}_{attempt}' + _safe_filename(str(choice.get('download_url') or ''),
                    '.mp4' if asset_request.kind == 'video' else '.jpg'))
                if choice.get('download_url'):
                    _download_sync(str(choice['download_url']),local_path)
                if not local_path.is_file() or not local_path.stat().st_size:
                    raise ComposeError('Asset file is missing or empty')
                if asset_request.kind == 'image':
                    from PIL import Image
                    with Image.open(local_path) as picture:
                        picture.verify()
                else:
                    probe_media(local_path)
                if request.editor.enabled and request.editor.asset_visual_review:
                    from ai_compose_asset_review import select_asset_window
                    review_root = asset_root/f'review_{index:02d}_{attempt}'
                    review_root.mkdir(exist_ok=True)
                    offset, observations = select_asset_window(request, asset_request, local_path, plan, words, review_root)
                    report.setdefault('visual_reviews', []).append({'asset_id':choice.get('asset_id'), 'windows':observations})
                    if offset is None:
                        continue
                    source_offset = offset
                candidate = choice
                break
            except Exception as exc:
                report.setdefault("download_decode_errors", []).append({
                    "asset_id": choice.get("asset_id"),
                    **failure_details(exc, choice.get("download_url"))})
                warnings.append(f'Candidate rejected for {asset_request.query!r}: {str(exc)[:200]}')
        if candidate is None:
            report["outcome"] = "no_candidates" if not candidates else "no_usable_candidate"
            if any(e.get("error_type") == "MediaHTTPBackoff" for e in report.get("download_decode_errors", [])):
                report["outcome"] = "host_cooldown"
            if eligible and len(report.get('visual_reviews', [])) == len(eligible) and not report.get('download_decode_errors'):
                report['outcome'] = 'visual_content_rejected'
                failed_searches.add(key)
            # Cache deterministic empty/score failures; allow transient download retries.
            if not eligible and not report.get("provider_errors") and not any(c.get("asset_id") in used for c in candidates):
                failed_searches.add(key)
            warnings.append(f'No usable relevant asset for {asset_request.query!r}; score floor {minimum_score:.3f}')
            continue
        candidate_score = float(candidate.get('score',0) or 0)
        used.add(str(candidate['asset_id']))
        try:
            x, y, width, height = _placement(asset_request)
            from ai_compose_editorial import speaker_safe
            if speaker_safe(request):
                x, y, width, height = .04, .66, .92, .14
            layer = ResolvedLayer(
                asset_id=str(candidate["asset_id"]), local_path=str(local_path), source=str(candidate.get("source", "unknown")), license=candidate.get("license"), attribution=candidate.get("attribution"), query=asset_request.query, kind=asset_request.kind, start=asset_request.start, end=asset_request.end, x=x, y=y, width=width, height=height, animation=("fade" if speaker_safe(request) else asset_request.animation), animation_seconds=asset_request.animation_seconds, opacity=asset_request.opacity, score=candidate_score, reason=asset_request.reason, source_offset=source_offset,
            )
            layers.append(layer)
            report["outcome"] = "resolved"
            manifest.append({
                **candidate,
                "local_path": str(local_path),
                "source_offset": source_offset,
                "visual_review": report.get("visual_reviews", []),
                "sha256": _sha256(local_path),
                "timeline": {"start": asset_request.start, "end": asset_request.end},
                "geometry": {"x": x, "y": y, "width": width, "height": height},
                "animation": layer.animation,
                "benefit": asset_request.benefit,
                "confidence": asset_request.confidence,
                "evidence_refs": asset_request.evidence_refs,
                "must_not_cover": asset_request.must_not_cover,
            })
        except Exception as exc:
            warnings.append(f"Asset {asset_request.query!r} was rejected: {str(exc)[:220]}")
    (root / "retrieval_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8")
    return layers, manifest, warnings


def _ass_time(seconds: float) -> str:
    total = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ")


def _graphic_text(value: str, width: int = 26, max_lines: int = 4) -> str:
    """Wrap graphic copy into bounded ASS lines so it cannot clip the frame."""

    clean = " ".join(str(value).split())
    lines = textwrap.wrap(clean, width=width, break_long_words=False, break_on_hyphens=False)
    lines = lines[:max_lines] or [clean[:width]]
    return r"\N".join(_ass_escape(line) for line in lines)


def _graphic_position(placement: str) -> tuple[int, int, int]:
    """Return a safe ASS anchor/alignment position for a graphic cue."""

    positions = {
        "top": (540, 150, 8),
        "upper_left": (90, 260, 7),
        "center": (540, 960, 5),
        "lower_left": (90, 1500, 1),
        "bottom": (540, 1750, 2),
    }
    return positions.get(placement, positions["center"])


def _graphic_tags(cue: Any) -> str:
    """Translate a typed graphic animation into bounded ASS tags."""

    x, y, alignment = _graphic_position(cue.placement)
    tags = [f"\\an{alignment}", f"\\pos({x},{y})"]
    if cue.animation in {"fade", "reveal"}:
        tags.append("\\fad(120,160)")
    elif cue.animation == "slide_up":
        tags.append(f"\\move({x},{y + 90},{x},{y},0,280)")
    elif cue.animation == "slide_down":
        tags.append(f"\\move({x},{y - 90},{x},{y},0,280)")
    elif cue.animation == "pop":
        tags.extend(["\\fscx85", "\\fscy85", "\\t(0,180,\\fscx100\\fscy100)", "\\fad(80,120)"])
    return "{" + "".join(tags) + "}"


def _graphic_style(cue: Any) -> str:
    return {
        "default": "GraphicDefault",
        "accent": "GraphicAccent",
        "warning": "GraphicWarning",
        "quote": "GraphicQuote",
        "cta": "GraphicCta",
    }.get(cue.style, "GraphicDefault")


def _make_ass(
    words: list[dict[str, Any]],
    segments: list[BaseSegment],
    duration: float,
    plan: CompositionPlan,
    destination: Path,
    edit_plan: EditPlan | None = None,
) -> None:
    mapped: list[dict[str, Any]] = []
    offsets: list[float] = []
    cursor = 0.0
    for segment in segments:
        offsets.append(cursor)
        cursor += segment.source_end - segment.source_start
    for word in words:
        start = float(word.get("start", 0.0))
        end = float(word.get("end", start + 0.1))
        for segment, offset in zip(segments, offsets):
            if start >= segment.source_start and start < segment.source_end:
                mapped.append({"start": offset + start - segment.source_start, "end": min(duration, offset + end - segment.source_start), "word": str(word.get("word", "")).strip()})
                break
    lines: list[tuple[float, float, str]] = []
    current: list[dict[str, Any]] = []
    for item in mapped:
        if current and (item["start"] - current[-1]["end"] > 0.75 or len(current) >= 7 or len(" ".join(x["word"] for x in current)) >= 42):
            lines.append((current[0]["start"], current[-1]["end"], " ".join(x["word"] for x in current)))
            current = []
        current.append(item)
    if current:
        lines.append((current[0]["start"], current[-1]["end"], " ".join(x["word"] for x in current)))
    content = [
        "[Script Info]", "ScriptType: v4.00+", "PlayResX: 1080", "PlayResY: 1920", "WrapStyle: 2", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]", "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,DejaVu Sans,48,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,3,1,2,80,80,170,1",
        "Style: Hook,DejaVu Sans,64,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,4,1,8,80,80,140,1",
        "Style: Outro,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,4,1,2,80,80,180,1",
        "Style: GraphicDefault,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,4,1,5,80,80,120,1",
        "Style: GraphicAccent,DejaVu Sans,66,&H0000FFFF,&H0000FFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,5,1,5,80,80,120,1",
        "Style: GraphicWarning,DejaVu Sans,58,&H0044AAFF,&H0044AAFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,5,1,5,80,80,120,1",
        "Style: GraphicQuote,DejaVu Sans,56,&H00FFFFFF,&H00FFFFFF,&H00333333,&HCC111111,0,1,0,0,100,100,0,0,1,4,1,5,90,90,120,1",
        "Style: GraphicCta,DejaVu Sans,60,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,5,1,2,80,80,150,1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in lines:
        if end > start:
            content.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(min(duration, end))},Default,,0,0,0,,{_ass_escape(text)}")
    explicit_graphics = list(edit_plan.graphics) if edit_plan is not None else []
    explicit_kinds = {cue.kind for cue in explicit_graphics}
    for cue in explicit_graphics:
        start = max(0.0, min(duration, float(cue.start)))
        end = max(start + 0.05, min(duration, float(cue.end)))
        if end <= start or not cue.text.strip():
            continue
        content.append(
            f"Dialogue: 10,{_ass_time(start)},{_ass_time(end)},{_graphic_style(cue)},,0,0,0,,"
            f"{_graphic_tags(cue)}{_graphic_text(cue.text)}"
        )
    if edit_plan is None and plan.hook_text and "hook" not in explicit_kinds:
        content.append(f"Dialogue: 5,0:00:00.00,{_ass_time(min(duration, 1.7))},Hook,,0,0,0,,{{\\fad(80,120)}}{_graphic_text(plan.hook_text)}")
    if edit_plan is None and plan.outro_text and "outro" not in explicit_kinds and duration > 1.0:
        start = max(0.0, duration - 2.4)
        content.append(f"Dialogue: 5,{_ass_time(start)},{_ass_time(duration)},Outro,,0,0,0,,{{\\fad(120,120)}}{_graphic_text(plan.outro_text)}")
    destination.write_text("\n".join(content) + "\n", encoding="utf-8")


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS or path.suffix.lower() not in IMAGE_EXTENSIONS | AUDIO_EXTENSIONS


def _choose_encoder() -> tuple[str, list[str]]:
    requested = os.getenv("AI_COMPOSE_VIDEO_CODEC", "auto").strip().lower()
    if requested in {"h264_nvenc", "hevc_nvenc", "libx264"}:
        return requested, ([] if requested != "libx264" else ["-preset", "ultrafast", "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0", "-tune", "zerolatency"])
    if os.getenv("AI_COMPOSE_USE_NVENC", "false").lower() in {"1", "true", "yes"}:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            try:
                result = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=30, check=False)
                if b"h264_nvenc" in result.stdout:
                    return "h264_nvenc", ["-preset", "p4", "-cq", "23"]
            except Exception:
                pass
    return "libx264", ["-preset", "ultrafast", "-x264-params", "ref=1:bframes=0:rc-lookahead=0:sync-lookahead=0:mbtree=0", "-tune", "zerolatency"]


def _sfx_path(library: str | None, kind: str) -> Path | None:
    setting = library or os.getenv("AI_COMPOSE_SFX_LIBRARY", "").strip()
    if not setting:
        return None
    root = Path(setting)
    if not root.exists():
        return None
    exact = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS and kind.lower() in path.stem.lower()]
    if exact:
        return exact[0]
    return None


def _download_freesound_sfx(kind: str, root: Path, warnings: list[str]) -> Path | None:
    token = os.getenv("FREESOUND_API_KEY", "").strip()
    if not token:
        return None
    try:
        import requests

        response = requests.get(
            "https://freesound.org/apiv2/search/text/",
            headers={"Authorization": f"Token {token}"},
            params={"query": kind, "filter": "duration:[0 TO 3]", "fields": "id,name,previews,license,username,url", "page_size": 5},
            timeout=(20, 60),
        )
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None
        item = results[0]
        url = item.get("previews", {}).get("preview-hq-mp3") or item.get("previews", {}).get("preview-lq-mp3")
        if not url:
            return None
        destination = root / "sfx" / f"freesound_{item.get('id')}.mp3"
        _download_sync(url, destination, max_bytes=50 * 1024 * 1024)
        warnings.append(f"SFX {kind} sourced from Freesound; retain license metadata for publishing ({item.get('license', 'unknown')})")
        return destination
    except Exception as exc:
        warnings.append(f"Freesound lookup failed for {kind}: {str(exc)[:180]}")
        return None


def _synthesize_voiceovers(request: AIComposeRequest, plan: CompositionPlan, root: Path, warnings: list[str]) -> list[tuple[Path, float, float, str]]:
    if request.voiceover.mode == "off":
        return []
    voiceovers = list(plan.voiceovers)
    if not voiceovers and request.voiceover.mode != "off" and request.voiceover.text:
        voiceovers = [VoiceoverPlan(text=request.voiceover.text, start=0.0, volume=request.voiceover.volume)]
    if not voiceovers:
        return []
    piper = shutil.which("piper")
    model = (request.voiceover.voice or os.getenv("AI_COMPOSE_TTS_MODEL", "")).strip()
    if not piper or not model:
        warnings.append("Voiceover was requested but piper and AI_COMPOSE_TTS_MODEL were not both available")
        return []
    outputs: list[tuple[Path, float, float, str]] = []
    for index, voiceover in enumerate(voiceovers[:3]):
        destination = root / f"voiceover_{index:02d}.wav"
        try:
            result = subprocess.run([piper, "--model", model, "--output_file", str(destination)], input=voiceover.text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180, check=False)
            if result.returncode != 0 or not destination.exists() or destination.stat().st_size == 0:
                raise ComposeError(result.stderr.decode("utf-8", errors="replace")[-500:])
            outputs.append((destination, voiceover.start, voiceover.volume, "voiceover"))
        except Exception as exc:
            warnings.append(f"Voiceover synthesis failed: {str(exc)[:220]}")
    return outputs


def _build_audio_inputs(request: AIComposeRequest, plan: CompositionPlan, root: Path, warnings: list[str]) -> list[tuple[Path, float, float, str]]:
    inputs: list[tuple[Path, float, float, str]] = []
    for event in plan.sfx_events:
        if request.sfx.mode == "off":
            continue
        path = _sfx_path(request.sfx.local_library or request.local_sfx_library, event.kind)
        if not path and request.asset_sources.freesound:
            path = _download_freesound_sfx(event.kind, root, warnings)
        if not path and request.editor.enabled and request.editor.procedural_sfx:
            from ai_compose_sound import synthesize
            path = synthesize(root / 'sfx' / (event.kind + '.wav'), event.kind, .35)
        if path:
            inputs.append((path, event.at, event.volume, "sfx"))
        else:
            warnings.append(f"No local SFX matched {event.kind}; event skipped")
    music = os.getenv("AI_COMPOSE_MUSIC_PATH", "").strip()
    music_allowed = not request.editor.enabled or (request.editor.music != 'off' and request.editorial_policy.music != 'off')
    if request.music_url and music_allowed:
        path = root / ("music" + _safe_filename(request.music_url, ".mp3"))
        try:
            _download_sync(request.music_url, path)
            inputs.append((path, 0.0, request.music_volume, "music"))
        except Exception as exc:
            warnings.append(f"Music download failed: {str(exc)[:220]}")
    elif music and Path(music).exists() and music_allowed:
        inputs.append((Path(music), 0.0, request.music_volume, "music"))
    elif request.editor.enabled and request.editor.music == 'procedural' and music_allowed:
        from ai_compose_sound import synthesize
        duration = sum(s.source_end-s.source_start for s in plan.base_segments)
        inputs.append((synthesize(root / 'score.wav', 'music', duration), 0., request.music_volume, 'music'))
    inputs.extend(_synthesize_voiceovers(request, plan, root, warnings))
    return inputs


def _piecewise_keyframe_expr(keyframes: list[Any], field: str, track_start: float) -> str:
    """Build a bounded FFmpeg expression that interpolates a camera field."""

    points = sorted(
        ((max(0.0, float(item.time) - track_start), float(getattr(item, field))) for item in keyframes),
        key=lambda item: item[0],
    )
    if not points:
        return "0"
    expression = f"{points[-1][1]:.5f}"
    for left, right in reversed(list(zip(points, points[1:]))):
        t0, value0 = left
        t1, value1 = right
        span = max(0.001, t1 - t0)
        interpolation = f"({value0:.5f}+({value1 - value0:.5f})*(t-{t0:.5f})/{span:.5f})"
        expression = f"if(lt(t,{t1:.5f}),{interpolation},{expression})"
    return expression


def _camera_visual_filter(camera: Any, width: int, height: int, fps: int) -> str:
    """Return a safe FFmpeg camera filter consuming the complete keyframe track.

    zoompan evaluates scale and position for each output frame while keeping
    the output dimensions stable. Track times are relative to the source clip.
    """

    zoom = _piecewise_keyframe_expr(camera.keyframes, 'scale', float(camera.start)).replace('t-', f'on/{fps}-').replace('lt(t,', f'lt(on/{fps},')
    x = _piecewise_keyframe_expr(camera.keyframes, 'x', float(camera.start)).replace('t-', f'on/{fps}-').replace('lt(t,', f'lt(on/{fps},')
    y = _piecewise_keyframe_expr(camera.keyframes, 'y', float(camera.start)).replace('t-', f'on/{fps}-').replace('lt(t,', f'lt(on/{fps},')
    return (f"fps={fps},scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height},zoompan=z='{zoom}':x='(iw-iw/zoom)*(.5+({x}))':"
            f"y='(ih-ih/zoom)*(.5+({y}))':d=1:s={width}x{height}:fps={fps}")



def _transition_name(kind: str) -> str | None:
    return {
        "dissolve": "fade",
        "swipe": "slideleft",
        "match": "smoothleft",
    }.get(kind)


def render_composition(
    source: Path,
    media: dict[str, Any],
    request: AIComposeRequest,
    plan: CompositionPlan,
    layers: list[ResolvedLayer],
    words: list[dict[str, Any]],
    root: Path,
    variant: int,
    warnings: list[str],
    edit_plan: EditPlan | None = None,
    prepared_voiceovers: list | None = None,
) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ComposeError("ffmpeg is required")
    segments = plan.base_segments
    duration = sum(max(0.05, segment.source_end - segment.source_start) for segment in segments)
    if duration <= 0.1:
        raise ComposeError("Composition plan contains no usable base duration")
    if edit_plan is None:
        policy = resolve_editorial_policy(request, media, words)
        edit_plan = compile_edit_plan(request, plan, layers, words, policy, media, variant=variant)
    output = root / f"variant_{variant:02d}.mp4"
    ass = root / f"variant_{variant:02d}.ass"
    motion_counts = {}
    if request.editor.enabled:
        from ai_compose_motion import write_motion_ass
        motion_counts = write_motion_ass(ass, request, plan, edit_plan, words)
    else:
        _make_ass(words, segments, duration, plan, ass, edit_plan)

    command: list[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-filter_complex_threads", "2" if request.editor.enabled else "1"]
    video_input_count = len(segments)
    input_index = 0
    for segment in segments:
        if segment.kind == 'narration':
            command += ['-f','lavfi','-i',f'color=c=0x101b2c:s={request.output_width}x{request.output_height}:r={request.fps}:d={segment.source_end-segment.source_start:.6f}']
        else:
            command += ["-ss", f"{segment.source_start:.3f}", "-t", f"{segment.source_end - segment.source_start:.3f}", "-i", str(source)]
        input_index += 1
    layer_indices: list[tuple[ResolvedLayer, int]] = []
    for layer in layers:
        path = Path(layer.local_path)
        command += (["-stream_loop", "-1", "-ss", str(layer.source_offset), "-i", str(path)] if _is_video(path) else ["-loop", "1", "-i", str(path)])
        layer_indices.append((layer, input_index))
        input_index += 1
    if prepared_voiceovers is None:
        audio_inputs = _build_audio_inputs(request, plan, root, warnings)
    else:
        music_sfx_request = request.model_copy(deep=True)
        music_sfx_request.voiceover.mode = 'off'
        audio_inputs = _build_audio_inputs(music_sfx_request, plan, root, warnings) + prepared_voiceovers
    voice_inputs = [item for item in audio_inputs if item[3] == 'voiceover']
    if request.editor.enabled and voice_inputs:
        import wave
        voice_words = []
        for index, (path, at, _volume, _kind) in enumerate(voice_inputs):
            with wave.open(str(path), 'rb') as voice:
                voice_duration = voice.getnframes()/voice.getframerate()
            if at+voice_duration > duration+.05:
                raise ComposeError('Generated narration exceeds its timeline slot; shorten the script')
            if request.captions_enabled and plan.captions_enabled and request.editor.caption_style != 'off':
                voice_root = root/f'voice_words_{index}'
                voice_root.mkdir(exist_ok=True)
                transcribed_voice = transcribe_chunked(path,0,voice_duration,request.language,voice_root)
                if not transcribed_voice:
                    raise ComposeError('Generated narration could not be transcribed for required captions')
                for word in transcribed_voice:
                    voice_words.append(dict(word,start=word['start']+at,end=word['end']+at,
                                            caption_section=f'narration:{index}'))
        if voice_words:
            # Voiceover timestamps already use program time, unlike source ASR.
            caption_plan = plan.model_copy(update={'base_segments':[BaseSegment(source_start=0,source_end=duration)]})
            from ai_compose_motion import write_motion_ass, map_words
            body_words = map_words(words, segments) if edit_plan.policy.get('source_audio') not in {'mute','replace'} else []
            motion_counts = write_motion_ass(ass, request, caption_plan, edit_plan, sorted(body_words+voice_words, key=lambda w:w['start']))
    audio_indices: list[tuple[Path, float, float, str, int]] = []
    for path, at, volume, kind in audio_inputs:
        command += (["-stream_loop", "-1"] if kind == 'music' else []) + ["-i", str(path)]
        audio_indices.append((path, at, volume, kind, input_index))
        input_index += 1
    if not media.get("has_audio"):
        command += ["-f", "lavfi", "-t", f"{duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        silence_index = input_index
        input_index += 1
    else:
        silence_index = None

    filters: list[str] = []
    width, height, fps = request.output_width, request.output_height, request.fps
    base_video_labels: list[str] = []
    base_audio_labels: list[str] = []
    camera_tracks_rendered = 0
    for i, segment in enumerate(segments):
        d = segment.source_end - segment.source_start
        camera = camera_for_clip(edit_plan, i)
        from ai_compose_editorial import speaker_safe
        safe = speaker_safe(request) and segment.kind == 'source'
        pane_height = max(2, int(height * .48 / 2)*2) if safe else height
        if camera and camera.keyframes:
            visual_filter = _camera_visual_filter(camera, width, pane_height, fps)
            camera_tracks_rendered += 1
        else:
            visual_filter = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,fps={fps}"
        if safe:
            if not (camera and camera.keyframes):
                visual_filter = f"scale={width}:{pane_height}:force_original_aspect_ratio=decrease,pad={width}:{pane_height}:(ow-iw)/2:(oh-ih)/2:color=0x101b2c,fps={fps}"
            visual_filter += f",pad={width}:{height}:0:{int(height*.16)//2*2}:color=0x101b2c"
        filters.append(f"[{i}:v]{visual_filter},setsar=1,setpts=PTS-STARTPTS[vseg{i}]")
        base_video_labels.append(f"[vseg{i}]")
        if segment.kind == 'narration':
            filters.append(f"anullsrc=r=48000:cl=stereo,atrim=duration={d:.6f},asetpts=PTS-STARTPTS[aseg{i}]")
        elif media.get("has_audio"):
            filters.append(f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo,volume={segment.audio_gain:.4f},apad,atrim=duration={d:.6f},asetpts=PTS-STARTPTS[aseg{i}]")
        else:
            filters.append(f"[{silence_index}:a]atrim=duration={d:.3f},asetpts=PTS-STARTPTS[aseg{i}]")
        base_audio_labels.append(f"[aseg{i}]")
    transitions_rendered = 0
    transition_specs = list(edit_plan.transitions[: max(0, len(segments) - 1)])
    can_xfade = (
        len(segments) > 1
        and len(transition_specs) == len(segments) - 1
        and all(float(item.duration) > 0.0 and _transition_name(item.kind) for item in transition_specs)
    )
    if len(segments) == 1:
        video_label = base_video_labels[0]
        audio_label = base_audio_labels[0]
    elif can_xfade:
        # Freeze the outgoing boundary during the incoming clip's first frames.
        # Unlike overlapping source clips, this preserves every source/program
        # timestamp, captions and dialogue sync, without a frozen output tail.
        durations = [min(float(t.duration), float(segments[i].source_end-segments[i].source_start)-.05,
                         float(segments[i+1].source_end-segments[i+1].source_start)-.05)
                     for i,t in enumerate(transition_specs)]
        if all(d > 0 for d in durations):
            padded = []
            for i, label in enumerate(base_video_labels):
                if i < len(durations):
                    padded_label = f'[vpad{i}]'
                    filters.append(f'{label}tpad=stop_mode=clone:stop_duration={durations[i]:.3f}{padded_label}')
                    padded.append(padded_label)
                else:
                    padded.append(label)
            current_label = padded[0]
            elapsed = float(segments[0].source_end-segments[0].source_start)
            for index, transition in enumerate(transition_specs, 1):
                output_label = f'[vtrans{index}]'
                filters.append(f'{current_label}{padded[index]}xfade='
                    f'transition={_transition_name(transition.kind)}:duration={durations[index-1]:.3f}:'
                    f'offset={elapsed:.3f}{output_label}')
                current_label = output_label
                elapsed += float(segments[index].source_end-segments[index].source_start)
                transitions_rendered += 1
            video_label = current_label
        else:
            filters.append("".join(base_video_labels) + f"concat=n={len(segments)}:v=1:a=0[vbase]")
            video_label = "[vbase]"
        filters.append("".join(base_audio_labels) + f"concat=n={len(segments)}:v=0:a=1[abase]")
        audio_label = "[abase]"
    else:
        filters.append("".join(base_video_labels) + f"concat=n={len(segments)}:v=1:a=0[vbase]")
        filters.append("".join(base_audio_labels) + f"concat=n={len(segments)}:v=0:a=1[abase]")
        video_label = "[vbase]"
        audio_label = "[abase]"

    if edit_plan.policy.get('source_audio') in {'mute', 'replace'}:
        filters.append(f"{audio_label}volume=0[mutedbase]")
        audio_label = '[mutedbase]'

    for index, (layer, layer_input) in enumerate(layer_indices, 1):
        layer_duration = max(0.05, min(duration, layer.end) - max(0.0, layer.start))
        if layer_duration <= 0.05:
            continue
        if layer.height is not None:
            scale = f"scale={max(2, int(width * layer.width))}:{max(2, int(height * layer.height))}:force_original_aspect_ratio=decrease,pad={max(2, int(width * layer.width))}:{max(2, int(height * layer.height))}:(ow-iw)/2:(oh-ih)/2:color=black@0"
        else:
            scale = f"scale={max(2, int(width * layer.width))}:-2"
        filters.append(f"[{layer_input}:v]{scale},format=rgba,colorchannelmixer=aa={layer.opacity:.3f},trim=duration={layer_duration:.3f},setpts=PTS-STARTPTS[ov{index}]")
        target_x, target_y = int(width * layer.x), int(height * layer.y)
        start = max(0.0, layer.start)
        until = min(duration, start + layer.animation_seconds)
        x_expr, y_expr = str(target_x), str(target_y)
        if layer.animation == "slide_up":
            y_expr = f"if(lt(t,{until:.3f}),{target_y}+h*(1-(t-{start:.3f})/{max(0.05, layer.animation_seconds):.3f}),{target_y})"
        elif layer.animation == "slide_down":
            y_expr = f"if(lt(t,{until:.3f}),{target_y}-h*(1-(t-{start:.3f})/{max(0.05, layer.animation_seconds):.3f}),{target_y})"
        elif layer.animation == "slide_left":
            x_expr = f"if(lt(t,{until:.3f}),{target_x}+w*(1-(t-{start:.3f})/{max(0.05, layer.animation_seconds):.3f}),{target_x})"
        elif layer.animation == "slide_right":
            x_expr = f"if(lt(t,{until:.3f}),{target_x}-w*(1-(t-{start:.3f})/{max(0.05, layer.animation_seconds):.3f}),{target_x})"
        elif layer.animation == "float":
            y_expr = f"{target_y}+12*sin(2*PI*(t-{start:.3f})/2)"
        if layer.animation in {"fade", "reveal"}:
            filters[-1] = filters[-1].replace(f"[ov{index}]", f"[ov{index}_raw]")
            filters.append(f"[ov{index}_raw]fade=t=in:st=0:d={min(layer.animation_seconds, layer_duration):.3f}:alpha=1[ov{index}]")
        filters.append(f"[ov{index}]setpts=PTS+{start:.3f}/TB[ovtimed{index}]")
        filters.append(f"[{video_label[1:-1]}][ovtimed{index}]overlay=x='{x_expr}':y='{y_expr}':eof_action=pass:shortest=0:enable='between(t,{start:.3f},{min(duration, layer.end):.3f})'[vlay{index}]")
        video_label = f"[vlay{index}]"

    graphics_rendered = len(edit_plan.graphics)
    should_burn_ass = ass.exists() and (
        bool(motion_counts.get('progress_bars_rendered'))
        or bool(motion_counts.get('caption_events'))
        or (plan.captions_enabled and request.captions_enabled and bool(words))
        or bool(edit_plan.graphics)
        or (edit_plan is None and (bool(plan.hook_text) or bool(plan.outro_text)))
    )
    if should_burn_ass:
        ass_path = str(ass).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        filters.append(f"{video_label}ass='{ass_path}'[vcaption]")
        video_label = "[vcaption]"

    mix_labels = [audio_label]
    duck_music = request.editor.enabled and any(item[3] == 'music' for item in audio_indices)
    voice_sidechains = [f'[voiceside{i}]' for i,item in enumerate(audio_indices,1) if item[3] == 'voiceover']
    for index, (_path, at, volume, kind, audio_index) in enumerate(audio_indices, 1):
        label = f"[aud{index}]"
        delay = max(0, int(round(at * 1000)))
        if kind == "music":
            filters.append(f"[{audio_index}:a]aresample=48000,volume={volume:.3f},atrim=duration={duration:.3f},afade=t=out:st={max(0,duration-.8):.3f}:d=0.8,asetpts=PTS-STARTPTS{label}")
        else:
            filters.append(f"[{audio_index}:a]aresample=48000,volume={volume:.3f},adelay={delay}|{delay},atrim=duration={duration:.3f}{label}")
        if duck_music and kind == 'voiceover':
            filters.append(f'{label}asplit=2[voicemix{index}][voiceside{index}]')
            label = f'[voicemix{index}]'
        if duck_music and kind == 'music':
            filters.append(f"{audio_label}asplit=2[dialoguemix{index}][sidechain{index}]")
            mix_labels[0] = f'[dialoguemix{index}]'
            sidechain = f'[sidechain{index}]'
            if voice_sidechains:
                filters.append(sidechain+''.join(voice_sidechains)+
                    f'amix=inputs={len(voice_sidechains)+1}:normalize=0:duration=longest[allvoice{index}]')
                sidechain = f'[allvoice{index}]'
            filters.append(f"{label}{sidechain}sidechaincompress=threshold=.025:ratio=8:attack=20:release=250[duck{index}]")
            label = f'[duck{index}]'
        mix_labels.append(label)
    if len(mix_labels) == 1:
        filters.append(f"{audio_label}aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    else:
        filters.append("".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")

    codec, codec_flags = _choose_encoder()
    if codec == 'libx264':
        if request.editor.enabled:
            preset = os.getenv('AI_COMPOSE_X264_PRESET','veryfast')
            if preset not in {'ultrafast','superfast','veryfast','faster','fast','medium','slow'}:
                raise ComposeError('Invalid AI_COMPOSE_X264_PRESET')
            codec_flags = ['-preset',preset,'-threads','4']
        codec_flags += ['-crf',str(request.crf)]
    command += ["-filter_complex", ";".join(filters), "-map", video_label, "-map", "[aout]", "-c:v", codec, *codec_flags, "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", f"{request.audio_bitrate}k", "-ar", "48000", "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output)]
    _run(command, max(900, int(duration * 60) + 600), f"AI composition variant {variant}")
    if not output.exists() or output.stat().st_size == 0:
        raise ComposeError("FFmpeg produced no output")
    result = probe_media(output)
    return {
        "variant": variant,
        "local_path": str(output),
        "size_bytes": output.stat().st_size,
        "duration": round(result["duration"], 3),
        "width": result["width"],
        "height": result["height"],
        "codec": result["video_codec"],
        "render_features": {
            "graphics_rendered": graphics_rendered,
            "camera_tracks_rendered": camera_tracks_rendered,
            "visual_layers_rendered": len(layer_indices),
            "transitions_rendered": transitions_rendered,
            "captions_rendered": bool(motion_counts.get('caption_events')) if request.editor.enabled else bool(plan.captions_enabled and request.captions_enabled and words),
            "narration_sections_rendered": sum(s.kind == 'narration' for s in segments),
            "sfx_rendered": sum(kind == 'sfx' for _, _, _, kind in audio_inputs),
            "voiceovers_rendered": sum(kind == 'voiceover' for _, _, _, kind in audio_inputs),
            "music_rendered": sum(kind == 'music' for _, _, _, kind in audio_inputs),
            **motion_counts,
        },
    }


def _upload_output(path: Path, request_id: str, variant: int) -> dict[str, Any]:
    try:
        import media_router as media
        from b2_helper import get_b2_manager

        bucket = str(getattr(media, "BACKBLAZE_BUCKET_NAME", "") or "")
        if not bucket:
            return {}
        token = media._render_token_for_queue(f"ai-compose:{request_id}:{variant}")
        file_name = f"{media.B2_PREFIX}/{token}.mp4"
        manager = get_b2_manager()
        manager.upload_file(path, bucket, file_name, "video/mp4", 1)
        base = str(getattr(media, "PUBLIC_API_BASE", os.getenv("PUBLIC_API_BASE", "")).rstrip("/"))
        return {"token": token, "b2_file_name": file_name, "video_url": f"{base}/media/rendered/{token}" if base else None}
    except Exception as exc:
        logger.warning("B2 upload skipped: %s", exc)
        return {"upload_warning": str(exc)[:300]}


async def run_composition(request: AIComposeRequest, job: dict[str, Any], update: Callable[..., Any]) -> None:
    if request.editor.enabled:
        from ai_compose_editor import run_editor
        return await run_editor(request, job, update)
    work_root = Path(os.getenv("AI_COMPOSE_WORK_DIR", "media_jobs")) / str(job["job_id"])
    work_root.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    await update(stage="source", progress=0.04)
    source = await acquire_source(request, work_root)
    media = await asyncio.to_thread(probe_media, source)
    start = min(request.start_seconds, max(0.0, media["duration"] - 0.05))
    end = min(media["duration"], request.end_seconds or media["duration"])
    if end <= start:
        raise ComposeError("Selected source window has no usable duration")
    await update(stage="transcription", progress=0.12, source_duration=media["duration"], source_start=start, source_end=end)
    words: list[dict[str, Any]] = []
    try:
        words = await asyncio.to_thread(transcribe_chunked, source, start, end - start, request.language, work_root)
    except Exception as exc:
        warnings.append(f"Transcription unavailable; visual/timing fallback used: {str(exc)[:240]}")
    await update(stage="visual_analysis", progress=0.28, word_count=len(words))
    frames = await asyncio.to_thread(extract_frames, source, end - start, work_root, 48, start)
    for frame in frames:
        frame["time"] = round(float(frame["time"]) + start, 3)
    policy = resolve_editorial_policy(request, media, words)
    evidence_path = work_root / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "media": media,
                "source_window": {"start": start, "end": end},
                "word_count": len(words),
                "words": words,
                "representative_frames": frames,
                "editorial_profile": policy.model_dump(mode="json"),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    await update(
        stage="editorial_profile",
        progress=0.34,
        frame_count=len(frames),
        editorial_profile=policy.model_dump(mode="json"),
    )
    await update(stage="direction", progress=0.40, frame_count=len(frames))
    director = LocalDirector(request)
    plan, planner_warnings = await asyncio.to_thread(director.plan, media, words, frames)
    warnings.extend(planner_warnings)
    # Clamp model-produced ranges to the real source/output envelope.
    clamped_segments: list[BaseSegment] = []
    for item in plan.base_segments:
        if item.source_end <= start or item.source_start >= end:
            continue
        segment_start = max(start, min(end - 0.05, item.source_start))
        segment_end = min(end, max(segment_start + 0.05, item.source_end))
        if segment_end > segment_start:
            clamped_segments.append(BaseSegment(source_start=segment_start, source_end=segment_end, reason=item.reason, score=item.score))
    plan.base_segments = clamped_segments
    if not plan.base_segments:
        plan = fallback_plan(request, media, words)
    fitted_segments: list[BaseSegment] = []
    remaining = min(request.target_seconds, sum(item.source_end - item.source_start for item in plan.base_segments))
    for item in plan.base_segments:
        if remaining <= 0.05:
            break
        segment_end = min(item.source_end, item.source_start + remaining)
        if segment_end - item.source_start > 0.05:
            fitted_segments.append(item.model_copy(update={"source_end": segment_end}))
            remaining -= segment_end - item.source_start
    plan.base_segments = fitted_segments or plan.base_segments
    output_duration = sum(item.source_end - item.source_start for item in plan.base_segments)
    valid_assets: list[AssetRequest] = []
    for item in plan.asset_requests:
        if item.end <= 0 or item.start >= output_duration:
            continue
        asset_start = max(0.0, min(max(0.0, output_duration - 0.05), item.start))
        asset_end = min(output_duration, item.end)
        if asset_end - asset_start > 0.05:
            valid_assets.append(item.model_copy(update={"start": asset_start, "end": asset_end}))
    plan.asset_requests = valid_assets
    valid_graphics = []
    for item in plan.graphics:
        if item.end <= 0 or item.start >= output_duration:
            continue
        graphic_start = max(0.0, min(max(0.0, output_duration - 0.05), item.start))
        graphic_end = min(output_duration, item.end)
        if graphic_end - graphic_start > 0.05:
            valid_graphics.append(item.model_copy(update={"start": graphic_start, "end": graphic_end}))
    plan.graphics = valid_graphics
    if policy.motion_graphics == "off":
        plan.graphics = []
    if policy.supplementary_visuals == "off":
        if valid_assets:
            warnings.append("Supporting visual requests were removed by the effective editorial policy.")
        plan.asset_requests = []
    if policy.voiceover == "off":
        plan.voiceovers = []
    if policy.sfx == "off":
        plan.sfx_events = []
    await update(stage="asset_retrieval", progress=0.54, asset_queries=len(plan.asset_requests))
    layers, manifest, asset_warnings = await asyncio.to_thread(resolve_assets, request, plan, work_root)
    warnings.extend(asset_warnings)
    await update(stage="rendering", progress=0.68, resolved_assets=len(layers))
    outputs: list[dict[str, Any]] = []
    variant_plans: dict[str, dict[str, Any]] = {}
    variant_reviews: dict[str, dict[str, Any]] = {}
    for variant in range(1, request.output_count + 1):
        variant_edit_plan = compile_edit_plan(
            request,
            plan,
            layers,
            words,
            policy,
            media,
            variant=variant,
        )
        plan_review = inspect_edit_plan(
            variant_edit_plan,
            request,
            resolved_layer_count=len(layers),
        )
        if not plan_review.ok:
            raise ComposeError(
                f"Compiled edit plan failed validation for variant {variant}: "
                f"{plan_review.model_dump_json()}"
            )
        variant_plans[str(variant)] = variant_edit_plan.model_dump(mode="json")
        variant_reviews[str(variant)] = {"plan": plan_review.model_dump(mode="json")}
        variant_layers = layers
        if variant > 1:
            if variant == 2:
                variant_layers = [
                    layer.model_copy(update={"animation": "fade" if layer.animation.startswith("slide") else layer.animation})
                    for layer in layers
                ]
            elif variant >= 3:
                variant_layers = [
                    layer.model_copy(update={"animation": "float" if layer.animation == "none" else layer.animation})
                    for layer in layers
                ]
        rendered = await asyncio.to_thread(
            render_composition,
            source,
            media,
            request,
            plan,
            variant_layers,
            words,
            work_root,
            variant,
            warnings,
            variant_edit_plan,
        )
        render_review = await asyncio.to_thread(
            inspect_render,
            Path(rendered["local_path"]),
            variant_edit_plan.duration,
            request,
            probe_media,
            edit_plan=variant_edit_plan,
            render_features=rendered.get("render_features"),
        )
        rendered["quality_review"] = render_review.model_dump(mode="json")
        variant_reviews[str(variant)]["render"] = render_review.model_dump(mode="json")
        rendered.update(_upload_output(Path(rendered["local_path"]), str(job["request_id"]), variant))
        outputs.append(rendered)
        await update(stage="rendering", progress=0.68 + 0.22 * variant / max(1, request.output_count), outputs=outputs)
    if request.refinement_iterations > 0:
        from ai_compose_refinement import refine_composition

        plan, layers, manifest, outputs, refinement_warnings = await refine_composition(
            request=request,
            job=job,
            update=update,
            source=source,
            media=media,
            words=words,
            frames=frames,
            work_root=work_root,
            plan=plan,
            layers=layers,
            manifest=manifest,
            outputs=outputs,
            warnings=warnings,
        )
        warnings.extend(refinement_warnings)

    # Refinement may have replaced every rendered variant. Rebuild the receipt
    # metadata from the winning plans/outputs rather than retaining stale draft
    # plans from before the last revision round.
    variant_plans = {}
    variant_reviews = {}
    for variant in range(1, request.output_count + 1):
        variant_edit_plan = compile_edit_plan(
            request,
            plan,
            layers,
            words,
            policy,
            media,
            variant=variant,
        )
        plan_review = inspect_edit_plan(
            variant_edit_plan,
            request,
            resolved_layer_count=len(layers),
        )
        variant_plans[str(variant)] = variant_edit_plan.model_dump(mode="json")
        variant_reviews[str(variant)] = {"plan": plan_review.model_dump(mode="json")}
        if len(outputs) >= variant:
            quality_value = outputs[variant - 1].get("quality_review")
            if not quality_value:
                quality_value = (
                    await asyncio.to_thread(
                        inspect_render,
                        Path(outputs[variant - 1]["local_path"]),
                        variant_edit_plan.duration,
                        request,
                        probe_media,
                        edit_plan=variant_edit_plan,
                        render_features=outputs[variant - 1].get("render_features"),
                    )
                ).model_dump(mode="json")
                outputs[variant - 1]["quality_review"] = quality_value
            variant_reviews[str(variant)]["render"] = quality_value

    final_edit_plan = compile_edit_plan(request, plan, layers, words, policy, media, variant=1)
    final_plan_review = inspect_edit_plan(final_edit_plan, request, resolved_layer_count=len(layers))
    final_render_reviews = [item.get("quality_review") for item in outputs if item.get("quality_review")]
    quality_review = {
        "plan": final_plan_review.model_dump(mode="json"),
        "variants": variant_reviews,
        "final_render_reviews": final_render_reviews,
    }
    receipt = AIComposePlan(
        plan=plan,
        resolved_layers=layers,
        warnings=warnings,
        asset_manifest=manifest,
        editorial_profile=policy.model_dump(mode="json"),
        edit_plan=final_edit_plan.model_dump(mode="json"),
        quality_review=quality_review,
        artifacts={
            "evidence_path": str(evidence_path),
            "variant_edit_plans": variant_plans,
            "source_media": media,
        },
    )
    receipt_path = work_root / "composition_receipt.json"
    receipt_path.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    await update(stage="complete", progress=1.0, status="complete", outputs=outputs, plan=receipt.model_dump(mode="json"), receipt_path=str(receipt_path), completed_at=_now(), work_dir=str(work_root))
