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
    ResolvedLayer,
    SfxEvent,
    VoiceoverPlan,
)

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
            _validate_public_url(current)
            async with client.stream("GET", current, headers={"User-Agent": "SH01-AI-Compose/1.0"}) as response:
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
        _validate_public_url(current)
        response = requests.get(
            current,
            headers={"User-Agent": "SH01-AI-Compose/1.0"},
            stream=True,
            allow_redirects=False,
            timeout=(20, 180),
        )
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
            [sys.executable, "-m", "media_transcribe", str(audio), str(output), language or ""],
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
    if request.fetch_assets and request.max_asset_layers and keywords:
        query = " ".join(keywords[:3])
        asset_requests.append(
            AssetRequest(query=query, start=min(2.5, max(0.0, output_duration - 1.0)), end=min(output_duration, max(4.5, output_duration * 0.30)), placement="picture_in_picture", reason="support_the_opening_topic")
        )
        if output_duration > 12 and len(keywords) > 3:
            asset_requests.append(
                AssetRequest(query=" ".join(keywords[3:6]), start=max(6.0, output_duration * 0.48), end=min(output_duration, max(9.0, output_duration * 0.68)), placement="bottom", x=0.0, y=0.53, width=1.0, height=0.47, reason="support_the_explanation")
            )

    hook = None
    if full_text:
        first = re.split(r"(?<=[.!?])\s+", full_text)[0].strip()
        hook = first[:150] if first else None
    sfx_events: list[SfxEvent] = []
    if request.sfx.mode != "off":
        for asset in asset_requests:
            sfx_events.append(SfxEvent(kind="whoosh", at=asset.start, volume=request.sfx.volume, reason="asset_entered"))
    return CompositionPlan(
        title=(request.brief[:120] or "AI composed short"),
        hook_text=hook if request.preset == "viral_short" else None,
        base_segments=segments,
        asset_requests=asset_requests,
        sfx_events=sfx_events,
        captions_enabled=request.captions_enabled,
        director_notes=["Fallback plan used because no local director endpoint was available.", f"Selected output duration: {output_duration:.2f}s"],
    )


class LocalDirector:
    def __init__(self, request: AIComposeRequest):
        self.request = request

    def _endpoint(self) -> str:
        return (self.request.llm_url or os.getenv("AI_COMPOSE_LLM_URL", "") or os.getenv("LLAMA_SERVER_URL", "")).strip().rstrip("/")

    def _model(self) -> str:
        return (self.request.llm_model or os.getenv("AI_COMPOSE_LLM_MODEL", "local-director")).strip()

    def _plan_transformers(self, prompt: str) -> CompositionPlan:
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
        messages = [
            {"role": "system", "content": "You are a precise video editor. Return only JSON matching the requested schema."},
            {"role": "user", "content": prompt},
        ]
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
                    max_new_tokens=max(512, int(os.getenv("AI_COMPOSE_HF_MAX_NEW_TOKENS", "1800"))),
                    do_sample=False,
                )
            else:
                input_ids = encoded.to(model.device)
                output = model.generate(
                    input_ids,
                    max_new_tokens=max(512, int(os.getenv("AI_COMPOSE_HF_MAX_NEW_TOKENS", "1800"))),
                    do_sample=False,
                )
            generated = output[0][input_ids.shape[-1]:]
            content = tokenizer.decode(generated, skip_special_tokens=True)
        else:
            encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            output = model.generate(**encoded, max_new_tokens=1800, do_sample=False)
            content = tokenizer.decode(output[0][encoded["input_ids"].shape[-1]:], skip_special_tokens=True)
        try:
            return CompositionPlan.model_validate(self._parse_json(content))
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def _parse_json(content: str) -> dict[str, Any]:
        value = content.strip()
        if value.startswith("```"):
            value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.I)
            value = re.sub(r"\s*```$", "", value)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", value, flags=re.S)
            if not match:
                raise ComposeError("Local director did not return JSON")
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ComposeError("Local director returned a non-object plan")
        return parsed

    def _prompt(self, request: AIComposeRequest, media: dict[str, Any], words: list[dict[str, Any]], frames: list[dict[str, Any]]) -> str:
        schema = CompositionPlan.model_json_schema()
        scene_summary = [{"time": frame["time"], "frame_available": True} for frame in frames[:48]]
        return (
            "You are a short-form video editor. Return ONLY JSON matching the supplied schema. "
            "You are planning, not executing shell commands. Never invent source timestamps outside the media. "
            "base_segments use absolute source seconds. asset_requests use output-timeline seconds. "
            "Use supplementary assets only when they clarify the spoken idea. Do not invent factual claims. "
            "Use adaptive SFX sparingly: only for meaningful visual entrances, transitions, or payoffs. "
            "If a voiceover request is present, emit voiceovers with exact output-timeline start times; otherwise leave voiceovers empty. "
            f"Brief: {request.brief}\nPreset: {request.preset}\nTarget output seconds: {request.target_seconds}\n"
            f"Voiceover mode: {request.voiceover.mode}\nSFX mode: {request.sfx.mode}\n"
            f"Media metadata: {json.dumps(media, separators=(',', ':'))}\n"
            f"Transcript: {transcript_text(words, 18000)}\n"
            f"Representative frames: {json.dumps(scene_summary, separators=(',', ':'))}\n"
            f"JSON schema: {json.dumps(schema, separators=(',', ':'))}"
        )

    def plan(self, media: dict[str, Any], words: list[dict[str, Any]], frames: list[dict[str, Any]]) -> tuple[CompositionPlan, list[str]]:
        endpoint = self._endpoint()
        warnings: list[str] = []
        hf_model = os.getenv("AI_COMPOSE_HF_MODEL", "").strip()
        if hf_model:
            try:
                prompt = self._prompt(self.request, media, words, frames)
                return self._plan_transformers(prompt), warnings
            except Exception as exc:
                if self.request.llm_required:
                    raise ComposeError(f"Local Transformers director failed: {exc}") from exc
                warnings.append(f"Local Transformers director failed; fallback planning was used: {str(exc)[:240]}")
        if not endpoint:
            if self.request.llm_required:
                raise ComposeError("AI_COMPOSE_LLM_URL is required when llm_required=true")
            warnings.append("No local director endpoint configured; deterministic fallback planning was used.")
            return fallback_plan(self.request, media, words), warnings
        prompt = self._prompt(self.request, media, words, frames)
        try:
            import requests

            if os.getenv("AI_COMPOSE_LLM_BACKEND", "openai").strip().lower() == "ollama" or endpoint.endswith("/api/chat"):
                response = requests.post(
                    endpoint if endpoint.endswith("/api/chat") else f"{endpoint}/api/chat",
                    json={"model": self._model(), "stream": False, "format": CompositionPlan.model_json_schema(), "messages": [{"role": "user", "content": prompt}]},
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
                        "messages": [{"role": "system", "content": "Produce only valid JSON."}, {"role": "user", "content": prompt}],
                        "response_format": {"type": "json_schema", "json_schema": {"name": "composition_plan", "schema": CompositionPlan.model_json_schema(), "strict": True}},
                    },
                    timeout=(30, 600),
                )
                response.raise_for_status()
                content = response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            plan = CompositionPlan.model_validate(self._parse_json(str(content)))
            return plan, warnings
        except Exception as exc:
            if self.request.llm_required:
                raise ComposeError(f"Local director failed: {exc}") from exc
            warnings.append(f"Local director failed; deterministic fallback planning was used: {str(exc)[:240]}")
            return fallback_plan(self.request, media, words), warnings


def _candidate_tokens(candidate: dict[str, Any]) -> set[str]:
    text = " ".join(str(candidate.get(key, "")) for key in ("title", "description", "tags", "filename", "query"))
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


def _search_remote(query: str, kind: str, request: AIComposeRequest) -> list[dict[str, Any]]:
    try:
        import requests
    except ImportError as exc:
        raise ComposeError("requests is required for remote asset search") from exc
    cfg = request.asset_sources
    results: list[dict[str, Any]] = []
    limit = cfg.max_candidates_per_query
    if cfg.pexels and os.getenv("PEXELS_API_KEY", "").strip():
        try:
            endpoint = "https://api.pexels.com/v1/videos/search" if kind == "video" else "https://api.pexels.com/v1/search"
            response = requests.get(endpoint, headers={"Authorization": os.getenv("PEXELS_API_KEY", "").strip()}, params={"query": query, "per_page": limit, "orientation": "portrait"}, timeout=(20, 60))
            response.raise_for_status()
            payload = response.json()
            for item in payload.get("videos" if kind == "video" else "photos", []):
                if kind == "video":
                    files = sorted(item.get("video_files", []), key=lambda value: (value.get("width", 0) < 1080, -value.get("width", 0)))
                    download_url = files[0].get("link") if files else None
                    title = f"Pexels video {item.get('id')}"
                    attribution = f"Video by {item.get('user', {}).get('name', 'unknown')} on Pexels"
                else:
                    download_url = item.get("src", {}).get("portrait") or item.get("src", {}).get("large2x")
                    title = item.get("alt") or f"Pexels photo {item.get('id')}"
                    attribution = f"Photo by {item.get('photographer', 'unknown')} on Pexels"
                if download_url:
                    results.append({"asset_id": f"pexels_{item.get('id')}", "source": "pexels", "download_url": download_url, "title": title, "description": item.get("alt", ""), "tags": query, "license": "Pexels", "attribution": attribution, "kind": kind, "score": _score_candidate(query, {"title": title, "description": item.get("alt", ""), "tags": query})})
        except Exception as exc:
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
            logger.warning("Pixabay search failed: %s", exc)
    if cfg.unsplash and kind == "image" and os.getenv("UNSPLASH_ACCESS_KEY", "").strip():
        try:
            response = requests.get("https://api.unsplash.com/search/photos", headers={"Authorization": f"Client-ID {os.getenv('UNSPLASH_ACCESS_KEY', '').strip()}"}, params={"query": query, "per_page": limit, "orientation": "portrait"}, timeout=(20, 60))
            response.raise_for_status()
            for item in response.json().get("results", []):
                image_url = item.get("urls", {}).get("regular")
                if image_url:
                    photographer = item.get("user", {}).get("name", "unknown")
                    results.append({"asset_id": f"unsplash_{item.get('id')}", "source": "unsplash", "download_url": image_url, "title": item.get("alt_description") or item.get("description") or "Unsplash photo", "description": item.get("alt_description") or "", "tags": query, "license": "Unsplash", "attribution": f"Photo by {photographer} on Unsplash", "kind": kind, "score": _score_candidate(query, {"title": item.get("alt_description", ""), "description": item.get("description", ""), "tags": query})})
        except Exception as exc:
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


def resolve_assets(request: AIComposeRequest, plan: CompositionPlan, root: Path) -> tuple[list[ResolvedLayer], list[dict[str, Any]], list[str]]:
    asset_root = root / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    library_setting = request.local_asset_library or os.getenv("AI_COMPOSE_ASSET_LIBRARY", "").strip()
    used: set[str] = set()
    layers: list[ResolvedLayer] = []
    manifest: list[dict[str, Any]] = []
    warnings: list[str] = []
    for index, asset_request in enumerate(plan.asset_requests[: request.max_asset_layers]):
        candidates: list[dict[str, Any]] = []
        if request.asset_sources.local_library and library_setting:
            candidates.extend(_search_local(Path(library_setting), asset_request.query, asset_request.kind))
        if request.fetch_assets:
            candidates.extend(_search_remote(asset_request.query, asset_request.kind, request))
        candidates.sort(key=lambda item: (item.get("asset_id") in used, -float(item.get("score", 0.0))))
        candidate = next((item for item in candidates if item.get("asset_id") not in used), None)
        if not candidate:
            warnings.append(f"No asset found for query: {asset_request.query}")
            continue
        used.add(str(candidate.get("asset_id")))
        try:
            local_path = Path(str(candidate.get("local_path"))) if candidate.get("local_path") else asset_root / f"asset_{index:02d}{_safe_filename(str(candidate.get('download_url') or ''), '.mp4' if asset_request.kind == 'video' else '.jpg')}"
            if candidate.get("download_url"):
                awaitable = _download_sync
                awaitable(str(candidate["download_url"]), local_path)
            elif not local_path.exists():
                raise ComposeError("Local asset disappeared before rendering")
            if not local_path.exists() or local_path.stat().st_size == 0:
                raise ComposeError("Downloaded asset is empty")
            x, y, width, height = _placement(asset_request)
            layer = ResolvedLayer(
                asset_id=str(candidate["asset_id"]), local_path=str(local_path), source=str(candidate.get("source", "unknown")), license=candidate.get("license"), attribution=candidate.get("attribution"), query=asset_request.query, kind=asset_request.kind, start=asset_request.start, end=asset_request.end, x=x, y=y, width=width, height=height, animation=asset_request.animation, animation_seconds=asset_request.animation_seconds, opacity=asset_request.opacity, score=float(candidate.get("score", 0.0)), reason=asset_request.reason,
            )
            layers.append(layer)
            manifest.append({**candidate, "local_path": str(local_path), "sha256": _sha256(local_path), "timeline": {"start": asset_request.start, "end": asset_request.end}, "geometry": {"x": x, "y": y, "width": width, "height": height}, "animation": asset_request.animation})
        except Exception as exc:
            warnings.append(f"Asset {asset_request.query!r} was rejected: {str(exc)[:220]}")
    return layers, manifest, warnings


def _ass_time(seconds: float) -> str:
    total = max(0, int(round(seconds * 100)))
    hours, rem = divmod(total, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, centis = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _ass_escape(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}").replace("\n", " ")


def _make_ass(words: list[dict[str, Any]], segments: list[BaseSegment], duration: float, plan: CompositionPlan, destination: Path) -> None:
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
        "Style: Outro,DejaVu Sans,54,&H00FFFFFF,&H00FFFFFF,&H00101010,&H90000000,1,0,0,0,100,100,0,0,1,4,1,2,80,80,180,1", "",
        "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for start, end, text in lines:
        if end > start:
            content.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(min(duration, end))},Default,,0,0,0,,{_ass_escape(text)}")
    if plan.hook_text:
        content.append(f"Dialogue: 5,0:00:00.00,{_ass_time(min(duration, 1.7))},Hook,,0,0,0,,{{\\fad(80,120)}}{_ass_escape(plan.hook_text)}")
    if plan.outro_text and duration > 1.0:
        start = max(0.0, duration - 2.4)
        content.append(f"Dialogue: 5,{_ass_time(start)},{_ass_time(duration)},Outro,,0,0,0,,{{\\fad(120,120)}}{_ass_escape(plan.outro_text)}")
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
    return next((path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS), None)


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
        if path:
            inputs.append((path, event.at, event.volume, "sfx"))
        else:
            warnings.append(f"No local SFX matched {event.kind}; event skipped")
    music = os.getenv("AI_COMPOSE_MUSIC_PATH", "").strip()
    if request.music_url:
        path = root / ("music" + _safe_filename(request.music_url, ".mp3"))
        try:
            _download_sync(request.music_url, path)
            inputs.append((path, 0.0, request.music_volume, "music"))
        except Exception as exc:
            warnings.append(f"Music download failed: {str(exc)[:220]}")
    elif music and Path(music).exists():
        inputs.append((Path(music), 0.0, request.music_volume, "music"))
    inputs.extend(_synthesize_voiceovers(request, plan, root, warnings))
    return inputs


def render_composition(source: Path, media: dict[str, Any], request: AIComposeRequest, plan: CompositionPlan, layers: list[ResolvedLayer], words: list[dict[str, Any]], root: Path, variant: int, warnings: list[str]) -> dict[str, Any]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ComposeError("ffmpeg is required")
    segments = plan.base_segments
    duration = sum(max(0.05, segment.source_end - segment.source_start) for segment in segments)
    if duration <= 0.1:
        raise ComposeError("Composition plan contains no usable base duration")
    output = root / f"variant_{variant:02d}.mp4"
    ass = root / f"variant_{variant:02d}.ass"
    _make_ass(words, segments, duration, plan, ass)

    command: list[str] = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    video_input_count = len(segments)
    input_index = 0
    for segment in segments:
        command += ["-ss", f"{segment.source_start:.3f}", "-t", f"{segment.source_end - segment.source_start:.3f}", "-i", str(source)]
        input_index += 1
    layer_indices: list[tuple[ResolvedLayer, int]] = []
    for layer in layers:
        path = Path(layer.local_path)
        command += (["-stream_loop", "-1", "-i", str(path)] if _is_video(path) else ["-loop", "1", "-i", str(path)])
        layer_indices.append((layer, input_index))
        input_index += 1
    audio_inputs = _build_audio_inputs(request, plan, root, warnings)
    audio_indices: list[tuple[Path, float, float, str, int]] = []
    for path, at, volume, kind in audio_inputs:
        command += ["-i", str(path)]
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
    for i, segment in enumerate(segments):
        d = segment.source_end - segment.source_start
        filters.append(f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps},setpts=PTS-STARTPTS[vseg{i}]")
        base_video_labels.append(f"[vseg{i}]")
        if media.get("has_audio"):
            filters.append(f"[{i}:a]aresample=48000,atrim=duration={d:.3f},asetpts=PTS-STARTPTS[aseg{i}]")
        else:
            filters.append(f"[{silence_index}:a]atrim=duration={d:.3f},asetpts=PTS-STARTPTS[aseg{i}]")
        base_audio_labels.append(f"[aseg{i}]")
    if len(segments) == 1:
        video_label = base_video_labels[0]
        audio_label = base_audio_labels[0]
    else:
        filters.append("".join(base_video_labels) + f"concat=n={len(segments)}:v=1:a=0[vbase]")
        filters.append("".join(base_audio_labels) + f"concat=n={len(segments)}:v=0:a=1[abase]")
        video_label = "[vbase]"
        audio_label = "[abase]"

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
        filters.append(f"[{video_label[1:-1]}][ov{index}]overlay=x='{x_expr}':y='{y_expr}':eof_action=pass:shortest=0:enable='between(t,{start:.3f},{min(duration, layer.end):.3f})'[vlay{index}]")
        video_label = f"[vlay{index}]"

    if plan.captions_enabled and request.captions_enabled and ass.exists() and (words or plan.hook_text or plan.outro_text):
        ass_path = str(ass).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
        filters.append(f"{video_label}ass='{ass_path}'[vcaption]")
        video_label = "[vcaption]"

    mix_labels = [audio_label]
    for index, (_path, at, volume, kind, audio_index) in enumerate(audio_indices, 1):
        label = f"[aud{index}]"
        delay = max(0, int(round(at * 1000)))
        if kind == "music":
            filters.append(f"[{audio_index}:a]aresample=48000,volume={volume:.3f},atrim=duration={duration:.3f},asetpts=PTS-STARTPTS{label}")
        else:
            filters.append(f"[{audio_index}:a]aresample=48000,volume={volume:.3f},adelay={delay}|{delay},atrim=duration={duration:.3f}{label}")
        mix_labels.append(label)
    if len(mix_labels) == 1:
        filters.append(f"{audio_label}aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")
    else:
        filters.append("".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=first:dropout_transition=0,aresample=48000,loudnorm=I=-14:TP=-1.5:LRA=11[aout]")

    codec, codec_flags = _choose_encoder()
    command += ["-filter_complex", ";".join(filters), "-map", video_label, "-map", "[aout]", "-c:v", codec, *codec_flags, "-pix_fmt", "yuv420p", "-r", str(fps), "-c:a", "aac", "-b:a", f"{request.audio_bitrate}k", "-ar", "48000", "-t", f"{duration:.3f}", "-movflags", "+faststart", str(output)]
    _run(command, max(900, int(duration * 60) + 600), f"AI composition variant {variant}")
    if not output.exists() or output.stat().st_size == 0:
        raise ComposeError("FFmpeg produced no output")
    result = probe_media(output)
    return {"variant": variant, "local_path": str(output), "size_bytes": output.stat().st_size, "duration": round(result["duration"], 3), "width": result["width"], "height": result["height"], "codec": result["video_codec"]}


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
    await update(stage="asset_retrieval", progress=0.54, asset_queries=len(plan.asset_requests))
    layers, manifest, asset_warnings = await asyncio.to_thread(resolve_assets, request, plan, work_root)
    warnings.extend(asset_warnings)
    await update(stage="rendering", progress=0.68, resolved_assets=len(layers))
    outputs: list[dict[str, Any]] = []
    for variant in range(1, request.output_count + 1):
        # Small deterministic variation for multiple outputs: alternate one animation.
        variant_layers = layers
        if variant > 1:
            variant_layers = [layer.model_copy(update={"animation": "fade" if layer.animation.startswith("slide") else layer.animation}) for layer in layers]
        rendered = await asyncio.to_thread(render_composition, source, media, request, plan, variant_layers, words, work_root, variant, warnings)
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

    receipt = AIComposePlan(plan=plan, resolved_layers=layers, warnings=warnings, asset_manifest=manifest)
    receipt_path = work_root / "composition_receipt.json"
    receipt_path.write_text(json.dumps(receipt.model_dump(mode="json"), indent=2, ensure_ascii=False), encoding="utf-8")
    await update(stage="complete", progress=1.0, status="complete", outputs=outputs, plan=receipt.model_dump(mode="json"), receipt_path=str(receipt_path), completed_at=_now(), work_dir=str(work_root))


