"""Resolve a general editing policy from the public AI Compose request.

The local director should not have to infer every operational rule from one
large free-form prompt.  This module turns the request, source evidence, and
backward-compatible options into a small, explicit policy object that can be
used by planning, rendering, audio, and QA.
"""
from __future__ import annotations

import re
import os
from typing import Any

from pydantic import BaseModel, Field

from ai_compose_schema import AIComposeRequest


class ResolvedEditorialPolicy(BaseModel):
    """Effective policy used by the internal workflow.

    This is an internal contract, not a new public endpoint.  Every field is
    serializable so it can be persisted in the job receipt and reused by a
    later refinement round.
    """

    content_type: str = "general"
    editing_intensity: str = "clean"
    cut_policy: str = "editorial"
    source_audio: str = "preserve"
    voiceover: str = "off"
    supplementary_visuals: str = "off"
    motion_graphics: str = "selective"
    camera_motion: str = "static"
    music: str = "off"
    sfx: str = "off"
    style_profile: str = "clean_editorial"
    review_level: str = "visual"
    speech_present: bool = False
    speech_word_count: int = 0
    speech_seconds: float = 0.0
    constraints: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


def _text(request: AIComposeRequest, words: list[dict[str, Any]]) -> str:
    transcript = " ".join(str(item.get("word", "")) for item in words)
    return f"{request.brief} {transcript}".lower()


def _speech_stats(words: list[dict[str, Any]]) -> tuple[bool, int, float]:
    valid = [
        item for item in words
        if str(item.get("word", "")).strip()
        and float(item.get("end", 0.0) or 0.0) > float(item.get("start", 0.0) or 0.0)
    ]
    if not valid:
        return False, 0, 0.0
    start = min(float(item.get("start", 0.0) or 0.0) for item in valid)
    end = max(float(item.get("end", 0.0) or 0.0) for item in valid)
    return len(valid) >= 3, len(valid), max(0.0, end - start)


def _infer_content_type(request: AIComposeRequest, text: str) -> str:
    requested = request.editorial_policy.content_type
    if requested != "auto":
        return requested
    rules = (
        ("screen_demo", r"\b(screen|software|app|website|click|cursor|dashboard|code|terminal|ui)\b"),
        ("product", r"\b(product|buy|sale|price|feature|review|unbox|offer|brand)\b"),
        ("documentary", r"\b(history|documentary|archive|war|investigation|evidence|timeline)\b"),
        ("tutorial", r"\b(how to|tutorial|step|steps|learn|lesson|guide|explain)\b"),
        ("music", r"\b(song|music|lyrics|beat|concert|performance|dance)\b"),
        ("podcast", r"\b(podcast|episode|conversation|host|guest)\b"),
        ("interview", r"\b(interview|question|answer|speaker)\b"),
        ("talking_head", r"\b(talking head|speaker|person|presenter|monologue)\b"),
    )
    for name, pattern in rules:
        if re.search(pattern, text, flags=re.I):
            return name
    return "general"


def _infer_intensity(request: AIComposeRequest, content_type: str) -> str:
    requested = request.editorial_policy.editing_intensity
    if requested != "auto":
        return requested
    if request.preset == "viral_short":
        return "viral"
    if request.preset == "documentary":
        return "cinematic"
    if request.preset == "product_promo":
        return "dynamic"
    if content_type in {"podcast", "talking_head", "interview"}:
        return "clean"
    return "clean"


def resolve_editorial_policy(
    request: AIComposeRequest,
    media: dict[str, Any] | None = None,
    words: list[dict[str, Any]] | None = None,
) -> ResolvedEditorialPolicy:
    """Return one deterministic policy for all later workflow stages.

    Explicit legacy request options win over inferred defaults.  In
    particular, ``fetch_assets=False`` and ``sfx.mode='off'`` are hard stops;
    the director cannot override them.
    """

    words = words or []
    text = _text(request, words)
    speech_present, word_count, speech_seconds = _speech_stats(words)
    content_type = _infer_content_type(request, text)
    intensity = _infer_intensity(request, content_type)
    requested = request.editorial_policy
    rationale: list[str] = []
    constraints: list[str] = []

    if requested.cut_policy == "auto":
        cut_policy = "preserve_existing_silence_cuts" if speech_present else "editorial"
    else:
        cut_policy = requested.cut_policy

    if requested.source_audio == "auto":
        source_audio = "preserve" if speech_present or bool((media or {}).get("has_audio")) else "replace"
    else:
        source_audio = requested.source_audio

    if request.voiceover.mode != "off":
        voiceover = request.voiceover.mode
        rationale.append("The legacy voiceover option was explicitly enabled.")
    elif requested.voiceover != "auto":
        voiceover = requested.voiceover
    else:
        voiceover = "off"
        if speech_present:
            constraints.append("Do not place generated voiceover over detected source dialogue.")

    if not request.fetch_assets or request.max_asset_layers <= 0 or requested.supplementary_visuals == "off":
        supplementary_visuals = "off"
        if not request.fetch_assets:
            constraints.append("Asset retrieval is disabled by the request.")
    else:
        supplementary_visuals = requested.supplementary_visuals
        if supplementary_visuals == "auto":
            supplementary_visuals = "prefer" if content_type not in {"podcast", "talking_head"} else "auto"

    if requested.motion_graphics == "auto":
        motion_graphics = "hero" if intensity in {"viral", "dynamic", "cinematic"} else "selective"
    else:
        motion_graphics = requested.motion_graphics

    if requested.camera_motion == "auto":
        camera_motion = "dynamic" if intensity in {"viral", "dynamic", "cinematic"} else (
            "punch_ins" if speech_present else "static"
        )
    else:
        camera_motion = requested.camera_motion

    if request.sfx.mode != "adaptive":
        sfx = request.sfx.mode
    elif requested.sfx != "auto":
        sfx = requested.sfx
    else:
        sfx = "adaptive" if intensity in {"viral", "dynamic", "cinematic"} else "transitions_only"
    if sfx == "aggressive" and request.preset == "clean_explainer":
        sfx = "transitions_only"
        constraints.append("Aggressive SFX was downgraded for clean_explainer.")

    if requested.music == "auto":
        music = "library" if request.music_url or os.getenv("AI_COMPOSE_MUSIC_PATH", "").strip() else "off"
    else:
        music = requested.music

    style_profile = requested.style_profile
    if style_profile == "auto":
        style_profile = {
            "podcast": "talking_head_editorial",
            "talking_head": "talking_head_editorial",
            "interview": "interview_editorial",
            "tutorial": "instructional_clean",
            "screen_demo": "screen_demo_focus",
            "product": "product_dynamic",
            "documentary": "documentary_cinematic",
            "music": "rhythmic_motion",
        }.get(content_type, "clean_editorial")

    if speech_present:
        rationale.append("Word-timed speech was detected; source dialogue is treated as the primary narrative.")
    if supplementary_visuals in {"prefer", "require"}:
        rationale.append("The policy permits transcript-anchored supporting visuals when a candidate improves comprehension.")
    if camera_motion in {"punch_ins", "dynamic"}:
        rationale.append("Camera movement is enabled as motivated emphasis, not as a constant effect.")
    rationale.append(f"Content type resolved as {content_type}; intensity resolved as {intensity}.")

    return ResolvedEditorialPolicy(
        content_type=content_type,
        editing_intensity=intensity,
        cut_policy=cut_policy,
        source_audio=source_audio,
        voiceover=voiceover,
        supplementary_visuals=supplementary_visuals,
        motion_graphics=motion_graphics,
        camera_motion=camera_motion,
        music=music,
        sfx=sfx,
        style_profile=style_profile,
        review_level=requested.review_level,
        speech_present=speech_present,
        speech_word_count=word_count,
        speech_seconds=round(speech_seconds, 3),
        constraints=constraints,
        rationale=rationale,
    )


def policy_prompt_context(policy: ResolvedEditorialPolicy) -> str:
    """Compact instructions safe to include in a small-model prompt."""

    return (
        f"Effective editorial policy: content={policy.content_type}; "
        f"intensity={policy.editing_intensity}; cuts={policy.cut_policy}; "
        f"audio={policy.source_audio}; voiceover={policy.voiceover}; "
        f"supporting_visuals={policy.supplementary_visuals}; "
        f"graphics={policy.motion_graphics}; camera={policy.camera_motion}; "
        f"music={policy.music}; sfx={policy.sfx}; style={policy.style_profile}.\n"
        "Every non-empty asset request must explain its benefit and be anchored "
        "to a concrete spoken or visual idea. Empty assets are valid only when "
        "the policy or evidence makes them unnecessary.\n"
        + "\n".join(f"Constraint: {item}" for item in policy.constraints)
    )
