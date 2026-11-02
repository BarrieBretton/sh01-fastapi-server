"""Versioned, renderer-neutral edit program for AI Compose.

The legacy ``CompositionPlan`` remains the model-facing contract.  This
module compiles it into an explicit program timeline so renderers do not have
to guess how source time, program time, layers, camera motion, and audio cues
relate to one another.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ai_compose_policy import ResolvedEditorialPolicy
from ai_compose_schema import AIComposeRequest, CompositionPlan, ResolvedLayer


class SourceClip(BaseModel):
    kind: Literal["source", "narration"] = "source"
    audio_gain: float = Field(default=1.0, ge=0.0, le=2.0)
    source_start: float = Field(ge=0.0)
    source_end: float = Field(gt=0.0)
    program_start: float = Field(ge=0.0)
    program_end: float = Field(gt=0.0)
    speed: float = Field(default=1.0, gt=0.1, le=4.0)
    reason: str = Field(default="selected_by_director", max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_windows(self) -> "SourceClip":
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        if self.program_end <= self.program_start:
            raise ValueError("program_end must be greater than program_start")
        return self


class CameraKeyframe(BaseModel):
    time: float = Field(ge=0.0)
    scale: float = Field(default=1.0, ge=1.0, le=1.35)
    x: float = Field(default=0.0, ge=-0.25, le=0.25)
    y: float = Field(default=0.0, ge=-0.25, le=0.25)


class CameraTrack(BaseModel):
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    keyframes: list[CameraKeyframe] = Field(min_length=1, max_length=8)
    motivation: str = Field(default="motivated_emphasis", max_length=300)
    subject: str | None = Field(default=None, max_length=120)

    @model_validator(mode="after")
    def validate_range(self) -> "CameraTrack":
        if self.end <= self.start:
            raise ValueError("camera track end must be greater than start")
        return self


class VisualLayerSpec(BaseModel):
    asset_id: str
    program_start: float = Field(ge=0.0)
    program_end: float = Field(gt=0.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float | None = Field(default=None, gt=0.0, le=1.0)
    z_index: int = 20
    animation: str = "none"
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    reason: str = Field(default="semantic_support", max_length=500)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)


class GraphicCue(BaseModel):
    kind: Literal["hook", "outro", "lower_third", "quote_card", "callout", "cta", "checklist", "stat", "timeline", "badge"]
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str = Field(min_length=1, max_length=500)
    placement: Literal["top", "upper_left", "center", "lower_left", "bottom"] = "center"
    animation: Literal["none", "fade", "slide_up", "slide_down", "pop", "reveal"] = "fade"
    style: Literal["default", "accent", "warning", "quote", "cta"] = "default"
    reason: str = Field(default="director_decision", max_length=300)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_range(self) -> "GraphicCue":
        if self.end <= self.start:
            raise ValueError("graphic cue end must be greater than start")
        return self


class AudioCue(BaseModel):
    kind: Literal["source_dialogue", "music", "voiceover", "sfx"]
    at: float = Field(ge=0.0)
    duration: float | None = Field(default=None, gt=0.0)
    gain: float = Field(default=1.0, ge=0.0, le=2.0)
    label: str | None = Field(default=None, max_length=200)
    reason: str = Field(default="audio_policy", max_length=300)


class TransitionSpec(BaseModel):
    at: float = Field(ge=0.0)
    kind: Literal["cut", "dissolve", "swipe", "match"] = "cut"
    duration: float = Field(default=0.0, ge=0.0, le=1.5)
    reason: str = Field(default="source_segment_boundary", max_length=300)


class EditPlan(BaseModel):
    """Canonical internal edit program consumed by renderers and QA."""

    schema_version: Literal["sh01/ai-edit@2"] = "sh01/ai-edit@2"
    duration: float = Field(gt=0.0)
    source_clips: list[SourceClip] = Field(min_length=1, max_length=64)
    camera_tracks: list[CameraTrack] = Field(default_factory=list, max_length=64)
    visual_layers: list[VisualLayerSpec] = Field(default_factory=list, max_length=64)
    graphics: list[GraphicCue] = Field(default_factory=list, max_length=128)
    audio_cues: list[AudioCue] = Field(default_factory=list, max_length=96)
    transitions: list[TransitionSpec] = Field(default_factory=list, max_length=64)
    captions_enabled: bool = True
    style_profile: str = "clean_editorial"
    variant: int = Field(default=1, ge=1, le=12)
    variant_strategy: str = "clean_editorial"
    policy: dict[str, Any] = Field(default_factory=dict)
    decision_log: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


def variant_profile(
    request: AIComposeRequest,
    policy: ResolvedEditorialPolicy,
    variant: int,
) -> tuple[str, str]:
    """Return a deterministic semantic strategy and camera mode per variant."""

    if variant <= 1:
        return "clean_editorial", policy.camera_motion
    if variant == 2:
        return "dynamic_punch_in", "dynamic"
    if variant == 3:
        return "supporting_visual_forward", policy.camera_motion
    return f"variant_{variant}", policy.camera_motion


def _camera_track(
    start: float,
    end: float,
    mode: str,
    index: int,
    variant: int,
    reason: str,
) -> CameraTrack | None:
    if mode == "static" or end - start < 2.2:
        return None
    if mode == "punch_ins":
        peak = 1.06 + (0.01 if variant > 1 else 0.0)
    else:
        peak = 1.10 + (0.015 if variant > 1 else 0.0)
    peak = min(1.28, peak)
    midpoint = start + (end - start) * 0.52
    direction = -0.018 if index % 2 else 0.018
    return CameraTrack(
        start=start,
        end=end,
        keyframes=[
            CameraKeyframe(time=start, scale=1.0, x=0.0, y=0.0),
            CameraKeyframe(time=midpoint, scale=peak, x=direction, y=0.0),
            CameraKeyframe(time=end, scale=1.02, x=direction / 2, y=0.0),
        ],
        motivation=reason,
    )


def compile_edit_plan(
    request: AIComposeRequest,
    plan: CompositionPlan,
    layers: list[ResolvedLayer],
    words: list[dict[str, Any]],
    policy: ResolvedEditorialPolicy,
    media: dict[str, Any],
    *,
    variant: int = 1,
) -> EditPlan:
    """Compile the legacy plan into a complete, explicit program timeline."""

    strategy, camera_mode = variant_profile(request, policy, variant)
    source_clips: list[SourceClip] = []
    camera_tracks: list[CameraTrack] = []
    transitions: list[TransitionSpec] = []
    cursor = 0.0
    for index, segment in enumerate(plan.base_segments):
        duration = max(0.05, segment.source_end - segment.source_start)
        clip = SourceClip(
            kind=segment.kind,
            audio_gain=segment.audio_gain,
            source_start=segment.source_start,
            source_end=segment.source_end,
            program_start=cursor,
            program_end=cursor + duration,
            reason=segment.reason,
            evidence_refs=[f"source_segment_{index}"],
        )
        source_clips.append(clip)
        track = None if segment.kind == 'narration' else _camera_track(cursor, cursor + duration, camera_mode, index, variant, segment.reason)
        if track:
            if request.editor.enabled:
                anchors = [w for w in words if segment.source_start+.4 < w['end'] < segment.source_end-.4
                           and str(w.get('word','')).rstrip().endswith(('.', '!', '?', ':'))]
                if anchors:
                    chosen = anchors[min(variant-1, len(anchors)-1)]
                    track.keyframes[1].time = cursor + chosen['end']-segment.source_start
                    track.motivation = 'phrase_boundary:' + str(chosen['word'])[:100]
            camera_tracks.append(track)
        if index:
            chosen = request.editor.transition if request.editor.enabled else 'auto'
            if chosen == 'auto':
                chosen = plan.transition_style or ('dissolve' if policy.editing_intensity == 'cinematic' else 'cut')
            if request.editor.enabled and 'transitions' in request.editor.required_features and chosen == 'cut':
                chosen = 'dissolve'
            transitions.append(
                TransitionSpec(
                    at=cursor,
                    kind=chosen,
                    duration=min(.18, duration/3, (plan.base_segments[index-1].source_end-plan.base_segments[index-1].source_start)/3) if chosen != 'cut' else 0.0,
                    reason="source_segment_boundary",
                )
            )
        cursor += duration

    visual_layers = [
        VisualLayerSpec(
            asset_id=layer.asset_id,
            program_start=layer.start,
            program_end=layer.end,
            x=layer.x,
            y=layer.y,
            width=layer.width,
            height=layer.height,
            z_index=20 + index,
            animation=layer.animation,
            opacity=layer.opacity,
            reason=layer.reason,
        )
        for index, layer in enumerate(layers)
    ]

    graphics: list[GraphicCue] = []
    if policy.motion_graphics != "off":
        # New directors can provide a complete, explicit graphics track. Keep
        # the legacy hook/outro fields as a backwards-compatible fallback, but
        # do not duplicate a matching explicit cue.
        explicit_kinds = {item.kind for item in plan.graphics}
        for item in plan.graphics:
            start = max(0.0, min(cursor - 0.05, item.start))
            end = min(cursor, max(start + 0.05, item.end))
            if end <= start:
                continue
            graphics.append(
                GraphicCue(
                    kind=item.kind,
                    start=round(start, 3),
                    end=round(end, 3),
                    text=item.text,
                    placement=item.placement,
                    animation=item.animation,
                    style=item.style,
                    reason=item.reason,
                    evidence_refs=item.evidence_refs,
                )
            )
        if plan.hook_text and "hook" not in explicit_kinds:
            graphics.append(
                GraphicCue(
                    kind="hook",
                    start=0.0,
                    end=min(cursor, 1.7),
                    text=plan.hook_text,
                    placement="top",
                    animation="slide_down",
                    style="accent",
                    reason="opening_hook",
                )
            )
        if plan.outro_text and "outro" not in explicit_kinds and cursor > 1.0:
            graphics.append(
                GraphicCue(
                    kind="outro",
                    start=max(0.0, cursor - 2.4),
                    end=cursor,
                    text=plan.outro_text,
                    placement="bottom",
                    animation="fade",
                    style="cta",
                    reason="closing_cta",
                )
            )

    audio_cues: list[AudioCue] = []
    if media.get("has_audio") and policy.source_audio not in {"mute", "replace"}:
        for clip in source_clips:
            if clip.kind == 'source' and clip.audio_gain > 0:
                audio_cues.append(AudioCue(kind="source_dialogue", at=clip.program_start,
                    duration=clip.program_end-clip.program_start, gain=clip.audio_gain, reason="preserve_source_audio"))
    for event in plan.sfx_events:
        audio_cues.append(AudioCue(kind="sfx", at=event.at, gain=event.volume, label=event.kind, reason=event.reason))
    for voiceover in plan.voiceovers:
        audio_cues.append(AudioCue(kind="voiceover", at=voiceover.start, gain=voiceover.volume, label=voiceover.text[:180], reason="director_voiceover"))
    if policy.music != "off":
        audio_cues.append(AudioCue(kind="music", at=0.0, duration=cursor, gain=request.music_volume, reason="music_policy"))

    decisions = [
        {"decision": "source_audio", "value": policy.source_audio, "reason": "resolved editorial policy"},
        {"decision": "camera_motion", "value": camera_mode, "reason": "variant and content policy"},
        {"decision": "supporting_visuals", "value": policy.supplementary_visuals, "reason": "asset policy and request flags"},
        {"decision": "variant_strategy", "value": strategy, "reason": "deterministic variant profile"},
    ]

    return EditPlan(
        duration=round(cursor, 3),
        source_clips=source_clips,
        camera_tracks=camera_tracks,
        visual_layers=visual_layers,
        graphics=graphics,
        audio_cues=audio_cues,
        transitions=transitions,
        captions_enabled=bool(plan.captions_enabled and request.captions_enabled),
        style_profile=policy.style_profile,
        variant=variant,
        variant_strategy=strategy,
        policy=policy.model_dump(mode="json"),
        decision_log=decisions,
    )


def camera_for_clip(edit_plan: EditPlan, clip_index: int) -> CameraTrack | None:
    """Find a camera track corresponding to a source clip by program window."""

    if clip_index >= len(edit_plan.source_clips):
        return None
    clip = edit_plan.source_clips[clip_index]
    return next(
        (
            track for track in edit_plan.camera_tracks
            if abs(track.start - clip.program_start) < 0.01
            and abs(track.end - clip.program_end) < 0.01
        ),
        None,
    )
