"""Typed contracts for the local AI video-composition workflow.

The model is allowed to propose these objects, but the renderer only accepts
validated instances.  The model never emits an FFmpeg command line.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AssetSourceConfig(BaseModel):
    local_library: bool = True
    pexels: bool = True
    pixabay: bool = True
    unsplash: bool = False
    freesound: bool = False
    wikimedia: bool = False  # public metadata API; no provider key
    max_candidates_per_query: int = Field(default=8, ge=1, le=20)


class VoiceoverOptions(BaseModel):
    mode: Literal["off", "hook_only", "hook_and_outro", "full"] = "off"
    text: str | None = Field(default=None, max_length=2400)
    voice: str | None = Field(default=None, max_length=300)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class SfxOptions(BaseModel):
    mode: Literal["off", "adaptive", "transitions_only", "aggressive"] = "adaptive"
    volume: float = Field(default=0.55, ge=0.0, le=2.0)
    local_library: str | None = None


class EditorialPolicy(BaseModel):
    """Generalized editorial controls normalized from a short user brief.

    These controls are deliberately additive.  Existing top-level ``preset``,
    ``voiceover`` and ``sfx`` fields remain supported; the policy resolver
    combines both forms before direction starts.
    """

    content_type: Literal[
        "auto", "talking_head", "podcast", "interview", "tutorial",
        "screen_demo", "product", "documentary", "vlog", "montage",
        "music", "animation",
    ] = "auto"
    editing_intensity: Literal["auto", "minimal", "clean", "dynamic", "viral", "cinematic"] = "auto"
    cut_policy: Literal["auto", "preserve_existing_silence_cuts", "editorial", "aggressive"] = "auto"
    source_audio: Literal["auto", "preserve", "replace", "mute"] = "auto"
    voiceover: Literal["auto", "off", "hook_only", "hook_and_outro", "full"] = "auto"
    supplementary_visuals: Literal["auto", "off", "prefer", "require"] = "auto"
    motion_graphics: Literal["auto", "off", "selective", "hero"] = "auto"
    camera_motion: Literal["auto", "static", "punch_ins", "dynamic"] = "auto"
    music: Literal["auto", "off", "library", "generated"] = "auto"
    sfx: Literal["auto", "off", "transitions_only", "adaptive", "aggressive"] = "auto"
    style_profile: str = Field(default="auto", min_length=1, max_length=120)
    review_level: Literal["mechanical", "visual", "strict"] = "visual"


class EditorOptions(BaseModel):
    """Automatic production controls. Explicit off switches remain authoritative."""

    enabled: bool = False  # opt in; existing clients retain their behavior
    quality_profile: Literal["standard", "polished"] = "standard"
    narrative_revision: Literal["design_only", "replan"] = "design_only"
    layout: Literal["auto", "speaker_safe", "director"] = "auto"
    caption_font_size: int = Field(default=62, ge=36, le=96)
    asset_visual_review: bool = False
    visual_review_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    caption_style: Literal["phrase", "highlight", "karaoke", "off"] = "highlight"
    theme: Literal["editorial", "bold", "minimal"] = "editorial"
    progress_bar: bool = True
    procedural_sfx: bool = True
    music: Literal["auto", "off", "library", "procedural"] = "auto"
    min_visual_layers: int = Field(default=0, ge=0, le=12)
    min_graphics: int = Field(default=1, ge=0, le=8)
    strict_requirements: bool = True
    planning_window_seconds: int = Field(default=120, ge=30, le=600)
    max_planning_windows: int = Field(default=24, ge=1, le=60)
    visual_review_required: bool = False
    publish_outputs: bool = False
    narration: Literal["off", "auto", "intro", "intro_outro", "intro_bridge_outro"] = "off"
    narration_budget_seconds: float = Field(default=12.0, ge=2.0, le=60.0)
    transition: Literal["auto", "cut", "dissolve", "swipe", "match"] = "auto"
    required_features: list[Literal["voiceover", "sfx", "music", "camera_motion", "transitions", "graphics", "supplementary_media", "captions"]] = Field(default_factory=list, max_length=8)


class AIComposeRequest(BaseModel):
    """Public request accepted by POST /media/ai-compose."""

    unlimited_resources: bool = False
    source_path: str | None = None
    source_url: str | None = None
    telegram_file_id: str | None = None
    request_id: str | None = Field(default=None, max_length=512)

    brief: str = Field(
        default="Create an informative, energetic short-form video from the strongest moments.",
        min_length=3,
        max_length=5000,
    )
    preset: Literal["viral_short", "clean_explainer", "documentary", "product_promo"] = "viral_short"
    target_seconds: float = Field(default=45.0, ge=5.0, le=1800.0)
    start_seconds: float = Field(default=0.0, ge=0.0)
    end_seconds: float | None = Field(default=None, gt=0.0)
    language: str | None = Field(default=None, min_length=2, max_length=12)
    output_count: int = Field(default=1, ge=1, le=3)
    # Bounded draft-review-revision cycles after the first render.
    refinement_iterations: int = Field(default=2, ge=0, le=4)
    refinement_strict: bool = False
    output_width: int = Field(default=1080, ge=256, le=3840)
    output_height: int = Field(default=1920, ge=256, le=3840)
    fps: Literal[24, 25, 30, 60] = 30
    crf: int = Field(default=21, ge=16, le=30)
    audio_bitrate: Literal[128, 160, 192, 256] = 192

    captions_enabled: bool = True
    fetch_assets: bool = True
    asset_sources: AssetSourceConfig = Field(default_factory=AssetSourceConfig)
    local_asset_library: str | None = None
    local_sfx_library: str | None = None
    max_asset_layers: int = Field(default=8, ge=0, le=12)
    editorial_policy: EditorialPolicy = Field(default_factory=EditorialPolicy)
    editor: EditorOptions = Field(default_factory=EditorOptions)

    voiceover: VoiceoverOptions = Field(default_factory=VoiceoverOptions)
    sfx: SfxOptions = Field(default_factory=SfxOptions)
    music_url: str | None = None
    music_volume: float = Field(default=0.08, ge=0.0, le=1.0)

    llm_url: str | None = Field(default=None, max_length=500)
    llm_model: str | None = Field(default=None, max_length=300)
    llm_required: bool = False
    visual_model: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_request(self) -> "AIComposeRequest":
        if sum(bool(value) for value in (self.source_path, self.source_url, self.telegram_file_id)) != 1:
            raise ValueError("Provide exactly one of source_url, telegram_file_id, or source_path")
        if self.end_seconds is not None and self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        if self.preset == "clean_explainer" and self.sfx.mode == "aggressive":
            raise ValueError("clean_explainer cannot use aggressive SFX")
        return self


class BaseSegment(BaseModel):
    source_start: float = Field(ge=0.0)
    source_end: float = Field(gt=0.0)
    reason: str = Field(default="selected_by_director", max_length=500)
    score: float | None = None
    kind: Literal["source", "narration"] = "source"
    audio_gain: float = Field(default=1.0, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_range(self) -> "BaseSegment":
        if self.source_end <= self.source_start:
            raise ValueError("source_end must be greater than source_start")
        return self


class AssetRequest(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    kind: Literal["video", "image"] = "video"
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    placement: Literal["full", "top", "bottom", "left", "right", "picture_in_picture", "split"] = "picture_in_picture"
    x: float = Field(default=0.08, ge=0.0, le=1.0)
    y: float = Field(default=0.54, ge=0.0, le=1.0)
    width: float = Field(default=0.84, gt=0.02, le=1.0)
    height: float | None = Field(default=None, gt=0.02, le=1.0)
    animation: Literal["none", "fade", "slide_up", "slide_down", "slide_left", "slide_right", "float", "reveal"] = "slide_up"
    animation_seconds: float = Field(default=0.28, ge=0.05, le=1.5)
    opacity: float = Field(default=1.0, ge=0.05, le=1.0)
    reason: str = Field(default="semantic_support", max_length=500)
    benefit: str | None = Field(default=None, max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)
    must_not_cover: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_range(self) -> "AssetRequest":
        if self.end <= self.start:
            raise ValueError("asset end must be greater than start")
        return self


class SfxEvent(BaseModel):
    kind: Literal["whoosh", "click", "pop", "impact", "riser", "swipe", "transition"]
    at: float = Field(ge=0.0)
    volume: float = Field(default=0.55, ge=0.0, le=2.0)
    reason: str = Field(default="visual_event", max_length=300)


class VoiceoverPlan(BaseModel):
    text: str = Field(min_length=1, max_length=2400)
    start: float = Field(default=0.0, ge=0.0)
    volume: float = Field(default=1.0, ge=0.0, le=2.0)


class NarrationSection(BaseModel):
    """Inserted narration; duration is measured from synthesized audio, never guessed."""

    position: Literal["intro", "bridge", "outro"]
    text: str = Field(min_length=1, max_length=600)
    after_segment: int | None = Field(default=None, ge=0, le=59)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)


class GraphicPlan(BaseModel):
    """A model-directed text/graphic cue rendered by the compositor."""

    kind: Literal["hook", "outro", "lower_third", "quote_card", "callout", "cta", "checklist", "stat", "timeline", "badge"] = "callout"
    start: float = Field(ge=0.0)
    end: float = Field(gt=0.0)
    text: str = Field(min_length=1, max_length=500)
    placement: Literal["top", "upper_left", "center", "lower_left", "bottom"] = "center"
    animation: Literal["none", "fade", "slide_up", "slide_down", "pop", "reveal"] = "fade"
    style: Literal["default", "accent", "warning", "quote", "cta"] = "default"
    reason: str = Field(default="director_decision", max_length=300)
    evidence_refs: list[str] = Field(default_factory=list, max_length=12)

    @model_validator(mode="after")
    def validate_range(self) -> "GraphicPlan":
        if self.end <= self.start:
            raise ValueError("graphic end must be greater than start")
        return self


class CompositionPlan(BaseModel):
    schema_version: Literal["sh01/ai-compose@1"] = "sh01/ai-compose@1"
    title: str = Field(default="AI composed short", max_length=300)
    hook_text: str | None = Field(default=None, max_length=240)
    outro_text: str | None = Field(default=None, max_length=240)
    base_segments: list[BaseSegment] = Field(min_length=1, max_length=60)
    asset_requests: list[AssetRequest] = Field(default_factory=list, max_length=12)
    sfx_events: list[SfxEvent] = Field(default_factory=list, max_length=24)
    voiceovers: list[VoiceoverPlan] = Field(default_factory=list, max_length=3)
    narration_sections: list[NarrationSection] = Field(default_factory=list, max_length=3)
    transition_style: Literal["cut", "dissolve", "swipe", "match"] | None = None
    graphics: list[GraphicPlan] = Field(default_factory=list, max_length=24)
    captions_enabled: bool = True
    director_notes: list[str] = Field(default_factory=list, max_length=20)


class ResolvedLayer(BaseModel):
    asset_id: str
    local_path: str
    source: str
    license: str | None = None
    attribution: str | None = None
    query: str
    kind: Literal["video", "image"]
    start: float
    end: float
    x: float
    y: float
    width: float
    height: float | None = None
    animation: str = "none"
    animation_seconds: float = 0.28
    opacity: float = 1.0
    score: float = 0.0
    reason: str = "semantic_support"
    source_offset: float = Field(default=0.0, ge=0.0)


class AIComposePlan(BaseModel):
    schema_version: Literal["sh01/ai-compose-result@1"] = "sh01/ai-compose-result@1"
    plan: CompositionPlan
    resolved_layers: list[ResolvedLayer] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    asset_manifest: list[dict[str, Any]] = Field(default_factory=list)
    editorial_profile: dict[str, Any] = Field(default_factory=dict)
    edit_plan: dict[str, Any] = Field(default_factory=dict)
    quality_review: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, Any] = Field(default_factory=dict)
