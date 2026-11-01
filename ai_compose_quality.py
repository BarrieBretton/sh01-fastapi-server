"""Deterministic quality checks for the renderer-neutral edit program."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from ai_compose_ir import EditPlan
from ai_compose_schema import AIComposeRequest


class QualityIssue(BaseModel):
    severity: str = "warning"
    category: str
    code: str
    message: str
    evidence_refs: list[str] = Field(default_factory=list)


class ReviewReport(BaseModel):
    schema_version: str = "sh01/ai-review@1"
    ok: bool
    score: float = Field(default=1.0, ge=0.0, le=1.0)
    issues: list[QualityIssue] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


def inspect_edit_plan(
    edit_plan: EditPlan,
    request: AIComposeRequest,
    *,
    resolved_layer_count: int = 0,
) -> ReviewReport:
    """Check timeline invariants before invoking a renderer."""

    issues: list[QualityIssue] = []
    clips = edit_plan.source_clips
    if not clips:
        issues.append(QualityIssue(severity="error", category="timeline", code="no_source_clips", message="The edit plan contains no source clips."))
    expected_program = 0.0
    for index, clip in enumerate(clips):
        if abs(clip.program_start - expected_program) > 0.06:
            issues.append(QualityIssue(severity="error", category="timeline", code="program_gap", message=f"Source clip {index} does not start at the previous program cursor.", evidence_refs=[f"source_segment_{index}"]))
        if clip.source_end <= clip.source_start:
            issues.append(QualityIssue(severity="error", category="timeline", code="invalid_source_window", message=f"Source clip {index} has an invalid source window."))
        expected_program = clip.program_end
    if abs(expected_program - edit_plan.duration) > 0.06:
        issues.append(QualityIssue(severity="error", category="timeline", code="duration_mismatch", message="Edit duration does not match the final source clip cursor."))

    for layer in edit_plan.visual_layers:
        if layer.program_end <= layer.program_start:
            issues.append(QualityIssue(severity="error", category="visual", code="invalid_layer_window", message=f"Layer {layer.asset_id} has an invalid program window."))
        if layer.x + layer.width > 1.001 or layer.y + (layer.height or 1.0) > 1.001:
            issues.append(QualityIssue(severity="warning", category="visual", code="layer_outside_frame", message=f"Layer {layer.asset_id} extends outside the normalized frame."))

    if edit_plan.policy.get("supplementary_visuals") == "require" and resolved_layer_count == 0:
        issues.append(QualityIssue(severity="warning", category="assets", code="required_asset_missing", message="The policy required supporting visuals but no asset layer was resolved."))
    if edit_plan.policy.get("source_audio") == "preserve" and not edit_plan.audio_cues:
        issues.append(QualityIssue(severity="warning", category="audio", code="audio_policy_unrepresented", message="The policy preserves source audio but the edit plan has no audio cue."))

    error_count = sum(issue.severity == "error" for issue in issues)
    score = max(0.0, 1.0 - (error_count * 0.25) - (sum(issue.severity == "warning" for issue in issues) * 0.05))
    return ReviewReport(
        ok=error_count == 0,
        score=round(score, 3),
        issues=issues,
        checks={
            "source_clip_count": len(clips),
            "visual_layer_count": len(edit_plan.visual_layers),
            "camera_track_count": len(edit_plan.camera_tracks),
            "transition_count": len(edit_plan.transitions),
            "audio_cue_count": len(edit_plan.audio_cues),
            "duration": edit_plan.duration,
            "requested_dimensions": [request.output_width, request.output_height],
        },
        recommended_actions=[issue.message for issue in issues if issue.severity != "error"],
    )


def inspect_render(
    path: Path,
    expected_duration: float,
    request: AIComposeRequest,
    probe: Callable[[Path], dict[str, Any]],
    *,
    run_decode: bool = True,
) -> ReviewReport:
    """Check a rendered file without importing the engine (avoids a cycle)."""

    issues: list[QualityIssue] = []
    try:
        media = probe(path)
    except Exception as exc:
        return ReviewReport(
            ok=False,
            score=0.0,
            issues=[QualityIssue(severity="error", category="technical", code="probe_failed", message=str(exc)[:500])],
        )
    duration = float(media.get("duration") or 0.0)
    if duration < max(0.1, expected_duration * 0.92):
        issues.append(QualityIssue(severity="error", category="technical", code="duration_short", message=f"Rendered duration {duration:.3f}s is shorter than expected {expected_duration:.3f}s."))
    if media.get("width") != request.output_width or media.get("height") != request.output_height:
        issues.append(QualityIssue(severity="error", category="technical", code="dimensions_mismatch", message=f"Rendered dimensions are {media.get('width')}x{media.get('height')}; expected {request.output_width}x{request.output_height}."))
    if not media.get("has_audio"):
        issues.append(QualityIssue(severity="warning", category="audio", code="missing_audio", message="Rendered output has no audio stream."))

    if run_decode and shutil.which("ffmpeg"):
        decoded = subprocess.run(
            [shutil.which("ffmpeg") or "ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        if decoded.returncode:
            detail = decoded.stderr.decode("utf-8", errors="replace")[-1000:]
            issues.append(QualityIssue(severity="error", category="technical", code="decode_errors", message=detail))

    error_count = sum(issue.severity == "error" for issue in issues)
    score = max(0.0, 1.0 - error_count * 0.3 - sum(issue.severity == "warning" for issue in issues) * 0.05)
    return ReviewReport(
        ok=error_count == 0,
        score=round(score, 3),
        issues=issues,
        checks={
            "duration": round(duration, 3),
            "width": media.get("width"),
            "height": media.get("height"),
            "has_audio": bool(media.get("has_audio")),
            "codec": media.get("video_codec"),
        },
        recommended_actions=[issue.message for issue in issues],
    )
