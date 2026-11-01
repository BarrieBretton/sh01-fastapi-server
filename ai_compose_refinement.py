"""Bounded draft-review-revision orchestration for AI composition.

The existing media/caption routes remain untouched. This module is imported
only by the new AI-compose workflow after its first deterministic render.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Callable

from ai_compose_schema import (
    AIComposeRequest,
    AssetRequest,
    BaseSegment,
    CompositionPlan,
    ResolvedLayer,
)
from ai_compose_engine import (
    ComposeError,
    LocalDirector,
    _upload_output,
    fallback_plan,
    probe_media,
    render_composition,
    resolve_assets,
)
from ai_compose_ir import compile_edit_plan
from ai_compose_policy import resolve_editorial_policy
from ai_compose_quality import inspect_edit_plan, inspect_render

logger = logging.getLogger("ai_compose_refinement")


def _director_configured(request: AIComposeRequest) -> bool:
    return bool(
        os.getenv("AI_COMPOSE_HF_MODEL", "").strip()
        or request.llm_url
        or os.getenv("AI_COMPOSE_LLM_URL", "").strip()
        or os.getenv("LLAMA_SERVER_URL", "").strip()
    )


def _quality_check(
    path: Path,
    expected_duration: float,
    request: AIComposeRequest,
) -> dict[str, Any]:
    """Run deterministic post-render checks before requesting a revision."""
    issues: list[str] = []

    try:
        media = probe_media(path)
    except Exception as exc:
        return {
            "ok": False,
            "issues": [f"probe_failed: {str(exc)[:300]}"],
        }

    duration = float(media.get("duration") or 0.0)
    if duration < max(0.1, expected_duration * 0.92):
        issues.append(f"duration_short:{duration:.3f}/{expected_duration:.3f}")

    if (
        media.get("width") != request.output_width
        or media.get("height") != request.output_height
    ):
        issues.append(
            f"dimensions:{media.get('width')}x{media.get('height')};"
            f"expected:{request.output_width}x{request.output_height}"
        )

    if not media.get("has_audio"):
        issues.append("missing_audio")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        decoded = subprocess.run(
            [ffmpeg, "-v", "error", "-i", str(path), "-f", "null", "-"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300,
            check=False,
        )
        if decoded.returncode:
            detail = decoded.stderr.decode("utf-8", errors="replace")[-1000:]
            issues.append(f"decode_errors:{detail}")

    return {
        "ok": not issues,
        "issues": issues,
        "duration": round(duration, 3),
        "width": media.get("width"),
        "height": media.get("height"),
        "has_audio": bool(media.get("has_audio")),
        "codec": media.get("video_codec"),
    }


def _normalise_plan(
    request: AIComposeRequest,
    plan: CompositionPlan,
    media: dict[str, Any],
    words: list[dict[str, Any]],
) -> CompositionPlan:
    """Clamp a revised model plan to the actual source/output envelope."""
    start = min(request.start_seconds, max(0.0, media["duration"] - 0.05))
    end = min(media["duration"], request.end_seconds or media["duration"])

    segments: list[BaseSegment] = []
    for item in plan.base_segments:
        if item.source_end <= start or item.source_start >= end:
            continue

        segment_start = max(start, min(end - 0.05, item.source_start))
        segment_end = min(end, max(segment_start + 0.05, item.source_end))

        if segment_end > segment_start:
            segments.append(
                BaseSegment(
                    source_start=segment_start,
                    source_end=segment_end,
                    reason=item.reason,
                    score=item.score,
                )
            )

    if not segments:
        plan = fallback_plan(request, media, words)
        segments = list(plan.base_segments)

    fitted: list[BaseSegment] = []
    remaining = min(
        request.target_seconds,
        sum(item.source_end - item.source_start for item in segments),
    )

    for item in segments:
        if remaining <= 0.05:
            break

        segment_end = min(item.source_end, item.source_start + remaining)
        if segment_end - item.source_start > 0.05:
            fitted.append(
                item.model_copy(update={"source_end": segment_end})
            )
            remaining -= segment_end - item.source_start

    plan.base_segments = fitted or segments
    output_duration = sum(
        item.source_end - item.source_start
        for item in plan.base_segments
    )

    valid_assets: list[AssetRequest] = []
    for item in plan.asset_requests:
        if item.end <= 0 or item.start >= output_duration:
            continue

        asset_start = max(
            0.0,
            min(max(0.0, output_duration - 0.05), item.start),
        )
        asset_end = min(output_duration, item.end)

        if asset_end - asset_start > 0.05:
            valid_assets.append(
                item.model_copy(update={"start": asset_start, "end": asset_end})
            )

    plan.asset_requests = valid_assets
    policy = resolve_editorial_policy(request, media, words)
    if policy.supplementary_visuals == "off":
        plan.asset_requests = []
    if policy.voiceover == "off":
        plan.voiceovers = []
    if policy.sfx == "off":
        plan.sfx_events = []
    return plan


def _revision_request(
    request: AIComposeRequest,
    plan: CompositionPlan,
    quality: dict[str, Any],
    round_number: int,
) -> AIComposeRequest:
    context = {
        "revision_round": round_number,
        "current_plan": plan.model_dump(mode="json"),
        "draft_quality": quality,
    }

    brief = (
        f"{request.brief}\n\n"
        "You are revising an existing video edit after a draft render. "
        "Preserve relevant content, but correct timing, hook, segment selection, "
        "asset placement, transitions, captions, and audio decisions where useful. "
        "Return a complete replacement plan, not a partial patch. "
        "Do not invent source timestamps outside the media. "
        "Here is the current plan and mechanical draft review:\n"
        f"{json.dumps(context, ensure_ascii=False, separators=(',', ':'))}"
    )

    # The base request has a 5000-character validation limit. model_copy does
    # not revalidate, so truncate explicitly before reusing the director.
    return request.model_copy(update={"brief": brief[:5000]})


def _revise_plan(
    request: AIComposeRequest,
    plan: CompositionPlan,
    quality: dict[str, Any],
    media: dict[str, Any],
    words: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    round_number: int,
) -> tuple[CompositionPlan, list[str]]:
    revision_request = _revision_request(request, plan, quality, round_number)
    return LocalDirector(revision_request).plan(media, words, frames)


async def _render_variants(
    request: AIComposeRequest,
    job: dict[str, Any],
    source: Path,
    media: dict[str, Any],
    plan: CompositionPlan,
    layers: list[ResolvedLayer],
    words: list[dict[str, Any]],
    work_root: Path,
    warnings: list[str],
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    policy = resolve_editorial_policy(request, media, words)

    for variant in range(1, request.output_count + 1):
        edit_plan = compile_edit_plan(
            request,
            plan,
            layers,
            words,
            policy,
            media,
            variant=variant,
        )
        plan_review = inspect_edit_plan(edit_plan, request, resolved_layer_count=len(layers))
        if not plan_review.ok:
            raise ComposeError(
                f"Revised edit plan failed validation for variant {variant}: "
                f"{plan_review.model_dump_json()}"
            )
        variant_layers = layers
        if variant > 1:
            variant_layers = [
                layer.model_copy(
                    update={
                        "animation": (
                            "fade"
                            if layer.animation.startswith("slide")
                            else layer.animation
                        )
                    }
                )
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
            edit_plan,
        )
        render_review = await asyncio.to_thread(
            inspect_render,
            Path(rendered["local_path"]),
            edit_plan.duration,
            request,
            probe_media,
        )
        rendered["quality_review"] = render_review.model_dump(mode="json")
        rendered.update(
            _upload_output(
                Path(rendered["local_path"]),
                str(job["request_id"]),
                variant,
            )
        )
        outputs.append(rendered)

    return outputs


async def refine_composition(
    *,
    request: AIComposeRequest,
    job: dict[str, Any],
    update: Callable[..., Any],
    source: Path,
    media: dict[str, Any],
    words: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    work_root: Path,
    plan: CompositionPlan,
    layers: list[ResolvedLayer],
    manifest: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    warnings: list[str],
) -> tuple[
    CompositionPlan,
    list[ResolvedLayer],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    rounds = int(request.refinement_iterations)

    if rounds <= 0:
        return plan, layers, manifest, outputs, []

    if not _director_configured(request):
        return plan, layers, manifest, outputs, [
            "Iterative refinement skipped because no local director backend is configured."
        ]

    refinement_warnings: list[str] = []
    current_plan = plan
    current_layers = layers
    current_manifest = manifest
    current_outputs = outputs

    for round_number in range(1, rounds + 1):
        await update(
            stage="quality_review",
            progress=min(0.97, 0.90 + 0.02 * round_number),
            refinement_round=round_number,
            refinement_total=rounds,
        )

        if not current_outputs:
            raise ComposeError("Cannot refine a composition without a rendered draft")

        expected_duration = sum(
            item.source_end - item.source_start
            for item in current_plan.base_segments
        )
        quality = await asyncio.to_thread(
            _quality_check,
            Path(current_outputs[0]["local_path"]),
            expected_duration,
            request,
        )

        try:
            revised_plan, planner_warnings = await asyncio.to_thread(
                _revise_plan,
                request,
                current_plan,
                quality,
                media,
                words,
                frames,
                round_number,
            )
        except Exception as exc:
            if request.refinement_strict:
                raise ComposeError(
                    f"Iterative director failed in round {round_number}: "
                    f"{str(exc)[:500]}"
                ) from exc

            refinement_warnings.append(
                f"Iterative director failed in round {round_number}; "
                f"keeping the best completed draft: {str(exc)[:240]}"
            )
            break

        refinement_warnings.extend(planner_warnings)
        current_plan = _normalise_plan(
            request,
            revised_plan,
            media,
            words,
        )

        await update(
            stage="asset_retrieval",
            progress=min(0.98, 0.93 + 0.01 * round_number),
            refinement_round=round_number,
            asset_queries=len(current_plan.asset_requests),
        )

        current_layers, current_manifest, asset_warnings = await asyncio.to_thread(
            resolve_assets,
            request,
            current_plan,
            work_root,
        )
        refinement_warnings.extend(asset_warnings)

        await update(
            stage="revision_render",
            progress=min(0.99, 0.95 + 0.01 * round_number),
            refinement_round=round_number,
            refinement_total=rounds,
        )

        current_outputs = await _render_variants(
            request,
            job,
            source,
            media,
            current_plan,
            current_layers,
            words,
            work_root,
            warnings,
        )

    return (
        current_plan,
        current_layers,
        current_manifest,
        current_outputs,
        refinement_warnings,
    )

