# AI Compose v3 renderer patch

This patch completes the missing execution half of the generalized edit
program. It is designed to be copied into the repository root on top of the
existing `feature/ai-compose-kaggle` AI Compose branch.

## What is now rendered

- Explicit model-directed `graphics` cues: hooks, lower thirds, quote cards,
  callouts, and CTAs with safe placements, styles, fades, slides, reveals, and
  pop animation.
- Per-frame camera pan/punch motion from the compiled camera keyframe track.
- Cinematic `dissolve`, `swipe`, and `match` transitions through FFmpeg
  `xfade`, with contiguous audio and output duration preservation.
- Existing captions, partial-frame layers, voiceover, music, SFX, and
  provenance artifacts remain on the same route.
- Deterministic fallback graphics when a local director does not provide
  explicit graphics but transcript evidence supports a hook or bridge.

## Proof of execution

The renderer returns `render_features` for each variant. The technical review
compares these counts with the compiled `EditPlan`; planned graphics, camera
tracks, or transitions cannot silently disappear into a receipt-only field.

The patch includes unit tests and a real FFmpeg smoke render. The Kaggle
notebook adds a D2 smoke test that submits a job, downloads the output, archives
the receipt, and asserts the render feature contract.

## Scope and safety

The LLM still returns typed JSON only. It never receives shell access or an
FFmpeg command channel. The renderer constructs the filter graph from bounded
Pydantic objects and keeps FFmpeg as the default backend. No existing caption
route is replaced or duplicated.

