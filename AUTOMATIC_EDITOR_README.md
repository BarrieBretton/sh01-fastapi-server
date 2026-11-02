# Automatic editor

Use `notebooks/sh01_automatic_editor.ipynb`: one integrated prompt-driven run,
rather than a long collection of disabled feature tests. Older notebooks remain
available for regression work.

## Run

1. Import the new notebook into Kaggle. Enable GPU and Internet; mount the video.
2. Edit the private `.env` heredoc before first execution, or edit the existing
   `.env`. Setup never overwrites an existing file.
3. Set `AI_COMPOSE_SOURCE_PATH` exactly (no trailing space in the variable name).
   The notebook derives `MEDIA_LOCAL_SOURCE_DIR` before starting Uvicorn.
4. Set your brief and target duration. Supply an asset library or provider keys
   for B-roll. Watch the MP4, inspect the receipt, then download the result ZIP
   or save Kaggle notebook outputs. Browser confirmation cannot be bypassed.

GPU model loading and real-provider calls still need verification in your Kaggle
image. Offline tests mock model/ASR boundaries and execute real FFmpeg renders.
Passing tests do not establish artistic quality, factual correctness or virality.

## Existing API integration

`POST /media/ai-compose` remains the entry point. Add this to its normal request:

```json
{
  "editor": {
    "enabled": true,
    "caption_style": "highlight",
    "theme": "editorial",
    "min_graphics": 2,
    "min_visual_layers": 1,
    "music": "auto",
    "visual_review_required": true,
    "strict_requirements": true,
    "publish_outputs": false
  }
}
```

Source, brief, dimensions, refinement, voiceover and asset options retain their
contracts. `editor.enabled` defaults to false for existing callers. Existing
caption endpoints are not replaced. `llm_required=true` now fails on missing or
invalid direction unless the operator explicitly enables
`AI_COMPOSE_ALLOW_DETERMINISTIC_RECOVERY=true`.

Production `app.py` retains its queue/storage wiring. Kaggle starts
`ai_compose_local_app:app` on loopback with one worker, reusing the same router
with SQLite receipts. This bootstrap does not initialize Redis, Postgres or cron.
It is not a replacement for the production application. Interrupted jobs are
marked failed on restart; receipts persist but GPU execution is not resumable.

## Execution

| Phase | Behavior and evidence |
| --- | --- |
| Understand | Isolated chunked Whisper, metadata, sampled frames, local VLM descriptions |
| Select | Bounded source-window direction plus a selection pass for long sources |
| Design | Separate model call for graphics, assets and sound; preserves selected source segments |
| Retrieve | Existing local/Pexels/Pixabay/Unsplash adapters, metadata threshold, alternate media-type attempt |
| Render | FFmpeg/libass cards, staggered checklists/timelines, badges, kinetic captions, animated camera, timed overlays |
| Sound | Visual-event SFX, optional procedural effects/bed, supplied music ducked under source audio, configured Piper narration |
| Review | Timeline/feature checks, decode validation, actual rendered-frame VLM review, bounded revision |
| Deliver | Highest-scoring valid draft, isolated originals, JSON evidence/receipts, final MP4s |

Transitions blend a held outgoing frame into the incoming clip at its original
program start, preserving source/audio/caption timing. This is not an overlapping
source edit that removes time from the timeline.

Captions and graphics are separate tracks. Caption off remains a hard stop. Long
or unreadable graphic copy fails the strict gate. New numbers absent from speech
are rejected; this limited check is not general fact checking. Checklist/timeline
labels use `|` separators.

Each draft has its own directory. Failed planning or rendering revisions preserve
the last valid draft when `refinement_strict=false`. Required B-roll that cannot be
resolved fails the job instead of silently delivering a plain clip. Variants share
the narrative and vary camera emphasis; they are not independently authored stories.

## Models, assets and audio

The notebook defaults to `Qwen/Qwen2.5-3B-Instruct` for text direction and
`Qwen/Qwen2.5-VL-3B-Instruct` for source/draft vision, using sequential 4-bit loading.
The vision adapter follows the public
[Qwen2.5-VL APIs](https://huggingface.co/docs/transformers/model_doc/qwen2_5_vl).
Eight sampled frames per call bound memory; they do not give complete temporal
understanding of every frame in a long video.

Alternatively leave `AI_COMPOSE_VISION_HF_MODEL` empty, configure
`AI_COMPOSE_VISION_URL` to a local OpenAI-compatible chat completions URL, and set
request `visual_model`. Required visual review must run and score at least 0.70.
This is a subjective model score, not calibrated quality measurement. Without
vision, explicitly set `visual_review_required=false`; the receipt states the gap.
No paid inference backend is selected automatically.

Set `AI_COMPOSE_ASSET_LIBRARY` or request `local_asset_library`. Use descriptive
filenames and `.mp4.json`/`.json` sidecars with `title`, `description`, `tags`,
`license` and `attribution`. Retrieval ranks metadata word overlap, not embedding
similarity. Frame review can catch bad placement but does not prove asset relevance.
Set `PEXELS_API_KEY`, `PIXABAY_API_KEY` or `UNSPLASH_ACCESS_KEY` for remote retrieval.
Provider metadata, attribution and file hashes are recorded; check the asset's
license for your intended use. Provider success is unverified in the offline suite.

Existing dialogue suppresses added narration unless source audio is explicitly
replaced/muted. Silent sources can use configured Piper (`piper` on PATH and
`AI_COMPOSE_TTS_MODEL`). Music `auto` uses a configured track; `procedural` opts
into a simple original synthesized bed, not AI music generation. SFX attach to
visual entrances; orphan effects are removed if retrieval fails.

Long sources are transcribed in chunks and planned in windows. The default bound
is 24 x 120-second planning windows. Select a range or raise the bounded window
budget for longer inputs. Only selected intervals are decoded for rendering;
Kaggle disk, GPU memory and session duration still impose practical limits.

## Verification and boundaries

```bash
python -m unittest discover -s tests -p 'test_ai_compose*.py' -v
python scripts/ai_compose_showcase.py /tmp/sh01-showcase
```

The showcase is an authored renderer fixture, not autonomous creative judgment.
Tests cover real captions/graphics/audio, delayed overlay pixels, dissolve timing,
required assets, numeric evidence, SQLite restart handling and failed revisions.

This implements the local editing/design/review subset inspired by the catalog.
It does not install every catalog project, run Remotion/HyperFrames, generate
arbitrary 3D scenes, avatars or lip-sync, provide a manual multitrack UI, or
guarantee professional output for every input. Those remain separate capabilities.
