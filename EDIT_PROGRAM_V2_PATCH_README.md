# SH01 AI Compose: generalized edit-program patch

This patch is the first implementation slice of the generalized workflow
described in the AI video-skills audit. It is intentionally additive and is
safe to apply on top of the existing `feature/ai-compose-kaggle` branch.

## Files

```text
ai_compose_schema.py       additive request/result fields
ai_compose_policy.py       content/audio/asset/motion policy resolver
ai_compose_ir.py           sh01/ai-edit@2 renderer-neutral timeline
ai_compose_quality.py      pre-render and post-render quality reports
ai_compose_engine.py       policy, evidence, camera tracks, QA integration
ai_compose_refinement.py   revision renders consume the same edit program
ai_compose.py              capabilities and schema route advertise the IR
tests/test_ai_compose_policy_ir_quality.py
```

## Compatibility guarantees

- Existing `/media/ai-compose` and job-status routes are retained.
- Existing caption and auto-editor routes are untouched.
- The legacy `CompositionPlan` is still accepted by the local director.
- `preset`, top-level `voiceover`, `sfx`, and `fetch_assets` remain supported.
- Existing queue, resource-guard, local-source, and B2 integrations remain
  outside the patch's control flow.
- The model still returns JSON only; it never receives an FFmpeg shell tool.

## New behavior

The job now persists an `evidence.json` artifact and a receipt containing:

- the resolved editorial profile;
- a versioned `sh01/ai-edit@2` edit program;
- source-to-program clip mappings;
- bounded camera tracks and semantic variant strategy;
- visual-layer and audio-cue decisions;
- pre-render plan validation;
- post-render technical quality review;
- asset benefit, confidence, evidence references, and provenance.

Dialogue-aware policy is enforced after the model responds. If source speech
is detected, source audio remains primary and automatic voiceover is removed
unless the user explicitly requested it. `fetch_assets=false` disables asset
retrieval even if a model incorrectly emits asset requests.

## Apply on the Windows repository

Copy the patch files into the repository root, replacing only the listed AI
Compose files. Do not replace `app.py`, `media_dispatcher.py`, caption routes,
or infrastructure files for this slice. Then run:

```powershell
python -m py_compile `
  ai_compose_schema.py `
  ai_compose_policy.py `
  ai_compose_ir.py `
  ai_compose_quality.py `
  ai_compose_engine.py `
  ai_compose_refinement.py `
  ai_compose.py

python -m unittest discover -s tests -p "test_ai_compose*.py" -v
git diff --check
```

Commit this as a separate change from unrelated infra work. The next slice
can add optional scene/face/audio analyzers and Remotion/HyperFrames workers
without changing this public request contract.
