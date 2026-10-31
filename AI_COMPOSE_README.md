# SH01 local AI composition patch

This package adds a local-first `ai_compose` workflow to the SH01 FastAPI
server. It is designed for a GPU Kaggle worker and uses:

1. `faster-whisper` for word-level transcription;
2. a small local director model (optional `AI_COMPOSE_HF_MODEL`, `llama.cpp`,
   or Ollama-compatible endpoint) for a typed composition plan;
3. local/Pexels/Pixabay/Unsplash asset retrieval with provenance;
4. deterministic FFmpeg composition, captions, audio mixing, and QC;
5. the existing SH01 durable queue and B2 rendered-media route when configured.

## Install

Unzip this package into the SH01 repository root and run:

```bash
python install_ai_compose.py
```

The installer is idempotent. It creates timestamped backups before changing
`app.py` and `media_dispatcher.py`. It refuses to patch if the expected
upstream markers are missing.

Then install the repository's normal requirements:

```bash
pip install -r requirements.txt
```

The core workflow adds no mandatory Python dependency beyond the existing SH01
requirements. For an in-process GPU director on Kaggle, install the optional
packages already normally present in Kaggle images:

```bash
pip install -U transformers accelerate bitsandbytes
```

Alternatively run a local `llama.cpp` or Ollama-compatible server and set
`AI_COMPOSE_LLM_URL`.

## Kaggle environment

Set these before starting Uvicorn:

```python
import os

os.environ.update({
    "MEDIA_LOCAL_SOURCE_DIR": "/kaggle/input/your-dataset",
    "AI_COMPOSE_WORK_DIR": "/kaggle/working/ai-compose-jobs",
    "AI_COMPOSE_MAX_SOURCE_BYTES": str(16 * 1024**3),
    "MEDIA_ALLOW_UNLIMITED_RESOURCES": "true",
    "MEDIA_QUEUE_CONSUMER_ENABLED": "true",
    "MEDIA_JOB_STORE_MODE": "remote",
    # Use one of these director backends:
    # "AI_COMPOSE_HF_MODEL": "Qwen/Qwen2.5-3B-Instruct",
    # "AI_COMPOSE_LLM_URL": "http://127.0.0.1:8080/v1/chat/completions",
})
```

For a local-only test without the Cloudflare/D1 queue, use:

```python
os.environ["AI_COMPOSE_LOCAL_FALLBACK"] = "true"
```

The route will then execute in the Uvicorn process. For long jobs, the durable
queue is preferred.

## Request

```bash
curl -X POST "$BASE/media/ai-compose" \
  -H "X-API-KEY: $MEDIA_CLIP_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "unlimited_resources": true,
    "source_path": "/kaggle/input/your-dataset/main.mp4",
    "brief": "Create a sharp, informative military-history short with a strong opening, relevant archival B-roll, captions, and restrained whoosh effects.",
    "preset": "viral_short",
    "target_seconds": 60,
    "output_count": 2,
    "captions_enabled": true,
    "fetch_assets": true,
    "asset_sources": {"local_library": true, "pexels": true, "pixabay": true},
    "local_asset_library": "/kaggle/input/asset-library",
    "local_sfx_library": "/kaggle/input/sfx-library",
    "sfx": {"mode": "adaptive"}
  }'
```

Poll the returned `status_url` using the same API key. Each completed output
contains a local path, a public `/media/rendered/{token}` URL when B2 is
configured, a composition receipt, the resolved asset manifest, and warnings.

When B2 is not configured, download a local result from:

```text
GET /media/ai-compose-files/{job_id}/variant_01.mp4
```

## Local asset sidecars

For each local asset, an optional sidecar can sit beside it:

```json
{
  "title": "Soldiers moving through a forest",
  "description": "Archival-style forest movement",
  "tags": ["military", "forest", "patrol"],
  "license": "CC0",
  "attribution": "Public domain archive"
}
```

The system records the sidecar data, file hash, query, timing, geometry, and
selected reason in `composition_receipt.json`.

## Important behavior

- The local model proposes JSON only; it never writes arbitrary shell commands.
- FFmpeg receives internally generated argv lists after Pydantic validation.
- Source video is analyzed in chunks and representative frames are cached in
  the per-job work directory.
- The renderer uses `h264_nvenc` only when explicitly enabled and detected;
  otherwise it safely uses the existing `libx264` path.
- External asset APIs are optional. Without keys, local assets still work.
- `source_path` must exist on the worker that claims the queue job. A Kaggle
  local path must therefore be consumed by the Kaggle worker, not a Render
  worker.

