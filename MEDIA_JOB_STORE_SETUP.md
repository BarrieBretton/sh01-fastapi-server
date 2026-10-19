# Media job store and shared queue

**Deployment note, 20 October 2026:** SH01's three async media job types use a canonical queue exposed by the dashboard-managed `infra-control` Cloudflare Worker and backed by D1. This supersedes the older instructions that sent workers directly to a Render master media-store endpoint.

The Git repository contains the **client and consumer**, not the deployed infra-control Worker implementation or its D1 migrations. Before reproducing the system, export/review the live Worker and D1 schema in the Cloudflare UI and add sanitized source and migrations to Git. See [`docs/PROJECT_HANDBOOK.md`](docs/PROJECT_HANDBOOK.md) for the external inventory and evidence gaps.

## Who runs which part

| Component | Setting and behavior |
| --- | --- |
| infra-control Cloudflare Worker | Serves `/internal/media-jobs/queue/*` from its D1 database. Check the exact deployed **D1 binding name** and schema in the dashboard; they are not recorded in Git. It validates the shared `MEDIA_JOB_STORE_API_KEY` received in `X-MEDIA-JOB-KEY`. |
| SH01 Render workers | `MEDIA_JOB_STORE_MODE=remote`, `CONTROL_PLANE_BASE_URL=https://infra-control.vivojaymail.workers.dev`, `MEDIA_JOB_STORE_API_KEY=<same queue key>`, `MEDIA_QUEUE_CONSUMER_ENABLED=true`. Each consumer polls for a claim; the canonical queue coordinates jobs across consumers. |
| Control-plane Render services | Keep `MEDIA_QUEUE_CONSUMER_ENABLED=false`. These services host `/infra/*` and other infrastructure routes; they must not claim media jobs. They do not need a fixed Postgres slot for the D1 queue. |
| High-resource local/Kaggle worker | Same remote queue URL/key and consumer setting, plus `MEDIA_ALLOW_UNLIMITED_RESOURCES=true`; restrict local `source_path` to `MEDIA_LOCAL_SOURCE_DIR`. That path must exist on whichever host claims the job. |

The queue is independent of the active Supabase/Postgres slot. **Do not add or set `MEDIA_QUEUE_POSTGRES_SLOT`** for this D1 deployment. The control plane's Postgres ring serves separate infrastructure and n8n workloads.

`MEDIA_JOB_STORE_MODE=local` and `MEDIA_JOB_STORE_SERVER_ENABLED` relate to older/local media record routes in `infra/media_job_store_routes.py`; they are not a substitute for the canonical D1 queue required by `store.queue_enqueue`, `store.queue_claim`, and related methods. Treat `MEDIA_JOB_STORE_API_KEY` as its own secret, distinct from both `CONTROL_PLANE_API_KEY` and `MEDIA_CLIP_API_KEY`.

## Validate the queue

In PowerShell, supply the secret through your current secret manager; the value is never committed.

```powershell
$Control = 'https://infra-control.vivojaymail.workers.dev'
$QueueKey = '<MEDIA_JOB_STORE_API_KEY>'

Invoke-RestMethod "$Control/internal/media-jobs/queue/health" `
  -Headers @{ 'X-MEDIA-JOB-KEY' = $QueueKey } -UseBasicParsing
```

An operator observed `{"ok":true,"queue":"cloudflare-d1"}` from this health endpoint. A health response confirms the deployed API responds; it does not prove the worker source, D1 migration history, or every operation can be recreated from this repository.

The worker-facing SH01 editor uses **another** header:

```powershell
Invoke-RestMethod 'https://sh01.vivojaymail.workers.dev/media/auto-edit-capabilities' `
  -Headers @{ 'X-API-KEY' = $MediaClipApiKey } -UseBasicParsing
```

On a consumer, verify `MEDIA_QUEUE_CONSUMER_ENABLED=true` and check startup logs for queue poll errors. Submit a short representative job through the SH01 proxy, poll `GET /media/auto-edit-jobs/{job_id}`, and inspect the final media/EDL before increasing workload. Use `unlimited_resources=false` on small Render instances; leave the higher-resource setting for an opted-in host.

## Failure and recovery

- If the queue endpoint is unavailable, SH01 refuses job registration or reports queue errors. Investigate the Cloudflare Worker and D1 dashboard before retrying with the **same request ID**.
- `queued` means accepted, not yet processing. `running` stages can include download, transcription, analysis and rendering. Read the returned `status_url` or poll by job ID.
- The resource checks reduce OOM risk but can reject a job; a successful Kaggle render is not proof that it fits in a 512 MiB Render worker.
- Heavy synchronous image/social paths can return a busy response via `media_admission.py`; they do not join the long-lived async media job queue.
- Keep D1 queue schema, deployed Worker code, keys' **names**, queue settings, and smoke-test date in `docs/PROJECT_HANDBOOK.md` as they change.
