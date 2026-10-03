# Media Job Store split

The SH01 media workers do not need Postgres/Supabase credentials.

## Control-plane/master service

Set:

```text
MEDIA_JOB_STORE_MODE=local
MEDIA_JOB_STORE_SERVER_ENABLED=true
MEDIA_JOB_STORE_API_KEY=<strong dedicated secret>
```

Keep the existing `INFRA_REGISTRY_JSON`, `SUPABASE_A_*`, `SUPABASE_B_*`,
`CONTROL_PLANE_API_KEY`, Render admin credentials, and other infrastructure
credentials on the control-plane/master service.

## SH01 workers

Set only:

```text
MEDIA_JOB_STORE_MODE=remote
CONTROL_PLANE_BASE_URL=https://sh01-fastapi-server-mtwu.onrender.com
MEDIA_JOB_STORE_API_KEY=<same dedicated secret>
MEDIA_JOB_STORE_SERVER_ENABLED=false
```

Do not copy `SUPABASE_A_*`, `SUPABASE_B_*`, `RENDER_API_KEY`, or the privileged
`CONTROL_PLANE_API_KEY` to SH01 workers for media-job persistence.

## Request path

Worker media routers call `media_job_store.store`. In remote mode it uses:

- `PUT /internal/media-jobs/{job_id}`
- `GET /internal/media-jobs/{job_id}`

with `X-MEDIA-JOB-KEY`.

The master handles those endpoints and writes/reads `control_plane.jobs`
through `infra.persistence`.

Writes are idempotent because `control_plane.jobs.id` is the primary key and
`put_job()` uses `ON CONFLICT (id) DO UPDATE`.

## Health check

Against the master/control-plane only:

```bash
curl -sS   "$CONTROL_PLANE_BASE_URL/internal/media-jobs/health"   -H "X-MEDIA-JOB-KEY: $MEDIA_JOB_STORE_API_KEY"
```
