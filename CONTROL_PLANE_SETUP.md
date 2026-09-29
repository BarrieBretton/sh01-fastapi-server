# SH01 control plane - corrected final overlay

This overlay supersedes the earlier one-shot package. It is designed around the two Cloudflare Workers you actually run today:

- `https://n8n-prod-1.vivojaymail.workers.dev/` fronts the n8n Render ring.
- `https://sh01.vivojaymail.workers.dev/` fronts the SH01/FastAPI Render ring.

The existing proxy/failover behavior of both Workers is preserved. The only routing additions are a preferred-backend control, maintenance mode, and KV persistence.

## 1. Files to copy into the repo

Replace/add these files from this package:

```text
infra/models.py
infra/routes.py
infra/state.py
infra/registry.py
infra/persistence.py
infra/jobs.py
infra/runtime_config.py
infra/render_provider.py
infra/b2ring.py
infra/router_client.py
infra/orchestrator.py
infra_registry.example.json
scripts/infra_smoke.ps1
CONTROL_PLANE_SETUP.md
cloudflare/n8n-prod-1-worker.js
cloudflare/sh01-worker.js
cloudflare/wrangler.n8n.example.jsonc
cloudflare/wrangler.sh01.example.jsonc
```

Keep your already-proven `infra/postgres.py` exactly as-is.

Your existing `app.py` already includes `infra_router`, so no app router change is required.

## 2. Render rings are independent

The registry remains flat for compatibility, but every Render slot has a role:

```json
"render-a": { "role": "n8n", ... },
"render-b": { "role": "n8n", ... },
"sh01-a":   { "role": "sh01", ... },
"sh01-b":   { "role": "sh01", ... }
```

You can add `render-c`, `render-d`, `sh01-c`, etc. No A/B toggle is hardcoded. The next target is selected from the sorted slots for that role.

Durable state becomes:

```json
{
  "render": {
    "n8n": "render-b",
    "sh01": "sh01-a"
  },
  "postgres": "supabase-b",
  "b2": "b2-a"
}
```

For compatibility, an old persisted string value such as `"render": "render-a"` is automatically interpreted as the n8n Render slot.

## 3. Master Render environment

Keep the existing Postgres migration variables on Session Pooler port 5432:

```text
SUPABASE_A_HOST=...
SUPABASE_A_PORT=5432
SUPABASE_A_USER=...
SUPABASE_A_DB=postgres
SUPABASE_A_PASSWORD=...

SUPABASE_B_HOST=...
SUPABASE_B_PORT=5432
SUPABASE_B_USER=...
SUPABASE_B_DB=postgres
SUPABASE_B_PASSWORD=...
```

The registry uses `runtime_port: 6543` when configuring n8n itself, so n8n continues to use the Supabase Transaction Pooler.

### n8n Render slots

```text
RENDER_A_API_KEY=...
RENDER_A_SERVICE_ID=srv-...
RENDER_A_BASE_URL=https://n8n-fbal.onrender.com

RENDER_B_API_KEY=...
RENDER_B_SERVICE_ID=srv-...
RENDER_B_BASE_URL=https://n8n-hbek.onrender.com
```

Add more slots by adding more registry entries and corresponding env pointers.

The Render API key needs permission to read the service, add/update service env vars, suspend/resume the service, and trigger/read deploys.

The control plane updates only the DB/B2 variables listed in the registry maps. It never updates `N8N_ENCRYPTION_KEY`; the same encryption key must remain on every n8n Render slot.

### SH01 Render slots

Add service IDs/API keys/base URLs for the SH01 backends you want visible to the control plane:

```text
SH01_A_API_KEY=...
SH01_A_SERVICE_ID=srv-...
SH01_A_BASE_URL=https://sh01-fastapi-server.onrender.com

SH01_B_API_KEY=...
SH01_B_SERVICE_ID=srv-...
SH01_B_BASE_URL=https://sh01.onrender.com
```

The n8n stack failover endpoint does not modify or cut over the SH01 ring. SH01 is modeled separately so it can be monitored/control-routed without accidentally coupling it to n8n.

### B2 slots

Your existing account can remain `b2-a`:

```text
BACKBLAZE_KEY_ID=...
BACKBLAZE_APPLICATION_KEY=...
BACKBLAZE_BUCKET_NAME=...
```

Second account/bucket:

```text
B2_B_KEY_ID=...
B2_B_APPLICATION_KEY=...
B2_B_BUCKET_NAME=...
```

For mirroring, source credentials need list/read; destination needs list/read/write. `prune_extra=true` additionally needs delete.

### Durable startup defaults

```text
ACTIVE_N8N_RENDER_SLOT=render-a
ACTIVE_SH01_RENDER_SLOT=sh01-a
ACTIVE_POSTGRES_SLOT=supabase-b
ACTIVE_B2_SLOT=b2-a
```

`ACTIVE_RENDER_SLOT` is still accepted as a backward-compatible fallback for the n8n role.

## 4. INFRA_REGISTRY_JSON

Use `infra_registry.example.json` as the shape and map the env pointer names to your actual variables.

The registry contains env-variable names, not secret values.

## 5. Upgrade the existing n8n Cloudflare Worker

Do not replace its failover engine with a generic proxy. Use `cloudflare/n8n-prod-1-worker.js`, which is your existing implementation plus the control layer.

Keep Worker var `BACKENDS` as a comma-separated list of every trusted n8n Render origin:

```text
BACKENDS=https://n8n-fbal.onrender.com,https://n8n-hbek.onrender.com
```

You can add C/D/E later by appending them.

Create/bind a Workers KV namespace as:

```text
ROUTER_STATE
```

Set Worker secret:

```text
ROUTER_CONTROL_KEY=<strong random secret>
```

The new protected endpoints are:

```text
GET  /__control/status
POST /__control/preferred
POST /__control/maintenance
```

`POST /__control/preferred` accepts only an origin already present in `BACKENDS`. It cannot turn the Worker into an arbitrary proxy.

Example payload:

```json
{"backend":"https://n8n-hbek.onrender.com"}
```

The effective proxy order becomes the preferred backend first, followed by every other configured backend in its original order. All original Render suspension detection, retry statuses, body replay, headers, and 30-second timeout remain.

Maintenance mode deliberately returns HTTP 503 for normal n8n traffic. The orchestrator uses this while suspending the active n8n writer and copying Postgres, preventing the Worker's normal failover behavior from sending writes to a stale standby during migration.

## 6. Upgrade the existing SH01 Cloudflare Worker

Use `cloudflare/sh01-worker.js`.

It preserves the SH01-specific behavior you already have, including `TIMEOUT_MS`, `x-forwarded-host`, and `x-failover-attempts`, while adding the same protected preferred/maintenance control contract.

Use a separate KV namespace for SH01. Both namespaces may use the binding name `ROUTER_STATE` because they belong to different Worker deployments.

Set its own secret `ROUTER_CONTROL_KEY`.

## 7. Master router-control variables

On the master Render service:

```text
N8N_ROUTER_CONTROL_BASE_URL=https://n8n-prod-1.vivojaymail.workers.dev
N8N_ROUTER_CONTROL_KEY=<n8n Worker secret>

SH01_ROUTER_CONTROL_BASE_URL=https://sh01.vivojaymail.workers.dev
SH01_ROUTER_CONTROL_KEY=<sh01 Worker secret>
```

The legacy `ROUTER_CONTROL_BASE_URL` / `ROUTER_CONTROL_KEY` are accepted only as n8n fallbacks so the first package does not immediately break, but migrate to the role-specific variables above.

## 8. Safe n8n failover sequence

A live `/infra/failover` performs:

```text
preflight
-> n8n Worker maintenance ON
-> snapshot all registered n8n Render service states
-> suspend EVERY currently running n8n Render service
-> verify zero registered n8n writers remain running
-> transactional Postgres public-schema migration
-> exact table-set and row-count comparison
-> B2 compare/mirror if configured
-> update target n8n DB runtime to target Supabase transaction pooler :6543
-> update target n8n B2 runtime if configured
-> resume target n8n Render service if suspended
-> deploy target n8n
-> wait for target /healthz/readiness
-> set n8n Worker preferred backend to target
-> persist durable active state while ingress is still frozen
-> n8n Worker maintenance OFF LAST
```

If the job fails after ingress is frozen, rollback restores the exact pre-failover n8n Render running/suspended topology, restores the old preferred backend, and disables maintenance. Any target that was started by the failed cutover but was previously suspended is suspended again. Durable active state is not committed until the entire cutover succeeds.

Only one full failover can run at a time in the process.

## 9. Deploy the repo overlay

After copying the corrected package into your existing working tree:

```powershell
git status
git diff
```

Then stage the complete control-plane change:

```powershell
git add `
  infra/models.py `
  infra/routes.py `
  infra/state.py `
  infra/registry.py `
  infra/persistence.py `
  infra/jobs.py `
  infra/runtime_config.py `
  infra/render_provider.py `
  infra/b2ring.py `
  infra/router_client.py `
  infra/orchestrator.py `
  infra_registry.example.json `
  scripts/infra_smoke.ps1 `
  CONTROL_PLANE_SETUP.md `
  cloudflare/n8n-prod-1-worker.js `
  cloudflare/sh01-worker.js `
  cloudflare/wrangler.n8n.example.jsonc `
  cloudflare/wrangler.sh01.example.jsonc

git diff --cached
git commit -m "feat: complete resilient infrastructure control plane"
git push
```

The Cloudflare dashboard Worker code/KV/secret changes are deployed separately from the GitHub repo unless you already deploy those Workers with Wrangler.

## 10. Comprehensive non-destructive test

After the master deploys and both Workers are upgraded:

```powershell
.\scripts\infra_smoke.ps1 -Base $BASE -Key $KEY
```

It checks:

- control-plane state
- every Postgres slot
- n8n and SH01 Render rings
- every B2 slot
- n8n router KV/control state
- SH01 router control state when configured
- full failover plan without executing it

Suspended standby Render slots may fail direct health and are warnings rather than test failures.

## 11. Full live test

When the dry run is correct:

```powershell
.\scripts\infra_smoke.ps1 `
  -Base $BASE `
  -Key $KEY `
  -ExecuteFailover `
  -SyncB2
```

Do not add `-PruneB2Extra` unless destination-only B2 objects should actually be permanently deleted.

During the Postgres migration the public n8n Worker intentionally serves maintenance 503s, because allowing writes during a database snapshot would make an exact cutover unsafe.

## 12. Useful API calls

Current state:

```powershell
Invoke-RestMethod -Method GET -Uri "$BASE/infra/status" -Headers @{ "X-API-KEY" = $KEY } | ConvertTo-Json -Depth 20
```

n8n router:

```powershell
Invoke-RestMethod -Method GET -Uri "$BASE/infra/router/n8n/status" -Headers @{ "X-API-KEY" = $KEY } | ConvertTo-Json -Depth 20
```

SH01 router:

```powershell
Invoke-RestMethod -Method GET -Uri "$BASE/infra/router/sh01/status" -Headers @{ "X-API-KEY" = $KEY } | ConvertTo-Json -Depth 20
```

Dry-run next n8n stack hop:

```powershell
$body = @{
  sync_b2 = $true
  prune_b2_extra = $false
  switch_router = $true
  quiesce_source = $true
  dry_run = $true
} | ConvertTo-Json

Invoke-RestMethod `
  -Method POST `
  -Uri "$BASE/infra/failover" `
  -Headers @{ "X-API-KEY" = $KEY } `
  -ContentType "application/json" `
  -Body $body | ConvertTo-Json -Depth 30
```

A live request returns a job ID. Poll `/infra/jobs/{job_id}`.

## 13. Boundaries / known separate issue

The stale Redis cron-master hostname in `app.py` is separate from this control plane. This overlay does not change it.

The Postgres migration still copies the `public` schema only, matching the migration you already proved. If you later depend on additional Supabase-managed schemas, define a separate migration policy before assuming they move with this failover.
