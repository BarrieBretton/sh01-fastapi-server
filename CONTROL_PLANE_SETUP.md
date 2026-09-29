# SH01 Control Plane Setup

This document describes the **current paired Render-slot control-plane architecture**.

> Historical note: an earlier version modeled n8n and SH01 as independent Render rings. That model is obsolete. The authoritative model is now:
>
> ```text
> render-a = [n8n-a, sh01-a]
> render-b = [n8n-b, sh01-b]
> ```

The two Cloudflare Workers currently used are:

- `https://n8n-prod-1.vivojaymail.workers.dev/`
- `https://sh01.vivojaymail.workers.dev/`

The Workers preserve their existing proxy/failover behavior while adding:

- preferred-backend control,
- maintenance mode,
- KV-backed router state,
- authenticated control endpoints.

---

## 1. Paired Render topology

Current topology:

```text
render-a
├── n8n-a   -> https://n8n-fbal.onrender.com
└── sh01-a  -> https://sh01-fastapi-server.onrender.com

render-b
├── n8n-b   -> https://n8n-hbek.onrender.com
└── sh01-b  -> https://sh01.onrender.com
```

Future slots follow the same pattern:

```text
render-c
├── n8n-c
└── sh01-c
```

The control plane stores a single active Render slot:

```json
{
  "render": "render-a",
  "postgres": "supabase-b",
  "b2": "b2-a"
}
```

The Render, Postgres, and B2 slot names are independent. `render-a` does not imply `supabase-a` or `b2-a`.

---

## 2. Repository files

The control-plane implementation lives primarily in:

```text
infra/models.py
infra/routes.py
infra/state.py
infra/registry.py
infra/persistence.py
infra/jobs.py
infra/runtime_config.py
infra/render_provider.py
infra/postgres.py
infra/b2ring.py
infra/router_client.py
infra/orchestrator.py

infra_registry.example.json

scripts/infra_smoke.ps1

cloudflare/n8n-prod-1-worker.js
cloudflare/sh01-worker.js
cloudflare/wrangler.n8n.example.jsonc
cloudflare/wrangler.sh01.example.jsonc
```

`app.py` already includes `infra_router`.

---

## 3. Master Render environment

### Control-plane auth

```text
CONTROL_PLANE_API_KEY=<strong secret>
```

The `/infra/*` API uses:

```text
X-API-KEY: <CONTROL_PLANE_API_KEY>
```

### Bootstrap state

Current bootstrap:

```text
ACTIVE_RENDER_SLOT=render-a
ACTIVE_POSTGRES_SLOT=supabase-b
ACTIVE_B2_SLOT=b2-a
```

These are only bootstrap/fallback values. Successful failovers persist durable state, which becomes authoritative across restarts.

Legacy variables such as:

```text
ACTIVE_N8N_RENDER_SLOT
ACTIVE_SH01_RENDER_SLOT
```

are obsolete for new configuration.

---

## 4. Render API configuration

Use one Render account/workspace API key:

```text
RENDER_API_KEY=<rnd_...>
```

The current paired services use:

```text
RENDER_A_N8N_SERVICE_ID=<srv-...>
RENDER_A_N8N_BASE_URL=https://n8n-fbal.onrender.com

RENDER_A_SH01_SERVICE_ID=<srv-...>
RENDER_A_SH01_BASE_URL=https://sh01-fastapi-server.onrender.com

RENDER_B_N8N_SERVICE_ID=<srv-...>
RENDER_B_N8N_BASE_URL=https://n8n-hbek.onrender.com

RENDER_B_SH01_SERVICE_ID=<srv-...>
RENDER_B_SH01_BASE_URL=https://sh01.onrender.com
```

The Render API key is account/workspace-scoped. It is not an n8n-specific or SH01-specific key.

The control plane needs permission to:

- inspect services,
- read/update service environment variables,
- suspend/resume services,
- trigger deploys,
- inspect deploy status.

The control plane never rewrites `N8N_ENCRYPTION_KEY`; keep the same encryption key on all n8n slots.

---

## 5. Supabase/Postgres configuration

The control plane uses port `5432` for migration/control-plane access:

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

The n8n runtime uses Supabase's transaction-pooler port:

```text
6543
```

This is represented in the registry as:

```json
"runtime_port": 6543
```

So:

```text
migration/control-plane connection -> 5432
n8n runtime connection             -> 6543
```

The current Postgres migration logic covers the `public` schema.

---

## 6. Backblaze B2 configuration

Current B2 A slot:

```text
BACKBLAZE_KEY_ID=...
BACKBLAZE_APPLICATION_KEY=...
BACKBLAZE_BUCKET_NAME=...
```

A future B2 B slot may use:

```text
B2_B_KEY_ID=...
B2_B_APPLICATION_KEY=...
B2_B_BUCKET_NAME=...
```

Do not register a B2 slot in `INFRA_REGISTRY_JSON` until its required env vars actually exist.

For mirroring:

```text
source      -> list/read
destination -> list/read/write
```

If `prune_extra=true` is used, destination delete permission is also required.

`prune_extra` is destructive.

---

## 7. Cloudflare router control

### n8n Worker

Worker:

```text
https://n8n-prod-1.vivojaymail.workers.dev
```

Normal variable:

```text
BACKENDS=https://n8n-fbal.onrender.com,https://n8n-hbek.onrender.com
```

Secret:

```text
ROUTER_CONTROL_KEY=<strong random secret>
```

Dedicated KV namespace bound as:

```text
ROUTER_STATE
```

### SH01 Worker

Worker:

```text
https://sh01.vivojaymail.workers.dev
```

Normal variables:

```text
BACKENDS=https://sh01-fastapi-server.onrender.com,https://sh01.onrender.com
TIMEOUT_MS=90000
```

Secret:

```text
ROUTER_CONTROL_KEY=<different strong random secret>
```

Dedicated KV namespace bound as:

```text
ROUTER_STATE
```

Use separate KV namespaces for the two Workers.

### Master router-control env vars

```text
N8N_ROUTER_CONTROL_BASE_URL=https://n8n-prod-1.vivojaymail.workers.dev
N8N_ROUTER_CONTROL_KEY=<same secret as n8n Worker>

SH01_ROUTER_CONTROL_BASE_URL=https://sh01.vivojaymail.workers.dev
SH01_ROUTER_CONTROL_KEY=<same secret as SH01 Worker>
```

The Worker control endpoints are:

```text
GET  /__control/status
POST /__control/preferred
POST /__control/maintenance
```

`POST /__control/preferred` only accepts a backend already present in `BACKENDS`.

---

## 8. `INFRA_REGISTRY_JSON`

The registry stores **env-var names**, not secret values.

Current shape:

```json
{
  "render": {
    "render-a": {
      "api_key_env": "RENDER_API_KEY",
      "n8n": {
        "service_id_env": "RENDER_A_N8N_SERVICE_ID",
        "base_url_env": "RENDER_A_N8N_BASE_URL",
        "health_path": "/healthz/readiness"
      },
      "sh01": {
        "service_id_env": "RENDER_A_SH01_SERVICE_ID",
        "base_url_env": "RENDER_A_SH01_BASE_URL",
        "health_path": "/"
      }
    },
    "render-b": {
      "api_key_env": "RENDER_API_KEY",
      "n8n": {
        "service_id_env": "RENDER_B_N8N_SERVICE_ID",
        "base_url_env": "RENDER_B_N8N_BASE_URL",
        "health_path": "/healthz/readiness"
      },
      "sh01": {
        "service_id_env": "RENDER_B_SH01_SERVICE_ID",
        "base_url_env": "RENDER_B_SH01_BASE_URL",
        "health_path": "/"
      }
    }
  },
  "postgres": {
    "supabase-a": {
      "host_env": "SUPABASE_A_HOST",
      "port_env": "SUPABASE_A_PORT",
      "user_env": "SUPABASE_A_USER",
      "database_env": "SUPABASE_A_DB",
      "password_env": "SUPABASE_A_PASSWORD",
      "runtime_port": 6543
    },
    "supabase-b": {
      "host_env": "SUPABASE_B_HOST",
      "port_env": "SUPABASE_B_PORT",
      "user_env": "SUPABASE_B_USER",
      "database_env": "SUPABASE_B_DB",
      "password_env": "SUPABASE_B_PASSWORD",
      "runtime_port": 6543
    }
  },
  "b2": {
    "b2-a": {
      "key_id_env": "BACKBLAZE_KEY_ID",
      "application_key_env": "BACKBLAZE_APPLICATION_KEY",
      "bucket_name_env": "BACKBLAZE_BUCKET_NAME"
    }
  }
}
```

In the Render dashboard, store raw JSON directly. Do not surround the value with literal single quotes.

---

## 9. Adding a new paired Render slot

Example: `render-c`.

Create:

```text
n8n-c
sh01-c
```

Add:

```text
RENDER_C_N8N_SERVICE_ID=<srv-...>
RENDER_C_N8N_BASE_URL=https://<n8n-c>.onrender.com

RENDER_C_SH01_SERVICE_ID=<srv-...>
RENDER_C_SH01_BASE_URL=https://<sh01-c>.onrender.com
```

Add to the registry:

```json
"render-c": {
  "api_key_env": "RENDER_API_KEY",
  "n8n": {
    "service_id_env": "RENDER_C_N8N_SERVICE_ID",
    "base_url_env": "RENDER_C_N8N_BASE_URL",
    "health_path": "/healthz/readiness"
  },
  "sh01": {
    "service_id_env": "RENDER_C_SH01_SERVICE_ID",
    "base_url_env": "RENDER_C_SH01_BASE_URL",
    "health_path": "/"
  }
}
```

Append the new n8n origin to the n8n Worker's `BACKENDS`.

Append the new SH01 origin to the SH01 Worker's `BACKENDS`.

No code change should be required.

---

## 10. Adding a new Supabase/Postgres slot

Example: `supabase-c`.

Add:

```text
SUPABASE_C_HOST=...
SUPABASE_C_PORT=5432
SUPABASE_C_USER=...
SUPABASE_C_DB=postgres
SUPABASE_C_PASSWORD=...
```

Registry:

```json
"supabase-c": {
  "host_env": "SUPABASE_C_HOST",
  "port_env": "SUPABASE_C_PORT",
  "user_env": "SUPABASE_C_USER",
  "database_env": "SUPABASE_C_DB",
  "password_env": "SUPABASE_C_PASSWORD",
  "runtime_port": 6543
}
```

Then redeploy/restart the master and run the non-destructive smoke test.

---

## 11. Adding a new B2 slot

Example: `b2-c`.

Add:

```text
B2_C_KEY_ID=...
B2_C_APPLICATION_KEY=...
B2_C_BUCKET_NAME=...
```

Registry:

```json
"b2-c": {
  "key_id_env": "B2_C_KEY_ID",
  "application_key_env": "B2_C_APPLICATION_KEY",
  "bucket_name_env": "B2_C_BUCKET_NAME"
}
```

Validate permissions before live sync/failover.

---

## 12. Paired failover sequence

A live paired failover performs:

```text
preflight
-> verify source/target Postgres
-> snapshot all n8n and SH01 Render service states
-> snapshot target runtime env
-> verify both Cloudflare router-control states
-> refuse to start if either router is already in maintenance
-> n8n Worker maintenance ON
-> SH01 Worker maintenance ON
-> suspend every running n8n Render service
-> verify zero registered n8n services remain running
-> suspend every running SH01 Render service
-> verify zero registered SH01 services remain running
-> migrate/compare Postgres if changing DB slot
-> compare/mirror B2 if configured
-> configure target n8n DB/B2 runtime
-> configure target SH01 B2 runtime
-> resume/deploy/health-check target n8n
-> resume/deploy/health-check target SH01
-> set n8n Worker preferred backend to target n8n
-> set SH01 Worker preferred backend to target SH01
-> persist durable active Render/Postgres/B2 state
-> n8n Worker maintenance OFF
-> SH01 Worker maintenance OFF
```

Both public entry points remain in maintenance until state and routing are committed.

Changing paired Render slots requires `switch_router=true`.

Changing Postgres slots requires `quiesce_source=true`.

---

## 13. Rollback sequence

On failure, rollback attempts to:

```text
both Workers maintenance ON
-> suspend currently-running n8n + SH01 services
-> restore target n8n env
-> restore target SH01 env
-> redeploy restored target services when required
-> restore exact pre-failover n8n running/suspended topology
-> restore exact pre-failover SH01 running/suspended topology
-> restore both Worker preferred backends
-> restore durable state if changed
-> both Workers maintenance OFF
```

---

## 14. Validation

Set:

```powershell
$BASE = "https://sh01-fastapi-server-mtwu.onrender.com"
$KEY = "<CONTROL_PLANE_API_KEY>"
```

Health:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/health" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

Status:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

n8n router:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/n8n/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

SH01 router:

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/sh01/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

---

## 15. Smoke test

Non-destructive:

```powershell
.\scripts\infra_smoke.ps1 -Base $BASE -Key $KEY
```

Do not run live failover until this output has been reviewed.

Live test, only after validation:

```powershell
.\scripts\infra_smoke.ps1 `
  -Base $BASE `
  -Key $KEY `
  -ExecuteFailover `
  -SyncB2
```

Do not add `-PruneB2Extra` unless destination-only B2 objects are intentionally disposable.

---

## 16. Security

Never commit real values for:

```text
CONTROL_PLANE_API_KEY
RENDER_API_KEY
N8N_ROUTER_CONTROL_KEY
SH01_ROUTER_CONTROL_KEY
SUPABASE_*_PASSWORD
BACKBLAZE_APPLICATION_KEY
B2_*_APPLICATION_KEY
```

Cloudflare `ROUTER_CONTROL_KEY` must be a Worker Secret.

---

## 17. Known limitations

- `_FAILOVER_LOCK` is process-local, not distributed.
- Workers KV is eventually consistent.
- Postgres migration currently targets the `public` schema.
- `prune_extra` is destructive.
- SH01 direct health currently depends on `/` returning 2xx.
- The target n8n service may begin workloads once resumed/deployed before public cutover.
- Multi-process control-plane deployment would require stronger distributed locking/state coordination.
