# sh01-fastapi-server

FastAPI application plus a registry-driven infrastructure control plane for running the SH01 stack across multiple infrastructure providers.

The control plane is designed so that new **paired Render slots**, **Supabase/Postgres slots**, and **Backblaze B2 slots** can be added mostly by configuration: create the provider resource, add environment variables, add a registry entry, update Cloudflare `BACKENDS` for new Render origins, redeploy, and run the smoke test.

`A` and `B` are only the slots that exist today; the control plane is intended to scale to `C`, `D`, and beyond without A/B-specific code.

---

## Current architecture

Render is modeled as **paired infrastructure slots**:

```text
render-a
├── n8n-a   -> https://n8n-fbal.onrender.com
└── sh01-a  -> https://sh01-fastapi-server.onrender.com

render-b
├── n8n-b   -> https://n8n-hbek.onrender.com
└── sh01-b  -> https://sh01.onrender.com
```

A future slot follows the same shape:

```text
render-c
├── n8n-c
└── sh01-c
```

A Render failover changes the **whole pair** together. Durable state stores one Render slot:

```json
{
  "render": "render-a",
  "postgres": "supabase-b",
  "b2": "b2-a"
}
```

Render, Postgres, and B2 slots are independent. The letters do not need to match across providers.

---

## Traffic flow

Two Cloudflare Workers front the Render services:

```text
Public n8n traffic
    |
    v
n8n-prod-1.vivojaymail.workers.dev
    |
    +--> preferred n8n Render backend
    +--> fallback n8n Render backends

Public SH01 traffic
    |
    v
sh01.vivojaymail.workers.dev
    |
    +--> preferred SH01 Render backend
    +--> fallback SH01 Render backends
```

Each Worker preserves the existing proxy/failover behavior and exposes a protected control API:

```text
GET  /__control/status
POST /__control/preferred
POST /__control/maintenance
```

Each Worker has:

```text
ROUTER_CONTROL_KEY   # Worker secret
ROUTER_STATE         # KV binding
```

Use separate KV namespaces for the n8n and SH01 Workers.

The master control plane uses:

```text
N8N_ROUTER_CONTROL_KEY
SH01_ROUTER_CONTROL_KEY
```

Each value must exactly match the corresponding Worker's `ROUTER_CONTROL_KEY`.

---

## Current Cloudflare configuration

### n8n Worker

```text
Worker: https://n8n-prod-1.vivojaymail.workers.dev

BACKENDS=https://n8n-fbal.onrender.com,https://n8n-hbek.onrender.com
ROUTER_CONTROL_KEY=<secret>
ROUTER_STATE=<dedicated n8n KV namespace binding>
```

### SH01 Worker

```text
Worker: https://sh01.vivojaymail.workers.dev

BACKENDS=https://sh01-fastapi-server.onrender.com,https://sh01.onrender.com
TIMEOUT_MS=90000
ROUTER_CONTROL_KEY=<different secret>
ROUTER_STATE=<dedicated SH01 KV namespace binding>
```

Do not reuse the same KV namespace for both Workers.

`POST /__control/preferred` only accepts origins already present in that Worker's `BACKENDS` list.

---

## Master control-plane environment variables

### Control-plane authentication

```text
CONTROL_PLANE_API_KEY=<strong secret>
```

Requests to `/infra/*` use:

```text
X-API-KEY: <CONTROL_PLANE_API_KEY>
```

### Bootstrap state

The current Workers were verified to be routing to the A Render pair, so the current bootstrap values are:

```text
ACTIVE_RENDER_SLOT=render-a
ACTIVE_POSTGRES_SLOT=supabase-b
ACTIVE_B2_SLOT=b2-a
```

These are bootstrap/fallback values. Successful failovers persist authoritative durable state, so `ACTIVE_RENDER_SLOT` should not be manually changed after every failover.

### Render API access

Use one Render account/workspace API key:

```text
RENDER_API_KEY=<Render API key>
```

Current paired service variables:

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

A Render API key is account/workspace-level; it is not an n8n-specific or SH01-specific key. The `srv-...` service ID identifies the specific Render service to control.

### Router control

```text
N8N_ROUTER_CONTROL_BASE_URL=https://n8n-prod-1.vivojaymail.workers.dev
N8N_ROUTER_CONTROL_KEY=<same value as n8n Worker ROUTER_CONTROL_KEY>

SH01_ROUTER_CONTROL_BASE_URL=https://sh01.vivojaymail.workers.dev
SH01_ROUTER_CONTROL_KEY=<same value as SH01 Worker ROUTER_CONTROL_KEY>
```

### Supabase/Postgres migration connections

The control plane uses port `5432` for migration tooling:

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

n8n runtime uses the Supabase Transaction Pooler port `6543`, configured through each registry entry's:

```json
"runtime_port": 6543
```

So:

```text
control-plane migration tooling -> 5432
n8n runtime                    -> 6543
```

### Backblaze B2

Current B2 A slot:

```text
BACKBLAZE_KEY_ID=...
BACKBLAZE_APPLICATION_KEY=...
BACKBLAZE_BUCKET_NAME=...
```

Optional future B2 B slot:

```text
B2_B_KEY_ID=...
B2_B_APPLICATION_KEY=...
B2_B_BUCKET_NAME=...
```

Do not register a B2 slot in `INFRA_REGISTRY_JSON` until all required credentials for it actually exist.

---

## `INFRA_REGISTRY_JSON`

The registry stores **environment-variable names**, not secret values.

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

In the Render dashboard, store JSON directly. Do **not** wrap the value in literal single quotes.

---

# Adding new infrastructure

The control plane is intended to make expansion configuration-driven.

## Add a new paired Render slot

Example: `render-c = [n8n-c, sh01-c]`.

### 1. Create both Render services

Create the n8n and SH01 services, then record their `srv-...` IDs and public base URLs.

### 2. Add four master environment variables

```text
RENDER_C_N8N_SERVICE_ID=<srv-...>
RENDER_C_N8N_BASE_URL=https://<n8n-c>.onrender.com
RENDER_C_SH01_SERVICE_ID=<srv-...>
RENDER_C_SH01_BASE_URL=https://<sh01-c>.onrender.com
```

No new Render API key is needed if the existing `RENDER_API_KEY` can manage the new services.

### 3. Add one registry entry

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

### 4. Add both origins to Cloudflare

Append the new n8n origin to the n8n Worker `BACKENDS` list and the new SH01 origin to the SH01 Worker `BACKENDS` list.

This is required because the preferred-backend endpoint refuses arbitrary origins.

### 5. Redeploy and validate

Redeploy/restart the master, run health/status checks, then run the non-destructive smoke test before any live cutover.

---

## Add a new Supabase/Postgres slot

Example: `supabase-c`.

### 1. Add env vars

```text
SUPABASE_C_HOST=...
SUPABASE_C_PORT=5432
SUPABASE_C_USER=...
SUPABASE_C_DB=postgres
SUPABASE_C_PASSWORD=...
```

### 2. Add the registry entry

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

### 3. Validate before failover

The migration connection must work on `5432`. n8n will use `6543` after cutover.

The current migration policy copies/verifies the `public` schema. If future workloads depend on additional schemas, define and test that migration policy first.

---

## Add a new Backblaze B2 slot

Example: `b2-c`.

### 1. Add credentials

```text
B2_C_KEY_ID=...
B2_C_APPLICATION_KEY=...
B2_C_BUCKET_NAME=...
```

### 2. Add the registry entry

```json
"b2-c": {
  "key_id_env": "B2_C_KEY_ID",
  "application_key_env": "B2_C_APPLICATION_KEY",
  "bucket_name_env": "B2_C_BUCKET_NAME"
}
```

### 3. Validate permissions

```text
source      -> list/read
destination -> list/read/write
```

If `prune_extra=true` is used, destination also needs delete permission.

`prune_extra` is destructive; do not enable it unless destination-only objects are intentionally disposable.

---

## Quick-reference expansion checklist

| Provider | Create | Add env vars | Add registry entry | Other required update |
|---|---|---|---|---|
| Render | paired n8n + SH01 services | service ID + base URL for each service | one `render-x` entry containing both services | append both origins to Worker `BACKENDS` |
| Supabase | new project/database | host/port/user/db/password | one `supabase-x` entry | migration port 5432, runtime port 6543 |
| B2 | new account/bucket | key ID/application key/bucket | one `b2-x` entry | validate permissions |

After any addition:

```text
1. update provider resources
2. add env vars
3. update INFRA_REGISTRY_JSON
4. update Worker BACKENDS if a Render slot was added
5. redeploy/restart master
6. GET /infra/health
7. GET /infra/status
8. check both router status endpoints
9. run non-destructive smoke test
10. inspect output
11. only then perform a live failover
```

---

# Paired failover sequence

A full paired failover runs approximately as follows:

```text
PRE-FLIGHT
    |
    +--> verify source + target Postgres health
    +--> snapshot every registered n8n service
    +--> snapshot every registered SH01 service
    +--> snapshot target runtime env
    +--> verify both router states
    +--> refuse to begin if either router is already in maintenance

FREEZE INGRESS
    |
    +--> n8n Worker maintenance ON
    +--> SH01 Worker maintenance ON

QUIESCE
    |
    +--> suspend every running n8n Render service
    +--> verify zero registered n8n services remain running
    +--> suspend every running SH01 Render service
    +--> verify zero registered SH01 services remain running

DATA
    |
    +--> migrate + compare Postgres if changing slots
    +--> compare or mirror B2 if configured

TARGET CONFIGURATION
    |
    +--> configure target n8n DB runtime
    +--> configure target n8n B2 runtime
    +--> configure target SH01 B2 runtime

START TARGET PAIR
    |
    +--> resume/deploy/health-check target n8n
    +--> resume/deploy/health-check target SH01

ROUTER CUTOVER
    |
    +--> n8n Worker preferred -> target n8n origin
    +--> SH01 Worker preferred -> target SH01 origin

COMMIT
    |
    +--> persist one active Render slot
    +--> persist target Postgres
    +--> persist target B2

OPEN INGRESS LAST
    |
    +--> n8n maintenance OFF
    +--> SH01 maintenance OFF
```

Important invariants:

```text
render-a = [n8n-a, sh01-a]
render-b = [n8n-b, sh01-b]
render-c = [n8n-c, sh01-c]
```

Changing paired Render slots requires router switching.

Changing Postgres slots requires quiescing the source.

---

## Rollback behavior

If execution fails, rollback attempts to:

1. place both Workers into maintenance,
2. suspend currently running n8n and SH01 services,
3. restore target n8n runtime env,
4. restore target SH01 runtime env,
5. redeploy restored target services if they were running before failover,
6. restore the exact pre-failover n8n running/suspended topology,
7. restore the exact pre-failover SH01 running/suspended topology,
8. restore both Worker preferred backends to the previous paired Render slot,
9. restore durable control-plane state if it had already changed,
10. disable maintenance after rollback completes.

Target environment values are snapshotted before mutation so rollback can restore them.

---

# Durable state

Current state shape:

```json
{
  "render": "render-a",
  "postgres": "supabase-b",
  "b2": "b2-a"
}
```

Legacy state such as:

```json
{
  "render": {
    "n8n": "render-b",
    "sh01": "sh01-a"
  }
}
```

is normalized to one paired Render slot for compatibility.

`ACTIVE_RENDER_SLOT`, `ACTIVE_POSTGRES_SLOT`, and `ACTIVE_B2_SLOT` are bootstrap defaults, not a manual failover ledger.

---

# Control-plane API and validation

Set:

```powershell
$BASE = "https://sh01-fastapi-server-mtwu.onrender.com"
$KEY = "<CONTROL_PLANE_API_KEY>"
```

## Health

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/health" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

## Full status

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

## n8n router status

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/n8n/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

Expected important fields include:

```text
persistence_available=true
maintenance=false
```

## SH01 router status

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/sh01/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

Expected important fields include:

```text
persistence_available=true
maintenance=false
```

---

# Testing

## Syntax/import checks

```powershell
python -m compileall .\infra
python -c "import infra.orchestrator; print('orchestrator import OK')"
```

Expected:

```text
orchestrator import OK
```

## Non-destructive smoke test

```powershell
.\scripts\infra_smoke.ps1 -Base $BASE -Key $KEY
```

Run this before the first live failover of any newly added infrastructure slot.

Suspended standby Render services can fail direct HTTP health checks because they are intentionally suspended; service-state checks should still identify their Render suspension state.

## Live failover

Only after the smoke/dry-run output is correct:

```powershell
.\scripts\infra_smoke.ps1 `
  -Base $BASE `
  -Key $KEY `
  -ExecuteFailover `
  -SyncB2
```

Do not add:

```text
-PruneB2Extra
```

unless deleting destination-only B2 objects is explicitly intended.

---

# Security rules

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

The registry should contain env-var names, not secret values.

`ROUTER_CONTROL_KEY` must be a Cloudflare Worker **Secret**, not a plaintext Worker variable.

---

# Important implementation details

- One Render API key can be used across all Render services in the permitted account/workspace.
- Render service IDs (`srv-...`) identify specific services.
- Both n8n and SH01 belong to one paired Render slot.
- n8n direct health uses `/healthz/readiness`.
- SH01 direct health currently uses `/`.
- Render deploy requests use `deployMode=deploy_only`.
- `N8N_ENCRYPTION_KEY` is not rewritten by the control plane; keep it identical across n8n slots.
- B2 comparison distinguishes source completeness from destination-only extras.
- `prune_extra` is destructive.
- Postgres migration currently targets the `public` schema.
- Workers KV is eventually consistent; it is not the sole database write-safety barrier.
- The hard migration barrier is that registered n8n writers are suspended and verified before the database copy.

---

# Known limitations

1. The failover lock is process-local. A multi-process deployment or overlapping restart would need a distributed/advisory lock.
2. Standalone mutation jobs and a full failover do not yet share one global distributed mutation lock.
3. Durable-state initialization can perform network/database work during startup.
4. Some state/job reads could become stale if multiple control-plane processes were introduced.
5. Render suspension-state handling assumes the API reports the expected `suspended` / `not_suspended` values.
6. Workers KV is eventually consistent.
7. `prune_extra` can delete destination-only B2 objects.
8. SH01 `/` must continue returning a public 2xx for direct health checks.
9. Postgres migration currently covers only the `public` schema.
10. Target n8n can begin executing workloads once resumed/deployed, before public Worker cutover. Other registered n8n services remain suspended during migration/cutover.

---

# Relevant repository files

```text
infra/
├── models.py
├── routes.py
├── state.py
├── registry.py
├── persistence.py
├── jobs.py
├── runtime_config.py
├── render_provider.py
├── postgres.py
├── b2ring.py
├── router_client.py
└── orchestrator.py

cloudflare/
├── n8n-prod-1-worker.js
├── sh01-worker.js
├── wrangler.n8n.example.jsonc
└── wrangler.sh01.example.jsonc

scripts/
└── infra_smoke.ps1

infra_registry.example.json
```

`infra/postgres.py` contains the already-proven Postgres migration logic.

---

# Historical migration note

The control plane originally modeled n8n and SH01 as independent Render rings. That is no longer the current architecture.

```text
OLD
render.n8n  -> independent n8n slot
render.sh01 -> independent SH01 slot

CURRENT
render-a -> [n8n-a, sh01-a]
render-b -> [n8n-b, sh01-b]
```

The paired model in this README is authoritative.

Older documentation that describes independent n8n/SH01 active Render slots should be treated as historical until updated.

---

## Current bootstrap snapshot

At the time the paired migration was configured:

```text
n8n Worker  -> https://n8n-fbal.onrender.com
SH01 Worker -> https://sh01-fastapi-server.onrender.com

ACTIVE_RENDER_SLOT=render-a
ACTIVE_POSTGRES_SLOT=supabase-b
ACTIVE_B2_SLOT=b2-a
```

Always verify live state after future failovers rather than assuming this snapshot remains current.
