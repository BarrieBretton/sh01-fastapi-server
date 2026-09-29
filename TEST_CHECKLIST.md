# Control Plane Test Checklist

Use this checklist after the paired Render control-plane code is deployed and the two Cloudflare Workers have:

- the current Worker code,
- `ROUTER_CONTROL_KEY` configured as a Secret,
- dedicated KV namespaces bound as `ROUTER_STATE`,
- complete `BACKENDS` lists.

The authoritative Render model is:

```text
render-a = [n8n-a, sh01-a]
render-b = [n8n-b, sh01-b]
```

Do not use older independent n8n/SH01 ring assumptions.

---

## Phase 1 - Static validation

- [ ] `python -m compileall .\infra` completes successfully.
- [ ] `python -c "import infra.orchestrator; print('orchestrator import OK')"` prints `orchestrator import OK`.
- [ ] `INFRA_REGISTRY_JSON` parses successfully.
- [ ] `INFRA_REGISTRY_JSON` has paired `render-*` entries containing both `n8n` and `sh01`.
- [ ] Registry contains only actually configured Postgres/B2/Render slots.
- [ ] No real secrets are committed to the repo.
- [ ] `ACTIVE_RENDER_SLOT` is present.
- [ ] Legacy `ACTIVE_N8N_RENDER_SLOT` and `ACTIVE_SH01_RENDER_SLOT` are not required for the new setup.

---

## Phase 2 - Environment checks

- [ ] `CONTROL_PLANE_API_KEY` exists.
- [ ] `RENDER_API_KEY` exists.
- [ ] Every registered Render service has a real `srv-...` ID.
- [ ] Every registered Render service has the correct base URL.
- [ ] Supabase migration ports are `5432`.
- [ ] Registry `runtime_port` is `6543` for n8n runtime.
- [ ] `ACTIVE_POSTGRES_SLOT` references a registered Postgres slot.
- [ ] `ACTIVE_B2_SLOT` references a registered B2 slot.
- [ ] `ACTIVE_RENDER_SLOT` references a registered paired Render slot.

Current expected bootstrap state before the first paired live failover:

```text
ACTIVE_RENDER_SLOT=render-a
ACTIVE_POSTGRES_SLOT=supabase-b
ACTIVE_B2_SLOT=b2-a
```

---

## Phase 3 - Cloudflare checks

### n8n Worker

- [ ] `BACKENDS` contains every trusted n8n Render origin.
- [ ] `ROUTER_CONTROL_KEY` is configured as a Secret.
- [ ] `ROUTER_STATE` is bound to a dedicated n8n KV namespace.
- [ ] Worker normal traffic still proxies successfully.

Current A/B origins:

```text
https://n8n-fbal.onrender.com
https://n8n-hbek.onrender.com
```

### SH01 Worker

- [ ] `BACKENDS` contains every trusted SH01 Render origin.
- [ ] `TIMEOUT_MS=90000`.
- [ ] `ROUTER_CONTROL_KEY` is configured as a Secret.
- [ ] `ROUTER_STATE` is bound to a dedicated SH01 KV namespace.
- [ ] Worker normal traffic still proxies successfully.

Current A/B origins:

```text
https://sh01-fastapi-server.onrender.com
https://sh01.onrender.com
```

- [ ] The n8n and SH01 Workers do not share the same KV namespace.

---

## Phase 4 - Public routing baseline

Before the first live failover, verify the current pair.

```powershell
curl.exe -sS -D - -o NUL "https://n8n-prod-1.vivojaymail.workers.dev/" |
  Select-String -Pattern "(?i)^x-backend-used:"

curl.exe -sS -D - -o NUL "https://sh01.vivojaymail.workers.dev/" |
  Select-String -Pattern "(?i)^x-backend-used:"
```

Current expected A-pair result:

```text
n8n  -> https://n8n-fbal.onrender.com
sh01 -> https://sh01-fastapi-server.onrender.com
```

- [ ] Both Workers point to the same Render slot letter before bootstrapping paired state.

---

## Phase 5 - Control-plane read endpoints

Set:

```powershell
$BASE = "https://sh01-fastapi-server-mtwu.onrender.com"
$KEY = "<CONTROL_PLANE_API_KEY>"
```

### Health

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/health" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

- [ ] Returns HTTP 200.
- [ ] Active state is readable.
- [ ] Render state is one paired slot string, not separate n8n/SH01 active slots.

### Full status

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

- [ ] Shows registered paired Render slots.
- [ ] Shows registered Postgres slots.
- [ ] Shows registered B2 slots.
- [ ] Shows the expected current active state.

---

## Phase 6 - Router-control endpoints

### n8n

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/n8n/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

Expected:

```text
persistence_available=true
maintenance=false
```

- [ ] `configured` contains all n8n Worker backends.
- [ ] preferred/effective ordering is sensible.

### SH01

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "$BASE/infra/router/sh01/status" `
  -Headers @{ "X-API-KEY" = $KEY } |
  ConvertTo-Json -Depth 20
```

Expected:

```text
persistence_available=true
maintenance=false
```

- [ ] `configured` contains all SH01 Worker backends.
- [ ] preferred/effective ordering is sensible.

---

## Phase 7 - Provider health checks

### Postgres

- [ ] Every registered Postgres slot health check succeeds.
- [ ] Active Postgres verification succeeds.
- [ ] Target Postgres is reachable before migration.

### Render

For every paired Render slot:

- [ ] n8n service status can be read through Render API.
- [ ] SH01 service status can be read through Render API.
- [ ] Active n8n direct readiness is healthy.
- [ ] Active SH01 direct `/` health is healthy.
- [ ] Suspended standby services are correctly recognized as suspended.

A suspended standby may fail direct HTTP health; that alone is not a test failure.

### B2

- [ ] Every registered B2 slot health check succeeds.
- [ ] No unconfigured B2 slot is present in the registry.

---

## Phase 8 - Non-destructive smoke test

Run:

```powershell
.\scripts\infra_smoke.ps1 -Base $BASE -Key $KEY
```

- [ ] Script completes without destructive actions.
- [ ] It checks control-plane state.
- [ ] It checks Postgres slots.
- [ ] It checks paired Render services.
- [ ] It checks B2 slots.
- [ ] It checks both routers.
- [ ] It produces a failover plan without executing it.
- [ ] Planned Render target is one paired slot.
- [ ] Planned cutover includes both n8n and SH01.
- [ ] No live failover has happened yet.

---

## Phase 9 - Dry-run failover plan

The plan should include, in order, the equivalent of:

```text
preflight
n8n_router_maintenance_on
sh01_router_maintenance_on
suspend_all_running_n8n
suspend_all_running_sh01
postgres_migrate_and_compare
b2_sync_or_compare
configure_target_n8n
configure_target_sh01
resume_deploy_healthcheck_target_n8n
resume_deploy_healthcheck_target_sh01
set_n8n_router_preferred
set_sh01_router_preferred
persist_active_state
n8n_router_maintenance_off
sh01_router_maintenance_off
```

- [ ] Changing Render slots without `switch_router=true` is rejected.
- [ ] Changing Postgres slots without `quiesce_source=true` is rejected.
- [ ] Failover refuses to start if either router is already in maintenance mode.

---

## Phase 10 - First live paired failover

Only proceed after all previous phases pass.

Example:

```powershell
.\scripts\infra_smoke.ps1 `
  -Base $BASE `
  -Key $KEY `
  -ExecuteFailover `
  -SyncB2
```

Do not use `-PruneB2Extra` during the first live test.

During failover verify:

- [ ] n8n Worker enters maintenance.
- [ ] SH01 Worker enters maintenance.
- [ ] Every registered running n8n service is suspended.
- [ ] Zero registered n8n services remain running before Postgres migration.
- [ ] Every registered running SH01 service is suspended.
- [ ] Zero registered SH01 services remain running before target startup.
- [ ] Postgres migration succeeds.
- [ ] Postgres comparison reports an exact match.
- [ ] B2 compare/mirror succeeds if enabled.
- [ ] Target n8n receives DB runtime port `6543`.
- [ ] Target n8n receives target B2 runtime when configured.
- [ ] Target SH01 receives target B2 runtime when configured.
- [ ] Target n8n deploy becomes live.
- [ ] Target n8n `/healthz/readiness` is 2xx.
- [ ] Target SH01 deploy becomes live.
- [ ] Target SH01 `/` health is 2xx.
- [ ] n8n Worker preferred backend changes to target n8n.
- [ ] SH01 Worker preferred backend changes to target SH01.
- [ ] Durable state commits one paired Render slot.
- [ ] Durable Postgres/B2 state is correct.
- [ ] n8n maintenance is disabled only after state/router commit.
- [ ] SH01 maintenance is disabled only after state/router commit.

---

## Phase 11 - Functional verification after cutover

- [ ] n8n UI loads.
- [ ] Workflows are present.
- [ ] Credentials decrypt.
- [ ] Active workflows start normally.
- [ ] Run a harmless manual workflow.
- [ ] Confirm the execution is written to the new active Postgres.
- [ ] SH01 public endpoints respond through the Worker.
- [ ] Worker headers show the new paired Render slot.

---

## Phase 12 - Durable-state restart test

After a successful failover:

- [ ] Restart/redeploy the master naturally.
- [ ] `GET /infra/status` reloads the same persisted active Render/Postgres/B2 state.
- [ ] No manual edit to `ACTIVE_RENDER_SLOT` was required.
- [ ] Both Workers still prefer the same paired Render slot.

---

## Phase 13 - Rollback test

Force a harmless failure after maintenance begins but before final completion.

Verify rollback:

- [ ] turns both Workers to maintenance,
- [ ] re-quiesces running n8n services,
- [ ] re-quiesces running SH01 services,
- [ ] restores target n8n runtime env if changed,
- [ ] restores target SH01 runtime env if changed,
- [ ] redeploys restored target services when necessary,
- [ ] restores the exact pre-failover n8n running set,
- [ ] restores the exact pre-failover SH01 running set,
- [ ] restores the previous n8n preferred backend,
- [ ] restores the previous SH01 preferred backend,
- [ ] restores durable state if already changed,
- [ ] disables maintenance only after rollback completes.

---

## Phase 14 - Expansion test for a new slot

Whenever adding `render-c`, `supabase-c`, `b2-c`, etc.:

- [ ] create provider resource,
- [ ] add required env vars,
- [ ] add registry entry,
- [ ] update Worker `BACKENDS` if Render,
- [ ] redeploy/restart master,
- [ ] run health/status/router checks,
- [ ] run non-destructive smoke test,
- [ ] inspect dry-run target selection,
- [ ] only then allow live failover.

---

## Known limitations to remember during testing

- Full failover lock is process-local.
- Workers KV is eventually consistent.
- Postgres migration currently covers the `public` schema.
- `prune_extra` is destructive.
- SH01 health assumes `/` returns 2xx.
- Target n8n may begin workload execution once resumed/deployed before public cutover.
