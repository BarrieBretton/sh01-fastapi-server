# Paired Render slot migration

This overlay changes the control-plane model from independent n8n/SH01 Render rings to paired Render slots:

```text
render-a = [n8n-a, sh01-a]
render-b = [n8n-b, sh01-b]
```

`ACTIVE_RENDER_SLOT` is now the single Render bootstrap/default. A successful failover switches both Cloudflare preferred backends to the same paired slot and persists one Render slot in durable state.

## Master Render env names

Use one Render account/workspace API key:

```text
RENDER_API_KEY=<Render API key>
```

Per-service IDs and URLs:

```text
RENDER_A_N8N_SERVICE_ID=<srv-... for n8n-a>
RENDER_A_N8N_BASE_URL=https://n8n-fbal.onrender.com
RENDER_A_SH01_SERVICE_ID=<srv-... for sh01-a>
RENDER_A_SH01_BASE_URL=https://sh01-fastapi-server.onrender.com

RENDER_B_N8N_SERVICE_ID=<srv-... for n8n-b>
RENDER_B_N8N_BASE_URL=https://n8n-hbek.onrender.com
RENDER_B_SH01_SERVICE_ID=<srv-... for sh01-b>
RENDER_B_SH01_BASE_URL=https://sh01.onrender.com

ACTIVE_RENDER_SLOT=render-a-or-render-b
```

Keep:

```text
ACTIVE_POSTGRES_SLOT=...
ACTIVE_B2_SLOT=...
N8N_ROUTER_CONTROL_BASE_URL=...
N8N_ROUTER_CONTROL_KEY=...
SH01_ROUTER_CONTROL_BASE_URL=...
SH01_ROUTER_CONTROL_KEY=...
```

Use `infra_registry.example.json` as the new `INFRA_REGISTRY_JSON` shape.

## Failover behavior

A paired failover:

1. Preflights Postgres, both Render services, and both Cloudflare routers.
2. Enables maintenance on both Workers.
3. Suspends every running n8n service and verifies zero n8n writers remain.
4. Migrates/verifies Postgres and optionally B2.
5. Writes target n8n runtime DB/B2 config.
6. Resumes/deploys/health-checks target n8n.
7. Resumes/deploys/health-checks target SH01.
8. Sets both Worker preferred backends to the corresponding target services.
9. Persists one active Render slot.
10. Disables maintenance on both Workers.

Rollback restores target n8n env, both service running topologies, both router preferences, and durable state.

## Legacy compatibility

Old persisted state like:

```json
{"render":{"n8n":"render-b","sh01":"sh01-a"}}
```

is normalized to the n8n side (`render-b`) because the new model requires one paired slot. Old `ACTIVE_N8N_RENDER_SLOT` / `ACTIVE_SH01_RENDER_SLOT` are accepted only as bootstrap fallbacks when `ACTIVE_RENDER_SLOT` is absent.
