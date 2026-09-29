# Final comprehensive test checklist

Run this only after the corrected repo overlay is deployed and both Cloudflare Workers have their new control layer + KV binding.

1. `GET /infra/health` returns 200 and nested render state.
2. `GET /infra/status` shows `render.n8n`, `render.sh01`, Postgres and B2 slots.
3. Every Postgres health endpoint succeeds.
4. `POST /infra/postgres/{active}/verify` succeeds.
5. Every Render service-status endpoint returns its Render suspension state.
6. Active n8n Render direct readiness returns healthy; suspended standbys may not.
7. Every B2 health endpoint succeeds.
8. `/infra/router/n8n/status` shows `persistence_available=true`, `maintenance=false`, all configured BACKENDS, and the expected preferred/effective order.
9. `/infra/router/sh01/status` shows the analogous SH01 routing state.
10. Failover dry-run selects only a role=n8n Render target and never a SH01 target.
11. Live failover enters n8n maintenance, records every n8n Render service state, suspends every running n8n slot, and verifies zero registered n8n writers remain running before Postgres migration.
12. Postgres migration and compare report an exact match.
13. B2 compare/mirror succeeds if enabled.
14. Target Render receives DB port 6543 runtime config and preserves `N8N_ENCRYPTION_KEY` because the control plane never writes that variable.
15. Target n8n readiness is 2xx before Cloudflare preferred backend changes.
16. Only the selected target n8n Render slot is resumed/deployed for the new active stack; Worker preferred backend becomes that target and maintenance returns to false.
17. Durable state commits the target Postgres/B2/n8n Render while ingress is still in maintenance, before maintenance is disabled.
18. n8n UI loads, workflows are present, credentials decrypt, and active workflows start normally.
19. Trigger a harmless manual workflow and confirm an execution is written to the new active Postgres.
20. Restart the master naturally later and verify `/infra/status` reloads the same durable active state.

21. Rollback test: force a harmless failure before router commit and verify the exact pre-failover running/suspended n8n Render topology is restored.
22. SH01 Render health uses `/` (public 2xx) rather than protected `/infra/health`.
