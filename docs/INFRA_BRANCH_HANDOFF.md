# Infrastructure branch handoff: `fix/infra-selection-guard`

**Snapshot:** 20 October 2026. This file describes the branch as a proposed change, not a deployed system. On this date the branch was one commit ahead of `main` (`ba124c2a7e5e90bbf9fcb09b0b92daefe38546f3`, based on `daa29fa1e2f11b39f7b7698784941438215d708e`). Recheck the refs and live services whenever resuming work; these SHAs are historical evidence.

## Why this branch exists

The existing SH01 control plane manages independent infrastructure rings: a **Render slot** is a paired n8n and SH01 service; the **Postgres ring** has two Supabase projects; the **B2 ring** currently has one account and bucket. Earlier, changing active-slot metadata or advancing an omitted ring target could select storage without moving traffic, moving data, or configuring Render services. A planned database switch used a `public` schema dump and restore at failover time, which required an available source and could overwrite the target. Two controller instances had only process-local failover locks. The October 17 SLA hard-stop incident also showed why a failed standby resume must not suspend the currently serving pair.

This branch stages safer selection, shared control-plane coordination, and a **planned one-way** switch to a previously provisioned Supabase logical subscriber. It is a work in progress: committing code is not the same as provisioning its external dependencies or proving production recovery.

## What the branch changes (code-confirmed)

| Area | Implemented behavior | Main files |
| --- | --- | --- |
| Selection | `/infra/select/*` previews or returns a same-slot no-op; a different active slot returns 409 instead of changing metadata alone. A live `/infra/failover` requires an explicit target; omitted ring targets remain on their current slots. | `infra/routes.py`, `infra/orchestrator.py`, `tests/test_selection_guard.py`, `tests/test_independent_ring_targets.py` |
| Render pair | The SLA watchdog considers both services as one candidate. It will not fast-promote a candidate if another registered n8n/SH01 service is not confirmed suspended; otherwise it uses the quiescing failover path or stops safely. An unknown suspension state cannot be treated as stopped. | `infra/sla.py`, `infra/orchestrator.py`, `tests/test_sla_recovery.py` |
| Coordinator | The infra-control Worker adds authenticated `/internal/infra-coordination/{status,acquire,renew,commit,release}` routes backed by a single D1 row. A generation and expiring lease coordinate the controllers. Controller mutations check that lease, and active-slot commits go to D1 first. | `cloudflare/control-plane-worker.js`, `cloudflare/infra_coordination.sql`, `infra/coordination.py`, `infra/state.py`, `infra/router_client.py`, `infra/render_provider.py` |
| Database standby | A setup script can initialize a subscription from a reachable publisher to an **empty** destination. A read-only endpoint inspects publication/table coverage, copy state and lag. A planned switch quiesces Render, writes a source marker, waits to read it on the subscriber, disables the subscription and adjusts owned sequences before restarting services. The old publisher is not automatically reopened after that one-way boundary. The destructive `/infra/postgres/migrate` endpoint now returns 409. | `scripts/setup_postgres_subscription.py`, `infra/pg_replica.py`, `infra/orchestrator.py`, `infra/routes.py`, `tests/test_pg_replica.py` |
| B2 | Omitted B2 targets stay put. The configured single bucket remains a single point of failure; the branch does not create or populate another bucket. | `infra/orchestrator.py`, `docs/INFRA_AUTOMATION_GATES.md` |

The new Worker source is checked into this branch; verify that its deployed dashboard version has the **same** routes, queue behavior and bindings before replacing any live Worker. The existing D1 media queue and router KV have different roles. D1 is the proposed coordinator; router KV is not a lock.

## Observed checks, not a production certification

- The operator ran `python -m unittest discover -s tests -v` on the branch in Windows: **29 tests passed**. Exception tracebacks printed during tests were intentional mocked failures.
- The D1 coordinator's JavaScript test passed in a separate Node-capable development workspace. Node was unavailable on the operator's office PC. Neither test connected to real Supabase, Render, router Workers or production D1.
- The branch was pushed. No PR, merge, production rollout, D1 seed, live subscription, or end-to-end failover drill was verified as of this snapshot. Do not infer deployment status merely from the branch existing.

## Activation gates, in dependency order

1. Confirm the live serving Render pair and actual n8n database via the routers, control-plane status, service environments and database writes. Confirm `b2-a` is still the only active bucket. Do not choose A/B by hostname, old env defaults or this document.
2. Back up both Supabase projects. Check that the intended destination project is empty and that both projects permit a direct logical replication connection. Review `docs/SUPABASE_REPLICA_RUNBOOK.md`; run its setup script **without** `--apply` first (the default is a read-only check), then use `--apply` only after the source/destination and backups are verified. A populated standby needs a different, reviewed migration.
3. Verify subscription table copy, schema parity, publication coverage and freshness with `GET /infra/postgres/replica-status?source=<publisher>&destination=<subscriber>` once the branch is deployed in staging. Arrange ongoing schema/publication maintenance and WAL monitoring. Match `N8N_ENCRYPTION_KEY` on both Render pairs.
4. Deploy the **infra-control** Worker with its existing `MEDIA_QUEUE_DB` binding and `CONTROL_PLANE_API_KEY` secret. Apply `cloudflare/infra_coordination.sql` to that D1 database and insert the one `global` row using **observed** active Render/Postgres/B2 slots. Verify coordinator status without exposing the key. Verify the media queue still works.
5. Configure **both** controller instances with the same `CONTROL_PLANE_BASE_URL`, `CONTROL_PLANE_API_KEY` and `INFRA_COORDINATION_ENABLED=true`; verify router status, controller health and a recent watchdog execution. Stage a planned failover and inspect service, database and job behavior before merging an auto-deploying production branch.

**Deployment trap:** the proposed Python code refuses live infra mutations while `INFRA_COORDINATION_ENABLED` is false or the coordinator is unavailable. A premature merge/deploy can disable automatic Render failover even if the currently serving pair stays online. If the flag is true but D1 is unseeded or inaccessible, controller health can also fail. Do not merge into auto-deploying `main` just to store these files; the pushed branch already preserves them.

## What this branch does not solve

- No guarantee of lossless automatic Postgres promotion when the source has already become unreachable: the write barrier requires a reachable publisher. Do not silently promote a possibly stale project.
- No automatically maintained reverse subscription following a successful promotion. Before switching back, rebuild and verify the former publisher as a new standby.
- The publication covers `public` tables; PostgreSQL logical replication does not automatically copy DDL or sequence state. Other schemas, external writers and Supabase-managed services require separate analysis.
- No B2 redundancy exists while only one bucket is provisioned.
- Router Worker changes use separate KV controls without a shared fencing-generation check. D1 coordinates controller decisions but does not prove end-to-end exactly-one-writer behavior. Verify this limitation in a staging drill before claiming unattended failover.
- No live PR status, deployed revision, worker binding, database plan, registry membership, Supabase subscription, alert route or watchdog schedule is established by this document; recheck each from its source.

## Resume checklist for a new AI or engineer

1. Read `AGENTS.md`, `docs/PROJECT_HANDBOOK.md`, **this file**, `docs/SUPABASE_REPLICA_RUNBOOK.md`, `docs/INFRA_AUTOMATION_GATES.md`, then the named modules. Some handbook paragraphs describe the *historical* `main` baseline; this document describes the branch proposal. Code and current live evidence take precedence over both.
2. Compare `main` and `fix/infra-selection-guard`, and inspect any intervening merges, PRs or commits. In PowerShell: `git fetch origin`; `git status --short --branch`; `git log -1 --oneline origin/fix/infra-selection-guard`; `git diff --stat origin/main...origin/fix/infra-selection-guard`. Keep unrelated working-tree files out of this branch.
3. Record a non-secret evidence table: current Render pair, both router targets and maintenance flags, active Postgres and B2 slots, D1 schema/seed/bindings, subscriber readiness, controller environment-variable **names** and deploy SHAs, last successful watchdog tick. Mark each fact as code-confirmed, operator-observed or unverified.
4. Re-run the Python tests on a suitable host; run `node --test cloudflare/tests/infra-coordination.test.mjs` on Kaggle or another Node-capable machine. These tests are prerequisites, not proof of live failover.
5. Complete the activation gates and staging drill before changing production traffic. Document results, rollback limits and any new wiring in `docs/PROJECT_HANDBOOK.md` as you go.

For full setup and one-way recovery details, use [the Supabase runbook](SUPABASE_REPLICA_RUNBOOK.md). For remaining system risks, use [automation gates](INFRA_AUTOMATION_GATES.md). Never commit keys, passwords, connection strings, signed URLs or unredacted exports.
