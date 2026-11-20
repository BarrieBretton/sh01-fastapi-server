# Snapchat integration: development roadmap

Updated 20 November 2026 (IST).

This is the canonical tracker for the SH01 Snapchat development lane. The first
pilot merged into both deployment branches on 18 November. The next milestone
is an authenticated status check through the SH01 router, with publishing paused.
Snap account authorization, publication and revenue still require live evidence.

## Repositories and reference documents

| Responsibility | Repository / location |
|---|---|
| Publishing, scheduling, credentials and durable jobs | `BarrieBretton/sh01-fastapi-server`, deployment branch `main`; `snapchat_feature/` |
| Operator controls and evidence | `BarrieBretton/SH01-Operations-Dashboard-Cloudflare`, deployment branch `master`; Producers > Snapchat publishing |
| This roadmap | Backend `docs/SNAPCHAT-ROADMAP.md`; dashboard has a pointer at the same path |
| Configuration, queue schema and operational instructions | [SNAPCHAT-SETUP.md](SNAPCHAT-SETUP.md) |
| Wider backend architecture and evidence register | [PROJECT_HANDBOOK.md](PROJECT_HANDBOOK.md) |

The separate personal/AQI dashboard is outside this implementation. The `*-infra-review`
directories used by the installer are working checkouts of these two repositories.
The packaged source baselines were backend `232a6c545287aa3d828db759b5fedaabdabc1824`
and dashboard `4cda8bc90e118cb50a7ec04f7baa1931304af0bc`; these are build evidence,
not a claim about the repositories' current remote heads or deployed revisions.

## Decisions to preserve

- This workflow runs through SH01 server endpoints/functions. It has no n8n dependency.
- Cadence comes from the existing `Workflow-Schedule` workbook tab, interpreted in
  `Asia/Kolkata` (IST). Do not create a separate competing schedule database.
- Begin with one configured Snapchat Public Profile and already-prepared videos.
- Keep the small Render service responsible for coordination and bounded transfers.
  Additional rendering/editing must have its own resource plan before implementation.
- Publishing is disabled by default (`SNAPCHAT_ENABLED=0`). Account activation and
  monetization are separate milestones with separate evidence.
- Reuse the existing Google integration, API authentication and shared media admission.
- Preserve workbook cell links and durable results. Never automatically repost a job
  whose earlier publication outcome is unknown.
- Snapchat work does not authorize changes to B2 deletion rules or existing workflows.

## Milestones

| ID | Milestone | Current evidence / status | Completion requirement |
|---|---|---|---|
| SC-01 | Server publishing foundation | Implemented; offline verified | OAuth, prepared-video upload, separate Story/Spotlight destinations, durable job/occurrence records and duplicate protection |
| SC-02 | Operations Desk integration | Implemented; DOM tests pass; browser layout unverified | Readiness, cell links, progress/history, analytics, visible loading and session locking |
| SC-03 | Repository deployment | Pilot merges verified on both branches. Authenticated Snapchat health is pending. | Record deployed backend/Worker revisions and a paused health response through the router |
| SC-04 | Snapchat access and account authorization | Pending confirmation; external prerequisite | Obtain the appropriate API access, configure profile/app, complete OAuth and verify profile read |
| SC-05 | First manual publication | Pending SC-03/04 | Publish owned test content; distinguish API acceptance from actual visibility; measure resource use |
| SC-06 | Scheduled production pilot | Pending SC-05 | Demonstrate correct IST execution and restart/duplicate behavior before expanding cadence |
| SC-07 | Reporting and operational improvements | Basic manual analytics implemented; further work deferred | Validate real metrics and define improvements from pilot evidence |
| SC-08 | Media preparation and effects | Research/design backlog | Select a viable external preparation approach before implementing edits, subtitles or effects |
| SC-09 | Multi-profile operation | Deferred until single-profile pilot is stable | Isolated credentials, account routing, limits, schedules and job identities tested |
| SC-10 | Monetization readiness and outcomes | Unverified; ongoing business lane | Current program requirements, original-content evidence, actual account eligibility and any payout confirmation |

### SC-01 / SC-02: delivered implementation

- [x] Fixed `/snapchat/*` backend routes and authenticated dashboard bridge.
- [x] OAuth state validation, encrypted token storage and refresh handling.
- [x] `Snapchat-Queue` readiness preview with exact worksheet/cell links.
- [x] Prepared MP4 validation, temporary disk storage, encryption and chunked upload.
- [x] Durable scheduling and jobs, input revalidation, and content-hash duplicate protection.
- [x] Distinct queued, accepted, provider-verified, failed and uncertain outcomes.
- [x] Cached follower/view/watch-time reporting; revenue remains unknown.
- [x] Loading feedback and clearing sensitive UI state when the dashboard locks.
- [x] Offline validation: 46 Snapchat Python tests, 345 dashboard Node tests,
  97 existing Clickouts tests and 11 cleanup tests; five installer tests.
- [x] A real prepared H.264 MP4 passed ffprobe without re-encoding.
- [ ] Verify final browser rendering and all controls on the deployed desk.

### SC-03 / SC-04: immediate next work

1. In **Producers → Snapchat publishing**, use **Refresh status**. For a direct
   read-only check, use the stable SH01 router command in the setup guide.
   Record the reported revision, publishing flag and next step. A response from
   a suspended direct Render hostname left this check open.
2. Keep publishing paused and Snapchat schedule timings blank during connection tests.
3. Create/configure the intended creator Public Profile and business OAuth app;
   obtain API access. Follow the official access instructions linked below.
4. Set the environment variables listed in the setup guide. Record variable names
   and successful checks only; never put credentials or signed source URLs in this tracker.
5. Initialize the queue/internal state without overwriting existing workbook data.
6. Complete OAuth and verify the exact configured profile. A successful read is
   insufficient evidence that publication is permitted or that monetization is enabled.

### SC-05 / SC-06: publication and scheduling acceptance

- [ ] Inspect one original, rights-cleared prepared video and its queue row.
- [ ] Enable publishing only for the pilot; submit one Spotlight manually.
- [ ] Record job ID, provider ID, API outcome, observed visibility and viewing link.
- [ ] Test a Public Story separately. If the response omits a stable identifier,
  verify it in Snapchat and retain `accepted` rather than claiming API-verified publication.
- [ ] Measure peak Render memory, temporary disk, transfer time and errors under
  realistic load. Successful endpoint discovery does not establish memory safety.
- [ ] Reconcile the first live analytics report; record its window and freshness.
- [ ] Set an agreed IST schedule and verify its first due occurrence.
- [ ] Test competing triggers/restarts and duplicate protection in a controlled
  environment. An uncertain live outcome requires reconciliation before any new attempt.
- [ ] Check paused, empty-queue, rejected-content, expired-source and authorization-error
  behavior. Existing offline tests do not replace this deployment acceptance.

| Destination | Exact workflow label | Example only |
|---|---|---|
| Spotlight | `Server::Snapchat-Spotlight` | `11:15pm` IST |
| Public Stories | `Server::Snapchat-Stories` | `8:30pm` IST |

Use one row per workflow. One due occurrence selects one ready, previously unsubmitted
item for that destination in worksheet order. The latest due time today catches up
once; yesterday is not replayed. An empty queue consumes its occurrence. Blank timings
pause new scheduled jobs, while already queued jobs can still run. The server must be
running to check times; the optional existing Worker cron can call the same tick route.

## Workflow and state boundaries

The pilot is one server-owned workflow with reusable stages; it does not need a
separate subworkflow engine. Its stages are: read schedule and queue, claim a durable
job, recheck input, transfer/inspect/encrypt/upload media, recheck approval, persist
publication intent, submit, then verify and report the result.

| Record / component | Purpose |
|---|---|
| `Workflow-Schedule` | Operator-owned cadence; existing unrelated workflow rows are preserved |
| `Snapchat-Queue` | Operator-owned source, destination, readiness and rights assertions |
| `_sh01_snapchat_state` | Internal encrypted credentials, occurrence claims, job results and metrics cache |
| Backend `core.py` / `store.py` | Input identity, schedule parsing and workbook state |
| Backend `client.py` / `service.py` / `router.py` | Provider adapter, execution, authenticated routes and timer |
| Dashboard `snapchat-worker.js` / `snapchat-view.*` | Fixed route bridge and operator UI |

Queue input rows are currently read-only: the worker does not rewrite their status
to `uploaded`. A durable job record prevents repeating a successfully submitted item.
Changing a queued input invalidates its fingerprint; inspect the earlier job before
assigning a replacement item ID. Prepared source URLs must survive until transfer.

## Deferred development: prioritize after pilot evidence

- **Reporting:** evaluate scheduled metrics collection, richer per-post history,
  alerts and stale-data indications against actual API availability and quota use.
- **Recovery:** design explicit reconciliation/cancellation operations for uncertain
  jobs and improved Story identity tracking, with auditable evidence.
- **Preparation:** evaluate external rendering, cropping, subtitles, thumbnails and
  reusable assets. Prefer a returned asset URL where the chosen service supports it;
  the present uploader still transfers the prepared video through SH01 temporary disk.
- **Snap-native features:** investigate supported captions/link overlays and Lens or
  Camera Kit options separately. Do not assume a headless, server-hosted video-filter
  API exists or that Snapchat will perform arbitrary editing from a source URL.
- **Multiple profiles:** add explicit account mapping and isolation only after the
  single-profile workflow is reliable. The current queue cannot select arbitrary accounts.
- **Revenue:** track originality/rights, audience performance and actual program
  eligibility. Do not infer revenue or approval from successful automated posting.

For each item selected from this backlog, record its scope, resource/quota budget,
acceptance tests and rollback plan before adding it to an active milestone.

## Evidence and handoff log

Keep this table current in the same PR as a Snapchat implementation change. Use
`implemented`, `offline verified`, `operator observed`, `blocked`, or `unverified`
precisely. Link relevant PRs, test results, redacted job evidence and workbook cells.

| Date (IST) | Milestone / decision | Evidence | Next action |
|---|---|---|---|
| 18 Nov 2026 | SC-01/02 pilot packaged | `apply_sh01_snapchat.py`; 504 offline tests and real MP4 metadata check | Confirm deployment and API access |
| 18 Nov 2026 | Dedicated roadmap added | This document; dashboard pointer; application code unchanged | Maintain this tracker through activation and subsequent PRs |
| 20 Nov 2026 | SC-03 pilot merges verified | [Backend PR 2](https://github.com/BarrieBretton/sh01-fastapi-server/pull/2) at `02adbdf` and [Operations Desk PR 1](https://github.com/BarrieBretton/SH01-Operations-Dashboard-Cloudflare/pull/1) at `4c05b3d` | Record authenticated health through the stable router |
| 20 Nov 2026 | SC-03/04 setup diagnostics | Offline tests cover an uninitialized workbook, missing Google configuration, app/profile changes and a new OAuth grant whose profile read has yet to succeed | Follow the returned `next_step` and record the first successful authorized profile read |

Before handing this lane to another engineer or AI, record the latest completed
milestone, active blocker, next concrete action, deployment revisions and unresolved
job IDs. Keep secrets, OAuth codes and authorization query strings out of all Markdown.

## External references to recheck at activation

These are reference links, not a promise that external eligibility requirements or
API behavior will stay unchanged. Recheck them when requesting access and before
expanding the workflow; retain the check date and relevant decision here.

- [Public Profile API access and OAuth](https://developers.snap.com/marketing-api/Public-Profile-API/GetStarted)
- [Media, Stories and Spotlight publishing](https://developers.snap.com/marketing-api/Public-Profile-API/ProfileAssetManagement)
- [Analytics](https://developers.snap.com/marketing-api/Public-Profile-API/Metrics)
- [Snapchat monetization program](https://help.snapchat.com/hc/en-us/articles/14669003687444-About-Snapchat-s-Monetization-Program)
- [Snap Star guidance](https://help.snapchat.com/hc/en-us/articles/7012407611412-How-do-I-become-a-Snap-Star)
- [Creator monetization policy](https://values.snap.com/policy/creator-monetization-policy)
