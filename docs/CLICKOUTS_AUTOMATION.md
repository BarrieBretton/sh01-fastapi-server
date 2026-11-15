# Clickouts deletion automation

## Scope and evidence

Implemented in the **SH01 FastAPI server**, based on remote `main` at
`c82f0a47ad4c9d75c5ff27e4ed2334703bbfa3f1`. This adds a queue consumer to
`clickouts_worker.py`, Google/Redis adapters to `clickouts_state.py`, and routes /
startup scheduling to `clickouts_routes.py`. The existing single-post deletion
route and its account opt-in / exact confirmation rules remain separate.

Offline tests exercise fakes. Installing the patch does not configure Render,
change a workbook, call Meta, delete posts, or verify live permissions. Existing
dependencies already cover this feature; no requirements change is needed.

The consumer supports **Instagram top-level media and Facebook Page posts**.
Other platforms and rows containing only a permalink are blocked. It does not
scrape URLs to guess provider IDs. Queued rows are treated as already authorised
for deletion: enable this only for the workbook containing confirmed removals.

## Setup

1. Apply the patch, run the tests below, and publish the FastAPI repository.
2. Retain the existing `SPREADSHEET_ID` and Google credentials configured by
   `sheets_helper.get_credentials()`. The Google identity must be able to edit
   this workbook. No new workbook or Apps Script is needed.
3. Retain a working **shared `REDIS_URL`** on every SH01 replica consuming this
   workbook. Configure Redis persistence and avoid eviction of the Clickouts keys.
   Missing/unavailable Redis prevents deletion. Process memory is not a substitute.
   Read-only preview does not require Redis. The status endpoint reports Redis
   failures separately while retaining successful queue and schedule reads.
4. Use the server's existing **`MEDIA_CLIP_API_KEY`** for the protected Clickouts
   endpoints. No additional API-key variable is required. An already configured
   `SOCIAL_INVENTORY_API_KEY` is also accepted for compatibility; it does not
   override or disable the media key on instances where both exist.
5. Verify the Meta token and app permissions described below. Use the existing
   `META_SYSVAR` (or `FB_PAGE_ACCESS_TOKEN` fallback) and `META_GRAPH_VERSION`.
6. For Operations Desk status updates, set the server's `OPS_INGEST_TOKEN` to the
   dashboard's existing `INGEST_TOKEN`. `INGEST_TOKEN` on the server is also
   accepted. Retain `OPS_DASHBOARD_URL` if configured; the existing fixed dashboard
   default is otherwise used. A scanner-only credential cannot report deletions.
7. Add **one** schedule row as shown below using `Workflow` and `Timings`.
   Inspect `POST /social/clickouts/preview`; it reads the queue without modifying
   rows or calling Meta. It validates row identities, not provider permissions.
8. To activate actual deletion, set **`SOCIAL_DELETE_ENABLED=1`** on the SH01
   service and restart/redeploy it. A matching workflow row with valid `Timings`
   supplies its daily cadence. The first due run can consume existing queued rows
   immediately. Clear `Timings` to pause the daily cadence.

No separate per-account registry is required by this consumer. Account ownership
and the permalink are verified against Meta immediately before deletion. A blank
`Associated Accounts` cell does not remove this ownership check.

### Workflow-Schedule row

| Workflow | Timings | Associated Accounts | Run as Job Today | Public Drive Link | Caption | Remarks |
| --- | --- | --- | --- | --- | --- | --- |
| Server::Clickouts | 9:00pm, 11:58pm | | | | | |

`Python::Clickouts` is an alias for the same job. Use **one row**, not both aliases.
Other `Server::`/`Python::` names do not execute arbitrary functions or Python code.

- Times are **Asia/Kolkata (UTC+05:30)**, matching the existing server scheduler.
- Accepted: `9:00pm`, `3:30pm`, `11:45pm`, `11:58pm`, `9:30 AM`, `10:00`,
  `18:30`. AM/PM is case-insensitive and can be attached directly to the time.
  Separate multiple times with commas, semicolons or newlines. Native Google
  Sheets time cells are supported; a minute is never rounded into the next day.
- Only `Workflow` and `Timings` are required by the daily scheduler.
  **`Run as Job Today` is ignored**, including blank, `no`, and `yes`. It belongs
  to the operator's separate one-time override workflow. Clickouts neither reads
  that column for scheduling decisions nor consumes, resets or writes it.
  Setting it to `yes` does not create an immediate or additional Clickouts run.
- A matching row with valid `Timings` schedules daily runs. Blank `Timings` or
  no matching workflow row means no daily run; invalid nonblank times produce a
  diagnostic error. `Run as Job Today = no` is **not** a pause control.
- One shared poll per minute across replicas checks the schedule. A due occurrence
  processes up to **3 rows per tick**, then resumes on subsequent minute ticks
  while there is eligible backlog. Each batch also stops starting new items after
  40 seconds; in-progress network calls have their own bounded timeouts.
- After a restart, the most recent due time **today** is caught up once. Yesterday's
  missed runs are not replayed. Newly queued rows after a completed occurrence wait
  until the next time, unless an authenticated manual batch is requested.
- An asleep or stopped Render service cannot run an in-process scheduler. An
  existing external scheduler can call `/social/clickouts/tick` to wake/check it;
  that route still honours the Sheet settings and shared execution state.

### Clickouts columns and outcomes

Required existing columns (case-insensitive): `URLS`, `platform`, `account_id`,
`post_id`, `request_id`. Preserve IDs as **plain text**, especially long Meta IDs.
`queued_at` and custom columns are retained. Facebook `post_id` must be the Page
post Graph ID, `PAGE_ID_POST_ID`; a video/media object ID is not guessed to be a
Page post ID.

On the first executing batch, missing helper headers are appended:
`status`, `deleted_at`, `last_attempt_at`, `error`, `dashboard_ack`.
Rows are **never deleted** from Clickouts. Only these helper columns are written.
Developer metadata binds updates to a row when rows are inserted or moved. Each
update also rechecks the headers, request identity and current status. If an
identity is edited during processing, work stops instead of guessing its target.

| Status | Meaning / next action |
| --- | --- |
| blank / pending / queued / queued_clickouts / awaiting_sheet | Eligible after validation and ownership checks. |
| processing | Durable attempt marker. Prepared attempts resume safely; attempts that might have sent DELETE require a receipt or manual review. |
| done | Meta explicitly confirmed deletion, or the same exact post has a stored successful deletion receipt. The row stays in the Sheet. |
| blocked | Validation or ownership/permission read failed before DELETE. Correct the reported problem, then set status to `pending` to try again. |
| manual_required | DELETE may have happened, or the provider rejected/unconfirmed it. Verify the provider result. Clearing the Sheet status alone does not bypass the persistent attempt record. |
| any other value | Left untouched. `cancelled` can be used to exclude a pending row. |

Missing or duplicate `request_id` values are reported in preview / run counts and
left untouched because the intended row cannot be identified safely. They do not
block valid rows later in the Sheet. Duplicate requests with distinct request IDs
for the same exact platform/account/post share one deletion receipt.

Provider GET failures, 404s, and inaccessible posts are **not** evidence of
deletion. Only a successful DELETE confirmation makes a row `done`. DELETE is
never automatically retried after a timeout, malformed reply or provider error.

The persistent attempt moves from `prepared` to `delete_sent` **before** DELETE.
An interrupted `prepared` attempt can resume after re-verification. An interrupted
`delete_sent` attempt becomes `manual_required`. If deletion succeeds but writing
`done` to Sheets fails, the stored receipt permits a **status-only** recovery on
the next run. If both that receipt write and the process fail, the Sheet's
processing marker prevents a blind retry; manual reconciliation is required.

`dashboard_ack` is `synced`, `pending`, or `not_configured`. `identity_mismatch`
means the dashboard's post/request mapping differs and needs review;
`unverified_receipt` means no successful consumer receipt exists for a row already
labelled `done`. Handwriting `done` does not fabricate a provider confirmation.
A failed/missing
dashboard notification does not undo `done`. A later batch retries notification
without deleting again; scheduled retries happen on a later due occurrence.

Rows and schedule sheets are bounded to 10,000 allocated rows, and relevant
headers to the first 52 columns. Larger grids return an explicit limit error;
they are not silently truncated. Reduce unused allocated rows or deliberately
review the bound before increasing it. API quota responses persist a cooldown
and honour longer `Retry-After` values. This controls load but does not guarantee
that other services sharing the same Google/Meta quota cannot exhaust it.

## Endpoints

All Clickouts routes accept `X-API-KEY: <MEDIA_CLIP_API_KEY>` using the existing
server credential. An already configured `SOCIAL_INVENTORY_API_KEY` also works.
Missing/incorrect credentials are rejected before storage access. This credential
reuse is limited to these Clickouts routes; the existing provider ownership,
server deletion enable switch, and Sheet scheduling checks still apply.

| Method and path | Behaviour |
| --- | --- |
| GET `/social/clickouts/status` | Independent Google client, queue, schedule and Redis checks; preserve available evidence when another check fails. No deletion. |
| POST `/social/clickouts/preview` | Read-only queue eligibility and blocked-reason counts. |
| POST `/social/clickouts/run` | Defaults to `{"dry_run":true}`. `{"dry_run":false,"max_items":3}` executes one explicit batch, independent of the daily schedule, only with server deletion enabled. `max_items` is 1–5. |
| POST `/social/clickouts/tick` | Check/run the configured daily occurrence; suitable for an external scheduler. |

Overlapping execution returns `busy`; disabled execution returns `disabled`.
Upstream cooldown returns `cooldown` with `retry_after`. Transport failures return
sanitised diagnostics. Clients must inspect the JSON result as well as HTTP status.

### Diagnosing a deployed status failure

The original status implementation could hide a Redis failure (after successful
Sheet reads) or Google client/credential setup failure behind
`clickouts_unavailable_no_automatic_delete_retry`. That message alone does not
establish which dependency failed.

The corrected endpoint returns `diagnostics_version: 2`, `ok`, `status` and
`checks.google_client`, `checks.queue`, `checks.schedule`, `checks.redis`.
`runtime` includes validated deployment/instance identifiers when Render already
supplies them, so responses from different instances can be distinguished. These
are optional diagnostics; no new settings are required.
It returns HTTP 200 for a successfully assembled diagnostic report, including
`status: degraded` / `ok: false` when a dependency failed. This is not an execution
readiness or provider-permission guarantee. Unavailable counts are `null`;
successful counts remain numeric. A Sheet quota response suppresses the next
schedule read in the same status call. No diagnostic call creates a lease,
writes rows, refreshes metrics, calls Meta, or executes deletion.

| Diagnostic code | Interpretation |
| --- | --- |
| `redis_connection_failed` / `redis_timeout` | The server could not complete the Redis read. Existing `REDIS_URL` connectivity/service health needs checking. |
| `redis_authentication_or_permission_failed` | Redis rejected the configured credentials or command permission. |
| `shared_redis_not_configured` / `redis_configuration_invalid` | The existing Redis setting is absent or cannot be parsed. |
| `redis_state_invalid_json` / `redis_state_invalid_shape` | Stored execution state needs investigation; it is not automatically erased. |
| `google_credentials_refresh_rejected` | The existing Google credentials failed refresh. |
| `google_credentials_invalid` / `google_credentials_unavailable` | The Google credential loader could not supply credentials. |
| `google_client_dependency_missing` / `google_client_initialization_failed` | Google client initialization failed before a Sheet request. |
| `sheets_permission_or_quota_denied` | Google returned 403; this code alone does not distinguish access from quota denial. |
| `sheets_invalid_request` / `spreadsheet_not_found_or_inaccessible` | Google rejected the read or could not expose the configured workbook. |

The report contains static error codes and limited exception categories, never
connection strings, token values, credential JSON, or raw provider error text.
Actual execution still requires shared Redis and the existing deletion gates.
No additional environment variable was introduced by this correction.

### Read-only verification from PowerShell

```powershell
$base = 'https://sh01.vivojaymail.workers.dev'
$secure = Read-Host 'Existing MEDIA_CLIP_API_KEY' -AsSecureString
$key = [System.Net.NetworkCredential]::new('', $secure).Password
try {
    Invoke-RestMethod -Method Post -Uri "$base/social/clickouts/preview" `
        -Headers @{ 'X-API-KEY' = $key } | ConvertTo-Json -Depth 8
} finally {
    Remove-Variable key, secure -ErrorAction SilentlyContinue
}
```

## Meta requirements verified against official documentation

- [Instagram IG Media](https://developers.facebook.com/docs/instagram-platform/reference/instagram-media/):
  the documented DELETE operation uses Facebook Login, a **Facebook User access
  token**, `instagram_basic` and **`instagram_manage_contents`**, with access to the
  linked Page. It covers eligible non-ad media and entire carousel albums, not
  individual carousel children. This consumer uses the root credential for IG,
  not the Page token returned by account discovery. A Page-only token or existing
  insights permission does not establish deletion permission.
- [Facebook Page Post](https://developers.facebook.com/docs/graph-api/reference/page-post/):
  DELETE requires applicable Page permissions/access and is documented as
  available only to select developers. The consumer obtains the exact Page's
  access token and verifies `from.id`, `id` and permalink. It cannot grant an app
  an entitlement that Meta has not provided.
- The configured `META_GRAPH_VERSION` is preserved (existing default `v24.0`).
  Your deployed app/version/token's deletion capability is **unverified** until a
  real authorised request succeeds. The patch does not silently upgrade Graph
  versions or claim live provider access.

## Offline tests

```powershell
python -m unittest discover -s tests -p "test_clickouts*.py"
python -m unittest discover -s tests -p "test_social_delete.py"
python -m py_compile clickouts_state.py clickouts_worker.py clickouts_routes.py app.py
```

Tests use fake Sheets/Redis/Meta transports; they do not require credentials or
perform live deletion. Cases include same-post duplication, wrong-owner refusal,
long numeric-ID rejection, loss of a DELETE response, loss of a Sheet update,
Redis/lease failures, row/column movement, protected routes, daily aliases,
restart catch-up, bounded backlog continuation and quota cooldown.
Schedule regressions cover every job-override value, a missing override column,
the supplied compact `h:mmpm` strings, noon/midnight, and native Sheet time values.
