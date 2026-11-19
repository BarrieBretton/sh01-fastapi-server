# n8n dispatch and Telegram queue coordination

Scope: only `task-scheduler-master` (cSoy04ojAzkzBkVM) and `Telegram Audio Render Worker - Final` (2R3VrBB58stPVz6q). No infrastructure slot changes, DB migration, B2 deletion, or content publication is performed by the installer. Active schedules resume normal work after explicit workflow installation.

## Architecture

`workflow_guards.py` mounts authenticated `/workflow-guards` endpoints. Authentication reuses the exact Telegram media key resolution from `media_router`; it is not a new secret. Google authentication reuses `clickouts_state.google_store`. SPREADSHEET_ID must match the workbook audited for these workflows. No service starts polling on import.

A separate hidden `_sh01_workflow_guards` tab and `SH01_WORKFLOW_GUARDS_REVISION` named range reuse the existing SheetsState protocol without modifying Clickouts state. Initialization verifies rejection of stale revision guards. Reads never initialize it. A missing/damaged/unverified store blocks mutations. GET health, explicit POST initialize, schedule claim, Telegram claim and commit are the complete API.

## Dispatch

The n8n filter uses the trigger timestamp in Asia/Kolkata, requires actual Sheet row numbers, and skips Server::/Python:: workflows. Workflow inventory is fetched once per due batch, names must resolve to exactly one ID, and each row retains its target and payload. No `.first()` selects a shared target for different rows.

Before delegation, reserve one durable receipt per IST date, scheduled minute and Sheet row. The server independently reads the schedule row and validates its name and time. Requests older than 120 seconds or more than 15 seconds in the future stop. Duplicate receipts never grant a second dispatch, including retries after a lost HTTP reply. Records contain only identity/time, not schedule payloads or secrets.

This is **at-most-once admission, not exactly-once delivery**. A crash after reservation, an ambiguous response, or a later child dispatch failure can leave an occurrence reserved but not executed. Claims are not automatically cleared. Investigate n8n execution history before an explicitly authorized recovery. Claimed dispatches receive `_sh01_dispatch_receipt` and `_sh01_workflow_id` alongside the original row inputs. The existing wait-for-child behavior is retained: a failed earlier child can leave later reserved rows undispatched.

Do not move/reorder schedule rows during their due minute. Row numbers identify occurrences; manual row rearrangement can change that identity. The workbook is authoritative for payloads; the existing n8n inventory credential remains authoritative for target lookup and must point to the intended workflow inventory.

## Telegram queue

The original select, render HTTP calls, auth credentials and trigger cadence remain. A coordinator claim re-reads the exact queue_id and its status; duplicate IDs stop. Each row has a 180-second ownership token. Competing patched workers skip owned rows. HTTP operations remain bounded at 60 seconds; failures let ownership expire, not reset the row.

Handlers recognize queued/accepted/running and retain job IDs, attempts, and timestamps through 404/409/429/5xx or malformed responses. Complete requires a URL and token. A confirmed failed/interrupted backend job is marked failed for manual review; it is not automatically resubmitted under the same stable job ID. The workflow does not publish content.

The final Sheet update is replaced by a coordinator commit. The coordinator verifies ownership, expiry, headers, sheet ID and the complete row snapshot. It commits owned render columns and the revision rotation atomically in one Sheets batchUpdate. Stale competing commits cannot overwrite newer state. Only render columns may change; publish/account/source fields are excluded. Ordinary string cell writes prevent formula injection.

External editors and unpatched writers do not participate in this protocol. A manual change between the final snapshot read and Sheets write is not protected by a transactional compare-and-set on cell contents. Do not sort/edit the queue during active processing. This is not a universal lock for arbitrary workbook users.

## Rollout and ring boundaries

1. Publish the backend files to main through a PR and wait for deployment.
2. Select a reachable coordinator base that will remain available across worker failovers. Prefer the stable SH01 proxy after C has been admitted and router maintenance cleared. Do not silently pin the coordinator to a disposable C origin. The installer defaults to the stable proxy and refuses workflow edits if the coordinator health/auth/workbook checks fail.
3. `--check-workflows` reads both definitions, validates the reviewed baseline and credential references, previews changes, and checks the coordinator without writes.
4. `--apply-workflows` explicitly initializes the hidden state, saves a local ZIP of the two live originals, deactivates ONLY these two workflows, checks for running/waiting/new/unknown executions, updates graphs while inactive, verifies them and reactivates only workflows that were originally active. If execution history cannot prove quiescence, it stops with the workflows paused. No execution is cancelled.
5. On partial failure they may remain paused. Rerun the installer to reconcile using its recovery journal; do not blindly reactivate old graphs. Workflow PUT is not a server-side compare-and-swap in n8n 2.20.0. Avoid concurrent editing/activation during installation; the installer re-reads before writes and checks active state.
6. Every independently running copy must be patched before it is allowed to run schedules again. C-only patching does not protect against unpatched A/B. Other independently triggered workflows (backup, office-hours, Telegram indexer, etc.) are outside this patch. Ring registration and choosing the authoritative Supabase database remain separate, unresolved operations.

The queue's shared D1 render-job identity/leases are additional protection; their live configuration must still be verified. No claim is made that these changes prove or repair historical duplicate publishing.

## Capacity and operations

The state tab has 10,000 record slots (dispatch occurrences plus one reusable record per queue_id). No automatic receipt pruning: full state stops execution rather than forgets deduplication. Monitor and archive with an explicit retention/recovery design before the limit is approached. Do not delete/recreate this tab or its named range to clear errors.

Each claim reads state and the relevant sheet; each mutation uses a single atomic batch. No added n8n polling cadence, no parallel scan increase, and no automatic provider retry loop. Quota errors stop execution. Claims for unrelated rows can contend on the shared revision; failed conflicts remain visible in n8n for review rather than silently claiming success.

The installer retains runtime keys only in process environment or secure interactive input. It never prints or stores API keys. Live workflow backups may contain embedded literals from your pre-existing code; treat the ZIP as private. The redacted audit attachment is only a baseline, never imported as executable credentials.

## Validation

Offline Python tests model atomic Sheets batches, stale owners, lost replies, duplicate claims, changed rows/headers, expiry, permanent dispatch receipts and malformed state. JavaScript tests run the actual generated n8n code under Node: distinct/same target rows, IST matching, excluded server workflows, queued/running status, transient failures, credential preservation and denied claims. Node is optional for the user's installer; it is not installed globally.

No live queue mutation, initialization, content execution, database switch, or B2 deletion was performed during development. Live Google guard verification occurs only when the operator runs workflow installation.
