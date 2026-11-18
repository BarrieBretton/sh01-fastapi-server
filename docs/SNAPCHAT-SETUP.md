# Snapchat publishing through SH01

This is an implemented, offline-tested pilot for **one configured Snapchat Public
Profile**, prepared videos, Spotlight and Public Stories. It runs entirely in SH01;
there is no n8n dependency. Publishing is paused by default. No live Snapchat
account, API approval, upload, publication, payout, or deployment was verified while
building this change.

## Before activation

1. Create your Snapchat creator account/Public Profile. Decide on an original
   creator-led content format; this integration cannot grant monetization eligibility.
2. Create a Snap Business organization and OAuth application through Ads Manager's
   Business Details. **Public Profile API access is currently allowlist-only.**
   Request access from Snap using your client ID and intended use. A normal Snap Kit
   Developer Portal client is not a substitute. Never send anyone the client secret.
3. Obtain your Public Profile UUID. Confirm the authorized account can manage it.
4. Deploy the backend patch on **main**, and the Operations Desk patch on **master**.
   Existing CI/hosting configuration is retained. There are no dependency additions.

Official references, checked 18 November 2026:

- https://developers.snap.com/marketing-api/Public-Profile-API/GetStarted
- https://developers.snap.com/marketing-api/Public-Profile-API/ProfileAssetManagement
- https://developers.snap.com/marketing-api/Public-Profile-API/Metrics
- https://help.snapchat.com/hc/en-us/articles/14669003687444-About-Snapchat-s-Monetization-Program
- https://help.snapchat.com/hc/en-us/articles/7012407611412-How-do-I-become-a-Snap-Star
- https://values.snap.com/policy/creator-monetization-policy

## Backend environment

Set the following on each media backend instance intended to serve this workflow.
All instances sharing the workbook must use the **same profile, application and
encryption key**. Keep secrets in the hosting environment, never Git or sheet cells.

| Variable | Value |
|---|---|
| `SNAPCHAT_CLIENT_ID` | The allowlisted business OAuth application's client ID |
| `SNAPCHAT_CLIENT_SECRET` | Its client secret |
| `SNAPCHAT_PROFILE_ID` | The Public Profile UUID, not the username |
| `SNAPCHAT_REDIRECT_URI` | For example `https://sh01.onrender.com/snapchat/oauth/callback`; register this exact URL with Snap |
| `SNAPCHAT_TOKEN_ENCRYPTION_KEY` | A new Fernet-compatible key; preserve it across deployments |
| `SNAPCHAT_SOURCE_HOSTS` | Exact comma-separated source hosts, e.g. `f005.backblazeb2.com`; no protocols, paths or wildcard |
| `SNAPCHAT_ENABLED` | Start with `0`; use `1` only after authorization and content review |

Generate the new encryption key locally with standard-library Python:

```powershell
python -c "import base64,secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

Copy that value into the hosting secret. Do not send it in chat. Losing/changing
this key requires reconnecting OAuth and makes existing encrypted job inputs unreadable.

Existing `SPREADSHEET_ID`, Google credentials, `MEDIA_CLIP_API_KEY`, FFmpeg/ffprobe,
and the shared media job/admission store are reused. A busy/unavailable media store
defers execution; it does not run uploads without coordination. Existing media jobs
retain their 176 MiB minimum memory headroom. This bounded uploader requests 96 MiB;
the existing cgroup reserve and operator override still apply. Actual peak memory
on your Render instance must be measured during the pilot.

The dashboard uses the existing `MEDIA_CLIP_API_KEY` and `OPS_INFRA_MEDIA_URL`.
If no media URL is configured, this bridge uses `https://sh01.onrender.com`.

## Connect and prepare content

Open **Operations desk → Producers → Snapchat publishing**.

1. Choose **Initialize Snapchat queue**. It creates `Snapchat-Queue` and the hidden,
   warning-protected `_sh01_snapchat_state` if missing. Existing tabs/data are never
   overwritten. No schedule or sample publishable item is inserted.
2. Choose **Connect Snapchat**, then follow the returned Snapchat authorization
   link. It expires after 10 minutes. On success, return to the desk and choose
   **Check API connection**. This proves access to the configured profile's read API;
   actual publishing permission is verified only by the first successful publication.
3. Add a prepared video row to `Snapchat-Queue`. The initial header is:

| item_id | source_url | destination | status | approved | rights_confirmed | description | locale |
|---|---|---|---|---|---|---|---|
| episode-001-spotlight | A reachable HTTPS MP4 URL on an allowed host | spotlight | ready | yes | yes | Your original description | en_IN |

`destination` is `spotlight` or `story`. An item ID must be unique in the queue,
including across destinations, and contain only letters, numbers, `_` or `-`.
Use distinct IDs if you deliberately publish an episode to both destinations.
Approval and rights fields are operator assertions, not automatic copyright or
monetization checks. Description is used for Spotlight; Stories publish the prepared
video without a text overlay in this pilot. Blank locale defaults to `en_IN`.

Ready requires `status=ready`, `approved=yes`, `rights_confirmed=yes` after trimming
and case normalization, a supported destination, and a valid allowed source URL.
The media check additionally requires a prepared H.264 MP4, 5–300 seconds, at most
200 MiB, and dimensions no greater than 3840 on either axis. This narrower prepared
format is the pilot's compatibility choice. The duration flag in evidence is not a
monetization determination. Editing, rendering, Lens effects, automatic subtitles,
image publishing, and multiple profiles are outside this first release.

Prefer vertical video prepared for Snapchat. A signed B2 URL must remain valid
until the scheduled transfer. The upload pipeline reads it once to bounded local
temporary storage, inspects metadata, encrypts incrementally and uploads sequential
16 MiB parts using an 8 KiB HTTP read buffer. There is no FFmpeg re-encoding and no
whole-video RAM buffer. Temporary source/encrypted files are removed on completion.
Budget up to roughly twice the input size in local disk, and source-download plus
upload bandwidth. Snapchat is not being used as an arbitrary cloud video editor.

## First publication, then the schedule

Keep Snapchat schedule timings blank during the initial test. After connection and
queue preview are correct, set `SNAPCHAT_ENABLED=1` and redeploy/restart the backend.
Choose **Load content queue**, inspect the exact sheet/cell link and readiness, then
**Queue publication** for one owned, prepared test video. Refresh status to inspect
the outcome. The button creates a durable job; it does not wait for the video transfer.

Once the pilot is verified, add these rows to your existing `Workflow-Schedule`:

| Workflow | Timings |
|---|---|
| Server::Snapchat-Spotlight | 11:15pm |
| Server::Snapchat-Stories | 8:30pm |

These times are examples, interpreted in **Asia/Kolkata (IST)**. Comma-separated
times and native Sheet time cells use the existing parser. Use one row per workflow.
Other workflow names do not execute arbitrary commands. Associated Accounts and
Run as Job Today are not read or modified by this pilot.

Each due occurrence queues **one** ready, previously unsubmitted item for that
destination, in worksheet order. The latest due occurrence today catches up once;
yesterday is never replayed. An empty queue consumes that occurrence, so newly added
items wait for the next scheduled time. Blank timings pause new scheduled jobs;
already queued jobs can still run. `SNAPCHAT_ENABLED=0` pauses new worker starts.
Changing an environment variable cannot cancel a request already dispatched.

The server checks once per minute while running. A stopped/asleep Render service
cannot promise exact-minute execution. Optionally set **`SNAPCHAT_TICK_ENABLED=1`**
on the existing Operations Desk Worker so its existing scheduled trigger sends an
authenticated tick. This patch does not create/alter Cloudflare cron configuration.
Both triggers share durable state and a poll cooldown. They do not depend on n8n.

## Results and recovery

- `queued`, `running`: persisted job and current processing stage.
- `accepted`: Snapchat acknowledged publication; live status is not yet verified.
- `published`: the Spotlight read API returned the exact configured profile/post ID
  with `LIVE`. The generated Snapchat viewing URL is separately marked unverified
  until a person/browser confirms it during the pilot.
- Stories: the documented posting response can omit a post ID. The job explicitly
  remains `accepted` / `story_accepted_verify_in_snapchat`; no invented permalink or
  false live verification is shown.
- `failed_before_publication`: no publication was dispatched. Inspect the reason.
  An unchanged, still-approved item can be explicitly requeued from a fresh preview.
- `waiting_retry`: pre-publication rate limiting, with bounded retry/backoff.
- `outcome_unknown`: publication intent was persisted, but the outcome could not be
  confirmed. **Never automatically retried.** Inspect Snapchat before any further
  action. A content hash also prevents reposting the same bytes under another item ID.
- A missing/expired source URL must be corrected before use. If a job already exists
  for an edited item, assign a new item ID after checking the previous job. The content
  hash still blocks a video with a previous publication intent.

Workbook headers, exact item identity, approval and input fingerprint are rechecked
after transfer and just before publication. Inputs are read-only. Execution results
live in internal state and the desk; source `status` is not rewritten to `uploaded`.
Retaining a B2 URL in this non-protected queue continues to protect it from the
separate B2 cleanup reference rules. This patch does not delete media or edit those rules.

The hidden state uses the existing atomic revision-guard/lease protocol. OAuth and
signed input URLs are encrypted with the separate environment key. Neither tokens,
encryption material nor source authorization query strings appear in desk responses.
Application callback logs omit the OAuth query; also avoid full-URL logging at any
external proxy you configure for the callback.

## Analytics and API operations

Refresh analytics uses a 30-minute cache/cooldown. It reports follower count and
Story/Spotlight views and watch time for the previous 28 complete UTC days. It does
not infer invitation eligibility, Snap Star verification or revenue, and it does not
assume this window equals Snapchat's in-app eligibility calculation.

All routes below require your existing `X-API-KEY: MEDIA_CLIP_API_KEY`, except the
one-time-state-protected OAuth callback. The desk uses its existing dashboard token.

| Route | Behavior |
|---|---|
| GET /snapchat/health | Read configuration, schedule and recent job evidence |
| POST /snapchat/setup | Initialize queue/internal state without publishing |
| POST /snapchat/connect | Create one-time OAuth link |
| GET /snapchat/oauth/callback | Validate state, exchange code and verify profile |
| POST /snapchat/connection/check | Refresh credentials if needed and verify profile read |
| POST /snapchat/preview | Read queue readiness; no media download or publish |
| POST /snapchat/run | Queue `{item_id, fingerprint}` from the latest preview |
| POST /snapchat/tick | Honor workbook timing and process a pending job |
| GET /snapchat/jobs/{job_id} | Read one job |
| GET /snapchat/metrics | Read cached metrics |
| POST /snapchat/metrics/refresh | Collect metrics within cooldown limits |

Read-only post-deployment check from PowerShell:

```powershell
& {
    $ErrorActionPreference = 'Stop'
    $secure = Read-Host 'Existing MEDIA_CLIP_API_KEY' -AsSecureString
    $key = [System.Net.NetworkCredential]::new('', $secure).Password
    try {
        Invoke-RestMethod -Uri 'https://sh01.onrender.com/snapchat/health' `
            -Headers @{ 'X-API-KEY' = $key } -TimeoutSec 60 | ConvertTo-Json -Depth 12
    }
    finally { Remove-Variable key, secure -ErrorAction SilentlyContinue }
}
```

## Validation scope

Offline tests cover exact queue identity and workbook links, eligibility inputs,
IST parsing, atomic occurrence/job claims, duplicates, encrypted OAuth state and
rotation, stream encryption/multipart boundaries, stale source rows, interrupted
publication, HTML upstream errors, route authentication and UI loading/escaping.
The full dashboard suite and existing Clickouts/cleanup suites are also run.
The application source has a generated Worker ready for deployment; Node/npm is
not required on your laptop to apply the patch. A real H.264 test MP4 passed the
ffprobe check. Browser automation was blocked by the test environment, so visual
layout remains a live-desk acceptance check; DOM/loading/escaping tests ran offline.
Mocks do not establish Snapchat allowlisting, live upload acceptance, real profile
reach, production scheduling, actual memory peak or monetization. Those are the
explicit acceptance checks for the first authorized pilot.
