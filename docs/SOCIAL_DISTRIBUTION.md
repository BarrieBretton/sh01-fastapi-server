# SH01 Social Distribution – Production Documentation

_Last updated: 2026-10-06_

This document records the social-distribution architecture, implementation decisions, live test findings, current platform status, account-onboarding model, database migrations, operational procedures, failure modes, and known follow-ups discussed during the Threads/X/Tumblr/n8n implementation work.

It is intentionally secret-free. Never paste real access tokens, API keys, token secrets, encryption keys, database passwords, or Render/Cloudflare control keys into this document.

---

## 1. Current platform status

| Platform | Backend | n8n production path | Media | Multi-account | Current status |
|---|---|---|---|---|---|
| Threads | `threads_feature/` | Active | Image + video | Yes | **Active / working** |
| Tumblr | `tumblr_feature/` | Active | Image + video implemented | Yes | **Active / image tested working** |
| X.com | `x_feature/` | Scaffold only | Image + video backend implemented | Yes | **Parked** |
| TikTok | Not implemented yet | Scaffold only | N/A | Planned | **Parked** |

The canonical distribution order is:

```text
incoming content item
    ↓
canonical-account normalization
    ↓
Threads
    ↓
X.com scaffold (no network call)
    ↓
Tumblr
    ↓
TikTok scaffold (no network call)
    ↓
combined distribution result
```

X and TikTok are deliberately non-blocking scaffolds. They record a skipped/scaffold result and allow later platforms to continue.

---

## 2. Stable public endpoint

The public SH01 entry point used by n8n is:

```text
https://sh01.vivojaymail.workers.dev
```

Cloudflare routes this to the preferred SH01 Render backend and can fail over to another configured backend.

Social workflows should call the stable Worker URL instead of a specific Render origin.

---

## 3. Shared authentication from n8n to SH01

The social HTTP nodes use an n8n HTTP Header Auth credential.

Recommended n8n credential:

```text
Name: SH01 Social API Key
Header name: X-API-Key
Header value: <SH01 internal API key>
```

An existing credential named `Threads SH01 API Key` is also valid if it contains the same expected `X-API-Key`.

Platform OAuth credentials do **not** belong in n8n. They stay on SH01/Render or, for Threads, encrypted in Postgres.

### Internal-key resolution

Threads currently accepts:

```text
THREADS_INTERNAL_API_KEY
or fallback X_API_KEY
```

Tumblr accepts, in order:

```text
TUMBLR_INTERNAL_API_KEY
SOCIAL_INTERNAL_API_KEY
THREADS_INTERNAL_API_KEY
```

X accepts, in order:

```text
X_INTERNAL_API_KEY
SOCIAL_INTERNAL_API_KEY
THREADS_INTERNAL_API_KEY
```

A future cleanup can standardize all platforms on `SOCIAL_INTERNAL_API_KEY`; the current fallbacks preserve compatibility.

---

# Threads

## 4. Threads architecture

Files:

```text
threads_feature/
├── __init__.py
├── config.py
├── db.py
├── models.py
├── router.py
└── service.py

sql/
├── 001_threads_feature.sql
└── 002_threads_refresh_hardening.sql
```

Primary endpoints:

```text
GET  /threads/health
GET  /threads/tokens/status

POST /threads/tokens/bootstrap
POST /threads/tokens/refresh/{account}
POST /threads/tokens/refresh-all
POST /threads/tokens/refresh-scheduled

POST /threads/publish
POST /threads/publish/image
POST /threads/publish/video
```

The unified route is the preferred production route:

```text
POST /threads/publish
```

The canonical n8n workflow uses only the unified route.

---

## 5. Threads configured accounts

Built-in account mapping currently includes:

```text
erika.devereux -> 25447546524875159
vlvt.ave       -> 25962539886681226
cyootstuff     -> 25634868322816151
```

`THREADS_ACCOUNTS_JSON` can override or extend this mapping without a code change.

Example:

```json
{
  "erika.devereux": "25447546524875159",
  "vlvt.ave": "25962539886681226",
  "cyootstuff": "25634868322816151",
  "new.account": "THREADS_USER_ID"
}
```

Account keys are normalized by stripping `@`, trimming, and lowercasing.

---

## 6. Threads database tables

`001_threads_feature.sql` creates:

```text
public.threads_auth
public.threads_publish_jobs
```

`threads_auth` stores encrypted access tokens and lifecycle metadata.

`threads_publish_jobs` stores publish/idempotency state and has:

```text
UNIQUE(account_key, idempotency_key)
```

RLS is enabled.

`002_threads_refresh_hardening.sql` adds:

```text
last_refresh_attempt_at
last_refresh_success_at
next_refresh_after
consecutive_refresh_failures
refresh_error_code
refresh_error_status
```

and an expiry index.

Because the infra Postgres migration copies `public`, these social tables follow Postgres failover.

---

## 7. Threads token encryption

Required static secret on every serving SH01 instance:

```text
THREADS_TOKEN_ENCRYPTION_KEY
```

It must be identical on every SH01 serving node.

If a different key is deployed, previously stored tokens cannot be decrypted.

Tokens are encrypted with Fernet before being written to Postgres.

---

## 8. Threads token bootstrap

A token is bootstrapped once for an account through:

```text
POST /threads/tokens/bootstrap
```

Example Bash shape:

```bash
curl -k -X POST \
  "https://sh01.vivojaymail.workers.dev/threads/tokens/bootstrap" \
  -H "X-API-Key: <SH01_KEY>" \
  -H "Content-Type: application/json" \
  --data-raw '{
    "account": "vlvt.ave",
    "access_token": "<FRESH_THREADS_TOKEN>"
  }'
```

Never commit or paste the actual access token into documentation.

Bootstrap verifies that the supplied token resolves to the expected Threads user ID before storing it.

---

## 9. Threads token refresh scheduler

The backend runs an internal refresh scheduler.

Defaults:

```text
THREADS_REFRESH_SCHEDULER_ENABLED=true
THREADS_REFRESH_INITIAL_DELAY_SECONDS=30
THREADS_REFRESH_CHECK_INTERVAL_SECONDS=21600
THREADS_REFRESH_THRESHOLD_DAYS=21
THREADS_REFRESH_MAX_ATTEMPTS=3
THREADS_REFRESH_RETRY_BASE_SECONDS=2
THREADS_REFRESH_ALERT_EXPIRY_DAYS=7
```

Every SH01 worker may start the scheduler, but a Postgres advisory lock:

```text
threads_token_refresh_scheduler_v1
```

ensures only one worker performs a scheduled sweep.

Optional Telegram alerting:

```text
TELEGRAM_BOT_TOKEN
THREADS_REFRESH_ALERT_TELEGRAM_CHAT_ID
```

A refresh failure does not immediately prevent publishing while the currently stored token is still actually valid.

---

## 10. Important Threads token-expiry caveat discovered in production

This is a critical operational finding.

`bootstrap_token()` currently uses:

```text
expires_in supplied by caller
OR
THREADS_LONG_LIVED_LIFETIME_SECONDS
```

with the default assumed lifetime:

```text
5,184,000 seconds (~60 days)
```

Therefore `/threads/tokens/status` currently reflects the backend's stored/assumed expiry metadata; it is **not proof that Meta still considers the token/session valid**.

This was observed with `vlvt.ave` and `cyootstuff`:

- SH01 reported tokens configured and apparently valid until December.
- Threads container creation returned HTTP 400.
- The actual Meta error for `vlvt.ave` was OAuth error code `190` stating that the session had already expired.
- Bootstrapping a fresh token fixed publishing.

### Operational rule

If Threads returns an OAuth/session-expired error, do not trust the stored `expires_at`. Generate a fresh Threads token and bootstrap it again.

### Follow-up hardening still recommended

The backend should eventually:

1. distinguish authoritative vs assumed token expiry;
2. derive actual expiry from Meta where possible;
3. optionally provide a live validation mode for token status;
4. avoid presenting assumed expiry as authoritative health.

Until that is implemented:

```text
/threads/tokens/status = database/lifecycle status
Meta API response       = authoritative live token status
```

---

## 11. Threads structured upstream errors

The unified Threads publish route was hardened so Meta 4xx responses are no longer surfaced as an opaque SH01 500.

Preferred response shape:

```json
{
  "detail": {
    "platform": "threads",
    "account": "vlvt.ave",
    "operation": "create_container",
    "upstream_status": 400,
    "meta": {
      "error": {
        "message": "...",
        "type": "OAuthException",
        "code": 190,
        "error_subcode": 0
      }
    }
  }
}
```

This is implemented for the unified:

```text
POST /threads/publish
```

Use the unified endpoint in automation.

---

## 12. Threads idempotency

A publish can provide an explicit:

```text
idempotency_key
```

If omitted, SH01 generates a deterministic content-based key.

A successfully published job is returned on retry with:

```text
reused=true
```

and no duplicate post is created.

Important limitation:

If the post is manually deleted on Threads later, SH01 still remembers the original successful idempotency result. Reusing the same key will return the stored successful job and will **not** recreate the externally deleted post.

That is intentional server-side idempotency. External deletion reconciliation is not implemented.

For a new logical publish after deletion, use a new idempotency key.

---

## 13. Threads verified behavior

Verified during implementation:

```text
erika.devereux  -> image publish works
vlvt.ave        -> fresh token bootstrap fixed expired-session failure
cyootstuff      -> same token-expiry remediation required
```

A successful Erika smoke test previously produced a Threads permalink and repeating the same idempotency key returned `reused=true`.

---

# Tumblr

## 14. Tumblr architecture

Files:

```text
tumblr_feature/
├── __init__.py
├── config.py
├── db.py
├── models.py
├── router.py
└── service.py

sql/
└── 004_tumblr_feature.sql
```

Endpoints:

```text
GET  /tumblr/health
POST /tumblr/publish
```

The older:

```text
POST /post_tumblr
```

remains for compatibility but is now protected with `X-API-Key`.

New callers should use:

```text
POST /tumblr/publish
```

---

## 15. Tumblr multi-account configuration

Preferred mapping:

```text
TUMBLR_ACCOUNTS_JSON
```

Example:

```json
{
  "erika.devereux": "ERIKA_DEVEREUX",
  "vlvt.ave": "VLVT_AVE",
  "cyootstuff": "CYOOTSTUFF"
}
```

Each prefix resolves five envvars:

```text
TUMBLR_CONSUMER_KEY_<PREFIX>
TUMBLR_CONSUMER_SECRET_<PREFIX>
TUMBLR_TOKEN_<PREFIX>
TUMBLR_TOKEN_SECRET_<PREFIX>
TUMBLR_BLOG_IDENTIFIER_<PREFIX>
```

The backend also auto-discovers the existing legacy prefixes:

```text
ERIKA_DEVEREUX
VLVT_AVE
CYOOTSTUFF
```

### Adding a Tumblr account

Example account:

```text
new.pretty.blog
```

Add:

```json
"new.pretty.blog": "NEW_PRETTY_BLOG"
```

to `TUMBLR_ACCOUNTS_JSON`, and configure:

```text
TUMBLR_CONSUMER_KEY_NEW_PRETTY_BLOG
TUMBLR_CONSUMER_SECRET_NEW_PRETTY_BLOG
TUMBLR_TOKEN_NEW_PRETTY_BLOG
TUMBLR_TOKEN_SECRET_NEW_PRETTY_BLOG
TUMBLR_BLOG_IDENTIFIER_NEW_PRETTY_BLOG
```

Then redeploy/restart the serving SH01 process because the account map is loaded at process startup.

No Python or n8n workflow change is required.

---

## 16. Tumblr account safety check

Before publishing, SH01 calls Tumblr user info and verifies that the configured blog identifier belongs to the authenticated Tumblr user.

This prevents a typo/mismatched envvar from silently posting to the wrong blog.

---

## 17. Tumblr post format

The backend uses Tumblr's Neue Post Format (NPF) route:

```text
POST /v2/blog/{blog-identifier}/posts
```

Native user-uploaded media is sent via multipart form data.

The NPF JSON part must be a normal multipart field named:

```text
json
```

with **no filename**.

Correct request construction:

```python
"json": (None, json.dumps(body), "application/json")
```

The actual media part uses the identifier referenced from the NPF content block.

An earlier implementation used a JSON filename (`post.json`), which caused Tumblr error `8005` ("we don't support this media format yet"). Removing the filename fixed the live image publish.

---

## 18. Tumblr image normalization

Remote image URLs are not trusted based only on URL suffix or CDN `Content-Type`.

For image posts, SH01:

1. downloads the media;
2. decodes the actual bytes with Pillow;
3. applies EXIF orientation;
4. converts/composites transparency safely;
5. re-encodes a canonical JPEG;
6. sends `image/jpeg`;
7. includes actual image width/height in the NPF media object.

This was added after a Pinterest `.jpg` URL produced Tumblr error `8005`.

The normalization makes CDN-served WebP/PNG/etc. far less likely to break Tumblr native uploads.

---

## 19. Tumblr video support

The backend implements native video upload through the same unified endpoint.

Accepted backend validation currently allows:

```text
video/mp4
video/quicktime
```

The server downloads the remote video and sends it as Tumblr multipart NPF media.

The implementation exists, but a live video smoke test was not recorded in this chat. Treat video as implemented-but-needing-live-validation before relying on it for production-critical posting.

---

## 20. Tumblr database/idempotency

`004_tumblr_feature.sql` creates:

```text
public.tumblr_publish_jobs
```

with:

```text
UNIQUE(account_key, idempotency_key)
```

Supported states:

```text
published
draft
queue
private
```

Important job states include:

```text
started
media_failed
publish_failed
publish_ambiguous
published
```

On retry:

- `published` => returns existing result with `reused=true`;
- definite media/publish failures can be retried;
- `started` or `publish_ambiguous` => automatic retry is refused because Tumblr may already have accepted the create request and retrying could duplicate a post.

---

## 21. Tumblr verified behavior

Image publishing was verified for:

```text
erika.devereux
vlvt.ave
```

The original unsupported-format issue was fixed by image normalization plus the corrected NPF multipart JSON field.

---

# X.com

## 22. X backend implementation

Files:

```text
x_feature/
├── __init__.py
├── config.py
├── db.py
├── models.py
├── router.py
└── service.py

sql/
└── 003_x_feature.sql
```

Endpoints:

```text
GET  /x/health
POST /x/publish
```

The legacy `/post_image` path was also protected with `X-API-Key`.

The backend supports:

```text
IMAGE
VIDEO
multi-account credentials
account identity validation
idempotency
chunked video upload
ambiguous-create protection
```

`public.x_publish_jobs` stores idempotent publish state.

---

## 23. X multi-account configuration

Preferred configuration:

```text
X_ACCOUNTS_JSON
```

Example:

```json
{
  "barbrett": "BARBRETT",
  "cyootstuff": "CYOOTSTUFF"
}
```

Each prefix resolves:

```text
X_API_KEY_<PREFIX>
X_API_KEY_SECRET_<PREFIX>
X_ACCESS_TOKEN_<PREFIX>
X_ACCESS_TOKEN_SECRET_<PREFIX>
```

Legacy compatibility also supports:

```text
X_DEFAULT_ACCOUNT
API_KEY
API_KEY_SECRET
ACCESS_TOKEN
ACCESS_TOKEN_SECRET
```

For long-term operation, use the generalized prefixed configuration.

Changing the account registry requires process restart/redeploy.

---

## 24. Why X is currently parked

The live X test reached:

```text
GET/POST X API v2
```

and X returned:

```text
reason: client-not-enrolled
required_enrollment: Appropriate Level of API Access
```

The developer project/app showed v2 capability, but the old Free access tier was deprecated and the current Developer Console offered pay-per-use enrollment.

Because zero spend is preferred, the decision was:

```text
do not enroll/pay right now
do not keep rotating credentials
park X
retain backend code and n8n scaffold
```

The canonical n8n flow performs **no X network call** today.

When X is re-enabled later:

1. resolve current X API commercial/enrollment requirements;
2. verify the app/project entitlement;
3. provision account-specific credentials;
4. test `/x/health`;
5. test `/x/publish`;
6. enable the n8n X branch.

Do not rely on historical Free/Basic/Pro assumptions; X API access models change.

---

# TikTok

## 25. TikTok status

No TikTok Content Posting backend was implemented in this work.

The canonical n8n flow contains only a scaffold node that records:

```text
attempted=false
skipped=true
scaffold=true
```

and performs no network call.

When implemented later, use current official TikTok Content Posting/OAuth documentation and follow the same architecture:

```text
platform account registry
server-side secrets/tokens
authenticated SH01 endpoint
idempotent job persistence
n8n only passes logical account + media
```

---

# n8n

## 26. One canonical sequential social workflow

The intended workflow is:

```text
social-distribution-sequential-canonical-account-map
```

The parent workflow supplies a compact item:

```json
[
  {
    "image_url": "...",
    "caption": "...",
    "account": "..."
  }
]
```

For videos:

```json
[
  {
    "video_url": "...",
    "caption": "...",
    "account": "..."
  }
]
```

`media_type` is inferred as `VIDEO` when a video URL is present; otherwise it defaults to `IMAGE`.

---

## 27. Canonical account mapping

n8n maps one business/canonical account to platform-specific handles.

Recommended pattern:

```javascript
const ACCOUNT_MAP = {
  'erika.devereux': {
    tumblr: 'erika.devereux',
    threads: 'erika.devereux',
    x: '',
    tiktok: '',
  },
  'vlvt.ave': {
    tumblr: 'vlvt.ave',
    threads: 'vlvt.ave',
    x: '',
    tiktok: '',
  },
  'cyootstuff': {
    tumblr: 'cyootstuff',
    threads: 'cyootstuff',
    x: '',
    tiktok: '',
  },
};
```

While X/TikTok are parked, their mappings should preferably be blank rather than accidentally point at another account.

The previously shown working workflow had `cyootstuff` X/TikTok fields temporarily mapped to `vlvt.ave`; correct those values before either parked platform is ever activated.

---

## 28. n8n normalization behavior

The normalization node:

- requires `input.account`;
- strips a leading `@`;
- lowercases the canonical account;
- looks up the platform map;
- infers IMAGE/VIDEO;
- validates required media URL;
- builds platform account names;
- enables Threads/Tumblr only when mappings exist;
- keeps X/TikTok hard-disabled;
- normalizes caption/text/title/tags;
- computes platform-specific idempotency keys when a source ID exists;
- leaves idempotency absent when no source ID exists so the backend can use content hashing.

Possible source-id aliases include:

```text
source_id
queue_id
item_id
row_id
row_number
post_id
id
```

---

## 29. n8n idempotency strategy

When an upstream source ID is available, the workflow can generate keys shaped like:

```text
<platform>:<canonical-account>:<platform-account>:<media-type>:<source-id>
```

This prevents the same queue/source item from being posted twice to the same platform account.

If no source ID exists, SH01's platform backend generates a deterministic content hash.

---

## 30. n8n sequential failure behavior

Threads and Tumblr HTTP nodes use:

```text
fullResponse=true
neverError=true
responseFormat=json
```

This is intentional.

A Threads 4xx/5xx is captured as a result instead of crashing the workflow, so the sequence can continue to Tumblr.

Final output contains per-platform status.

Conceptually:

```json
{
  "ok": false,
  "media_type": "IMAGE",
  "accounts": {
    "threads": "...",
    "x": "...",
    "tumblr": "...",
    "tiktok": "..."
  },
  "results": {
    "threads": {
      "attempted": true,
      "ok": false,
      "statusCode": 400,
      "response": {}
    },
    "x": {
      "attempted": false,
      "ok": true,
      "skipped": true,
      "scaffold": true
    },
    "tumblr": {
      "attempted": true,
      "ok": true
    },
    "tiktok": {
      "attempted": false,
      "ok": true,
      "skipped": true,
      "scaffold": true
    }
  }
}
```

The overall production result considers the currently active platforms (Threads and Tumblr).

---

## 31. Dynamic account onboarding design

The core design goal is:

```text
adding a social account should not require editing backend code
```

There are two layers:

```text
n8n canonical account
    ↓
platform-specific logical handle
    ↓
SH01 account registry
    ↓
server-side platform credentials/tokens
```

### Threads

1. add/extend `THREADS_ACCOUNTS_JSON`;
2. redeploy/restart SH01;
3. bootstrap the new account token once;
4. add the canonical-to-Threads mapping in n8n.

### Tumblr

1. add/extend `TUMBLR_ACCOUNTS_JSON`;
2. add the five prefixed Tumblr envvars;
3. redeploy/restart SH01;
4. add the canonical-to-Tumblr mapping in n8n.

### X

When re-enabled:

1. add/extend `X_ACCOUNTS_JSON`;
2. add four prefixed X credential envvars;
3. redeploy/restart SH01;
4. map the canonical account in n8n;
5. enable the X branch.

### TikTok

Not implemented yet. Follow the same pattern when built.

---

# SH01 + infrastructure ring integration

## 32. Social features use the active Postgres runtime

Threads, Tumblr and X use the same SH01 runtime DB variables:

```text
DB_POSTGRESDB_HOST
DB_POSTGRESDB_PORT
DB_POSTGRESDB_DATABASE
DB_POSTGRESDB_USER
DB_POSTGRESDB_PASSWORD
DB_POSTGRESDB_SSL_ENABLED
```

The infra control plane now injects the active Postgres runtime into SH01 as well as n8n.

Runtime Supabase pooler configuration uses:

```text
port 6543
```

Migration tooling uses:

```text
port 5432
```

This means social idempotency/token tables move with the active `public` schema during Postgres failover.

---

## 33. Static secrets vs dynamic infra env

The infra controller updates dynamic runtime values such as:

```text
DB_POSTGRESDB_*
B2 runtime values
```

It does **not** create every static social credential automatically.

Static social secrets must exist on every SH01 Render slot that may serve traffic.

Examples:

```text
THREADS_TOKEN_ENCRYPTION_KEY
THREADS_INTERNAL_API_KEY / SOCIAL_INTERNAL_API_KEY
Tumblr account credentials
X account credentials if X is ever enabled
optional Telegram refresh-alert settings
```

Before enabling failover, keep static secrets/configuration synchronized across all serving SH01 nodes.

---

## 34. Render failover deployment mode

The current implementation uses:

```json
{
  "deployMode": "build_and_deploy"
}
```

for Render deploy triggers.

This is important because a failover/redeploy must build the latest repository commit rather than relying on a stale prebuilt artifact.

Any older documentation saying:

```text
deployMode=deploy_only
```

is outdated.

---

## 35. Current infra state used during this work

The active state during the implementation was:

```json
{
  "render": "render-a",
  "postgres": "supabase-b",
  "b2": "b2-a"
}
```

Treat this only as an implementation-time snapshot. Always query `/infra/status` before assuming it is still current.

The stable public SH01 Worker remains:

```text
https://sh01.vivojaymail.workers.dev
```

---

# Database migrations

## 36. Social migrations

Run migrations against the currently active Supabase target on migration port `5432`.

Files:

```text
sql/001_threads_feature.sql
sql/002_threads_refresh_hardening.sql
sql/003_x_feature.sql
sql/004_tumblr_feature.sql
```

Created tables:

```text
public.threads_auth
public.threads_publish_jobs
public.x_publish_jobs
public.tumblr_publish_jobs
```

RLS is enabled on these tables.

Because the repository ignores `sql/` in the current local setup, new SQL migration files may require:

```powershell
git add -f sql/<migration>.sql
```

Do not use `git add .` in this repository when helper/patch files are present.

---

# Operational tests

## 37. Threads health/token status

```powershell
Invoke-RestMethod `
  -Method GET `
  -Uri "https://sh01.vivojaymail.workers.dev/threads/tokens/status" `
  -Headers @{"X-API-Key"=$ThreadsKey} |
  ConvertTo-Json -Depth 30
```

Remember that stored expiry is not necessarily authoritative live expiry.

---

## 38. Threads publish smoke test

Use a fresh logical idempotency key for a fresh smoke test:

```json
{
  "account": "vlvt.ave",
  "media_type": "IMAGE",
  "image_url": "https://example.com/test.jpg",
  "text": "Example caption",
  "idempotency_key": "threads-vlvt-smoke-001"
}
```

A successful retry of the same key should return `reused=true`.

---

## 39. Tumblr health

```powershell
Invoke-RestMethod `
  -Uri "https://sh01.vivojaymail.workers.dev/tumblr/health" `
  -Headers @{"X-API-Key"=$ThreadsKey} |
  ConvertTo-Json -Depth 20
```

---

## 40. Tumblr image smoke test

Example body:

```json
{
  "account": "erika.devereux",
  "media_type": "IMAGE",
  "image_url": "https://example.com/test.jpg",
  "text": "Tumblr API smoke test",
  "tags": ["test", "automation"],
  "state": "published",
  "idempotency_key": "tumblr-smoke-erika-001"
}
```

Retrying a successful key should return `reused=true`.

---

# Troubleshooting

## 41. Threads: HTTP 400 with apparently healthy token

Symptoms:

```text
/threads/tokens/status says configured
stored expires_at is in the future
POST /threads/publish returns 400
```

Inspect the structured response body.

If Meta says:

```text
OAuthException
code=190
Session has expired
```

generate a fresh Threads token and bootstrap the affected account.

Do not assume the stored 60-day expiry is authoritative.

---

## 42. Tumblr: error 8005 unsupported media format

Two fixes are already in the backend:

1. normalize downloaded images into canonical JPEG using Pillow;
2. send the NPF `json` multipart part with no filename:

```python
"json": (None, json.dumps(body), "application/json")
```

If this error returns again, first verify the deployed commit includes both fixes.

---

## 43. Corporate certificate interception

On the corporate laptop, direct curl may fail TLS verification because of the corporate certificate chain.

For one-off diagnostics in Bash, `curl -k` was used to bypass certificate verification.

Example:

```bash
curl -k -i ...
```

Use this only for controlled diagnostics. Do not make disabling TLS verification a normal production behavior.

---

## 44. n8n HTTP node credential

Both active social nodes can use the same header credential:

```text
Post Threads
Post Tumblr
```

Credential:

```text
Header: X-API-Key
Value: <SH01 internal key>
```

Tumblr OAuth keys/tokens stay on SH01; they are not stored in the n8n workflow.

---

# Security

## 45. Never commit

Do not commit:

```text
Threads access tokens
THREADS_TOKEN_ENCRYPTION_KEY
THREADS_INTERNAL_API_KEY
SOCIAL_INTERNAL_API_KEY
Tumblr consumer secrets
Tumblr token secrets
X API secrets/tokens
database passwords
Cloudflare router-control keys
Render API key
B2 application keys
Telegram bot token
```

Do not paste live secrets into chat logs, READMEs, issue descriptions, workflow sticky notes, or Git history.

---

## 46. Local helper files

During implementation many one-shot helper scripts and workflow exports were intentionally left untracked.

Examples include:

```text
install_x_social_publish_patch.py
install_tumblr_social_distribution_patch.py
fix_tumblr_media_normalization.py
fix_tumblr_npf_multipart_json_part.py
fix_threads_upstream_error_propagation.py
social-publish-unified.json
social-distribution-sequential.json
```

Do not accidentally stage them with:

```text
git add .
```

Stage only intended production files.

---

# Current implementation history

## 47. Relevant commits

Key commits from this implementation sequence:

```text
da46acc4  Add unified Threads publishing and token management
df6559fc  Integrate SH01 Postgres runtime with infra ring
be4abe93  Build latest commit during Render failover
03d6a0ea  Harden Threads token refresh lifecycle
6a00dc2c  Harden Threads post-refresh verification
a96247f3  Add authenticated idempotent X publishing
0e96c8dc  Add unified multi-account Tumblr publishing
5d74cf59  Normalize Tumblr image uploads
76f409af  Fix Tumblr NPF multipart upload
7942c5d0  Surface Threads upstream API errors
5ee18d03  Handle Threads publish API errors
```

Use Git history for the final authoritative SHA if additional commits have landed after this documentation update.

---

# Known follow-ups

## 48. Recommended remaining work

1. **Threads token expiry hardening**
   - distinguish assumed and authoritative expiry;
   - optionally perform live token verification;
   - avoid reporting assumed expiry as definitive health.

2. **Tumblr video smoke test**
   - backend path exists;
   - perform one real MP4/MOV live post and idempotency retry.

3. **Static-secret parity**
   - verify every SH01 Render serving slot has identical required social static secrets before exercising infra failover.

4. **X**
   - keep scaffolded until current API enrollment/pay-per-use is intentionally enabled.

5. **TikTok**
   - implement only when ready, using current official Content Posting API requirements.

6. **Canonical n8n map**
   - keep X/TikTok values blank while parked;
   - avoid accidental cross-account mappings.

---

# Final operating model

```text
Parent workflow
    |
    | { media URL, caption, canonical account }
    v
Canonical social distributor (n8n)
    |
    +--> map canonical account -> platform handles
    |
    +--> Threads -> SH01 /threads/publish
    |
    +--> X scaffold (no call)
    |
    +--> Tumblr -> SH01 /tumblr/publish
    |
    +--> TikTok scaffold (no call)
    |
    v
Combined platform result

SH01
    |
    +--> active Postgres runtime
    |      ├── Threads encrypted auth
    |      ├── Threads publish jobs
    |      ├── Tumblr publish jobs
    |      └── X publish jobs
    |
    +--> server-side platform credentials
    |
    +--> platform APIs

Cloudflare Worker
    |
    +--> preferred SH01 Render backend
    +--> fallback SH01 Render backend
```

The intended separation is:

```text
n8n       = orchestration + canonical account routing
SH01      = auth, platform API behavior, idempotency, credential isolation
Postgres  = durable social auth/job state
Cloudflare/infra ring = stable ingress + failover
```

That separation should be preserved as additional platforms are added.
