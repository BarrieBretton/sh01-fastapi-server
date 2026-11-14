## Canonical home

These modules run in this existing SH01 FastAPI application. No separate Python service is required. They are mounted by app.py.

> **Current instructions (5 November 2026):** [Implementation and acceptance](../docs/IMPLEMENTATION-ACCEPTANCE.md). The historical setup and deployment observations below apply to their original revision. For this upgrade, retain the existing verified Sheets cutover, use the single minute cron, and follow the current guide for Meta pacing, companion integration and live checks.

<!-- SH01 DIRECT SHEETS FOLLOW-UP BEGIN -->
> **Current storage instructions:** the existing Cloudflare Worker calls the Google Sheets API directly using its existing Google OAuth credentials. No Apps Script project, gateway URL or gateway secret is required. Follow [../SHEETS-MIGRATION.md](../SHEETS-MIGRATION.md): deploy with `SHEETS_STORAGE_READY=0`, verify the live write guard and all-table copy, then set `SHEETS_STORAGE_READY=1`. Cloud deployment and cutover remain pending until verified. Existing optional business-sheet write flags and platform credentials remain in their current scopes. Older storage/deployment instructions below are historical where they differ from this guide.
<!-- SH01 DIRECT SHEETS FOLLOW-UP END -->

# SH01 owner social-media inventory companion

This is an optional Python router for the existing SH01 FastAPI server. It reads configured owner accounts on Facebook, Instagram and Tumblr, persists post metadata and fingerprints in the Operations Dashboard, and suggests duplicate **groups for human review**. It does not publish, edit or delete social posts. It does not append links to captions.

## Status and evidence

| Status | Evidence |
| --- | --- |
| Code-confirmed | Bounded owner-media API pagination, signed resume cursor, separate image/video comparison, optional crop verification, protected dashboard ingestion. |
| Locally tested | Unit tests use synthetic images, ordered video fingerprints, a real ffmpeg-generated test video, mock APIs and mock dashboard ingestion. |
| Unverified | Live Meta/Tumblr authorization, account ownership/IDs, provider API results, deployed server resources, dashboard production integration, crop-match precision and recall on the owner's real posts. |

The existing SH01 app and deployment were not changed by preparing these files. There is no claim that this endpoint is currently deployed or that every historical post can be obtained from platform APIs.

## 1. Install in the SH01 Python repository

Preferred for the inspected SH01 checkout: run the documentation follow-up patch
from the dashboard checkout with `--fastapi-repo` pointing to that separate repo.
It installs **social_inventory.py and social_delete.py**, registers both routers,
adds `oauthlib>=3.3,<4` directly to the existing Docker-consumed requirements.txt,
and writes README/handbook/subsystem documentation in that repo. It preserves unrelated app.py edits after validating the FastAPI constructor
and router bindings, and refuses conflicting requirements.txt edits. No cloud operation is performed.
The manual inventory-only setup below is an alternative; include the deletion
router separately if needed. See the complete [configuration register](../CONFIGURATION-MATRIX.md).


Copy `social_inventory.py` and `requirements-social.txt` next to the existing SH01 `app.py`. Keep the test file in your development checkout.

Use your existing project virtual environment, or create an isolated one for local verification:

```bash
python -m venv .venv-social
.venv-social/bin/python -m pip install -r requirements-social.txt
# Needed only for the tests' FastAPI client:
.venv-social/bin/python -m pip install httpx
# Optional geometric verification of cropped images:
.venv-social/bin/python -m pip install 'opencv-python-headless>=4.10,<5'
```

`ffmpeg` and `ffprobe` must already be available on the server PATH to fingerprint videos. They are existing SH01 media dependencies, not Python packages. Without them, video post metadata is still ingested and each video gets an explicit `ffmpeg_not_installed` item error. Do not install these packages into the system Python environment.

In `app.py`, after `app = FastAPI(...)`, add exactly:

```python
from social_inventory import router as social_inventory_router
app.include_router(social_inventory_router)
```

Include the additional requirements in the SH01 deployment build. For the normal isolated Render build environment:

```bash
python -m pip install -r requirements.txt -r requirements-social.txt
```

Add the optional OpenCV package to that environment only if you enable ORB. Do not add it globally. Test memory usage on the actual instance before increasing limits. A single scan per Python process is allowed; multiple Uvicorn workers/replicas have independent limits.

## 2. Configure server secrets and account references

| Variable | Meaning |
| --- | --- |
| `SOCIAL_INVENTORY_API_KEY` | Separate long secret required in the caller's `X-API-KEY` header. Also signs resume cursors. |
| `OPS_DASHBOARD_URL` | Dashboard HTTPS origin only, e.g. `https://YOUR-DASHBOARD.workers.dev`; no path/query. |
| `INGEST_TOKEN` | Same secret as the dashboard Worker `INGEST_TOKEN`; sent in `X-INGEST-KEY`. This companion deliberately uses `INGEST_TOKEN`, not the older error collector's `OPS_INGEST_TOKEN` name. |
| `SOCIAL_ACCOUNTS_JSON` | JSON registry below: account IDs, platform, handle and environment-variable names only. |
| `META_GRAPH_VERSION` | Explicit Graph API version; default `v24.0` matches the existing dashboard baseline. This is not a claim that v24.0 is the newest version. Verify your app permissions against the selected version. |
| `SOCIAL_SCAN_MAX_SECONDS` | Media discovery/comparison time budget; default 120, clamped to 15–300 seconds. A separate 30 seconds is reserved for dashboard delivery. |
| `SOCIAL_ENABLE_ORB` | `1` opts into OpenCV feature fingerprints and geometric comparison; default `0`. Requires optional package. |

Example account registry (replace placeholder identifiers; the values below are **not** real account IDs):

```json
{
  "facebook-owned-1": {
    "platform": "facebook",
    "provider_id": "123456789012345",
    "handle": "owned-page",
    "token_env": "FB_OWNED_PAGE_TOKEN",
    "workflow_id": "OPTIONAL_WORKFLOW_ID"
  },
  "instagram-owned-1": {
    "platform": "instagram",
    "provider_id": "17840000000000000",
    "handle": "owned-instagram",
    "token_env": "IG_OWNED_ACCESS_TOKEN"
  },
  "tumblr-owned-1": {
    "platform": "tumblr",
    "provider_id": "owned-blog.tumblr.com",
    "handle": "owned-blog",
    "oauth_env": {
      "consumer_key": "TUMBLR_CONSUMER_KEY",
      "consumer_secret": "TUMBLR_CONSUMER_SECRET",
      "token": "TUMBLR_OAUTH_TOKEN",
      "token_secret": "TUMBLR_OAUTH_TOKEN_SECRET"
    }
  }
}
```

The registry key is the dashboard `account_id`. Optional `provider_id` supplies the actual Meta numeric ID or Tumblr blog identifier. Without it, the key is used as the provider ID. Registry IDs must match the account IDs used by the dashboard account register. `platform` is lowercase. `workflow_id` is optional and is forwarded verbatim as an identifier; no workflow is changed.

Store every referenced platform token/secret in the server's secret environment. Tumblr can alternatively use an authorized OAuth 2 bearer token via `token_env` instead of `oauth_env`. This companion does not obtain, refresh or expand tokens.

### Platform access

- **Facebook:** a Page you administer, with an authorized Page access token that can read its posts and requested metrics/video sources. Uses `/{PAGE_ID}/posts` and, for native videos whose attachment has no source, `/{VIDEO_ID}?fields=source`. A thumbnail is never substituted for a video fingerprint. Personal profiles, arbitrary public accounts and external embedded video hosts are unsupported.
- **Instagram:** the existing **Facebook Login** Graph API path for an authorized professional account, `graph.facebook.com/{version}/{IG_USER_ID}/media`. Instagram Login tokens using `graph.instagram.com` are not interchangeable. The token needs the provider permissions for reading the owner account's media and requested engagement fields. Stories and private/personal-account scraping are unsupported.
- **Tumblr:** official API v2 `GET /blog/{blog}/posts?npf=true`, authenticated with the configured owner's OAuth credentials. NPF image/video blocks in the post's own body are read. Reblog trail media and externally hosted video URLs are not downloaded. `note_count` is deliberately not mislabeled as likes; unavailable metrics stay absent.

Provider permission sets and available fields must be verified for your app/token. A permission or field error is returned as a safe error category; the service does not quietly substitute unrelated public-feed data.

## 3. Deploy the dashboard ingestion update first

The Worker must provide these authenticated routes. All use `X-INGEST-KEY: <INGEST_TOKEN>`:

| Method and route | Contract |
| --- | --- |
| `GET /api/ingest/fingerprints?platform=instagram&account_id=instagram-owned-1&offset=0&limit=100` | `{"fingerprints":[{"post_id":"...","fingerprint":{...}}],"has_more":false}` |
| `POST /api/ingest/posts` | `{"posts":[...]}` with up to **10** posts per request from this companion. |
| `POST /api/ingest/duplicate-groups` | `{"groups":[...]}` with up to **10** candidate groups per request. |

Post rows contain `platform`, `account_id`, `account_handle`, `post_id`, `permalink`, `media_type` (`IMAGE`, `VIDEO`, `CAROUSEL`), `published_at`, `caption`, optional `workflow_id`, available `metrics` (`likes`, `comments`, `views`, `shares`) and optional `fingerprint`.

A post with a missing or unsupported canonical permalink is skipped individually, counted in `missing_permalink_skipped`, and reported as `missing_or_unsupported_permalink` with its post ID. It cannot invalidate the other posts in an ingest batch. A missing publication date is `null`; missing metrics are omitted, not replaced with zero. Caption URLs and credential-shaped assignments are stripped. Provider media URLs and tokens are never included in ingestion payloads. The dashboard can use the canonical post permalink and caption for review. This module does not supply browser media previews.

Each duplicate row has stable `group_id`, `platform`, `account_id`, `media_type`, `algorithm`, `confidence: "candidate"`, 2–100 sorted post IDs in `members`, and bounded numeric/method `evidence`. Groups use deterministic, disjoint greedy complete-link clustering: each candidate must match **every** existing member before joining. If A matches B and B matches C but A does not match C, all three cannot form one group. Three mutually matching posts produce one group of three, so the dashboard can retain a chosen number from a larger group.

Each group starts with a post visited by this scan. Cached-to-cached comparisons required to verify its other members also consume the comparison budget. Evidence records the verified `pair_count` (`n × (n − 1) / 2`), method counts and applicable `score_bounds` with minimum/maximum Hamming distances, video duration differences, ORB inlier ratios or exact SHA equality indicators. These are observed comparison measures, not calibrated confidence scores. Group IDs hash sorted members, account, platform and algorithm/version; output is deterministic for the same input and completed budget.

A time/pair/member cap sets `comparison_truncated`; any emitted partial group still contains only mutually verified members. This greedy method does not claim to find a largest possible cluster or cover every matching relationship. Groups are disjoint within one scan. A scan does not automatically remove older suggestions, which can overlap later groups as the available inventory changes; their lifecycle is managed by the dashboard review queue.

On any ingest failure, the resume cursor replays the original request's starting position, so idempotent dashboard upserts can recover partial or unacknowledged delivery. Counts record acknowledged batches; a network failure after the server committed a batch can undercount acknowledged rows. The same scan may be safely replayed.

## 4. Call the scan from Python or n8n

Request:

```json
{"platform":"instagram","account_id":"instagram-owned-1","max_posts":20}
```

Route: `POST https://YOUR-SH01-ORIGIN/social/inventory/scan`.

Header: `X-API-KEY` = server `SOCIAL_INVENTORY_API_KEY`. This is not the dashboard browser token, media editor key, control-plane key or queue key. Never put secrets in query strings.

Python caller, with the origin and key already loaded into the process environment:

```python
import json
import os
import urllib.request

body = {"platform": "instagram", "account_id": "instagram-owned-1", "max_posts": 20}
# For the next bounded call only, add body["cursor"] = previous_result["resume_cursor"].
req = urllib.request.Request(
    os.environ["SH01_ORIGIN"].rstrip("/") + "/social/inventory/scan",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json", "X-API-KEY": os.environ["SOCIAL_INVENTORY_API_KEY"]},
    method="POST",
)
with urllib.request.urlopen(req, timeout=400) as response:
    result = json.load(response)
print(json.dumps(result, indent=2))  # Safe summary, no tokens or media URLs.
```

For n8n use an HTTP Request node with the same JSON and a Header Auth credential. Set its timeout above the configured scan budget plus 30 seconds; an upstream proxy may impose a shorter timeout. Start with `max_posts: 5` on the deployed path. A proxy timeout does not prove the server scan stopped: allow the scan budget to expire before retrying. Do not create an unbounded pagination loop.

The response includes:

- `counts`: visited provider records, media records, unsupported non-media records, newly fingerprinted/reused fingerprints, metadata-only carousels, failed items and acknowledged ingested posts/groups.
- `coverage.provider_exhausted`: this API traversal returned no further page, **not** proof of complete account history.
- `coverage.pagination_truncated`, `stop_reason`, `cache_truncated`, `comparison_truncated`, comparison count and limits.
- `resume_cursor`: opaque signed account/platform-bound checkpoint, valid for 24 hours. It includes a provider cursor or offset and within-page position; it does not contain a paging URL or token. Rotating the inventory key invalidates old cursors.
- `errors`: bounded safe categories and affected post IDs. One inaccessible media item does not prevent other posts from being ingested. Its metadata is still saved.

`max_posts` is 1–100; default 20. It counts provider records, including unsupported non-media posts. API page size is 20. Missing/invalid caller key returns 401, an unconfigured endpoint returns 503, and a concurrent scan in the same process returns 409. Unknown accounts and invalid cursors return 400. Partial scan failures appear in a 200 summary with `ok: false`; callers must inspect the summary.

A failed media item is recorded and the cursor advances. To retry it, replay that scan's **input cursor** (or start again at the newest page), after fixing the cause. Do not mistake the next-page cursor for a failed-item retry queue.

## Bounds, comparison behavior and limitations

| Bound | Value |
| --- | --- |
| Per request provider records | Up to 100 |
| Cached fingerprints loaded | Up to 500 for the account; `cache_truncated` reports more |
| Candidate pair comparisons | Up to 15,000, including cached-to-cached verification inside groups seeded by this scan |
| Candidate groups per scan / members per group | Up to 200 / 100 |
| Media download total | 200 MiB, including provider/cache response bytes |
| Image / video file cap | 10 MiB / 40 MiB |
| Image input dimensions | At most 20 million pixels; animated images explicitly unsupported |
| Video input | MP4/MOV or Matroska/WebM containers; 0.5–900 seconds; at most 4096 × 2160 input pixels |
| Video samples | Five ordered positions: 10%, 30%, 50%, 70%, 90%; each decode has a 15-second subprocess timeout |
| HTTPS connections | 12-second socket timeout, also constrained by remaining scan budget |
| Redirects | At most three for media, with scheme/host/public-address revalidation on every hop; no authenticated API redirects |

DNS is checked for public addresses and the TLS connection is pinned to a checked address, retaining the original hostname for certificate verification. The media host suffix allowlist is fixed in source: Facebook `fbcdn.net`/`fbsbx.com`, Instagram `cdninstagram.com`/`fbcdn.net`, Tumblr `media.tumblr.com`/`vt.tumblr.com`. There is no request parameter for an arbitrary media URL. Newly introduced provider CDN hosts fail closed until the allowlist is reviewed.

Videos are downloaded to a temporary server directory then decoded locally; files are deleted when the operation completes or errors. ffmpeg is limited to local file/pipe protocols and the specified media container demuxers. Its stderr and provider error bodies are not exposed. Browser clients receive summary metadata only.

Fingerprints use algorithm `sh01-social-perceptual`, version `1`. Images have one full-frame dHash and 18 overlapping crop dHashes. A low-contrast guard limits featureless-image false positives. When enabled and installed, OpenCV adds up to 96 ORB features; a geometric inlier check can support near-crop candidates. Videos compare only with videos, with duration tolerance `max(1 second, 4% of the shorter duration)` and ordered frame hashes. Carousels are metadata-only and are never grouped automatically. Exact downloaded-file SHA-256 equality is reported separately in evidence.

These are **candidate suggestions**, not calibrated confidence scores or a guarantee of duplicate detection. Heavy crops, overlays, retiming, trims, new framing, repeated textures and genuinely different similar-looking material can cause misses or false positives. Enabling ORB does not make the method an accuracy guarantee. If OpenCV is absent the service falls back to the dHash path. The response reports whether ORB was requested; it does not assert that every item had enough ORB features.

Cached fingerprints are reused only for the same algorithm/version/media type. They are keyed by post ID; this assumes a platform post's media is immutable. A changed/edited post requires its stored fingerprint to be removed or the algorithm version incremented before recomputation. Enabling ORB alone does not recompute older dHash-only fingerprints. Cache and pair caps mean duplicate coverage can be partial even after provider pagination finishes.

Pagination represents a moving provider feed. New/deleted posts can change Tumblr offsets or API cursor results between calls. Replays are idempotent but cannot establish complete historical coverage. No provider call was made with live credentials during preparation.

## Verification

From the dashboard package root, with Python dependencies installed in your virtual environment:

```bash
python -m unittest discover -s server_extensions -p 'test_social_inventory.py' -v
python -m py_compile social_inventory.py
```

The tests do not require live credentials, platform traffic, Node or a dashboard deployment. The real synthetic video test skips explicitly if ffmpeg/ffprobe are unavailable. Optional ORB requires an additional environment with OpenCV; provider integration and actual deployment resource usage require a separately authorized staging smoke test.

## Provider reference links

- [Meta Instagram with Facebook Login](https://developers.facebook.com/docs/instagram-platform/instagram-api-with-facebook-login/)
- [Meta Page posts reference](https://developers.facebook.com/docs/graph-api/reference/page/posts/)
- [Tumblr official API v2 source](https://github.com/tumblr/docs/blob/master/api.md)
- [Tumblr official NPF format](https://github.com/tumblr/docs/blob/master/npf-spec.md)

Meta reference pages were unavailable to the documentation retrieval tool during preparation. The implementation uses the existing dashboard's Graph API version baseline and must be checked against your actual app/token before activation.


## Deletion companion

`social_delete.py` is separate from the read/ingest scanner. Install it beside
`social_inventory.py`; include `social_delete.router` in app.py. The optional
cross-repo patch installs both. It exposes POST `/social/posts/delete` with the
same inventory X-API-KEY. Default `dry_run:true` verifies registry resolution and
remote ownership; it does not delete a post. Actual execution requires
`SOCIAL_DELETE_ENABLED=1`, account `allow_delete:true`, `dry_run:false` and the
exact preview confirmation. Leave deletion disabled during setup. See
[workflow integration](../workflows/README.md) and [rollout tracker](../DEPLOYMENT-TRACKER.md).
No automatic scanner/deleter schedule is installed. Existing clickouts consumers
must be considered before appending any real deletion links.


## Installed companion and resumable operation

The coordinated FastAPI PR mounts `/social/inventory/scan`, `/social/inventory/status` and `/social/posts/delete` and installs bounded root-logger ERROR telemetry. `SOCIAL_ACCOUNTS_JSON` is an object keyed by dashboard account ID (the existing config contract), not an array. Use the dashboard repo `run_social_scan.py` for private atomic checkpoints. Use `/social/inventory/status` with the social key for configured-account counts and FFmpeg readiness; it does not test provider credentials. No deletion flag is enabled by installation.
