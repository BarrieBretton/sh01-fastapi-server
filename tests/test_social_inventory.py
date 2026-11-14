"""Run: python -m unittest discover -s server_extensions -p 'test_social_inventory.py'"""
import asyncio
import base64
import copy
import io
import json
import os
import random
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
import social_inventory as s

CFG = {"platform": "instagram", "account_id": "123", "provider_id": "123", "handle": "owned"}


def scene(seed=1):
    rng = random.Random(seed)
    im = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(im)
    for _ in range(90):
        x, y = rng.randrange(370), rng.randrange(270)
        draw.rectangle((x, y, x + rng.randrange(5, 30), y + rng.randrange(5, 30)), fill=tuple(rng.randrange(256) for _ in range(3)))
    return im


def png(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def raw_post(post_id):
    return {"id": str(post_id), "media_type": "IMAGE", "media_url": "https://cdninstagram.com/a.jpg?sig=secret", "permalink": f"https://www.instagram.com/p/{post_id}/?access_token=secret", "timestamp": "2026-11-01T01:00:00Z", "caption": "hello"}


class FakeProvider:
    def __init__(self, posts):
        self.posts, self.calls = posts, []

    def page(self, cursor, limit):
        self.calls.append((cursor, limit))
        start = int(cursor or 0)
        rows = self.posts[start:start + limit]
        return rows, str(start + len(rows)) if start + len(rows) < len(self.posts) else None

    def media_url(self, post):
        return post.media_url


class FakeHTTP:
    def __init__(self, data=None, failure=None):
        self.data, self.failure = data or png(scene()), failure
        self.calls = 0

    def request(self, *a, **k):
        self.calls += 1
        if self.failure:
            raise s.SafeError(self.failure)
        return self.data


class FakeDashboard:
    def __init__(self, cached=None, fail=False):
        self.cached, self.fail = cached or {}, fail
        self.posts, self.groups = [], []

    def cache(self, cfg):
        return dict(self.cached), False

    def ingest(self, kind, rows):
        if self.fail:
            raise s.SafeError("upstream_transport_error")
        getattr(self, kind).extend(copy.deepcopy(rows))
        return len(rows)


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SOCIAL_INVENTORY_API_KEY": "test-key-not-production", "SOCIAL_ENABLE_ORB": "0"})
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def test_multicrops_match_center_crop_and_reject_unrelated(self):
        im = scene(13)
        a = s.image_fingerprint(im)
        b = s.image_fingerprint(im.crop((40, 30, 360, 270)))
        self.assertIsNotNone(s.compare_fingerprints(a, b))
        self.assertIsNone(s.compare_fingerprints(a, s.image_fingerprint(scene(92))))

    def test_blank_images_do_not_suggest_perceptual_duplicates(self):
        a = s.image_fingerprint(Image.new("RGB", (100, 100), "black"))
        b = s.image_fingerprint(Image.new("RGB", (100, 100), "white"))
        self.assertIsNone(s.compare_fingerprints(a, b))

    def test_image_and_video_never_compare(self):
        a = s.image_fingerprint(scene())
        b = {"algorithm": s.ALGORITHM, "version": 1, "media_type": "VIDEO", "duration": 12,
             "sample_positions": [.1, .3, .5, .7, .9], "frames": [a] * 5}
        self.assertIsNone(s.compare_fingerprints(a, b))

    def test_ordered_video_samples_duration_and_blank_guard(self):
        frames = [s.image_fingerprint(scene(seed)) for seed in range(5)]
        a = {"algorithm": s.ALGORITHM, "version": 1, "media_type": "VIDEO", "duration": 12,
             "sample_positions": [.1, .3, .5, .7, .9], "frames": frames}
        self.assertIsNotNone(s.compare_fingerprints(a, copy.deepcopy(a)))
        b = dict(a, duration=20)
        self.assertIsNone(s.compare_fingerprints(a, b))
        b = dict(a, frames=list(reversed(frames)))
        self.assertIsNone(s.compare_fingerprints(a, b))
        blank = s.image_fingerprint(Image.new("RGB", (100, 100), "white"))
        self.assertIsNone(s.compare_fingerprints(dict(a, frames=[blank] * 5), dict(a, frames=[blank] * 5)))

    def test_url_allowlist_private_mixed_dns_and_rebinding_pinning(self):
        def dns(*a, **k):
            return [(2, 1, 6, "", ("93.184.216.34", 443))]
        self.assertEqual(s.validate_url("https://scontent.cdninstagram.com/a", s.CDNS["instagram"], suffix=True, resolver=dns)[1], "93.184.216.34")
        for url in ("http://cdninstagram.com/a", "https://cdninstagram.com.evil.test/a", "https://u:p@cdninstagram.com/a", "https://cdninstagram.com:8443/a", "https://localhost/a"):
            with self.assertRaises(s.SafeError):
                s.validate_url(url, s.CDNS["instagram"], suffix=True, resolver=dns)
        for address in ("127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "fc00::1"):
            def bad_dns(*a, **k):
                return dns() + [(2, 1, 6, "", (address, 443))]
            with self.assertRaises(s.SafeError):
                s.validate_url("https://cdninstagram.com/a", s.CDNS["instagram"], suffix=True, resolver=bad_dns)

    def test_redirect_is_validated_without_connecting_to_target(self):
        class Response:
            status = 302
            def getheader(self, name):
                return "https://169.254.169.254/latest/meta-data/" if name == "Location" else None
        class Connection:
            calls = 0
            def __init__(self, *a):
                Connection.calls += 1
            def request(self, *a, **k): pass
            def getresponse(self): return Response()
            def close(self): pass
        real_validate = s.validate_url
        def validate(url, allowed, **kw):
            return real_validate(url, allowed, resolver=lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))], **kw)
        with patch.object(s, "PinnedHTTPS", Connection), patch.object(s, "validate_url", validate):
            with self.assertRaises(s.SafeError):
                s.SafeHTTP(s.Budget()).request("https://cdninstagram.com/a", s.CDNS["instagram"], suffix=True, redirects=3)
        self.assertEqual(Connection.calls, 1)

    def test_scan_bound_and_signed_resume_without_lost_posts(self):
        provider, dash, http = FakeProvider([raw_post(i) for i in range(1, 28)]), FakeDashboard(), FakeHTTP()
        first = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=7), cfg=CFG, provider=provider, dashboard=dash, http=http)
        self.assertEqual(first["counts"]["visited"], 7)
        self.assertTrue(first["coverage"]["pagination_truncated"])
        self.assertNotIn("secret", json.dumps(first))
        second = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=20, cursor=first["resume_cursor"]), cfg=CFG, provider=provider, dashboard=dash, http=http)
        self.assertEqual(second["counts"]["visited"], 20)
        self.assertTrue(second["coverage"]["provider_exhausted"])
        self.assertEqual(len({p["post_id"] for p in dash.posts}), 27)
        self.assertFalse(second["coverage"]["full_history_verified"])
        self.assertTrue(all(c[1] == 20 for c in provider.calls))

    def test_cursor_rejects_tamper_wrong_account_and_expiry(self):
        cursor = s.encode_cursor({"after": "opaque", "skip": 2}, "123", "instagram")
        for c, account in ((cursor[:-1] + "z", "123"), (cursor, "different")):
            with self.assertRaises(s.SafeError):
                s.decode_cursor(c, account, "instagram")
        with patch.object(s.time, "time", return_value=0):
            old = s.encode_cursor({"after": None, "skip": 0}, "123", "instagram")
        with self.assertRaises(s.SafeError):
            s.decode_cursor(old, "123", "instagram")

    def test_failed_item_ingests_metadata_continues_and_no_raw_urls(self):
        dash = FakeDashboard()
        result = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=2), cfg=CFG,
                        provider=FakeProvider([raw_post(1), raw_post(2)]), dashboard=dash, http=FakeHTTP(failure="media_source_unavailable"))
        self.assertEqual(result["counts"]["failed_items"], 2)
        self.assertEqual(result["counts"]["posts_ingested"], 2)
        self.assertTrue(all("fingerprint" not in p for p in dash.posts))
        self.assertNotIn("access_token", json.dumps(dash.posts))
        self.assertNotIn("cdninstagram", json.dumps(result))

    def test_reuse_only_same_algorithm_version_and_type(self):
        fp = s.image_fingerprint(scene())
        for bad in (dict(fp, version=0), dict(fp, algorithm="old"), dict(fp, media_type="VIDEO")):
            http = FakeHTTP()
            r = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=1), cfg=CFG, provider=FakeProvider([raw_post(1)]), dashboard=FakeDashboard({"1": bad}), http=http)
            self.assertEqual(r["counts"]["reused"], 0)
            self.assertEqual(http.calls, 1)
        http = FakeHTTP()
        r = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=1), cfg=CFG, provider=FakeProvider([raw_post(1)]), dashboard=FakeDashboard({"1": fp}), http=http)
        self.assertEqual(r["counts"]["reused"], 1)
        self.assertEqual(http.calls, 0)

    def test_carousels_metadata_only_no_download(self):
        post, http, dash = dict(raw_post(1), media_type="CAROUSEL_ALBUM"), FakeHTTP(), FakeDashboard()
        result = s.scan(s.ScanRequest(platform="instagram", account_id="123"), cfg=CFG, provider=FakeProvider([post]), dashboard=dash, http=http)
        self.assertEqual(result["counts"]["carousels_metadata_only"], 1)
        self.assertEqual(http.calls, 0)
        self.assertEqual(dash.groups, [])

    def test_ingestion_failure_resumes_at_request_checkpoint(self):
        result = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=2), cfg=CFG,
                        provider=FakeProvider([raw_post(1), raw_post(2)]), dashboard=FakeDashboard(fail=True), http=FakeHTTP())
        self.assertEqual(result["coverage"]["stop_reason"], "dashboard_ingest_failed_replay")
        self.assertEqual(s.decode_cursor(result["resume_cursor"], "123", "instagram"), {"after": None, "skip": 0})

    def test_tumblr_npf_and_facebook_video_not_thumbnail(self):
        cfg = dict(CFG, platform="tumblr")
        raw = {"id": 1, "timestamp": 1700000000, "post_url": "https://www.tumblr.com/owned/1", "note_count": 200,
               "content": [{"type": "video", "media": {"url": "https://vt.tumblr.com/video.mp4"}}]}
        p = s.normalize_post(raw, cfg)
        self.assertEqual(p.metadata["media_type"], "VIDEO")
        self.assertEqual(p.metadata["metrics"], {"notes": 200})
        fb = {"id": "1_2", "permalink_url": "https://www.facebook.com/page/posts/2", "attachments": {"data": [{"media_type": "video", "target": {"id": "42"}, "media": {"image": {"src": "https://fbcdn.net/thumbnail.jpg"}}}]}}
        p = s.normalize_post(fb, dict(CFG, platform="facebook"))
        self.assertEqual(p.metadata["media_type"], "VIDEO")
        self.assertIsNone(p.media_url)
        self.assertEqual(p.video_id, "42")

    def test_complete_link_triple_identical_is_one_stable_group(self):
        fp = s.image_fingerprint(scene())
        records = {"a": fp, "b": fp, "c": fp}
        groups, count, truncated = s.build_groups(records, {"c"}, CFG, s.Budget())
        self.assertEqual(count, 3)  # Includes cached a-b verification.
        self.assertFalse(truncated)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["members"], ["a", "b", "c"])
        self.assertEqual(groups[0]["confidence"], "candidate")
        self.assertEqual(groups[0]["evidence"]["pair_count"], 3)
        self.assertEqual(groups[0]["evidence"]["score_bounds"]["hamming"], {"min": 0, "max": 0})
        reverse, _, _ = s.build_groups(dict(reversed(list(records.items()))), {"c"}, CFG, s.Budget())
        self.assertEqual(groups, reverse)

    def test_complete_link_bridge_does_not_merge_nonmatching_endpoints(self):
        def fp(h):
            return {"algorithm": s.ALGORITHM, "version": s.VERSION, "media_type": "IMAGE", "hashes": [h], "gray_stddev": 50}
        records = {"a": fp("0000000000000000"), "b": fp("000000000000000f"), "c": fp("00000000000000ff")}
        self.assertIsNotNone(s.compare_fingerprints(records["a"], records["b"]))
        self.assertIsNotNone(s.compare_fingerprints(records["b"], records["c"]))
        self.assertIsNone(s.compare_fingerprints(records["a"], records["c"]))
        groups, _, truncated = s.build_groups(records, set(records), CFG, s.Budget())
        self.assertFalse(truncated)
        self.assertEqual([g["members"] for g in groups], [["a", "b"]])
        self.assertEqual(groups[0]["evidence"]["pair_count"], 1)
        self.assertEqual(groups[0]["evidence"]["score_bounds"]["hamming"], {"min": 4, "max": 4})

    def test_complete_link_groups_are_disjoint_and_cover_multiple_clusters(self):
        first, second = s.image_fingerprint(scene(13)), s.image_fingerprint(scene(92))
        records = {"a": first, "b": first, "c": second, "d": second}
        groups, _, truncated = s.build_groups(records, set(records), CFG, s.Budget())
        self.assertFalse(truncated)
        self.assertEqual([g["members"] for g in groups], [["a", "b"], ["c", "d"]])
        self.assertEqual(len({member for group in groups for member in group["members"]}), 4)

    def test_complete_link_budget_keeps_verified_members_only(self):
        fp = s.image_fingerprint(scene())
        records = {"a": fp, "b": fp, "c": fp}
        with patch.object(s, "MAX_PAIRS", 1):
            groups, count, truncated = s.build_groups(records, {"a"}, CFG, s.Budget())
        self.assertEqual(count, 1)
        self.assertTrue(truncated)
        self.assertEqual([g["members"] for g in groups], [["a", "b"]])
        self.assertEqual(groups[0]["evidence"]["pair_count"], 1)
        class TimedBudget:
            def __init__(self): self.calls = 0
            def remaining(self):
                self.calls += 1
                if self.calls >= 4:
                    raise s.SafeError("scan_time_budget")
                return 1
        groups, count, truncated = s.build_groups(records, {"a"}, CFG, TimedBudget())
        self.assertEqual(count, 1)
        self.assertTrue(truncated)
        self.assertEqual([g["members"] for g in groups], [["a", "b"]])

    def test_complete_link_member_cap_reports_partial_coverage(self):
        fp = s.image_fingerprint(scene())
        with patch.object(s, "MAX_GROUP_MEMBERS", 3):
            groups, count, truncated = s.build_groups({key: fp for key in "abcd"}, {"a"}, CFG, s.Budget())
        self.assertEqual(count, 3)
        self.assertTrue(truncated)
        self.assertEqual(groups[0]["members"], ["a", "b", "c"])

    def test_meta_paging_uses_cursor_not_secret_bearing_next_url(self):
        class JSONHTTP:
            def __init__(self): self.calls = []
            def json(self, url, hosts, **kwargs):
                self.calls.append((url, hosts, kwargs))
                return {"data": [raw_post(1)], "paging": {"next": "https://evil.invalid/?access_token=do-not-follow", "cursors": {"after": "opaque-cursor"}}}
        http = JSONHTTP()
        with patch.dict(os.environ, {"OWNED_TEST_TOKEN": "bearer-secret-only-in-header"}):
            provider = s.Provider(dict(CFG, token_env="OWNED_TEST_TOKEN"), http)
            _, cursor = provider.page(None, 20)
            provider.page(cursor, 20)
        self.assertEqual(cursor, "opaque-cursor")
        self.assertTrue(all("bearer-secret" not in call[0] and "evil.invalid" not in call[0] for call in http.calls))
        self.assertIn("after=opaque-cursor", http.calls[1][0])
        self.assertEqual(http.calls[0][2]["headers"]["Authorization"], "Bearer bearer-secret-only-in-header")

    def test_ingestion_batches_ten_and_tracks_partial_acknowledgements(self):
        class JSONHTTP:
            def __init__(self): self.calls = []
            def json(self, url, hosts, **kwargs):
                self.calls.append(kwargs["payload"])
                if len(self.calls) == 2:
                    raise s.SafeError("upstream_transport_error")
                return {"accepted": len(kwargs["payload"]["posts"])}
        http = JSONHTTP()
        with patch.dict(os.environ, {"OPS_DASHBOARD_URL": "https://dashboard.example.test", "INGEST_TOKEN": "ingest-test"}):
            dashboard = s.Dashboard(http)
            with self.assertRaises(s.IngestError) as caught:
                dashboard.ingest("posts", [raw_post(i) for i in range(25)])
        self.assertEqual(caught.exception.accepted, 10)
        self.assertEqual(len(http.calls[0]["posts"]), 10)

    def test_missing_permalink_skips_only_that_post_before_ingestion(self):
        bad = dict(raw_post(1), permalink=None)
        http, dash = FakeHTTP(), FakeDashboard()
        result = s.scan(s.ScanRequest(platform="instagram", account_id="123", max_posts=2), cfg=CFG,
                        provider=FakeProvider([bad, raw_post(2)]), dashboard=dash, http=http)
        self.assertEqual(result["counts"]["missing_permalink_skipped"], 1)
        self.assertEqual([p["post_id"] for p in dash.posts], ["2"])
        self.assertEqual(http.calls, 1)
        self.assertIn({"stage": "item", "code": "missing_or_unsupported_permalink", "post_id": "1"}, result["errors"])

    def test_canonical_permalink_matches_dashboard_shapes(self):
        self.assertEqual(s.permalink("https://blog.tumblr.com/post/123/slug?token=x", "tumblr"), "https://www.tumblr.com/blog/123")
        self.assertEqual(s.permalink("https://m.facebook.com/watch/?v=123&access_token=secret", "facebook"), "https://www.facebook.com/watch?v=123")
        self.assertEqual(s.permalink("https://instagram.com/reel/AbC/?igsh=x", "instagram"), "https://www.instagram.com/reel/AbC/")
        for url in ("https://instagram.com/profile", "https://instagram.com.evil.test/p/xyz", "https://www.instagram.com:444/p/x"):
            self.assertIsNone(s.permalink(url, "instagram"))

    def test_route_auth_and_validation(self):
        app = FastAPI()
        app.include_router(s.router)
        with TestClient(app) as client:
            body = {"platform": "instagram", "account_id": "123"}
            self.assertEqual(client.post("/social/inventory/scan", json=body).status_code, 401)
            for invalid in (dict(body, max_posts=101), dict(body, url="https://arbitrary.test")):
                self.assertEqual(client.post("/social/inventory/scan", json=invalid, headers={"X-API-KEY": "test-key-not-production"}).status_code, 422)
            with patch.object(s, "scan", return_value={"ok": True}):
                self.assertEqual(client.post("/social/inventory/scan", json=body, headers={"X-API-KEY": "test-key-not-production"}).json(), {"ok": True})

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg not available")
    def test_real_ffmpeg_synthetic_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=5", "-t", "2", "-c:v", "mpeg4", "-threads", "1", "-y", str(path)], check=True)
            fp = s.fingerprint_bytes(path.read_bytes(), "VIDEO", s.Budget())
            self.assertTrue(s.valid_fingerprint(fp, "VIDEO"))
            self.assertEqual(len(fp["frames"]), 5)
            self.assertIsNotNone(s.compare_fingerprints(fp, copy.deepcopy(fp)))


if __name__ == "__main__":
    unittest.main()
