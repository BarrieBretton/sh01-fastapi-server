"""Offline HTTP identity, response cleanup, cooldown and diagnostic regressions."""
import asyncio
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import ai_compose_http as http
from ai_compose_commons import search
from ai_compose_engine import _download_sync, _download_async, resolve_assets
from ai_compose_schema import AIComposeRequest, CompositionPlan


class Response:
    def __init__(self, status=200, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    def close(self):
        self.closed = True

    def raise_for_status(self):
        if self.status_code >= 400:
            error = RuntimeError('HTTP failure')
            error.response = self
            raise error

    def iter_content(self, size):
        yield b'fixture'

    def json(self):
        return {'query': {'pages': {}}}


class MediaHTTPTests(unittest.TestCase):
    def setUp(self):
        http._cooldowns.clear()
        self.addCleanup(http._cooldowns.clear)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.url = 'https://upload.wikimedia.org/file.jpg'
        self.validation = patch('ai_compose_engine._validate_public_url')
        self.validation.start()
        self.addCleanup(self.validation.stop)

    def test_search_and_download_share_identity_and_close_responses(self):
        first, second = Response(), Response()
        get = Mock(side_effect=[first, second])
        with patch.dict('sys.modules', {'requests': types.SimpleNamespace(get=get)}):
            search('military', 'image')
            _download_sync(self.url, self.root / 'image.jpg')
        self.assertTrue(first.closed and second.closed)
        self.assertEqual((self.root / 'image.jpg').read_bytes(), b'fixture')
        for call in get.call_args_list:
            self.assertEqual(call.kwargs['headers']['User-Agent'], http.USER_AGENT)

    def test_429_blocks_host_across_candidates_until_deadline(self):
        response = Response(429, {'Retry-After': '600'})
        get = Mock(side_effect=[response, Response()])
        with patch.dict('sys.modules', {'requests': types.SimpleNamespace(get=get)}), \
             patch.object(http.time, 'monotonic', return_value=100) as clock:
            with self.assertRaises(http.MediaHTTPBackoff) as error:
                _download_sync(self.url, self.root / 'first')
            self.assertEqual(error.exception.retry_after_seconds, 600)
            clock.return_value = 200
            with self.assertRaises(http.MediaHTTPBackoff):
                _download_sync('https://upload.wikimedia.org/other.webm', self.root / 'second')
            self.assertEqual(get.call_count, 1)
            http.check_cooldown('https://another.example/file')
            clock.return_value = 701
            _download_sync(self.url, self.root / 'third')
            self.assertEqual(get.call_count, 2)
        self.assertTrue(response.closed)
        self.assertFalse((self.root / 'first').exists())

    def test_retry_after_date_invalid_and_503(self):
        with patch.object(http.time, 'time', return_value=0):
            self.assertEqual(http.retry_after_seconds('Thu, 01 Jan 1970 00:10:00 GMT'), 600)
        self.assertEqual(http.retry_after_seconds('invalid'), 600)
        with self.assertRaises(http.MediaHTTPBackoff):
            http.observe_response(self.url, Response(503, {'Retry-After': '60'}))
        with self.assertRaises(http.MediaHTTPBackoff):
            http.check_cooldown(self.url)

    def test_missing_retry_after_defaults_to_ten_minutes(self):
        with self.assertRaises(http.MediaHTTPBackoff) as error:
            http.observe_response(self.url, Response(429))
        self.assertEqual(error.exception.retry_after_seconds, 600)

    def test_redirects_close_and_validate_each_hop(self):
        first, second = Response(302, {'location': '/next.jpg'}), Response()
        get = Mock(side_effect=[first, second])
        with patch.dict('sys.modules', {'requests': types.SimpleNamespace(get=get)}), \
             patch('ai_compose_engine._validate_public_url') as validate:
            _download_sync(self.url, self.root / 'image')
        self.assertTrue(first.closed and second.closed)
        self.assertEqual(validate.call_count, 2)

    def test_reports_include_status_and_delay_without_url_secrets(self):
        request = AIComposeRequest(source_path='/tmp/source.mp4', fetch_assets=True)
        plan = CompositionPlan(base_segments=[{'source_start': 0, 'source_end': 5}],
            asset_requests=[{'query': 'military', 'kind': 'image', 'start': 0, 'end': 2}])
        candidate = {'asset_id': 'commons_1', 'title': 'Military', 'score': 1,
                     'download_url': self.url + '?token=PRIVATE'}
        get = Mock(return_value=Response(429, {'Retry-After': '600'}))
        with patch.dict('sys.modules', {'requests': types.SimpleNamespace(get=get)}), \
             patch('ai_compose_engine._search_remote', return_value=[candidate]), \
             patch.object(http.time, 'monotonic', return_value=100):
            for round_number in range(2):
                root = self.root / str(round_number)
                resolve_assets(request, plan, root)
                text = (root / 'retrieval_report.json').read_text(encoding='utf-8')
                report = json.loads(text)[0]
                error = report['download_decode_errors'][0]
                self.assertEqual(report['outcome'], 'host_cooldown')
                self.assertEqual(error['http_status'], 429)
                self.assertEqual(error['retry_after_seconds'], 600)
                self.assertEqual(error['host'], 'upload.wikimedia.org')
                self.assertNotIn('PRIVATE', text)
        self.assertEqual(get.call_count, 1)

    def test_other_http_status_is_retained_without_cooldown(self):
        response = Response(404)
        get = Mock(return_value=response)
        with patch.dict('sys.modules', {'requests': types.SimpleNamespace(get=get)}):
            with self.assertRaises(RuntimeError) as caught:
                _download_sync(self.url, self.root / 'image')
        self.assertEqual(http.failure_details(caught.exception, self.url)['http_status'], 404)
        http.check_cooldown(self.url)
        self.assertTrue(response.closed)

    def test_async_download_uses_same_identity_and_cooldown(self):
        response = Response(429, {'Retry-After': '600'})
        class Stream:
            async def __aenter__(self):
                return response
            async def __aexit__(self, *args):
                response.close()
        stream = Mock(return_value=Stream())
        class Client:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def stream(self, *args, **kwargs):
                return stream(*args, **kwargs)
        fake = types.SimpleNamespace(AsyncClient=lambda **kwargs: Client())
        with patch.dict('sys.modules', {'httpx': fake}):
            with self.assertRaises(http.MediaHTTPBackoff):
                asyncio.run(_download_async(self.url, self.root / 'image'))
        self.assertEqual(stream.call_args.kwargs['headers']['User-Agent'], http.USER_AGENT)
        self.assertTrue(response.closed)
        with self.assertRaises(http.MediaHTTPBackoff):
            http.check_cooldown(self.url)


if __name__ == '__main__':
    unittest.main()
