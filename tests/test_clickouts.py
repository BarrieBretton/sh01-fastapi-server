"""Offline deletion, scheduling, auth and Google adapter regression tests."""
import ast
import copy
import json
import os
import sys
import time
import types
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import clickouts_routes as routes
import clickouts_worker as worker
from clickouts_state import ClickoutError, GoogleClickouts, RedisState


def row(request='r1', post='200', **extra):
    return {'urls': 'https://www.instagram.com/p/ABC/', 'platform': 'instagram',
            'account_id': '100', 'post_id': post, 'request_id': request, 'status': '', **extra}


def schedule(workflow='Server::Clickouts', timings='10:00, 18:30', job_override=''):
    return [['Workflow', 'Timings', 'Associated Accounts', 'Run as Job Today'],
            [workflow, timings, '', job_override]]


class Redis:
    def __init__(self): self.data = {}; self.fail = False
    def get(self, key): return self.data.get(key)
    def set(self, key, value, nx=False, ex=None):
        if self.fail: raise OSError('private redis URL must not escape')
        if nx and key in self.data: return False
        self.data[key] = value
        return True
    def eval(self, script, count, key, owner):
        if self.fail: raise OSError('private redis URL must not escape')
        if self.data.get(key) != owner: return 0
        if "'del'" in script: self.data.pop(key)
        return 1


class Store:
    def __init__(self, rows):
        self.rows = copy.deepcopy(rows); self.schedule = schedule(); self.writes = []
        self.fail_done = False; self.fail_processing = False; self.prepared = False
        self.schedule_reads = 0
    def schedule_rows(self): self.schedule_reads += 1; return self.schedule
    def read(self): return copy.deepcopy(self.rows)
    def prepare(self): self.prepared = True
    def current(self, row): return next(r for r in self.rows if r['request_id'] == row['request_id'])
    def bind(self, row): self.verify(row)
    def verify(self, row):
        current = self.current(row)
        if any(current.get(k, '') != row.get(k, '') for k in (*GoogleClickouts.REQUIRED, 'status')):
            raise ClickoutError('queued_row_identity_changed')
    def update(self, row, **changes):
        self.verify(row)
        if changes.get('status') == 'done' and self.fail_done: raise ClickoutError('sheets_request_failed')
        if changes.get('status') == 'processing' and self.fail_processing: raise ClickoutError('sheets_request_failed')
        self.current(row).update(changes); row.update(changes); self.writes.append(changes)


class Provider:
    def __init__(self): self.deleted = []; self.verified = []; self.error = None; self.verify_error = None
    def verify(self, row):
        self.verified.append(row['post_id'])
        if self.verify_error: raise self.verify_error
    def delete(self, row):
        self.deleted.append(row['post_id'])
        if self.error: raise self.error
        return {'status': 'done', 'post_id': row['post_id'], 'deleted_at': '2026-11-15T10:00:00Z'}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {}, clear=True); self.env.start()
        self.redis = Redis(); self.state = RedisState(self.redis, 'book')
        self.store = Store([row()]); self.provider = Provider()
    def tearDown(self): self.env.stop()
    def service(self, state=None, ack=None):
        return worker.ClickoutsWorker(self.store, state or self.state,
            provider_factory=lambda: self.provider, ack=ack or (lambda r: 'not_configured'), enabled=lambda: True)
    def test_preview_is_read_only_and_never_calls_provider(self):
        result = self.service().preview()
        self.assertEqual(result['eligible'], 1); self.assertFalse(self.store.prepared)
        self.assertEqual(self.redis.data, {}); self.assertEqual(self.provider.verified, [])
        self.assertEqual(self.store.writes, [])
    def test_done_only_after_confirmation_preserves_row_and_id(self):
        self.store.rows[0]['post_id'] = '17954543838036274'
        self.service().run()
        self.assertEqual(len(self.store.rows), 1)
        self.assertEqual(self.store.rows[0]['status'], 'done')
        self.assertEqual(self.store.rows[0]['post_id'], '17954543838036274')
        self.assertEqual(self.provider.deleted, ['17954543838036274'])
        self.service(RedisState(self.redis, 'book')).run()
        self.assertEqual(len(self.provider.deleted), 1)
    def test_permission_or_owner_failure_before_delete_is_blocked(self):
        self.provider.verify_error = ClickoutError('provider_owner_or_post_mismatch')
        self.service().run()
        self.assertEqual(self.store.rows[0]['status'], 'blocked'); self.assertEqual(self.provider.deleted, [])
    def test_ambiguous_delete_never_retried_even_if_status_cleared(self):
        self.provider.error = TimeoutError('secret')
        self.service().run()
        self.assertEqual(self.store.rows[0]['status'], 'manual_required')
        self.store.rows[0]['status'] = ''
        self.service(RedisState(self.redis, 'book')).run()
        self.assertEqual(len(self.provider.deleted), 1)
        self.assertNotIn('secret', str(self.store.rows))
    def test_deleted_then_sheet_failure_recovers_without_another_delete(self):
        self.store.fail_done = True
        with self.assertRaises(ClickoutError): self.service().run()
        self.assertEqual(self.store.rows[0]['status'], 'processing')
        self.store.fail_done = False
        self.service(RedisState(self.redis, 'book')).run()
        self.assertEqual(self.store.rows[0]['status'], 'done'); self.assertEqual(len(self.provider.deleted), 1)
    def test_processing_marker_failure_prevents_delete(self):
        self.store.fail_processing = True
        with self.assertRaises(ClickoutError): self.service().run()
        self.assertEqual(self.provider.deleted, [])
        self.store.fail_processing = False
        self.service().run()
        self.assertEqual(self.store.rows[0]['status'], 'done')
        self.assertEqual(len(self.provider.deleted), 1)
    def test_sheet_processing_survives_redis_data_loss(self):
        self.store.rows[0]['status'] = 'processing'
        self.service(RedisState(Redis(), 'book')).run()
        self.assertEqual(self.provider.deleted, []); self.assertEqual(self.store.rows[0]['status'], 'manual_required')
    def test_receipt_record_failure_after_delete_remains_manual(self):
        original = self.provider.delete
        def delete(row):
            receipt = original(row); self.redis.fail = True; return receipt
        self.provider.delete = delete
        with self.assertRaisesRegex(ClickoutError, 'redis_request_failed'): self.service().run()
        self.redis.fail = False
        # Simulate lease expiry after Redis recovers.
        self.redis.data.pop(self.state.prefix + 'lease', None)
        self.service(RedisState(self.redis, 'book')).run()
        self.assertEqual(len(self.provider.deleted), 1); self.assertEqual(self.store.rows[0]['status'], 'manual_required')
    def test_two_requests_for_same_post_delete_once(self):
        self.store.rows.append(row('r2'))
        self.service().run()
        self.assertEqual([r['status'] for r in self.store.rows], ['done', 'done'])
        self.assertEqual(self.provider.deleted, ['200'])
    def test_invalid_request_duplicates_do_not_starve_valid_rows(self):
        self.store.rows = [row(), row(), row('good', '201')]
        result = self.service().run(1)
        self.assertEqual(result['invalid_rows'], 2); self.assertEqual(self.provider.deleted, ['201'])
    def test_old_numeric_ids_and_unsupported_platforms_never_deleted(self):
        self.store.rows = [row('numeric', post=17954543838036274), row('other', platform='tumblr')]
        self.service().run()
        self.assertEqual(self.provider.deleted, []); self.assertTrue(all(r['status'] == 'blocked' for r in self.store.rows))
    def test_busy_lease_and_lost_lease_prevent_delete(self):
        self.state.acquire()
        other = RedisState(self.redis, 'book')
        self.assertEqual(self.service(other).run()['status'], 'busy')
        self.state.release()
        def verify(row): self.redis.data[self.state.prefix + 'lease'] = 'replacement'
        self.provider.verify = verify
        with self.assertRaisesRegex(ClickoutError, 'lease_lost'): self.service().run()
        self.assertEqual(self.provider.deleted, [])
        self.assertEqual(self.redis.data[self.state.prefix + 'lease'], 'replacement')
    def test_global_disabled_does_not_touch_sheet(self):
        service = self.service(); service.enabled = lambda: False
        self.assertEqual(service.run()['status'], 'disabled'); self.assertFalse(self.store.prepared)
    def test_verification_rate_limit_defers_and_survives_restart(self):
        self.provider.verify_error = ClickoutError('meta_rate_limited', 300)
        self.service().run()
        result = self.service(RedisState(self.redis, 'book')).run()
        self.assertEqual(result['status'], 'cooldown'); self.assertEqual(len(self.provider.verified), 1)
        self.assertEqual(self.store.rows[0]['status'], '')
    def test_delete_rejection_not_marked_done_or_retried(self):
        self.provider.error = ClickoutError('meta_rate_limited', 300)
        self.service().run()
        self.assertEqual(self.store.rows[0]['status'], 'manual_required')
        self.assertEqual(self.service().run()['status'], 'cooldown')
    def test_changed_row_after_owner_check_is_not_deleted(self):
        self.provider.verify = lambda r: self.store.rows[0].update(post_id='999')
        with self.assertRaises(ClickoutError): self.service().run()
        self.assertEqual(self.provider.deleted, [])
    def test_callback_retry_never_repeats_delete(self):
        os.environ['OPS_INGEST_TOKEN'] = 'fake'
        self.service(ack=lambda r: 'pending').run()
        self.service(ack=lambda r: 'synced').run()
        self.assertEqual(self.store.rows[0]['dashboard_ack'], 'synced')
        self.assertEqual(self.provider.deleted, ['200'])
    def test_handwritten_done_without_receipt_cannot_report_provider_deletion(self):
        os.environ['OPS_INGEST_TOKEN'] = 'fake'
        self.store.rows[0]['status'] = 'done'
        ack = Mock()
        self.service(ack=ack).run()
        ack.assert_not_called()
        self.assertEqual(self.store.rows[0]['status'], 'done')
        self.assertEqual(self.store.rows[0]['dashboard_ack'], 'unverified_receipt')
    def test_schedule_resumes_batches_and_runs_once_per_daily_time(self):
        self.store.rows = [row(str(i), str(200+i)) for i in range(4)]
        at = datetime(2026, 11, 15, 5, 0, tzinfo=timezone.utc)  # 10:30 IST, catches 10:00.
        self.service().run(2, True, at)
        self.redis.data.pop(self.state.prefix + 'next_poll')
        self.service(RedisState(self.redis, 'book')).run(2, True, at)
        self.redis.data.pop(self.state.prefix + 'next_poll')
        self.store.rows.append(row('new', '999'))
        self.assertEqual(self.service().run(2, True, at)['status'], 'waiting')
        self.assertEqual(len(self.provider.deleted), 4)
        self.redis.data.pop(self.state.prefix + 'next_poll')
        evening = datetime(2026, 11, 15, 13, 0, tzinfo=timezone.utc)
        self.service().run(2, True, evening)
        self.assertEqual(self.provider.deleted[-1], '999')
    def test_missing_timings_or_unregistered_schedule_cannot_execute(self):
        self.store.schedule = schedule(timings='')
        self.assertEqual(self.service().run(scheduled=True)['status'], 'paused')
        self.redis.data.pop(self.state.prefix + 'next_poll')
        self.store.schedule = schedule('Python::arbitrary_code')
        self.assertEqual(self.service().run(scheduled=True)['status'], 'paused')
        self.assertEqual(self.provider.deleted, [])
    def test_due_clickouts_runs_with_blank_no_or_yes_job_override_and_keeps_cells(self):
        at = datetime(2026, 11, 16, 15, 30, tzinfo=timezone.utc)  # 9:00pm IST.
        for override in ('', 'no', 'yes', False, True, 0, 1):
            with self.subTest(override=override):
                store = Store([row()]); store.schedule = schedule(timings='9:00pm', job_override=override)
                original = copy.deepcopy(store.schedule)
                provider = Provider()
                service = worker.ClickoutsWorker(store, RedisState(Redis(), 'book'),
                    provider_factory=lambda: provider, ack=lambda r:'not_configured', enabled=lambda:True)
                result = service.run(scheduled=True, now=at)
                self.assertEqual(result['done'], 1)
                self.assertEqual(provider.deleted, ['200'])
                self.assertEqual(store.schedule, original)
    def test_yes_override_does_not_execute_before_time_or_repeat_completed_occurrence(self):
        self.store.schedule = schedule(timings='9:00pm', job_override='yes')
        before = datetime(2026, 11, 16, 15, 29, tzinfo=timezone.utc)
        self.assertEqual(self.service().run(scheduled=True, now=before)['status'], 'waiting')
        self.assertEqual(self.provider.deleted, [])
        self.redis.data.pop(self.state.prefix+'next_poll')
        due = datetime(2026, 11, 16, 15, 30, tzinfo=timezone.utc)
        self.service().run(scheduled=True, now=due)
        self.store.rows.append(row('later', '201'))
        for override in ('no', 'yes'):
            self.store.schedule = schedule(timings='9:00pm', job_override=override)
            self.redis.data.pop(self.state.prefix+'next_poll')
            self.assertEqual(self.service().run(scheduled=True, now=due)['status'], 'waiting')
        self.assertEqual(self.provider.deleted, ['200'])
    def test_poll_is_shared_across_replicas(self):
        self.service().run(scheduled=True, now=datetime(2026, 11, 15, 1, tzinfo=timezone.utc))
        self.service(RedisState(self.redis, 'book')).run(scheduled=True)
        self.assertEqual(self.store.schedule_reads, 1)


class ScheduleAndProviderTests(unittest.TestCase):
    def test_aliases_times_and_disabled_empty_time(self):
        self.assertTrue(worker.read_schedule(schedule(' Python :: Clickouts '))['enabled'])
        self.assertEqual(worker.parse_times('9:30 AM; 6:00 PM\n00:00'), [0, 570, 1080])
        self.assertEqual(worker.parse_times(.5), [720])
        self.assertFalse(worker.read_schedule(schedule(timings=''))['enabled'])
        for bad in ['24:00', '12:99', '0:00PM', 'every hour', '', '10:00<script>']:
            with self.assertRaises(ClickoutError): worker.parse_times(bad)
        with self.assertRaises(ClickoutError): worker.read_schedule(schedule() + [schedule('Python::Clickouts')[1]])
    def test_only_workflow_and_timings_headers_are_required(self):
        self.assertEqual(worker.read_schedule([['Workflow','Timings'], ['Server::Clickouts','11:58pm']])['times'], [1438])
        values = [['Timings','Run as Job Today','Workflow','Run as Job Today'], ['3:30pm','yes','Python::Clickouts','no']]
        self.assertEqual(worker.read_schedule(values)['times'], [930])
    def test_job_override_values_do_not_change_daily_schedule(self):
        expected = worker.read_schedule(schedule(timings='11:45pm'))
        for override in ('', 'yes', 'no', 'TRUE', 'FALSE', True, False, 0, 1, 'custom one-time args'):
            with self.subTest(override=override):
                self.assertEqual(worker.read_schedule(schedule(timings='11:45pm', job_override=override)), expected)
        for blank in ('', '  ', None):
            data = worker.read_schedule(schedule(timings=blank, job_override='yes'))
            self.assertFalse(data['enabled']); self.assertEqual(data['reason'], 'workflow_timings_empty')
        with self.assertRaises(ClickoutError): worker.read_schedule(schedule(timings='invalid', job_override='no'))
    def test_screenshot_time_formats_and_noon_midnight(self):
        expected = {'9:00pm':1260, '3:30pm':930, '11:00pm':1380, '11:45pm':1425,
                    '6:00pm':1080, '3:00pm':900, '11:58pm':1438, '7:00pm':1140,
                    '8:48pm':1248, '9:48pm':1308, '7:30pm':1170, '9:30pm':1290,
                    '10:30pm':1350, '11:15pm':1395, '10:40pm':1360,
                    '12:00am':0, '12:00pm':720, ' 11:58 PM ':1438}
        for text, minute in expected.items():
            with self.subTest(text=text):
                self.assertEqual(worker.parse_times(text), [minute])
                self.assertEqual(worker.read_schedule(schedule(timings=text))['times'], [minute])
        self.assertEqual(worker.parse_times('9:00pm;11:58pm\n9:00pm'), [1260,1438])
    def test_native_sheet_time_never_rounds_to_another_minute_or_day(self):
        for minute in (0, 1, 720, 930, 1260, 1438, 1439):
            self.assertEqual(worker.parse_times(minute/1440), [minute])
        self.assertEqual(worker.parse_times((23*3600+59*60+59)/86400), [1439])
        self.assertEqual(worker.read_schedule(schedule(timings=0))['times'], [0])
    def provider(self):
        with patch.dict(os.environ, {'META_SYSVAR': 'x'*32}, clear=True): return worker.MetaDelete()
    def test_owner_permalink_and_exact_id_verified(self):
        provider = self.provider(); provider.request = Mock(return_value={
            'id': '200', 'owner': {'id': '100'}, 'permalink': row()['urls']})
        provider.verify(row())
        for changed in [{'id': '999'}, {'owner': {'id': '999'}}, {'permalink': 'https://www.instagram.com/p/OTHER/'}]:
            provider.request.return_value = {'id':'200', 'owner':{'id':'100'}, 'permalink':row()['urls'], **changed}
            with self.assertRaises(ClickoutError): provider.verify(row())
    def test_correct_token_for_instagram_and_facebook(self):
        provider = self.provider(); provider.request = Mock(return_value={'id': '100', 'access_token': 'page-token'})
        self.assertEqual(provider.token(row()), 'x'*32)
        fb = row(platform='facebook', post_id='100_200', urls='https://www.facebook.com/100/posts/200')
        self.assertEqual(provider.token(fb), 'page-token'); provider.token(fb)
        provider.request.assert_called_once()
        worker.validate(fb)
        with self.assertRaises(ClickoutError): worker.validate({**fb, 'post_id': '999_200'})
    def test_explicit_confirmation_required_from_provider(self):
        provider = self.provider()
        for data in [{'success': False}, {'success': 1}, {'success': True, 'deleted_id':'999'}, {}]:
            provider.request = Mock(return_value=data)
            with self.assertRaises(ClickoutError): provider.delete(row())
        provider.request = Mock(return_value={'success':True, 'deleted_id':'200'})
        self.assertEqual(provider.delete(row())['status'], 'done')
    def test_http_transport_no_redirects_or_retries_and_safe_errors(self):
        provider = self.provider()
        response = Mock(status_code=429, headers={'Retry-After':'900'})
        response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
        response.iter_content.return_value = [b'{"error":{"code":4,"message":"secret"}}']
        session = Mock(); session.__enter__ = Mock(return_value=session); session.__exit__ = Mock(return_value=False)
        session.request.return_value = response
        module = types.SimpleNamespace(Session=Mock(return_value=session))
        with patch.dict(sys.modules, {'requests':module}):
            with self.assertRaises(ClickoutError) as error: provider.request('200', 'private', 'DELETE')
        self.assertEqual(error.exception.retry_after, 900); self.assertNotIn('secret', str(error.exception))
        session.request.assert_called_once()
        self.assertFalse(session.request.call_args.kwargs['allow_redirects']); self.assertFalse(session.trust_env)
    def test_dashboard_ack_uses_ingest_credential_not_scanner_key(self):
        import social_inventory as inventory
        http = Mock(); http.json.side_effect = [{'post':row()}, {'ok':True}]
        with patch.dict(os.environ, {'OPS_INGEST_TOKEN':'test-ingest'}, clear=True), patch.object(inventory, 'SafeHTTP', return_value=http):
            self.assertEqual(worker.acknowledge(row()), 'synced')
        self.assertEqual(http.json.call_args.kwargs['headers'], {'X-INGEST-KEY':'test-ingest'})
    def test_dashboard_ack_rejects_unrelated_request_identity(self):
        import social_inventory as inventory
        http = Mock(); http.json.return_value = {'post':row(request='other')}
        with patch.dict(os.environ, {'OPS_INGEST_TOKEN':'test-ingest'}, clear=True), patch.object(inventory, 'SafeHTTP', return_value=http):
            self.assertEqual(worker.acknowledge(row()), 'identity_mismatch')
        http.json.assert_called_once()


class RouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(routes.router); self.client = TestClient(app)
        self.env = patch.dict(os.environ, {'SOCIAL_INVENTORY_API_KEY':'delete-key', 'MEDIA_CLIP_API_KEY':'scanner-key'}, clear=True)
        self.env.start()
    def tearDown(self): self.env.stop()
    def test_auth_before_any_storage_work_and_default_preview(self):
        with patch.object(routes, 'worker') as factory:
            for path in ['/preview', '/run', '/tick']:
                self.assertEqual(self.client.post('/social/clickouts'+path, json={}, headers={'X-API-KEY':'wrong-key'}).status_code, 401)
            factory.assert_not_called()
            factory.return_value.preview.return_value = {'dry_run': True}
            reply = self.client.post('/social/clickouts/run', json={}, headers={'X-API-KEY':'delete-key'})
            self.assertTrue(reply.json()['dry_run']); factory.return_value.run.assert_not_called()
    def test_existing_media_key_works_without_a_social_key_on_every_endpoint(self):
        os.environ.pop('SOCIAL_INVENTORY_API_KEY')
        headers = {'X-API-KEY': 'scanner-key'}
        with patch.object(routes, 'worker') as factory:
            factory.return_value.preview.return_value = {'dry_run': True}
            factory.return_value.state.get.return_value = None
            factory.return_value.run.return_value = {'status': 'disabled', 'done': 0}
            self.assertEqual(self.client.get('/social/clickouts/status', headers=headers).status_code, 200)
            self.assertEqual(self.client.post('/social/clickouts/preview', headers=headers).status_code, 200)
            result = self.client.post('/social/clickouts/run', json={}, headers=headers)
            self.assertTrue(result.json()['dry_run']); factory.return_value.run.assert_not_called()
            self.assertEqual(self.client.post('/social/clickouts/run', json={'dry_run':False}, headers=headers).status_code, 200)
            self.assertEqual(self.client.post('/social/clickouts/tick', headers=headers).status_code, 200)
    def test_partially_configured_social_key_does_not_override_existing_media_key(self):
        with patch.object(routes, 'worker') as factory:
            factory.return_value.preview.return_value = {'dry_run': True}
            for key in ('scanner-key', 'delete-key'):
                self.assertEqual(self.client.post('/social/clickouts/preview', headers={'X-API-KEY':key}).status_code, 200)
    def test_missing_wrong_and_unrelated_keys_fail_before_storage(self):
        with patch.object(routes, 'worker') as factory:
            for key in (None, '', 'wrong', 'control-key'):
                headers = {'X-API-KEY':key} if key is not None else {}
                self.assertEqual(self.client.get('/social/clickouts/status', headers=headers).status_code, 401)
            with patch.dict(os.environ, {'CONTROL_PLANE_API_KEY':'control-key', 'MEDIA_CLIP_API_KEY':' ', 'SOCIAL_INVENTORY_API_KEY':''}, clear=True):
                self.assertEqual(self.client.get('/social/clickouts/status', headers={'X-API-KEY':'control-key'}).status_code, 503)
            factory.assert_not_called()
    def test_media_key_rotation_takes_effect_without_cached_auth(self):
        os.environ.pop('SOCIAL_INVENTORY_API_KEY')
        with patch.object(routes, 'worker') as factory:
            factory.return_value.preview.return_value = {'dry_run': True}
            os.environ['MEDIA_CLIP_API_KEY'] = 'replacement-key'
            self.assertEqual(self.client.post('/social/clickouts/preview', headers={'X-API-KEY':'scanner-key'}).status_code, 401)
            self.assertEqual(self.client.post('/social/clickouts/preview', headers={'X-API-KEY':'replacement-key'}).status_code, 200)
    def test_media_auth_cannot_bypass_disabled_deletion(self):
        os.environ.pop('SOCIAL_INVENTORY_API_KEY')
        store = Mock(); state = Mock()
        service = worker.ClickoutsWorker(store, state)
        with patch.object(routes, 'worker', return_value=service):
            for path, body in [('/run', {'dry_run':False}), ('/tick', {})]:
                response = self.client.post('/social/clickouts'+path, json=body, headers={'X-API-KEY':'scanner-key'})
                self.assertEqual(response.json()['status'], 'disabled')
        store.read.assert_not_called(); state.acquire.assert_not_called()
    def test_explicit_batch_limits_and_opaque_failures(self):
        with patch.object(routes, 'worker') as factory:
            headers = {'X-API-KEY':'delete-key'}
            self.assertEqual(self.client.post('/social/clickouts/run', json={'max_items':100}, headers=headers).status_code, 422)
            factory.side_effect = RuntimeError('private connection string')
            result = self.client.post('/social/clickouts/run', json={'dry_run':False}, headers=headers)
            self.assertEqual(result.status_code, 503); self.assertNotIn('private', result.text)
    def test_app_wires_router_without_changing_existing_routes(self):
        source = (Path(__file__).resolve().parents[1]/'app.py').read_text()
        ast.parse(source)
        for name in ('clickouts_router', 'social_inventory_router', 'social_delete_router'):
            self.assertIn('app.include_router('+name+')', source)
    def test_scheduler_is_opt_in_single_instance_and_stops_on_shutdown(self):
        scheduler = Mock()
        fake = types.SimpleNamespace(BackgroundScheduler=Mock(return_value=scheduler))
        with patch.dict(sys.modules, {'apscheduler.schedulers.background': fake}), patch.object(routes, '_scheduler', None):
            routes.start_clickouts_scheduler()
            fake.BackgroundScheduler.assert_not_called()
            os.environ['SOCIAL_DELETE_ENABLED'] = '1'
            routes.start_clickouts_scheduler(); routes.start_clickouts_scheduler()
            fake.BackgroundScheduler.assert_called_once()
            self.assertEqual(scheduler.add_job.call_args.kwargs['seconds'], 60)
            self.assertEqual(scheduler.add_job.call_args.kwargs['max_instances'], 1)
            routes.stop_clickouts_scheduler()
            scheduler.shutdown.assert_called_once_with(wait=False)


class StatusDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI(); app.include_router(routes.router); self.client = TestClient(app)
        self.env = patch.dict(os.environ, {'MEDIA_CLIP_API_KEY': 'existing-key', 'SPREADSHEET_ID': 'book'}, clear=True)
        self.env.start()
        self.store = Store([row()])
        self.service = worker.ClickoutsWorker(self.store, None)
        self.headers = {'X-API-KEY': 'existing-key'}
    def tearDown(self): self.env.stop()
    def status(self, redis_error=None, last=None):
        state = Mock()
        state.get.return_value = last
        if redis_error: state.get.side_effect = redis_error
        with patch.object(routes, 'worker', return_value=self.service), patch.object(routes, 'redis_state', return_value=state):
            response = self.client.get('/social/clickouts/status', headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.store.writes, []); self.assertFalse(self.store.prepared)
        state.acquire.assert_not_called(); state.put.assert_not_called()
        return response.json()
    def test_failed_redis_read_preserves_queue_schedule_and_marks_degraded(self):
        data = self.status(ClickoutError('redis_connection_failed'))
        self.assertEqual(data['rows'], 1); self.assertEqual(data['eligible'], 1)
        self.assertTrue(data['schedule']['enabled']); self.assertFalse(data['ok'])
        self.assertEqual(data['checks']['redis']['code'], 'redis_connection_failed')
        self.assertIsNone(data['last_run']); self.assertEqual(data['diagnostics_version'], 2)
    def test_google_initialization_failure_does_not_hide_redis_result(self):
        state = Mock(); state.get.return_value = {'status':'completed', 'done':2}
        with patch.object(routes, 'worker', side_effect=ClickoutError('google_credentials_refresh_rejected')), patch.object(routes, 'redis_state', return_value=state):
            data = self.client.get('/social/clickouts/status', headers=self.headers).json()
        self.assertIsNone(data['rows']); self.assertIsNone(data['eligible'])
        self.assertEqual(data['checks']['google_client']['code'], 'google_credentials_refresh_rejected')
        self.assertEqual(data['checks']['queue']['status'], 'skipped')
        self.assertEqual(data['checks']['redis']['status'], 'ok')
        self.assertEqual(data['last_run']['done'], 2)
    def test_invalid_schedule_preserves_known_queue_counts(self):
        self.store.schedule = schedule(timings='not a time')
        data = self.status()
        self.assertEqual(data['rows'], 1); self.assertEqual(data['checks']['queue']['status'], 'ok')
        self.assertIsNone(data['schedule'])
        self.assertEqual(data['checks']['schedule']['code'], 'invalid_clickouts_timings_use_HH_MM')
    def test_429_does_not_issue_another_sheets_read_for_schedule(self):
        self.store.read = Mock(side_effect=ClickoutError('sheets_rate_limited', 300))
        data = self.status()
        self.assertEqual(self.store.schedule_reads, 0)
        self.assertEqual(data['checks']['queue']['retry_after'], 300)
        self.assertEqual(data['checks']['schedule']['code'], 'sheets_cooldown')
    def test_preview_and_default_run_need_no_redis_configuration(self):
        with patch.object(routes, 'google_store', return_value=self.store), patch.object(routes, 'redis_state') as state_factory:
            for path in ('/preview', '/run'):
                response = self.client.post('/social/clickouts'+path, json={}, headers=self.headers)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()['eligible'], 1)
            state_factory.assert_not_called()
        self.assertEqual(self.store.writes, [])
    def test_real_execution_still_requires_redis(self):
        with patch.object(routes, 'google_store') as google, patch.object(routes, 'redis_state', side_effect=ClickoutError('redis_connection_failed')):
            response = self.client.post('/social/clickouts/run', json={'dry_run':False}, headers=self.headers)
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()['detail'], 'redis_connection_failed')
            google.assert_not_called()
    def test_unexpected_exceptions_and_state_extras_never_leak_secrets(self):
        self.store.read = Mock(side_effect=KeyError('token=private-secret&url=redis://user:pass@host'))
        data = self.status(last={'status':'completed', 'done':2, 'private':'private-secret', 'checked_at':'private-secret'})
        self.assertNotIn('private-secret', json.dumps(data)); self.assertNotIn('redis://', json.dumps(data))
        self.assertEqual(data['checks']['queue']['error_type'], 'KeyError')
        self.assertEqual(data['last_run'], {'status':'completed', 'done':2})
    def test_missing_keys_stop_before_diagnostic_dependencies(self):
        with patch.object(routes, 'worker') as service, patch.object(routes, 'redis_state') as state:
            self.assertEqual(self.client.get('/social/clickouts/status').status_code, 401)
            service.assert_not_called(); state.assert_not_called()
    def test_healthy_diagnostics_are_not_a_claim_of_delete_permission(self):
        data = self.status()
        self.assertTrue(data['ok']); self.assertFalse(data['enabled'])
        self.assertEqual(data['provider_permissions'], 'not_checked')
    def test_corrupt_redis_state_is_explicit(self):
        data = self.status(last='private-secret')
        self.assertEqual(data['checks']['redis']['code'], 'redis_state_invalid_shape')
        self.assertNotIn('private-secret', json.dumps(data))
    def test_runtime_reports_only_valid_existing_render_identifiers(self):
        with patch.dict(os.environ, {'RENDER_GIT_COMMIT':'a'*40, 'RENDER_SERVICE_ID':'srv-example',
                                     'RENDER_INSTANCE_ID':'redis://private:secret@host'}):
            data = self.status()
        self.assertEqual(data['runtime']['commit'], 'a'*40)
        self.assertEqual(data['runtime']['service_id'], 'srv-example')
        self.assertIsNone(data['runtime']['instance_id'])
        self.assertNotIn('private:secret', json.dumps(data))


class DependencyDiagnosticsTests(unittest.TestCase):
    def test_redis_errors_have_specific_codes_without_connection_details(self):
        variants = [(ConnectionError, 'redis_connection_failed'), (TimeoutError, 'redis_timeout'),
                    (type('AuthenticationError', (Exception,), {}), 'redis_authentication_or_permission_failed')]
        for kind, code in variants:
            client = Mock(); client.get.side_effect = kind('redis://private:secret@host')
            with self.assertRaisesRegex(ClickoutError, code): RedisState(client, 'book').get('last_result')
        client = Mock(); client.get.return_value = 'secret invalid json'
        with self.assertRaisesRegex(ClickoutError, 'redis_state_invalid_json'): RedisState(client, 'book').get('last_result')
    def test_google_factory_classifies_wrapped_refresh_errors_and_build_errors(self):
        import clickouts_state as state_module
        get_credentials = Mock(return_value=object()); build = Mock(return_value=object())
        modules = {
            'httplib2':types.SimpleNamespace(Http=Mock()),
            'google_auth_httplib2':types.SimpleNamespace(AuthorizedHttp=Mock()),
            'googleapiclient':types.ModuleType('googleapiclient'),
            'googleapiclient.discovery':types.SimpleNamespace(build=build),
            'sheets_helper':types.SimpleNamespace(get_credentials=get_credentials),
        }
        with patch.dict(sys.modules, modules):
            self.assertIsInstance(state_module.google_store('book'), GoogleClickouts)
            wrapped = RuntimeError('private credential payload')
            wrapped.__cause__ = type('RefreshError', (Exception,), {})('private refresh token')
            get_credentials.side_effect = wrapped
            with self.assertRaisesRegex(ClickoutError, 'google_credentials_refresh_rejected'):
                state_module.google_store('book')
            get_credentials.side_effect = None; build.side_effect = RuntimeError('private URL')
            with self.assertRaisesRegex(ClickoutError, 'google_client_initialization_failed'):
                state_module.google_store('book')
    def test_redis_factory_handles_bad_url_without_echoing_it(self):
        redis = types.SimpleNamespace(from_url=Mock(side_effect=ValueError('redis://private:secret@host')))
        with patch.dict(sys.modules, {'redis':redis}), patch.dict(os.environ, {'REDIS_URL':'private'}, clear=True):
            with self.assertRaisesRegex(ClickoutError, 'redis_configuration_invalid'):
                routes.redis_state('book')


if __name__ == '__main__':
    unittest.main()
