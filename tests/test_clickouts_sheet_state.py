"""Offline protocol tests: actual Sheets request shapes, conflicts and recovery."""
import copy
import json
import os
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from clickouts_state import ClickoutError, GoogleClickouts
from clickouts_sheet_state import SheetsState
from clickouts_worker import ClickoutsWorker, post_key
from test_clickouts import Store, Provider, row, schedule


class HTTPError(Exception):
    def __init__(self, status):
        super().__init__('private provider message')
        self.resp = type('Response', (dict,), {'status': status})({'retry-after': '120'})


class Request:
    def __init__(self, action): self.action = action
    def execute(self, num_retries):
        assert num_retries == 0
        return self.action()


class API:
    def __init__(self):
        self.sheets, self.ranges = {}, {}
        self.calls = []
        self.before = None
        self.lose = False
        self.quota = False
        self.ignore_guard = False
    def spreadsheets(self): return self
    def get(self, **kwargs):
        def action():
            self.calls.append(('get', kwargs))
            if kwargs.get('ranges') and not self.sheets: raise HTTPError(400)
            sheets = [{'properties': copy.deepcopy(v['properties']),
                       'data': [{'startRow': 0, 'rowData': copy.deepcopy(v['rows'])}]} for v in self.sheets.values()]
            return {'sheets': sheets, 'namedRanges': copy.deepcopy(list(self.ranges.values()))}
        return Request(action)
    def batchUpdate(self, **kwargs):
        def action():
            self.calls.append(('write', kwargs))
            if self.quota: raise HTTPError(429)
            if self.before:
                hook, self.before = self.before, None
                hook()
            sheets, ranges = copy.deepcopy(self.sheets), copy.deepcopy(self.ranges)
            for request in kwargs['body']['requests']:
                if 'addSheet' in request:
                    props = request['addSheet']['properties']; sid = props['sheetId']
                    if sid in sheets or any(s['properties']['title'] == props['title'] for s in sheets.values()): raise HTTPError(400)
                    sheets[sid] = {'properties': props, 'rows': []}
                elif 'addNamedRange' in request:
                    named = request['addNamedRange']['namedRange']; nid = named['namedRangeId']
                    if nid in ranges or any(n['name'] == named['name'] for n in ranges.values()):
                        if not self.ignore_guard: raise HTTPError(400)
                        ranges = {i:n for i,n in ranges.items() if n['name'] != named['name']}
                    ranges[nid] = named
                elif 'updateNamedRange' in request:
                    named = request['updateNamedRange']['namedRange']; nid = named['namedRangeId']
                    if nid not in ranges:
                        if not self.ignore_guard: raise HTTPError(400)
                    else: ranges[nid]['range'] = named['range']
                elif 'deleteNamedRange' in request:
                    nid = request['deleteNamedRange']['namedRangeId']
                    if nid not in ranges and not self.ignore_guard: raise HTTPError(400)
                    ranges.pop(nid, None)
                elif 'updateCells' in request:
                    data = request['updateCells']; start = data['start']; rows = sheets[start['sheetId']]['rows']
                    i = start['rowIndex']
                    while len(rows) <= i: rows.append({'values': []})
                    for offset, new_row in enumerate(data['rows']):
                        old = rows[i+offset]['values']; new_values = copy.deepcopy(new_row['values'])
                        old[:len(new_values)] = new_values
                else: raise AssertionError(request)
            self.sheets, self.ranges = sheets, ranges
            if self.lose:
                self.lose = False
                raise ConnectionError('private credentials must not escape')
            return {'replies': [{} for _ in kwargs['body']['requests']]}
        return Request(action)


class SheetStateTests(unittest.TestCase):
    def setUp(self):
        self.cooldowns = patch.dict(GoogleClickouts._cooldowns, {}, clear=True); self.cooldowns.start()
        self.api = API(); self.now = [1000.0]
        self.state = self.fresh()
    def tearDown(self): self.cooldowns.stop()
    def fresh(self): return SheetsState(GoogleClickouts(self.api, 'book'), clock=lambda:self.now[0])
    def test_missing_status_is_read_only_and_first_execution_initializes_verified_hidden_state(self):
        self.assertFalse(self.state.load()); self.assertIsNone(self.state.get('last_result'))
        self.assertFalse(any(c[0] == 'write' for c in self.api.calls))
        self.assertTrue(self.state.acquire()); self.assertTrue(self.state.control['verified'])
        self.assertTrue(next(iter(self.api.sheets.values()))['properties']['hidden'])
        self.state.put('last_result', {'status':'completed','done':2}); self.state.release()
        self.assertEqual(self.fresh().get('last_result')['done'], 2)
    def test_concurrent_claim_is_fenced_and_losing_batch_changes_no_records(self):
        self.state.initialize(); rival = self.fresh()
        self.api.before = lambda:self.assertTrue(rival.acquire())
        self.assertFalse(self.state.acquire())
        self.assertFalse(self.fresh().acquire())
        rival.put('post:a', {'phase':'delete_sent'}); rival.release()
        self.assertTrue(self.state.acquire())
        self.assertFalse(self.state.claim('post:a', {'phase':'prepared'}))
        self.assertEqual(self.state.get('post:a')['phase'], 'delete_sent')
    def test_expired_owner_cannot_change_state_or_release_replacement_lease(self):
        self.state.acquire(); old = self.state.owner
        self.now[0] += 181; replacement = self.fresh(); self.assertTrue(replacement.acquire())
        replacement.put('post:a', {'phase':'delete_sent'})
        with self.assertRaisesRegex(ClickoutError, 'lease_lost'): self.state.put('post:b', {})
        with self.assertRaisesRegex(ClickoutError, 'revision_conflict'): self.state.release()
        fresh = self.fresh(); fresh.load()
        self.assertNotEqual(fresh.control['lease']['owner'], old)
        self.assertEqual(fresh.get('post:a')['phase'], 'delete_sent')
    def test_lost_commit_response_reconciles_exact_revision_without_repeating_write(self):
        self.state.acquire(); self.api.lose = True
        before = len([c for c in self.api.calls if c[0]=='write'])
        self.assertTrue(self.state.claim('post:a', {'phase':'delete_sent'}))
        self.assertEqual(len([c for c in self.api.calls if c[0]=='write'])-before, 1)
        self.assertEqual(self.fresh().get('post:a')['phase'], 'delete_sent')
    def test_stale_revision_rejected_atomically_even_with_other_cell_updates(self):
        self.state.acquire(); stale = self.fresh(); stale.load(); stale.owner = self.state.owner
        self.state.put('one', 1)
        with self.assertRaisesRegex(ClickoutError, 'revision_conflict'): stale.put('two', 2)
        fresh = self.fresh(); self.assertEqual(fresh.get('one'), 1); self.assertIsNone(fresh.get('two'))
    def test_live_guard_probe_fails_closed_if_service_accepts_a_stale_guard(self):
        self.api.ignore_guard = True
        with self.assertRaisesRegex(ClickoutError, 'guard_not_enforced'): self.state.acquire()
        with self.assertRaisesRegex(ClickoutError, 'corrupt'): self.fresh().acquire()
    def test_quota_failure_has_no_immediate_reconciliation_release_or_new_client_request(self):
        self.state.acquire(); self.api.quota = True
        with self.assertRaisesRegex(ClickoutError, 'sheets_rate_limited'): self.state.put('one', 1)
        count = len(self.api.calls)
        with self.assertRaisesRegex(ClickoutError, 'sheets_cooldown'): self.state.release()
        with self.assertRaisesRegex(ClickoutError, 'sheets_cooldown'): self.fresh().load()
        self.assertEqual(len(self.api.calls), count)
    def test_no_state_reset_on_corrupt_cells_or_missing_guard(self):
        self.state.acquire(); self.api.ranges.clear(); count=len(self.api.calls)
        with self.assertRaisesRegex(ClickoutError, 'corrupt'): self.fresh().acquire()
        self.assertFalse(any(c[0]=='write' for c in self.api.calls[count:]))
    def test_confirmed_delete_survives_sheet_update_failure_and_restart_without_second_delete(self):
        store=Store([row()]); provider=Provider(); store.fail_done=True
        service=ClickoutsWorker(store,self.state,provider_factory=lambda:provider,ack=lambda r:'not_configured')
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ClickoutError): service.run()
            self.assertEqual(provider.deleted,['200']); store.fail_done=False
            result=ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider,ack=lambda r:'not_configured').run()
        self.assertEqual(result['done'],1); self.assertEqual(provider.deleted,['200'])
        self.assertEqual(store.rows[0]['status'],'done')
    def test_uncertain_delete_is_not_reissued_by_another_instance(self):
        store=Store([row()]); provider=Provider(); provider.error=RuntimeError('lost response')
        ClickoutsWorker(store,self.state,provider_factory=lambda:provider,ack=lambda r:'not_configured').run()
        self.assertEqual(store.rows[0]['status'],'manual_required')
        ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider).run()
        self.assertEqual(len(provider.deleted),1)
    def test_delete_sent_claim_survives_expired_lease_and_blocks_duplicate_sheet_row(self):
        self.state.acquire(); self.state.claim(post_key(row()), {'status':'processing','phase':'delete_sent'})
        self.now[0]+=181; store=Store([row()]); provider=Provider()
        ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider).run()
        self.assertEqual(provider.deleted,[]); self.assertEqual(store.rows[0]['status'],'manual_required')
    def test_no_matching_schedule_never_initializes_state(self):
        store=Store([row()]); store.schedule=schedule(timings='')
        self.assertEqual(ClickoutsWorker(store,self.state).run(scheduled=True)['status'],'paused')
        self.assertFalse(any(c[0]=='write' for c in self.api.calls))
    def test_daily_batches_resume_across_instances_and_completed_occurrence_does_not_repeat(self):
        store=Store([row('r'+str(i),str(200+i)) for i in range(4)])
        store.schedule=schedule(timings='10:30am',job_override='no'); provider=Provider()
        at=datetime(2026,11,16,5,0,tzinfo=timezone.utc)
        with patch('clickouts_worker.time.time',side_effect=lambda:self.now[0]):
            first=ClickoutsWorker(store,self.state,provider_factory=lambda:provider,ack=lambda r:'not_configured')
            self.assertEqual(first.run(scheduled=True,now=at)['done'],3)
            other=ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider,ack=lambda r:'not_configured')
            self.assertEqual(other.run(scheduled=True,now=at)['status'],'waiting')
            self.now[0]+=61
            resumed=ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider,ack=lambda r:'not_configured')
            self.assertEqual(resumed.run(scheduled=True,now=at)['done'],1)
            store.rows.append(row('later','999')); self.now[0]+=61
            later=ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider,ack=lambda r:'not_configured')
            self.assertEqual(later.run(scheduled=True,now=at)['status'],'waiting')
        self.assertEqual(provider.deleted,['200','201','202','203'])
    def test_request_arriving_during_provider_delete_cannot_execute_same_post(self):
        store=Store([row()]); provider=Provider(); original=provider.delete
        def delete(target):
            rival=ClickoutsWorker(store,self.fresh(),provider_factory=lambda:provider)
            self.assertEqual(rival.run()['status'],'busy')
            return original(target)
        provider.delete=delete
        ClickoutsWorker(store,self.state,provider_factory=lambda:provider,ack=lambda r:'not_configured').run()
        self.assertEqual(provider.deleted,['200'])


if __name__ == '__main__': unittest.main()
