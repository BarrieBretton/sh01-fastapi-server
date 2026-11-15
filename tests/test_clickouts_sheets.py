"""Sheets adapter tests with row-aware metadata and real request shapes; no Google calls."""
import copy
import re
import unittest

from clickouts_state import ClickoutError, GoogleClickouts


class Request:
    def __init__(self, action): self.action = action
    def execute(self, num_retries):
        assert num_retries == 0
        return self.action()


class Sheets:
    def __init__(self):
        self.headers = ['URLS', 'platform', 'account_id', 'post_id', 'request_id', 'custom']
        self.rows = [['https://www.instagram.com/p/ABC/', 'instagram', '100', '200', 'r1', '=1+2']]
        self.markers = {}; self.updates = []; self.calls = []; self.width = 26
    def spreadsheets(self): return self
    def values(self): return self
    def developerMetadata(self): return self
    def get(self, **args):
        self.calls.append(('get', args))
        if 'range' not in args:
            return Request(lambda: {'sheets': [{'properties': {'sheetId': i, 'title': title,
                'gridProperties': {'columnCount': self.width, 'rowCount': 1000}}}
                for i, title in [(1, 'clickouts'), (2, 'Workflow-Schedule')]]})
        if 'Workflow-Schedule' in args['range']:
            return Request(lambda: {'values': [['Workflow', 'Timings', 'Run as Job Today']]})
        return Request(lambda: {'values': copy.deepcopy([self.headers, *self.rows])})
    def update(self, **args):
        def action():
            self.headers.extend(args['body']['values'][0])
            return {'updatedRows': 1}
        self.calls.append(('update', args)); return Request(action)
    def search(self, **args):
        return Request(lambda: {'matchedDeveloperMetadata': [{'developerMetadata': {
            'metadataId': key, 'metadataValue': value['value']}} for key, value in self.markers.items()]})
    def batchUpdate(self, **args):
        def action():
            request = args['body']['requests'][0]
            if 'appendDimension' in request:
                self.width += request['appendDimension']['length']; return {'replies': [{}]}
            data = request['createDeveloperMetadata']['developerMetadata']
            index = data['location']['dimensionRange']['startIndex'] - 1
            mid = len(self.markers) + 10
            self.markers[mid] = {'value': data['metadataValue'], 'row': self.rows[index]}
            return {'replies': [{'createDeveloperMetadata': {'developerMetadata': {'metadataId': mid}}}]}
        return Request(action)
    def batchGetByDataFilter(self, **args):
        def action():
            ranges = []
            for data_filter in args['body']['dataFilters']:
                if 'gridRange' in data_filter:
                    value = self.headers
                else:
                    marker = self.markers[data_filter['developerMetadataLookup']['metadataId']]
                    value = marker['row']
                ranges.append({'dataFilters': [data_filter], 'valueRange': {'values': [copy.deepcopy(value)]}})
            return {'valueRanges': list(reversed(ranges))}  # Order cannot be assumed.
        return Request(action)
    def batchUpdateByDataFilter(self, **args):
        def action():
            self.updates.append(args)
            item = args['body']['data'][0]
            target = self.markers[item['dataFilter']['developerMetadataLookup']['metadataId']]['row']
            values = item['values'][0]
            target.extend([''] * (len(values) - len(target)))
            for i, value in enumerate(values):
                if value is not None: target[i] = value
            return {'totalUpdatedRows': 1}
        return Request(action)


class SheetsTests(unittest.TestCase):
    def setUp(self):
        GoogleClickouts._cooldowns.clear()
        self.api = Sheets(); self.store = GoogleClickouts(self.api, 'book')
    def test_prepare_only_appends_missing_headers_and_preserves_custom_cells(self):
        rows = self.store.read(); self.store.prepare()
        self.assertEqual(self.api.headers[:6], ['URLS', 'platform', 'account_id', 'post_id', 'request_id', 'custom'])
        self.store.bind(rows[0]); self.store.update(rows[0], status='processing', error='')
        self.assertEqual(self.api.rows[0][:6], ['https://www.instagram.com/p/ABC/', 'instagram', '100', '200', 'r1', '=1+2'])
        self.assertEqual(self.api.updates[0]['body']['valueInputOption'], 'RAW')
        self.assertEqual(self.api.updates[0]['body']['data'][0]['values'][0][:6], [None]*6)
    def test_metadata_follows_row_movement_and_reused_after_restart(self):
        rows = self.store.read(); self.store.prepare(); self.store.bind(rows[0])
        self.api.rows.insert(0, ['unrelated'])
        self.store.update(rows[0], status='done')
        self.assertEqual(self.api.rows[0], ['unrelated'])
        self.assertEqual(self.api.rows[1][self.api.headers.index('status')], 'done')
        fresh = GoogleClickouts(self.api, 'book'); fresh_rows = fresh.read(); fresh.prepare(); fresh.bind(fresh_rows[1])
        self.assertEqual(len(self.api.markers), 1)
    def test_reordered_columns_stop_before_status_write(self):
        rows = self.store.read(); self.store.prepare(); self.store.bind(rows[0])
        self.api.headers.reverse()
        with self.assertRaisesRegex(ClickoutError, 'headers_changed'): self.store.update(rows[0], status='done')
        self.assertEqual(self.api.updates, [])
    def test_changed_post_id_is_rejected(self):
        rows = self.store.read(); self.store.prepare(); self.store.bind(rows[0])
        self.api.rows[0][3] = '999'
        with self.assertRaisesRegex(ClickoutError, 'identity_changed'): self.store.update(rows[0], status='done')
    def test_status_edit_is_not_overwritten(self):
        rows = self.store.read(); self.store.prepare(); self.store.bind(rows[0])
        self.store.update(rows[0], status='processing')
        self.api.rows[0][self.api.headers.index('status')] = 'cancelled'
        with self.assertRaisesRegex(ClickoutError, 'status_changed'): self.store.update(rows[0], status='done')
    def test_header_validation_and_grid_bounded_reads(self):
        self.store.read(); self.store.schedule_rows()
        ranges = [args['range'] for name, args in self.api.calls if 'range' in args]
        self.assertTrue(all(re.search(r'!A1:Z1000$', value) for value in ranges))
        self.api.headers.append('POST_ID')
        with self.assertRaisesRegex(ClickoutError, 'headers_missing_or_duplicated'): self.store.read()
    def test_no_completed_read_cache_and_no_unowned_writes(self):
        rows = self.store.read(); self.store.prepare(); self.store.bind(rows[0])
        with self.assertRaisesRegex(ClickoutError, 'unowned_column'): self.store.update(rows[0], post_id='999')
        self.api.rows.append(['url', 'instagram', '100', '201', 'r2'])
        self.assertEqual(len(self.store.read()), 2)
    def test_rate_limit_honors_retry_after_without_automatic_retry(self):
        class Failure(Exception): pass
        failure = Failure('private Google response')
        failure.resp = type('Response', (dict,), {'status':429})({'retry-after':'900'})
        def fail(): raise failure
        with self.assertRaises(ClickoutError) as result: self.store.call(Request(fail))
        self.assertEqual(result.exception.retry_after, 900)
        self.assertEqual(str(result.exception), 'sheets_rate_limited')


if __name__ == '__main__': unittest.main()
