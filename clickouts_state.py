"""Bounded Google Sheets access and shared Redis state for Clickouts deletion."""
from __future__ import annotations

import hashlib
import json
import uuid


class ClickoutError(Exception):
    """Static, credential-free diagnostic only."""
    def __init__(self, code, retry_after=0):
        super().__init__(code)
        self.retry_after = retry_after


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def column(index):
    result = ''
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


class RedisState:
    """One workbook lease; persistent attempt/receipt records are never auto-expired.

    Redis must be shared by every server consuming this workbook. Sheet processing
    markers remain a second barrier if Redis data is lost. No destructive retries.
    """
    def __init__(self, client, workbook):
        self.client = client
        self.prefix = 'sh01:clickouts:v1:' + digest(workbook) + ':'
        self.owner = None

    def acquire(self):
        owner = uuid.uuid4().hex
        if not self.client.set(self.prefix + 'lease', owner, nx=True, ex=180):
            return False
        self.owner = owner
        return True

    def guard(self):
        if not self.owner or not self.client.eval(
            "if redis.call('get',KEYS[1]) == ARGV[1] then "
            "return redis.call('expire',KEYS[1],180) else return 0 end",
            1, self.prefix + 'lease', self.owner,
        ):
            raise ClickoutError('worker_lease_lost')

    def release(self):
        if self.owner:
            try:
                self.client.eval(
                    "if redis.call('get',KEYS[1]) == ARGV[1] then "
                    "return redis.call('del',KEYS[1]) else return 0 end",
                    1, self.prefix + 'lease', self.owner,
                )
            finally:
                self.owner = None

    def get(self, key):
        raw = self.client.get(self.prefix + key)
        return json.loads(raw) if raw else None

    def put(self, key, value):
        self.guard()
        self.client.set(self.prefix + key, json.dumps(value, separators=(',', ':')))

    def claim(self, key, value):
        self.guard()
        return bool(self.client.set(self.prefix + key, json.dumps(value), nx=True))


class GoogleClickouts:
    REQUIRED = ('urls', 'platform', 'account_id', 'post_id', 'request_id')
    OWNED = ('status', 'deleted_at', 'last_attempt_at', 'error', 'dashboard_ack')
    META_KEY = 'sh01_clickout_identity_v1'
    LIMIT = 10000

    def __init__(self, service, workbook):
        self.service, self.workbook = service, workbook
        self.headers, self.metadata = [], {}
        self.sheet_id, self.title = None, None
        self.book = None

    def call(self, request):
        try:
            return request.execute(num_retries=0)
        except Exception as exc:
            status = getattr(getattr(exc, 'resp', None), 'status', None)
            # No provider response bodies, URLs or credentials in diagnostics.
            delay = 0
            if status in (429, 503):
                from clickouts_worker import retry_seconds
                delay = retry_seconds(getattr(exc, 'resp', {}).get('retry-after'), 60)
            raise ClickoutError('sheets_rate_limited' if status == 429 else 'sheets_request_failed', delay) from None

    def properties(self, title):
        if self.book is None:
            self.book = self.call(self.service.spreadsheets().get(
                spreadsheetId=self.workbook, fields='sheets.properties'))
        matches = [x['properties'] for x in self.book.get('sheets', [])
                   if x['properties']['title'].strip().casefold() == title.casefold()]
        if len(matches) != 1:
            raise ClickoutError('required_tab_missing_or_ambiguous')
        return matches[0]

    def schedule_rows(self):
        props = self.properties('Workflow-Schedule')
        height = props['gridProperties']['rowCount']
        if height > 10000:
            raise ClickoutError('schedule_grid_row_limit')
        width = min(props['gridProperties']['columnCount'], 52)
        title = props['title'].replace("'", "''")
        result = self.call(self.service.spreadsheets().values().get(
            spreadsheetId=self.workbook, range=f"'{title}'!A1:{column(width)}{height}",
            valueRenderOption='UNFORMATTED_VALUE'))
        values = result.get('values', [])
        return values

    def read(self):
        props = self.properties('clickouts')
        self.sheet_id, self.title = props['sheetId'], props['title']
        self.width = props.get('gridProperties', {}).get('columnCount', 26)
        width = min(self.width, 52)
        height = props['gridProperties']['rowCount']
        if height > self.LIMIT:
            raise ClickoutError('clickouts_grid_row_limit')
        quoted = "'" + self.title.replace("'", "''") + "'"
        values = self.call(self.service.spreadsheets().values().get(
            spreadsheetId=self.workbook, range=f'{quoted}!A1:{column(width)}{height}',
            valueRenderOption='UNFORMATTED_VALUE')).get('values', [])
        self.headers = [str(x).strip().casefold() for x in (values[0] if values else [])]
        nonempty = [x for x in self.headers if x]
        if len(nonempty) != len(set(nonempty)) or any(x not in self.headers for x in self.REQUIRED):
            raise ClickoutError('clickouts_headers_missing_or_duplicated')
        result = []
        for number, cells in enumerate(values[1:], 2):
            if not any(x != '' for x in cells):
                continue
            row = {h: cells[i] if i < len(cells) else '' for i, h in enumerate(self.headers) if h}
            row['_row'] = number
            result.append(row)
        return result

    def prepare(self):
        """Execution only: append helper headers; never rewrite existing headers/rows."""
        missing = [x for x in self.OWNED if x not in self.headers]
        if missing:
            end = len(self.headers) + len(missing)
            if end > 52:
                raise ClickoutError('clickouts_column_limit')
            if end > self.width:
                self.call(self.service.spreadsheets().batchUpdate(spreadsheetId=self.workbook, body={
                    'requests': [{'appendDimension': {'sheetId': self.sheet_id, 'dimension': 'COLUMNS',
                                                      'length': end - self.width}}]}))
            quoted = "'" + self.title.replace("'", "''") + "'"
            self.call(self.service.spreadsheets().values().update(
                spreadsheetId=self.workbook, range=f'{quoted}!{column(len(self.headers)+1)}1:{column(end)}1',
                valueInputOption='RAW', body={'values': [missing]}))
            self.headers.extend(missing)
        found = self.call(self.service.spreadsheets().developerMetadata().search(
            spreadsheetId=self.workbook,
            body={'dataFilters': [{'developerMetadataLookup': {'metadataKey': self.META_KEY}}]}))
        self.metadata = {}
        for item in found.get('matchedDeveloperMetadata', []):
            data = item['developerMetadata']
            self.metadata.setdefault(data['metadataValue'], []).append(data['metadataId'])

    @staticmethod
    def identity(row):
        return digest(json.dumps([row.get(k) for k in GoogleClickouts.REQUIRED], separators=(',', ':')))

    def bind(self, row):
        key = self.identity(row)
        found = self.metadata.get(key, [])
        if len(found) > 1:
            raise ClickoutError('ambiguous_row_marker')
        if found:
            row['_metadata'] = found[0]
        else:
            result = self.call(self.service.spreadsheets().batchUpdate(spreadsheetId=self.workbook, body={
                'requests': [{'createDeveloperMetadata': {'developerMetadata': {
                    'metadataKey': self.META_KEY, 'metadataValue': key, 'visibility': 'DOCUMENT',
                    'location': {'dimensionRange': {'sheetId': self.sheet_id, 'dimension': 'ROWS',
                                 'startIndex': row['_row'] - 1, 'endIndex': row['_row']}}}}}]}))
            row['_metadata'] = result['replies'][0]['createDeveloperMetadata']['developerMetadata']['metadataId']
            self.metadata[key] = [row['_metadata']]
        self.verify(row)

    def verify(self, row):
        header_filter = {'gridRange': {'sheetId': self.sheet_id, 'startRowIndex': 0,
            'endRowIndex': 1, 'startColumnIndex': 0, 'endColumnIndex': len(self.headers)}}
        result = self.call(self.service.spreadsheets().values().batchGetByDataFilter(
            spreadsheetId=self.workbook, body={'dataFilters': [self.filter(row), header_filter],
                'valueRenderOption': 'UNFORMATTED_VALUE'}))
        ranges = result.get('valueRanges', [])
        matching = [r for r in ranges if any(f.get('developerMetadataLookup', {}).get('metadataId') == row['_metadata']
                                             for f in r.get('dataFilters', []))]
        def is_header(data_filter):
            grid = data_filter.get('gridRange', {})
            return (grid.get('sheetId') == self.sheet_id and grid.get('startRowIndex', 0) == 0
                    and grid.get('endRowIndex') == 1 and grid.get('startColumnIndex', 0) == 0
                    and grid.get('endColumnIndex') == len(self.headers))
        header_ranges = [r for r in ranges if any(is_header(f) for f in r.get('dataFilters', []))]
        header_values = header_ranges[0].get('valueRange', {}).get('values', []) if len(header_ranges) == 1 else []
        if len(header_values) != 1 or [str(x).strip().casefold() for x in header_values[0]] != self.headers:
            raise ClickoutError('clickouts_headers_changed')
        values = matching[0].get('valueRange', {}).get('values', []) if len(matching) == 1 else []
        if len(values) != 1:
            raise ClickoutError('queued_row_moved_or_removed')
        cells = values[0]
        current = {h: cells[i] if i < len(cells) else '' for i, h in enumerate(self.headers) if h}
        if self.identity(current) != self.identity(row):
            raise ClickoutError('queued_row_identity_changed')
        if str(current.get('status', '')).strip().casefold() != str(row.get('status', '')).strip().casefold():
            raise ClickoutError('queued_row_status_changed')

    @staticmethod
    def filter(row):
        return {'developerMetadataLookup': {'metadataId': row['_metadata']}}

    def update(self, row, **changes):
        if any(key not in self.OWNED for key in changes):
            raise ClickoutError('unowned_column_write')
        self.verify(row)
        values = [None] * len(self.headers)  # Sheets null skips unowned cells.
        for key, value in changes.items():
            values[self.headers.index(key)] = value
        result = self.call(self.service.spreadsheets().values().batchUpdateByDataFilter(
            spreadsheetId=self.workbook, body={'valueInputOption': 'RAW', 'data': [{
                'dataFilter': self.filter(row), 'majorDimension': 'ROWS', 'values': [values]}]}))
        if result.get('totalUpdatedRows') != 1:
            raise ClickoutError('queued_row_update_not_confirmed')
        row.update(changes)


def google_store(workbook):
    # Build a private bounded transport per worker call; google clients are not thread-safe.
    import httplib2
    from google_auth_httplib2 import AuthorizedHttp
    from googleapiclient.discovery import build
    from sheets_helper import get_credentials
    http = AuthorizedHttp(get_credentials(), http=httplib2.Http(timeout=12))
    service = build('sheets', 'v4', http=http, cache_discovery=False)
    return GoogleClickouts(service, workbook)
