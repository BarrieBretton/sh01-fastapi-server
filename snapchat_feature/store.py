from __future__ import annotations

from contextlib import contextmanager
import uuid
from clickouts_state import google_store
from clickouts_routes import configured_workbook
from clickouts_sheet_state import SheetsState
from .core import HEADERS, SnapError, queue_rows, column


class SnapchatState(SheetsState):
    TITLE = '_sh01_snapchat_state'
    MARKER = 'sh01_snapchat_state_v1'
    RANGE_NAME = 'SH01_SNAPCHAT_REVISION'
    LEASE_SECONDS = 900

    def batch(self, requests):
        requests = list(requests)
        for req in list(requests):
            if 'addSheet' in req:
                requests.append({'addProtectedRange': {'protectedRange': {
                    'range': {'sheetId': req['addSheet']['properties']['sheetId']}, 'warningOnly': True,
                    'description': 'Internal encrypted Snapchat job state. Do not edit.'}}})
        return super().batch(requests)


class Workbook:
    def __init__(self, config, google=None):
        self.config = config
        self.google = google or google_store(configured_workbook())
        self.state = SnapchatState(self.google)

    def queue(self):
        props = self.google.properties('Snapchat-Queue')
        grid = props.get('gridProperties', {})
        if grid.get('rowCount', 0) > 10000 or grid.get('columnCount', 0) > 52:
            raise SnapError('snapchat_queue_grid_limit', status=409)
        title = props['title'].replace("'", "''")
        data = self.google.call(self.google.service.spreadsheets().values().get(
            spreadsheetId=self.google.workbook, range=f"'{title}'!A1:{column(grid['columnCount'])}{grid['rowCount']}",
            valueRenderOption='UNFORMATTED_VALUE'))
        return queue_rows(data.get('values', []), self.google.workbook, props['sheetId'], props['title'], self.config)

    def setup(self):
        # Explicit setup only. Existing data/header mismatches are never overwritten.
        book = self.google.call(self.google.service.spreadsheets().get(spreadsheetId=self.google.workbook, fields='sheets.properties'))
        matches = [s for s in book.get('sheets', []) if s['properties']['title'].strip().casefold() == 'snapchat-queue']
        if len(matches) > 1: raise SnapError('snapchat_queue_ambiguous', status=409)
        if not matches:
            sid = int(uuid.uuid4().hex[:7], 16)
            self.google.call(self.google.service.spreadsheets().batchUpdate(spreadsheetId=self.google.workbook,
                body={'requests': [{'addSheet': {'properties': {'sheetId': sid, 'title': 'Snapchat-Queue', 'gridProperties': {'rowCount': 1000, 'columnCount': 26, 'frozenRowCount': 1}}}},
                    {'updateCells': {'start': {'sheetId': sid, 'rowIndex': 0, 'columnIndex': 0},
                     'rows': [{'values': [{'userEnteredValue': {'stringValue': x}} for x in HEADERS]}], 'fields': 'userEnteredValue'}}]}))
        self.google.book = None
        self.queue()  # Validate existing tab; no rows or schedules are seeded.
        self.state.initialize()
        return {'status': 'ready', 'queue_sheet': 'Snapchat-Queue', 'publishing_enabled': self.config.enabled}


@contextmanager
def locked(state):
    if not state.acquire(): raise SnapError('snapchat_busy', status=409, retry_after=60)
    try: yield state
    finally:
        try: state.release()
        except Exception: pass  # Lease expiry recovers; never repeat a provider write.
