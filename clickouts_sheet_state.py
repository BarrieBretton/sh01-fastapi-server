"""Clickouts coordination in the existing workbook; no Redis or new credentials.

Every mutation rotates a named-range revision guard in the same atomic Sheets
batch as its data. A stale guard is tested during initialization before execution.
The hidden tab is internal state; users manage Workflow-Schedule and clickouts.
"""
from __future__ import annotations

import copy
import json
import time
import uuid

from clickouts_state import ClickoutError


class SheetsState:
    TITLE = '_sh01_clickouts_state'
    MARKER = 'sh01_clickouts_state_v1'
    RANGE_NAME = 'SH01_CLICKOUTS_STATE_REVISION'
    LIMIT = 10001
    LEASE_SECONDS = 180

    def __init__(self, store, clock=time.time):
        self.store, self.clock = store, clock
        self.owner = None
        self.control = None
        self.rows = {}
        self.values = {}
        self.sheet_id = None

    @staticmethod
    def cell(value):
        return {'userEnteredValue': {'stringValue': value}}

    def grid(self):
        return {'sheetId': self.sheet_id, 'startRowIndex': 0, 'endRowIndex': 1,
                'startColumnIndex': 1, 'endColumnIndex': 2}

    def guard_requests(self, old, new):
        return [
            {'updateNamedRange': {'namedRange': {'namedRangeId': old, 'range': self.grid()}, 'fields': 'range'}},
            {'deleteNamedRange': {'namedRangeId': old}},
            {'addNamedRange': {'namedRange': {'namedRangeId': new, 'name': self.RANGE_NAME, 'range': self.grid()}}},
        ]

    def write_row(self, index, values):
        return {'updateCells': {'start': {'sheetId': self.sheet_id, 'rowIndex': index, 'columnIndex': 0},
                               'rows': [{'values': [self.cell(v) for v in values]}], 'fields': 'userEnteredValue'}}

    def batch(self, requests):
        return self.store.call(self.store.service.spreadsheets().batchUpdate(
            spreadsheetId=self.store.workbook, body={'requests': requests}))

    def load(self):
        """One snapshot contains the guard, control and records. Never writes."""
        try:
            data = self.store.call(self.store.service.spreadsheets().get(
                spreadsheetId=self.store.workbook, ranges=[f"'{self.TITLE}'!A1:C{self.LIMIT}"],
                fields='namedRanges,sheets(properties,data(startRow,rowData(values(userEnteredValue))))'))
        except ClickoutError as exc:
            if str(exc) != 'sheets_invalid_request':
                raise
            # A missing tab makes the ranged get invalid. Distinguish it from a
            # damaged existing tab; never reset existing state on an API error.
            meta = self.store.call(self.store.service.spreadsheets().get(
                spreadsheetId=self.store.workbook, fields='sheets.properties,namedRanges'))
            if any(s['properties']['title'] == self.TITLE for s in meta.get('sheets', [])):
                raise
            if any(n.get('name') == self.RANGE_NAME for n in meta.get('namedRanges', [])):
                raise ClickoutError('clickouts_state_tab_missing_with_guard')
            self.control = None
            self.rows, self.values = {}, {}
            return False
        sheets = [s for s in data.get('sheets', []) if s['properties']['title'] == self.TITLE]
        if len(sheets) != 1:
            raise ClickoutError('clickouts_state_tab_invalid')
        sheet = sheets[0]
        props = sheet['properties']
        if props['gridProperties']['rowCount'] != self.LIMIT or props['gridProperties']['columnCount'] < 3:
            raise ClickoutError('clickouts_state_grid_changed')
        self.sheet_id = props['sheetId']
        blocks = sheet.get('data', [])
        if len(blocks) != 1 or blocks[0].get('startRow', 0) != 0:
            raise ClickoutError('clickouts_state_snapshot_invalid')
        rows = []
        for row in blocks[0].get('rowData', []):
            cells = []
            for cell in row.get('values', []):
                value = cell.get('userEnteredValue', {})
                if value and set(value) != {'stringValue'}:
                    raise ClickoutError('clickouts_state_cell_invalid')
                cells.append(value.get('stringValue', ''))
            rows.append(cells)
        try:
            if rows[0][0] != self.MARKER or any(rows[0][2:]):
                raise ValueError()
            control = json.loads(rows[0][1])
            if (type(control.get('revision')) is not int or control['revision'] < 0
                    or not isinstance(control.get('guard'), str)
                    or type(control.get('verified')) is not bool):
                raise ValueError()
            ranges = [n for n in data.get('namedRanges', []) if n.get('name') == self.RANGE_NAME]
            if len(ranges) != 1 or ranges[0].get('namedRangeId') != control['guard'] or ranges[0].get('range') != self.grid():
                raise ValueError()
            values, indexes = {}, {}
            for index, cells in enumerate(rows[1:], 1):
                if not any(cells):
                    continue
                if len(cells) < 2 or not cells[0] or cells[0] in values or any(cells[2:]):
                    raise ValueError()
                values[cells[0]] = json.loads(cells[1])
                indexes[cells[0]] = index
            lease = control.get('lease')
            if lease is not None and (not isinstance(lease, dict) or not isinstance(lease.get('owner'), str)
                    or type(lease.get('until')) not in (int, float)):
                raise ValueError()
        except (IndexError, KeyError, ValueError, TypeError, AttributeError):
            raise ClickoutError('clickouts_state_corrupt') from None
        self.control, self.values, self.rows = control, values, indexes
        return True

    def initialize(self):
        if not self.load():
            self.sheet_id = int(uuid.uuid4().hex[:7], 16)
            control = {'revision': 0, 'guard': uuid.uuid4().hex, 'verified': False, 'lease': None}
            requests = [
                {'addSheet': {'properties': {'sheetId': self.sheet_id, 'title': self.TITLE, 'hidden': True,
                                            'gridProperties': {'rowCount': self.LIMIT, 'columnCount': 3}}}},
                {'addNamedRange': {'namedRange': {'namedRangeId': control['guard'], 'name': self.RANGE_NAME,
                                                 'range': self.grid()}}},
                self.write_row(0, [self.MARKER, json.dumps(control)]),
            ]
            try:
                self.batch(requests)
            except ClickoutError as exc:
                if exc.retry_after:
                    raise
                # A competing initializer or lost response may already have
                # created the complete tab. Accept only a valid readable state.
                if not self.load():
                    raise exc
            else:
                self.control, self.values, self.rows = control, {}, {}
        if not self.control['verified']:
            old = self.control['guard']
            self.commit({})  # Rotate away from the guard used by the stale probe.
            try:
                self.batch(self.guard_requests(old, uuid.uuid4().hex) + [self.write_row(0, [self.MARKER, '{}', 'STALE_GUARD_ACCEPTED'])])
            except ClickoutError as exc:
                if str(exc) != 'sheets_invalid_request':
                    raise
            else:
                raise ClickoutError('clickouts_state_guard_not_enforced')
            self.load()
            if not self.control['verified']:
                self.commit({}, verified=True)

    def commit(self, changes, **control_changes):
        """No blind write retry. Only an exact committed revision reconciles loss."""
        old = self.control
        new = {**old, **control_changes, 'revision': old['revision'] + 1, 'guard': uuid.uuid4().hex}
        indexes = dict(self.rows)
        requests = self.guard_requests(old['guard'], new['guard'])
        for key, value in changes.items():
            if key not in indexes:
                indexes[key] = max(indexes.values(), default=0) + 1
            if indexes[key] >= self.LIMIT:
                raise ClickoutError('clickouts_state_capacity_reached')
            packed = json.dumps(value, separators=(',', ':'), allow_nan=False)
            if len(packed) > 30000 or len(key) > 200:
                raise ClickoutError('clickouts_state_record_too_large')
            requests.append(self.write_row(indexes[key], [key, packed]))
        requests.append(self.write_row(0, [self.MARKER, json.dumps(new, separators=(',', ':'))]))
        try:
            self.batch(requests)
        except ClickoutError as exc:
            if exc.retry_after:
                raise  # No immediate reads after quota rejection.
            self.control = None
            if self.load() and self.control['guard'] == new['guard']:
                return  # Exact durable commit, response was lost.
            if str(exc) == 'sheets_invalid_request':
                raise ClickoutError('clickouts_state_revision_conflict') from None
            raise ClickoutError('clickouts_state_write_unconfirmed') from None
        self.control, self.rows = new, indexes
        self.values.update(copy.deepcopy(changes))

    def acquire(self):
        self.initialize()
        lease = self.control.get('lease')
        if lease and lease['until'] > self.clock():
            return False
        owner = uuid.uuid4().hex
        try:
            self.commit({}, lease={'owner': owner, 'until': self.clock() + self.LEASE_SECONDS})
        except ClickoutError as exc:
            if str(exc) == 'clickouts_state_revision_conflict':
                return False
            raise
        self.owner = owner
        return True

    def guard(self):
        lease = (self.control or {}).get('lease') or {}
        if not self.owner or lease.get('owner') != self.owner or lease.get('until', 0) <= self.clock():
            raise ClickoutError('worker_lease_lost')
        # Short batches use the acknowledged lease without redundant Google
        # calls. Every durable mutation also checks the revision at Google.
        if lease['until'] - self.clock() < 60:
            self.commit({}, lease={'owner': self.owner, 'until': self.clock() + self.LEASE_SECONDS})

    def release(self):
        if not self.owner:
            return
        try:
            lease = (self.control or {}).get('lease') or {}
            if lease.get('owner') == self.owner:
                self.commit({}, lease=None)
        finally:
            self.owner = None

    def get(self, key):
        if self.control is None and not self.load():
            return None
        return copy.deepcopy(self.values.get(key))

    def put(self, key, value):
        self.guard()
        self.commit({key: value})

    def claim(self, key, value):
        self.guard()
        if key in self.values:
            return False
        self.commit({key: value})
        return True
