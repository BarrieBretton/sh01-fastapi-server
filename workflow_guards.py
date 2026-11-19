"""Shared workbook coordination for patched n8n workflows; no background jobs."""
from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from clickouts_sheet_state import SheetsState
from clickouts_state import ClickoutError, google_store

REVISION = 'workflow-guards-1'
IST = ZoneInfo('Asia/Kolkata')
WORKBOOK = '1Ol-Zu6mQIku0PeI39sOrmq7NNjU7cSbEVY17BInAdf8'
SCHEDULE_ID = 626043440
QUEUE_TITLE = 'telegram_audio_social_queue'
OWNED = ('status', 'claimed_at', 'render_job_id', 'render_token', 'render_video_url',
         'render_b2_file_name', 'render_size_bytes', 'rendered_at', 'attempt_count', 'last_error')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class GuardState(SheetsState):
    TITLE = '_sh01_workflow_guards'
    MARKER = 'sh01_workflow_guards_v1'
    RANGE_NAME = 'SH01_WORKFLOW_GUARDS_REVISION'

    def __init__(self, store, clock=time.time):
        super().__init__(store, clock)
        self.extra = []

    def batch(self, requests):
        # Commit the protected Sheet update and state revision in ONE transaction.
        return super().batch(requests + self.extra)


def authorize(x_api_key: str | None = Header(default=None, alias='X-API-KEY')):
    # Same credential as the existing Telegram media HTTP nodes.
    from media_router import _require_media_key
    _require_media_key(x_api_key)


router = APIRouter(prefix='/workflow-guards', tags=['Workflow coordination'],
                   dependencies=[Depends(authorize)])


def context():
    workbook = os.getenv('SPREADSHEET_ID', '').strip()
    if workbook != WORKBOOK:
        raise HTTPException(503, 'workflow_guards_workbook_mismatch')
    return GuardState(google_store(workbook))


def snapshot(state, *, schedule=False):
    meta = state.store.call(state.store.service.spreadsheets().get(
        spreadsheetId=state.store.workbook, fields='sheets.properties'))
    sheets = [s['properties'] for s in meta.get('sheets', [])
              if (s['properties']['sheetId'] == SCHEDULE_ID if schedule
                  else s['properties']['title'] == QUEUE_TITLE)]
    if len(sheets) != 1:
        raise HTTPException(409, 'workflow_sheet_missing_or_ambiguous')
    sheet = sheets[0]
    title = "'" + sheet['title'].replace("'", "''") + "'"
    data = state.store.call(state.store.service.spreadsheets().values().get(
        spreadsheetId=state.store.workbook, range=title,
        valueRenderOption='UNFORMATTED_VALUE')).get('values', [])
    if not data:
        raise HTTPException(409, 'workflow_sheet_empty')
    headers = [str(x).strip() for x in data[0]]
    nonempty = [h for h in headers if h]
    if len(set(nonempty)) != len(nonempty):
        raise HTTPException(409, 'duplicate_sheet_headers')
    rows = []
    for index, values in enumerate(data[1:], 2):
        if any(v != '' for v in values):
            row = {h: values[col] if col < len(values) else ''
                   for col, h in enumerate(headers) if h}
            row['row_number'] = index
            rows.append(row)
    return sheet, headers, rows


def clock_minute(value):
    match = re.fullmatch(r'(\d{1,2})(?::(\d{2}))?(am|pm)',
                         re.sub(r'\s+', '', str(value).lower()))
    if not match:
        raise HTTPException(409, 'invalid_schedule_time')
    hour, minute = int(match[1]), int(match[2] or 0)
    if not 1 <= hour <= 12 or minute > 59:
        raise HTTPException(409, 'invalid_schedule_time')
    return (hour % 12 + (12 if match[3] == 'pm' else 0)) * 60 + minute


def occurrence(row, stamp, now=None):
    try:
        dt = datetime.fromisoformat(stamp.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            raise ValueError()
    except (ValueError, AttributeError):
        raise HTTPException(422, 'invalid_schedule_timestamp') from None
    now = now or datetime.now(timezone.utc)
    age = (now - dt).total_seconds()
    if not -15 <= age <= 120:
        raise HTTPException(409, 'schedule_occurrence_outside_dispatch_window')
    dt = dt.astimezone(IST)
    minute = clock_minute(row.get('timings'))
    if minute != dt.hour * 60 + dt.minute:
        raise HTTPException(409, 'schedule_time_mismatch')
    # Same scheduled row cannot be dispatched again after a target-name edit.
    return 'dispatch:' + digest([dt.date().isoformat(), minute, row['row_number']])


def safe_commit(state, changes, **control):
    try:
        state.commit(changes, **control)
    except ClickoutError as exc:
        # A lost reply followed by another revision is deliberately not retried.
        raise HTTPException(409 if 'conflict' in str(exc) else 503,
                            'coordination_commit_not_confirmed') from None


class Dispatch(BaseModel):
    row_number: int = Field(ge=2)
    workflow_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    workflow_name: str = Field(min_length=1, max_length=256)
    scheduled_at: str = Field(max_length=80)


@router.get('/health')
def health():
    state = context()
    # Read only; installation must explicitly initialize before workflow changes.
    present = state.load()
    return {'revision': REVISION, 'workbook_matches': True, 'state_present': present,
            'state_verified': bool(present and state.control['verified'])}


@router.post('/initialize')
def initialize():
    state = context()
    state.initialize()
    return {'revision': REVISION, 'state_verified': state.control['verified']}


@router.post('/schedule/claim')
def schedule_claim(request: Dispatch):
    state = context()
    if not state.load() or not state.control['verified']:
        raise HTTPException(503, 'workflow_guards_not_initialized')
    _, _, rows = snapshot(state, schedule=True)
    matches = [r for r in rows if r['row_number'] == request.row_number]
    if len(matches) != 1 or matches[0].get('workflow') != request.workflow_name:
        raise HTTPException(409, 'schedule_row_changed')
    row = matches[0]
    if request.workflow_name.lower().startswith(('server::', 'python::')):
        raise HTTPException(409, 'server_workflow_not_an_n8n_target')
    key = occurrence(row, request.scheduled_at)
    if key in state.values:
        return {'allowed': False, 'reason': 'occurrence_already_reserved'}
    payload = copy.deepcopy(row)
    payload['job'] = str(payload.pop('run as job today', '')).strip().lower() == 'yes'
    payload['gdrive_link'] = payload.pop('public drive link', '')
    payload['_sh01_workflow_id'] = request.workflow_id
    payload['_sh01_dispatch_receipt'] = key
    safe_commit(state, {key: {'kind': 'dispatch_reserved', 'workflow_id': request.workflow_id,
                             'row_number': request.row_number, 'at': request.scheduled_at}})
    # Never grant a second dispatch after response loss/retry, even to same caller.
    return {'allowed': True, 'payload': payload}


class QueueClaim(BaseModel):
    queue_id: str = Field(min_length=1, max_length=512)
    action: str = Field(pattern=r'^(submit|poll)$')


class QueueCommit(BaseModel):
    guard: str = Field(pattern=r'^[a-f0-9]{32}$')
    queue_id: str = Field(min_length=1, max_length=512)
    changes: dict


def find_queue(state, queue_id):
    sheet, headers, rows = snapshot(state)
    if not set(('queue_id',) + OWNED).issubset(headers):
        raise HTTPException(409, 'queue_headers_missing')
    matches = [r for r in rows if str(r.get('queue_id', '')) == queue_id]
    if len(matches) != 1:
        raise HTTPException(409, 'queue_identity_missing_or_duplicated')
    return sheet, headers, matches[0]


@router.post('/telegram/claim')
def queue_claim(request: QueueClaim):
    state = context()
    if not state.load() or not state.control['verified']:
        raise HTTPException(503, 'workflow_guards_not_initialized')
    sheet, headers, row = find_queue(state, request.queue_id)
    expected = 'staged' if request.action == 'submit' else 'rendering'
    if str(row.get('status', '')).lower() != expected:
        return {'allowed': False, 'reason': 'queue_row_changed'}
    key = 'queue:' + digest(request.queue_id)
    existing = state.values.get(key, {})
    if existing.get('until', 0) > state.clock():
        return {'allowed': False, 'reason': 'queue_row_owned'}
    token = uuid.uuid4().hex
    safe_commit(state, {key: {'guard': token, 'until': state.clock() + 180,
                             'row_hash': digest(row), 'sheet_id': sheet['sheetId'],
                             'headers': headers}})
    return {'allowed': True, 'row': {**row, 'action': request.action}, 'guard': token}


def commit_queue(state, request):
    key = 'queue:' + digest(request.queue_id)
    claim = state.values.get(key, {})
    if claim.get('guard') != request.guard or claim.get('until', 0) <= state.clock() + 5:
        raise HTTPException(409, 'queue_claim_expired_or_replaced')
    sheet, headers, row = find_queue(state, request.queue_id)
    if (claim['row_hash'] != digest(row) or claim['sheet_id'] != sheet['sheetId']
            or claim['headers'] != headers):
        raise HTTPException(409, 'queue_row_changed_since_claim')
    changes = request.changes
    if set(changes) - set(OWNED):
        raise HTTPException(422, 'unsupported_queue_columns')
    for name, value in changes.items():
        if type(value) not in (str, int, float) or len(str(value)) > 8000:
            raise HTTPException(422, 'invalid_queue_value')
    if changes.get('status', row.get('status')) not in ('staged', 'rendering', 'rendered', 'failed'):
        raise HTTPException(422, 'invalid_render_queue_status')
    # Other patched executions must rotate this same state guard before writing.
    state.extra = [
        {'updateCells': {'start': {'sheetId': sheet['sheetId'],
                                  'rowIndex': row['row_number'] - 1,
                                  'columnIndex': headers.index(name)},
                         'rows': [{'values': [state.cell(str(value))]}],
                         'fields': 'userEnteredValue'}}
        for name, value in changes.items()
    ]
    if claim['until'] <= state.clock() + 5:
        state.extra = []
        raise HTTPException(409, 'queue_claim_expired_during_verification')
    try:
        safe_commit(state, {key: {'guard': request.guard, 'until': 0, 'completed': True}})
    finally:
        state.extra = []
    return {'ok': True, 'queue_id': request.queue_id}


@router.post('/telegram/commit')
def queue_commit(request: QueueCommit):
    state = context()
    if not state.load() or not state.control['verified']:
        raise HTTPException(503, 'workflow_guards_not_initialized')
    return commit_queue(state, request)
