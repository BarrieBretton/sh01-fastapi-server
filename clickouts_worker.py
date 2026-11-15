"""Clickouts queue consumer. Provider DELETE is issued at most once per attempt.

The sheet supplies queued identities, not executable code. No permalink scraping,
row deletion, missing-as-zero inference, or automatic retry of ambiguous deletes.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode

from clickouts_state import ClickoutError, digest

IST = timezone(timedelta(hours=5, minutes=30), 'Asia/Kolkata')
PENDING = {'', 'pending', 'queued', 'queued_clickouts', 'awaiting_sheet'}


def stamp(now=None):
    return (now or datetime.now(timezone.utc)).isoformat().replace('+00:00', 'Z')


def retry_seconds(value, default=60):
    try:
        delay = float(value) if re.fullmatch(r'\d+(?:\.\d+)?', str(value)) else (
            parsedate_to_datetime(str(value)) - datetime.now(timezone.utc)).total_seconds()
        return max(default, delay)
    except (ValueError, TypeError, OverflowError):
        return default


def status(row):
    return str(row.get('status', '')).strip().casefold()


def validate(row):
    for field in ('account_id', 'post_id', 'request_id'):
        # Reject numeric cells: large IDs may already have lost precision in Sheets.
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ClickoutError('missing_or_numeric_' + field)
    platform = row.get('platform')
    if platform not in ('instagram', 'facebook'):
        raise ClickoutError('unsupported_platform')
    if not re.fullmatch(r'\d{1,40}', row['account_id']):
        raise ClickoutError('invalid_account_id')
    pattern = r'\d{1,40}' if platform == 'instagram' else r'\d{1,40}_\d{1,40}'
    if not re.fullmatch(pattern, row['post_id']):
        raise ClickoutError('facebook_page_post_id_required' if platform == 'facebook' else 'invalid_post_id')
    if platform == 'facebook' and row['post_id'].split('_')[0] != row['account_id']:
        raise ClickoutError('account_post_mismatch')
    if not re.fullmatch(r'[A-Za-z0-9:_-]{1,160}', row['request_id']):
        raise ClickoutError('invalid_request_id')
    from social_inventory import permalink
    if not isinstance(row.get('urls'), str) or not permalink(row['urls'], platform):
        raise ClickoutError('invalid_permalink')


def post_key(row):
    return 'post:' + digest(':'.join(row[k] for k in ('platform', 'account_id', 'post_id')))


def parse_times(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool) and 0 <= value < 1:
        minutes = round(value * 1440) % 1440
        return [minutes]
    result = set()
    for item in re.split(r'[,;\n]+', str(value).strip()):
        match = re.fullmatch(r'\s*(\d{1,2}):(\d{2})(?::00)?\s*(AM|PM)?\s*', item, re.I)
        if not match:
            raise ClickoutError('invalid_clickouts_timings_use_HH_MM')
        hour, minute = int(match[1]), int(match[2])
        if minute > 59 or (match[3] and not 1 <= hour <= 12) or (not match[3] and hour > 23):
            raise ClickoutError('invalid_clickouts_timings_use_HH_MM')
        if match[3]:
            hour = hour % 12 + (12 if match[3].upper() == 'PM' else 0)
        result.add(hour * 60 + minute)
    return sorted(result)


def read_schedule(values):
    if not values:
        raise ClickoutError('workflow_schedule_empty')
    headers = [str(x).strip().casefold() for x in values[0]]
    for key in ('workflow', 'timings', 'run as job today'):
        if headers.count(key) != 1:
            raise ClickoutError('workflow_schedule_headers_invalid')
    rows = []
    for cells in values[1:]:
        row = {h: cells[i] if i < len(cells) else '' for i, h in enumerate(headers)}
        workflow = re.sub(r'\s+', '', str(row.get('workflow', ''))).casefold()
        if workflow in {'server::clickouts', 'python::clickouts'}:
            rows.append(row)
    if not rows:
        return {'enabled': False, 'times': [], 'timezone': 'Asia/Kolkata', 'reason': 'workflow_not_listed'}
    if len(rows) != 1:
        raise ClickoutError('duplicate_clickouts_schedule_rows')
    row = rows[0]
    enabled = str(row['run as job today']).strip().casefold() in {'yes', 'true', '1'}
    return {'enabled': enabled, 'times': parse_times(row['timings']) if enabled else [],
            'timezone': 'Asia/Kolkata', 'reason': None if enabled else 'workflow_paused'}


class MetaDelete:
    def __init__(self):
        from social_meta_accounts import credential
        self.root_token = credential()  # Facebook Login user token for Instagram deletion.
        self.version = os.getenv('META_GRAPH_VERSION', 'v24.0')
        if not re.fullmatch(r'v\d+\.\d+', self.version):
            raise ClickoutError('invalid_graph_version')
        self.tokens = {}

    def request(self, path, token, method='GET', fields=None):
        import requests
        url = 'https://graph.facebook.com/' + self.version + '/' + path
        if fields:
            url += '?' + urlencode({'fields': fields})
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.request(method, url, headers={'Authorization': 'Bearer ' + token},
                                     timeout=(3, 12), allow_redirects=False, stream=True) as response:
                    raw = bytearray()
                    deadline = time.monotonic() + 15
                    for chunk in response.iter_content(8192):
                        raw.extend(chunk)
                        if len(raw) > 65536 or time.monotonic() > deadline:
                            raise ClickoutError('provider_response_unconfirmed')
                    data = json.loads(raw)
                    error = data.get('error', {}) if isinstance(data, dict) else {}
                    code = error.get('code') if isinstance(error, dict) else None
                    if response.status_code == 429 or code in {4, 17, 32, 613, 80001, 80002, 80004}:
                        # A rejected GET is retryable; any failed DELETE remains manual.
                        raise ClickoutError('meta_rate_limited', retry_seconds(response.headers.get('Retry-After')))
                    if response.status_code in (401, 403) or code in {10, 190, 200, 294}:
                        raise ClickoutError('meta_permissions_or_token')
                    if response.status_code != 200 or error or not isinstance(data, dict):
                        raise ClickoutError('meta_request_rejected')
                    return data
        except ClickoutError:
            raise
        except Exception:
            raise ClickoutError('provider_outcome_unconfirmed') from None

    def token(self, row):
        if row['platform'] == 'instagram':
            return self.root_token
        account = row['account_id']
        if account not in self.tokens:
            page = self.request(account, self.root_token, fields='id,access_token')
            token = page.get('access_token')
            if str(page.get('id')) != account or not isinstance(token, str) or not token:
                raise ClickoutError('facebook_page_token_unavailable')
            self.tokens[account] = token
        return self.tokens[account]

    def verify(self, row):
        from social_inventory import permalink
        instagram = row['platform'] == 'instagram'
        data = self.request(row['post_id'], self.token(row),
                            fields='id,owner,permalink' if instagram else 'id,from,permalink_url')
        owner = data.get('owner' if instagram else 'from')
        if (str(data.get('id')) != row['post_id'] or not isinstance(owner, dict)
                or str(owner.get('id')) != row['account_id']):
            raise ClickoutError('provider_owner_or_post_mismatch')
        if permalink(data.get('permalink' if instagram else 'permalink_url'), row['platform']) != permalink(row['urls'], row['platform']):
            raise ClickoutError('provider_permalink_mismatch')

    def delete(self, row):
        data = self.request(row['post_id'], self.token(row), method='DELETE')
        if data.get('success') is not True or ('deleted_id' in data and str(data['deleted_id']) != row['post_id']):
            raise ClickoutError('provider_did_not_confirm_deletion')
        return {'status': 'done', 'deleted_at': stamp(), 'post_id': row['post_id']}


def acknowledge(row):
    """Optional Operations Desk receipt; failure never repeats a provider deletion."""
    key = os.getenv('OPS_INGEST_TOKEN') or os.getenv('INGEST_TOKEN')
    if not key:
        return 'not_configured'
    try:
        from social_inventory import Dashboard, SafeHTTP, Budget
        dashboard = Dashboard(SafeHTTP(Budget(seconds=12, max_bytes=65536)))
        mapping = dashboard.http.json(dashboard.base + '/api/ingest/resolve-post', dashboard.hosts,
            method='POST', headers={'X-INGEST-KEY': key},
            payload={'platform': row['platform'], 'permalink': row['urls']}).get('post', {})
        if any(mapping.get(k) != row[k] for k in ('platform', 'account_id', 'post_id', 'request_id')):
            return 'identity_mismatch'
        result = dashboard.http.json(dashboard.base + '/api/ingest/deletion-result', dashboard.hosts,
            method='POST', headers={'X-INGEST-KEY': key},
            payload={'request_id': row['request_id'], 'status': 'deleted', 'reason': 'clickouts_provider_confirmed'})
        return 'synced' if result.get('ok') is True else 'pending'
    except Exception:
        return 'pending'


class ClickoutsWorker:
    def __init__(self, store, state, provider_factory=MetaDelete, ack=acknowledge, enabled=None):
        self.store, self.state, self.provider_factory, self.ack = store, state, provider_factory, ack
        self.enabled = enabled or (lambda: os.getenv('SOCIAL_DELETE_ENABLED') == '1')

    def preview(self):
        rows = self.store.read()
        counts = Counter(status(row) or 'pending' for row in rows)
        reasons, eligible = Counter(), 0
        ids = Counter(row.get('request_id') for row in rows if isinstance(row.get('request_id'), str))
        for row in rows:
            if status(row) not in PENDING:
                continue
            try:
                validate(row)
                if ids[row['request_id']] != 1:
                    raise ClickoutError('duplicate_request_id')
                eligible += 1
            except ClickoutError as exc:
                reasons[str(exc)] += 1
        return {'dry_run': True, 'enabled': self.enabled(), 'rows': len(rows), 'eligible': eligible,
                'status_counts': dict(counts), 'blocked_counts': dict(reasons),
                'provider_permissions': 'not_checked', 'schedule': read_schedule(self.store.schedule_rows())}

    def run(self, max_items=3, scheduled=False, now=None):
        if not self.enabled():
            return {'status': 'disabled', 'done': 0}
        if not self.state.acquire():
            return {'status': 'busy', 'done': 0}
        try:
            return self._run(max_items, scheduled, now or datetime.now(timezone.utc))
        except ClickoutError as exc:
            if exc.retry_after:
                self.state.put('cooldown', time.time() + exc.retry_after)
            raise
        finally:
            self.state.release()

    def _run(self, max_items, scheduled, now):
        delay = (self.state.get('cooldown') or 0) - time.time()
        if delay > 0:
            return {'status': 'cooldown', 'retry_after': round(delay), 'done': 0}
        occurrence = None
        if scheduled:
            # One schedule poll per minute across replicas, including idle periods.
            if (self.state.get('next_poll') or 0) > time.time():
                return {'status': 'waiting', 'done': 0}
            self.state.put('next_poll', time.time() + 60)
            schedule = read_schedule(self.store.schedule_rows())
            if not schedule['enabled']:
                return {'status': 'paused', 'done': 0}
            local = now.astimezone(IST)
            due = [minute for minute in schedule['times'] if minute <= local.hour * 60 + local.minute]
            if not due:
                return {'status': 'waiting', 'done': 0}
            # After a restart, catch up the most recent time today (never yesterday).
            occurrence = local.date().isoformat() + ':' + str(max(due))
            if self.state.get('occurrence_done') == occurrence:
                return {'status': 'waiting', 'done': 0}
        rows = self.store.read()
        identifiers = Counter(row.get('request_id') for row in rows if isinstance(row.get('request_id'), str))
        invalid = sum(1 for row in rows if status(row) in PENDING and
                      identifiers.get(row.get('request_id')) != 1)
        candidates = [row for row in rows if identifiers.get(row.get('request_id')) == 1 and
                      (status(row) in PENDING | {'processing'} or
                      (status(row) == 'done' and row.get('dashboard_ack', '') in ('pending', 'not_configured', '')))]
        # Deferred acknowledgements never starve queued deletions; no retries without an ingest key.
        candidates.sort(key=lambda row: status(row) == 'done')
        ack_configured = bool(os.getenv('OPS_INGEST_TOKEN') or os.getenv('INGEST_TOKEN'))
        candidates = [r for r in candidates if status(r) != 'done' or ack_configured]
        if not candidates:
            if occurrence:
                self.state.put('occurrence_done', occurrence)
            return {'status': 'idle', 'done': 0, 'remaining': 0, 'invalid_rows': invalid}
        self.state.guard()
        self.store.prepare()
        counts = Counter(invalid_rows=invalid)
        processed = 0
        provider = None
        started = time.monotonic()
        for row in candidates[:max_items]:
            if time.monotonic() - started >= 40 or not self.enabled():
                break
            self.state.guard()
            # Duplicate request IDs cannot be safely bound to a Sheet row.
            if identifiers.get(row.get('request_id')) != 1:
                counts['invalid'] += 1
                processed += 1
                continue
            self.store.bind(row)
            if status(row) == 'done':
                try:
                    validate(row)
                    receipt = self.state.get(post_key(row))
                    if not receipt or receipt.get('status') != 'done' or receipt.get('post_id') != row['post_id']:
                        raise ClickoutError('unverified_receipt')
                except ClickoutError:
                    self.store.update(row, dashboard_ack='unverified_receipt')
                    counts['unverified_receipts'] += 1
                    processed += 1
                    continue
                ack = self.ack(row)
                self.state.guard()
                self.store.update(row, dashboard_ack=ack)
                counts['acknowledgements'] += 1
                processed += 1
                continue
            try:
                validate(row)
            except ClickoutError as exc:
                self.store.update(row, status='blocked', error=str(exc))
                counts['blocked'] += 1
                processed += 1
                continue
            key = post_key(row)
            receipt = self.state.get(key)
            if receipt and receipt.get('status') == 'done':
                self.finish(row, receipt)
                counts['done'] += 1
                processed += 1
                continue
            prepared = bool(receipt and receipt.get('status') == 'processing' and receipt.get('phase') == 'prepared')
            if not prepared and (receipt or status(row) == 'processing'):
                self.store.update(row, status='manual_required', error='previous_attempt_unconfirmed')
                counts['manual_required'] += 1
                processed += 1
                continue
            try:
                provider = provider or self.provider_factory()
                provider.verify(row)
            except Exception as exc:
                if isinstance(exc, ClickoutError) and exc.retry_after:
                    self.state.put('cooldown', time.time() + exc.retry_after)
                    counts['deferred'] += 1
                    break
                code = str(exc) if isinstance(exc, ClickoutError) else 'provider_verification_failed'
                self.state.guard()
                self.store.update(row, status='blocked', error=code)
                counts['blocked'] += 1
                processed += 1
                continue
            self.state.guard()
            self.store.verify(row)
            if not self.enabled():
                break
            # Durable record BEFORE a possibly destructive call. It is never auto-cleared.
            attempt = {'status': 'processing', 'phase': 'prepared', 'request_id': row['request_id'], 'at': stamp(now)}
            if not prepared and not self.state.claim(key, attempt):
                raise ClickoutError('post_already_claimed')
            self.store.update(row, status='processing', last_attempt_at=stamp(now), error='')
            self.store.verify(row)  # Require an acknowledged, readable processing marker.
            self.state.guard()
            if not self.enabled():
                break
            # Only this phase may have sent DELETE. Prepared attempts can safely resume
            # after a Sheets quota error; armed attempts require manual reconciliation.
            self.state.put(key, {**attempt, 'phase': 'delete_sent'})
            try:
                receipt = provider.delete(row)
                if receipt.get('status') != 'done' or receipt.get('post_id') != row['post_id']:
                    raise ClickoutError('provider_did_not_confirm_deletion')
            except Exception as exc:
                self.state.guard()
                code = str(exc) if isinstance(exc, ClickoutError) else 'provider_outcome_unconfirmed'
                self.state.put(key, {'status': 'manual_required', 'error': code})
                self.store.update(row, status='manual_required', error=code)
                counts['manual_required'] += 1
                processed += 1
                if isinstance(exc, ClickoutError) and exc.retry_after:
                    self.state.put('cooldown', time.time() + exc.retry_after)
                    break
                continue
            self.state.put(key, receipt)  # Recover status-only after restart / Sheets failure.
            self.finish(row, receipt)
            counts['done'] += 1
            processed += 1
        remaining = len(candidates) - processed
        if occurrence and remaining == 0:
            self.state.put('occurrence_done', occurrence)
        result = {'status': 'completed' if not remaining else 'more_pending', 'done': 0,
                  **counts, 'remaining': remaining, 'checked_at': stamp(now)}
        self.state.put('last_result', result)
        return result

    def finish(self, row, receipt):
        self.state.guard()
        # Record success before optional dashboard notification; callback failures cannot undo it.
        self.store.update(row, status='done', deleted_at=receipt['deleted_at'], error='', dashboard_ack='pending')
        ack = self.ack(row)
        self.state.guard()
        self.store.update(row, dashboard_ack=ack)
