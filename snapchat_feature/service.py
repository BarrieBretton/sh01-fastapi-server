from __future__ import annotations

import hashlib
import re
import secrets
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode

from clickouts_state import ClickoutError
from .core import Config, SnapError, digest, job_id, public_job, schedules, utcnow
from .store import Workbook, locked
from .client import SnapClient, TOKEN, download, probe, encrypt_file


def safe_code(exc):
    return str(exc) if isinstance(exc, (SnapError, ClickoutError)) and re.fullmatch(r'[a-z][a-z0-9_]{0,100}', str(exc)) else 'snapchat_operation_failed'


def activation_step(health):
    if health.get('state_error'):
        return 'check_workbook_access'
    if health['workbook_state'] == 'setup_required':
        return 'initialize_workbook'
    if health['missing_configuration'] or health.get('configuration_error'):
        return 'configure_snapchat'
    if health.get('authorization_error'):
        return 'check_encryption_key'
    if health['authorization'] == 'reconnect_required':
        return 'reconnect_snapchat'
    if health['authorization'] != 'credentials_stored':
        return 'connect_snapchat'
    if health['api_access'] != 'profile_read_verified_previously':
        return 'check_profile_access'
    return 'review_queue'


class Service:
    def __init__(self, config=None, workbook=None, client_factory=SnapClient, clock=time.time):
        self.config = config or Config.env()
        self._book = workbook
        self.client_factory, self.clock = client_factory, clock

    @property
    def book(self):
        if self._book is None:
            self._book = Workbook(self.config)
        return self._book

    @property
    def state(self):
        return self.book.state

    def client(self):
        return self.client_factory(self.config, self.state)

    def health(self):
        result = {'revision': 'snapchat-v2', 'enabled': self.config.enabled, 'missing_configuration': self.config.missing(),
                  'profile_id': self.config.profile_id, 'workbook_state': 'unknown', 'authorization': 'unknown', 'api_access': 'not_verified',
                  'monetization_eligibility': 'not_determined', 'checked_at': utcnow(), 'schedules': [], 'jobs': []}
        if not result['missing_configuration']:
            try:
                self.config.validate()
            except SnapError as exc:
                result['configuration_error'] = safe_code(exc)
        try:
            book = self.book
        except Exception as exc:
            result['state_error'] = result['schedule_error'] = safe_code(exc)
        else:
            try:
                state = book.state
                if state.load() and state.control['verified']:
                    result['workbook_state'] = 'initialized'
                    self._read_health_snapshot(result, state)
                else:
                    result['workbook_state'] = 'setup_required'
            except Exception as exc:
                result['state_error'] = safe_code(exc)
            try:
                result['schedules'] = schedules(book.google.schedule_rows())
            except Exception as exc:
                result['schedule_error'] = safe_code(exc)
        result['next_step'] = activation_step(result)
        return result

    def _read_health_snapshot(self, result, state):
        token = state.get('oauth_tokens')
        if not token:
            result['authorization'] = 'not_connected'
        elif token.get('refresh_pending'):
            result['authorization'] = 'reconnect_required'
        elif not result['missing_configuration'] and 'configuration_error' not in result:
            try:
                data = self.config.unseal(token['encrypted'])
            except SnapError as exc:
                result['authorization_error'] = safe_code(exc)
            else:
                same_account = data.get('client_id') == self.config.client_id and data.get('profile_id') == self.config.profile_id
                result['authorization'] = 'credentials_stored' if same_account else 'reconnect_required'
        connection = state.get('connection') or {}
        if (result['authorization'] == 'credentials_stored' and connection.get('id') == self.config.profile_id
                and connection.get('client_id') == self.config.client_id):
            result['connection'] = {key: connection[key] for key in ('id', 'display_name', 'checked_at') if key in connection}
            result['api_access'] = 'profile_read_verified_previously'
        lease = state.control.get('lease') or {}
        jobs = [public_job(v) for k, v in state.values.items() if k.startswith('job:') and v.get('profile_id') == self.config.profile_id]
        for j in jobs:
            if j['status'] in {'running', 'publishing'} and lease.get('until', 0) <= self.clock():
                j['status'] = 'outcome_unknown' if j['status'] == 'publishing' else 'interrupted_before_publication'
        result['jobs'] = sorted(jobs, key=lambda j: j['created_at'], reverse=True)[:100]
        result['last_tick'] = state.get('last_tick')
        result['metrics'] = state.get('metrics:' + self.config.profile_id)

    def preview(self):
        self.state.load()
        result = []
        for r in self.book.queue():
            item = {k: r[k] for k in ('item_id', 'destination', 'description', 'locale', 'fingerprint', 'source_locator', 'reference', 'problems')}
            old = self.state.get('job:' + job_id(self.config.profile_id, r))
            item['existing_job'] = public_job(old) if old else None
            item['ready'] = not r['problems'] and (not old or old['status'] in {'failed_before_publication', 'cancelled'})
            result.append(item)
        return {'rows': result, 'enabled': self.config.enabled, 'profile_id': self.config.profile_id,
                'checked_at': utcnow(), 'media_validation': 'Actual format/duration checked before upload; preview does not download media.'}

    def connect(self):
        self.config.validate()
        with locked(self.state):
            nonce = secrets.token_urlsafe(32)
            self.state.put('oauth_pending', {'state_hash': hashlib.sha256(nonce.encode()).hexdigest(),
                'expires_at': self.clock() + 600, 'profile_id': self.config.profile_id, 'client_id': self.config.client_id})
        return {'authorize_url': 'https://accounts.snapchat.com/login/oauth2/authorize?' + urlencode({
            'response_type': 'code', 'client_id': self.config.client_id, 'redirect_uri': self.config.redirect_uri,
            'scope': 'snapchat-profile-api', 'state': nonce}), 'expires_in': 600}

    def callback(self, code, nonce):
        self.config.validate()
        if not code or len(code) > 4000 or not nonce or len(nonce) > 200: raise SnapError('oauth_response_invalid', status=400)
        with locked(self.state):
            value = self.state.get('oauth_pending') or {}
            if value.get('expires_at', 0) < self.clock() or value.get('profile_id') != self.config.profile_id or value.get('client_id') != self.config.client_id or not secrets.compare_digest(value.get('state_hash', ''), hashlib.sha256(nonce.encode()).hexdigest()):
                raise SnapError('oauth_state_invalid_or_expired', status=400)
            self.state.put('oauth_pending', {})  # Consume before exchanging the one-time code.
            client = self.client()
            result = client.transport(TOKEN, method='POST', form={'grant_type': 'authorization_code', 'code': code,
                'client_id': self.config.client_id, 'client_secret': self.config.client_secret, 'redirect_uri': self.config.redirect_uri})
            client.save_tokens(result)
            profile = self._verify_profile(client)
        return {'status': 'connected', 'profile': profile, 'publishing_enabled': self.config.enabled}

    def check_connection(self):
        self.config.validate()
        with locked(self.state):
            profile = self._verify_profile(self.client())
        return {'status': 'profile_read_verified', 'profile': profile, 'publishing_permission': 'requires_first_publish_test'}

    def _verify_profile(self, client):
        profile = client.profile()
        self.state.put('connection', {**profile, 'client_id': self.config.client_id})
        return profile

    def require_enabled(self):
        self.config.validate()
        if not self.config.enabled: raise SnapError('snapchat_publishing_disabled', status=409)
        tokens = self.state.get('oauth_tokens')
        if not tokens or tokens.get('refresh_pending'): raise SnapError('snapchat_authorization_required', status=409)
        identity = self.config.unseal(tokens['encrypted'])
        if identity.get('profile_id') != self.config.profile_id or identity.get('client_id') != self.config.client_id:
            raise SnapError('snapchat_account_changed_reconnect', status=409)

    def enqueue(self, row, occurrence=None, occurrence_key=None, workflow=None):
        if row['problems']: raise SnapError('queue_item_not_ready', status=409)
        ident = job_id(self.config.profile_id, row)
        old = self.state.get('job:' + ident)
        if old:
            # Exactly the same immutable item is reused; a new run cannot bypass
            # an uncertain publication, or silently replace content under an ID.
            if old.get('fingerprint') != row['fingerprint']: raise SnapError('queue_item_changed_use_new_id', status=409)
            return old
        now = utcnow()
        job = {'job_id': ident, 'profile_id': self.config.profile_id, 'item_id': row['item_id'], 'destination': row['destination'],
               'fingerprint': row['fingerprint'], 'source_locator': row['source_locator'], 'reference': row['reference'],
               'candidate_encrypted': self.config.seal(row), 'status': 'queued', 'stage': 'queued', 'attempts': 0,
               'created_at': now, 'updated_at': now, 'occurrence': occurrence, 'publish_started': False, 'retry_at': 0}
        changes = {'job:' + ident: job}
        if occurrence_key:
            changes[occurrence_key] = {'due': occurrence, 'workflow': workflow, 'job_id': ident, 'status': 'queued'}
        self.state.guard()
        self.state.commit(changes)
        return job

    def run(self, item_id, fingerprint):
        with locked(self.state):
            self.require_enabled()
            matches = [r for r in self.book.queue() if r['item_id'] == item_id]
            if len(matches) != 1 or matches[0]['fingerprint'] != fingerprint: raise SnapError('queue_changed_refresh_preview', status=409)
            job = self.enqueue(matches[0])
            if job['status'] == 'failed_before_publication' and not job.get('publish_started'):
                job.update(status='queued', stage='queued', retry_at=0, error=None, attempts=0, updated_at=utcnow())
                self.state.put('job:' + job['job_id'], job)
            return public_job(job)

    def tick(self):
        if not self.config.enabled: return {'status': 'paused'}
        with locked(self.state):
            self.require_enabled()
            if self.state.get('next_poll_at') and self.state.get('next_poll_at') > self.clock(): return {'status': 'cooldown'}
            self.state.put('next_poll_at', self.clock() + 60)
            schedule = schedules(self.book.google.schedule_rows())
            rows = None; created = []
            for entry in schedule:
                if entry.get('error') or not entry['due']: continue
                key = 'occurrence:' + digest([self.config.profile_id, entry['workflow'], entry['due']])[:40]
                if self.state.get(key): continue
                if rows is None: rows = self.book.queue()
                candidates = [r for r in rows if r['destination'] == entry['destination'] and not r['problems'] and not self.state.get('job:' + job_id(self.config.profile_id, r))]
                if candidates:
                    job = self.enqueue(candidates[0], entry['due'], key, entry['workflow'])
                    created.append(public_job(job))
                else:
                    self.state.put(key, {'due': entry['due'], 'workflow': entry['workflow'], 'job_id': None, 'status': 'no_ready_content'})
            self.state.put('last_tick', {'checked_at': utcnow(), 'status': 'checked', 'queued': len(created)})
            return {'status': 'checked', 'jobs': created}

    def fresh_row(self, job):
        rows = [r for r in self.book.queue() if r['item_id'] == job['item_id']]
        if len(rows) != 1 or rows[0]['problems'] or rows[0]['fingerprint'] != job['fingerprint']:
            raise SnapError('queue_item_changed_or_unapproved', status=409)
        return rows[0]

    def has_work(self):
        self.state.load()
        return any(k.startswith('job:') and v.get('profile_id') == self.config.profile_id and (
            v.get('status') in {'running', 'publishing'} or
            v.get('status') in {'queued', 'waiting_retry'} and v.get('retry_at', 0) <= self.clock() or
            v.get('status') == 'accepted' and v.get('destination') == 'spotlight' and v.get('next_verify_at', 0) <= self.clock()
        ) for k, v in self.state.values.items())

    def process_one(self, downloader=download, prober=probe, encrypter=encrypt_file):
        if not self.config.enabled: return {'status': 'paused'}
        with locked(self.state):
            self.require_enabled()
            jobs = [v for k, v in self.state.values.items() if k.startswith('job:') and v['profile_id'] == self.config.profile_id]
            for j in jobs:
                if j['status'] == 'publishing':
                    j.update(status='outcome_unknown', error='publish_interrupted_do_not_repost', updated_at=utcnow())
                    self.state.put('job:' + j['job_id'], j)
                elif j['status'] == 'running':
                    j.update(status='queued', stage='recovering_before_publication')
                    self.state.put('job:' + j['job_id'], j)
            pending = sorted([j for j in jobs if j['status'] in {'queued', 'waiting_retry'} and j.get('retry_at', 0) <= self.clock()], key=lambda j: j['created_at'])
            if not pending:
                self.verify_one(jobs)
                return {'status': 'idle'}
            job = pending[0]; key = 'job:' + job['job_id']; deadline = time.monotonic() + 660
            def guard():
                if time.monotonic() >= deadline: raise SnapError('snapchat_job_time_limit')
                self.state.guard()
            def stage(name, part=0):
                guard(); job.update(stage=name, updated_at=utcnow())
                if part: job['stage'] += '_part_' + str(part)
                self.state.put(key, job)
            try:
                job.update(status='running', attempts=job['attempts'] + 1, owner=self.state.owner, error=None)
                stage('validating_queue'); row = self.fresh_row(job)
                client = self.client(); client.profile()
                with tempfile.TemporaryDirectory(prefix='sh01-snapchat-') as tmp:
                    source, encrypted = Path(tmp) / 'source.mp4', Path(tmp) / 'encrypted.mp4'
                    stage('downloading'); details = downloader(row['source_url'], source, self.config, guard)
                    stage('validating_media'); details.update(prober(source)); job['media'] = details
                    content_key = 'content:' + digest([self.config.profile_id, job['destination'], details['sha256']])
                    prior = self.state.get(content_key)
                    if prior:
                        raise SnapError('content_already_submitted_do_not_repost', status=409)
                    stage('encrypting'); encryption_key, iv = encrypter(source, encrypted, guard)
                    media = client.upload(encrypted, encryption_key, iv, guard, stage)
                    job['media_id'] = media
                    self.fresh_row(job)  # Recheck approval after transfer, immediately before publication.
                    guard()
                    job.update(status='publishing', stage='publishing', publish_started=True, updated_at=utcnow())
                    self.state.commit({key: job, content_key: {'job_id': job['job_id'], 'attempted_at': utcnow()}})
                    response = client.publish(media, row)
                    job.update(status='accepted', stage='accepted_by_snapchat', updated_at=utcnow(), provider_request_id=response.get('request_id'))
                    if row['destination'] == 'spotlight':
                        provider_id = response.get('spotlight_id')
                        if not isinstance(provider_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', provider_id):
                            raise SnapError('snapchat_accepted_without_post_id')
                        job['provider_id'] = provider_id
                        job['published_url'] = 'https://www.snapchat.com/spotlight/' + provider_id
                        job['link_verified'] = False  # Provider identity verification is distinct from browser URL verification.
                    else:
                        job['stage'] = 'story_accepted_verify_in_snapchat'
                    self.state.put(key, job)
                self.verify_one([job])
                return public_job(self.state.get(key))
            except Exception as exc:
                # Every failure after dispatch intent is terminal for automatic
                # publishing. An HTTP error alone is never interpreted as "not posted".
                job['error'] = safe_code(exc); job['updated_at'] = utcnow()
                if job.get('publish_started'):
                    job['status'] = 'outcome_unknown'
                elif isinstance(exc, SnapError) and exc.retry_after and job['attempts'] < 3:
                    job.update(status='waiting_retry', retry_at=self.clock() + exc.retry_after)
                else: job['status'] = 'failed_before_publication'
                self.state.put(key, job)
                return public_job(job)

    def verify_one(self, jobs):
        candidates = [j for j in jobs if j['destination'] == 'spotlight' and j['status'] == 'accepted' and j.get('provider_id') and j.get('next_verify_at', 0) <= self.clock()]
        if not candidates: return
        job = candidates[0]; key = 'job:' + job['job_id']
        job['next_verify_at'] = self.clock() + 600
        self.state.put(key, job)
        try:
            status = self.client().verify(job['provider_id'])
            job.update(provider_status=status, verified_at=utcnow(), updated_at=utcnow(), verification_error=None)
            if status == 'LIVE': job.update(status='published', stage='provider_live_verified')
            elif status == 'REJECTED': job.update(status='rejected', stage='provider_rejected')
        except Exception as exc: job['verification_error'] = safe_code(exc)
        self.state.put(key, job)

    def refresh_metrics(self):
        self.config.validate()
        with locked(self.state):
            old = self.state.get('metrics:' + self.config.profile_id)
            if old and old.get('checked_at', 0) + 1800 > self.clock(): return old
            cooldown = self.state.get('metrics_next_at') or 0
            if cooldown > self.clock(): raise SnapError('snapchat_metrics_cooldown', status=429, retry_after=int(cooldown - self.clock()))
            self.state.put('metrics_next_at', self.clock() + 1800)
            metrics = self.client().metrics()
            self.state.put('metrics:' + self.config.profile_id, metrics)
            return metrics

    def job(self, ident):
        self.state.load(); job = self.state.get('job:' + ident)
        if not job or job.get('profile_id') != self.config.profile_id: raise SnapError('snapchat_job_not_found', status=404)
        return public_job(job)
