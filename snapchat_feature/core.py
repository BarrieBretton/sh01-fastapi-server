from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlsplit, urlunsplit

from cryptography.fernet import Fernet, InvalidToken
from clickouts_worker import parse_times

IST = timezone(timedelta(hours=5, minutes=30), 'Asia/Kolkata')
WORKFLOWS = {'Server::Snapchat-Spotlight': 'spotlight', 'Server::Snapchat-Stories': 'story'}
HEADERS = ['item_id', 'source_url', 'destination', 'status', 'approved', 'rights_confirmed', 'description', 'locale']
ID = re.compile(r'^[A-Za-z0-9_-]{1,100}$')


class SnapError(RuntimeError):
    def __init__(self, code, *, status=503, retry_after=0):
        super().__init__(code)
        self.status, self.retry_after = status, retry_after


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def utcnow():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Config:
    client_id: str = ''
    client_secret: str = ''
    redirect_uri: str = ''
    profile_id: str = ''
    encryption_key: str = ''
    source_hosts: tuple = ()
    enabled: bool = False
    max_bytes: int = 200 * 1024 * 1024

    @classmethod
    def env(cls):
        return cls(
            client_id=os.getenv('SNAPCHAT_CLIENT_ID', '').strip(),
            client_secret=os.getenv('SNAPCHAT_CLIENT_SECRET', '').strip(),
            redirect_uri=os.getenv('SNAPCHAT_REDIRECT_URI', '').strip(),
            profile_id=os.getenv('SNAPCHAT_PROFILE_ID', '').strip(),
            encryption_key=os.getenv('SNAPCHAT_TOKEN_ENCRYPTION_KEY', '').strip(),
            source_hosts=tuple(sorted(set(x.strip().lower() for x in os.getenv('SNAPCHAT_SOURCE_HOSTS', '').split(',') if x.strip()))),
            enabled=os.getenv('SNAPCHAT_ENABLED', '').strip().lower() in {'1', 'true'},
        )

    def missing(self):
        fields = {'SNAPCHAT_CLIENT_ID': self.client_id, 'SNAPCHAT_CLIENT_SECRET': self.client_secret,
                  'SNAPCHAT_REDIRECT_URI': self.redirect_uri, 'SNAPCHAT_PROFILE_ID': self.profile_id,
                  'SNAPCHAT_TOKEN_ENCRYPTION_KEY': self.encryption_key, 'SNAPCHAT_SOURCE_HOSTS': self.source_hosts}
        return [name for name, value in fields.items() if not value]

    def validate(self):
        if self.missing():
            raise SnapError('snapchat_configuration_incomplete')
        if not re.fullmatch(r'[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}', self.profile_id):
            raise SnapError('snapchat_profile_id_invalid')
        try:
            u = urlsplit(self.redirect_uri)
            if u.scheme != 'https' or not u.hostname or u.username or u.password or u.port not in (443, None) or u.query or u.fragment or u.path != '/snapchat/oauth/callback':
                raise ValueError()
            Fernet(self.encryption_key.encode())
        except (ValueError, TypeError):
            raise SnapError('snapchat_redirect_or_encryption_key_invalid') from None
        if any(not re.fullmatch(r'[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?', h) or '.' not in h for h in self.source_hosts):
            raise SnapError('snapchat_source_hosts_invalid')

    def seal(self, data):
        return Fernet(self.encryption_key.encode()).encrypt(json.dumps(data, separators=(',', ':')).encode()).decode()

    def unseal(self, data):
        try:
            return json.loads(Fernet(self.encryption_key.encode()).decrypt(data.encode()))
        except (InvalidToken, ValueError, TypeError, AttributeError):
            raise SnapError('snapchat_encrypted_state_unreadable') from None


def source_locator(url, hosts):
    try:
        u = urlsplit(url)
        if u.scheme != 'https' or u.hostname not in hosts or u.port not in (None, 443) or u.username or u.password or u.fragment or len(url) > 12000:
            raise ValueError()
        if any(ord(c) < 32 for c in url):
            raise ValueError()
        return urlunsplit(('https', u.netloc, u.path, '', ''))
    except (ValueError, TypeError):
        raise SnapError('source_url_not_allowed', status=400) from None


def queue_rows(values, workbook, sheet_id, title, config):
    if not values:
        raise SnapError('snapchat_queue_headers_missing', status=409)
    names = [str(x).strip().lower() for x in values[0]]
    if any(names.count(x) != 1 for x in HEADERS):
        raise SnapError('snapchat_queue_headers_invalid', status=409)
    rows, counts = [], {}
    for index, cells in enumerate(values[1:], 2):
        if not any(cells):
            continue
        row = {name: str(cells[names.index(name)]) if names.index(name) < len(cells) else '' for name in HEADERS}
        row['item_id'] = row['item_id'].strip()
        row['destination'] = row['destination'].strip().lower()
        row['locale'] = row['locale'].strip() or 'en_IN'
        problems = []
        if not ID.fullmatch(row['item_id']): problems.append('invalid_item_id')
        if row['destination'] not in {'spotlight', 'story'}: problems.append('destination_must_be_spotlight_or_story')
        for field, required in [('status', 'ready'), ('approved', 'yes'), ('rights_confirmed', 'yes')]:
            if row[field].strip().lower() != required: problems.append(field + '_must_be_' + required)
        if len(row['description']) > 160: problems.append('description_exceeds_160_characters')
        if not re.fullmatch(r'[a-z]{2}_[A-Z]{2}', row['locale']): problems.append('invalid_locale')
        try: locator = source_locator(row['source_url'], config.source_hosts)
        except SnapError as exc: problems.append(str(exc)); locator = ''
        reference = {'sheet': title, 'sheet_id': sheet_id, 'row': index, 'column': 'source_url',
                     'cell': f'{column(names.index("source_url") + 1)}{index}'}
        reference['link'] = f'https://docs.google.com/spreadsheets/d/{quote(workbook, safe="")}/edit#gid={sheet_id}&range={reference["cell"]}'
        row.update(fingerprint=digest({k: row[k] for k in HEADERS}), source_locator=locator,
                   reference=reference, problems=problems)
        counts[row['item_id']] = counts.get(row['item_id'], 0) + 1
        rows.append(row)
    for row in rows:
        if counts[row['item_id']] > 1: row['problems'].append('duplicate_item_id')
    return rows


def column(n):
    result = ''
    while n:
        n, rem = divmod(n - 1, 26); result = chr(65 + rem) + result
    return result


def schedules(values, now=None):
    now = (now or datetime.now(timezone.utc)).astimezone(IST)
    if not values: raise SnapError('workflow_schedule_empty', status=409)
    headers = [str(x).strip().lower() for x in values[0]]
    if any(headers.count(x) != 1 for x in ('workflow', 'timings')):
        raise SnapError('workflow_schedule_headers_invalid', status=409)
    result = []
    for label, destination in WORKFLOWS.items():
        matches = [r for r in values[1:] if len(r) > headers.index('workflow') and str(r[headers.index('workflow')]).strip().casefold() == label.casefold()]
        item = {'workflow': label, 'destination': destination, 'timezone': 'Asia/Kolkata', 'times': [], 'due': None, 'next_at': None}
        if len(matches) > 1:
            item['error'] = 'duplicate_schedule_rows'; result.append(item); continue
        raw = matches[0][headers.index('timings')] if matches and len(matches[0]) > headers.index('timings') else ''
        if raw is not None and (not isinstance(raw, str) or raw.strip()):
            try: item['times'] = parse_times(raw)
            except Exception: item['error'] = 'invalid_schedule_time'
        minute = now.hour * 60 + now.minute
        due = [m for m in item['times'] if m <= minute]
        if due: item['due'] = now.date().isoformat() + f'T{due[-1] // 60:02d}:{due[-1] % 60:02d}+05:30'
        future = [m for m in item['times'] if m > minute]
        if item['times']:
            day = now if future else now + timedelta(days=1)
            m = future[0] if future else item['times'][0]
            item['next_at'] = day.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0).isoformat()
        result.append(item)
    return result


def job_id(profile, row):
    return digest([profile, row['item_id'], row['destination']])[:40]


def public_job(job):
    allowed = ('job_id', 'profile_id', 'item_id', 'destination', 'status', 'stage', 'error', 'reference',
               'source_locator', 'created_at', 'updated_at', 'attempts', 'provider_id', 'provider_status',
               'published_url', 'link_verified', 'verified_at', 'verification_error', 'media', 'occurrence', 'retry_at')
    return {k: job[k] for k in allowed if k in job}
