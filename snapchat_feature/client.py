"""Bounded transfers to fixed official Snap hosts. No redirects or video re-encoding."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from social_inventory import validate_url, PinnedHTTPS
from .core import SnapError, source_locator

API = 'https://businessapi.snapchat.com'
TOKEN = 'https://accounts.snapchat.com/login/oauth2/access_token'
CHUNK = 16 * 1024 * 1024


class MultipartFields(io.BytesIO):
    """Small control payload matching Snap's documented multipart FINALIZE call."""
    def __init__(self, fields):
        boundary = 'sh01' + uuid.uuid4().hex
        self.content_type = 'multipart/form-data; boundary=' + boundary
        payload = ''.join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n' for k, v in fields.items())
        payload = (payload + f'--{boundary}--\r\n').encode()
        self.length = len(payload)
        super().__init__(payload)


class MultipartChunk:
    """http.client reads this in 8 KiB pieces; never buffers the whole upload."""
    def __init__(self, file, size, part, guard=lambda: None):
        boundary = 'sh01' + uuid.uuid4().hex
        self.content_type = 'multipart/form-data; boundary=' + boundary
        self.head = io.BytesIO((f'--{boundary}\r\nContent-Disposition: form-data; name="action"\r\n\r\nADD\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="part_number"\r\n\r\n{part}\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="video.enc.mp4"\r\nContent-Type: application/octet-stream\r\n\r\n').encode())
        self.tail = io.BytesIO(f'\r\n--{boundary}--\r\n'.encode())
        self.length = len(self.head.getvalue()) + size + len(self.tail.getvalue())
        self.file, self.remaining, self.guard = file, size, guard

    def read(self, size=8192):
        self.guard()
        size = min(65536, size if size and size > 0 else 8192)
        data = self.head.read(size)
        if data: return data
        if self.remaining:
            data = self.file.read(min(size, self.remaining))
            if not data: raise SnapError('encrypted_upload_truncated')
            self.remaining -= len(data)
            return data
        return self.tail.read(size)


def request(url, *, method='GET', token=None, data=None, form=None, multipart=None):
    allowed = ('businessapi.snapchat.com', 'accounts.snapchat.com')
    try: host, address, path = validate_url(url, allowed, all_addresses=True)
    except Exception: raise SnapError('snapchat_endpoint_rejected') from None
    headers = {'User-Agent': 'SH01-Snapchat/1', 'Accept': 'application/json'}
    if token: headers['Authorization'] = 'Bearer ' + token
    body = None
    if data is not None: body = json.dumps(data).encode(); headers['Content-Type'] = 'application/json'
    if form is not None: body = urlencode(form).encode(); headers['Content-Type'] = 'application/x-www-form-urlencoded'
    if multipart is not None:
        body = multipart; headers['Content-Type'] = multipart.content_type; headers['Content-Length'] = str(multipart.length)
    conn = PinnedHTTPS(host, address, 30)
    try:
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        if r.status != 200:
            retry = 0
            if r.status in (429, 503):
                try: retry = max(60, min(3600, int(r.getheader('Retry-After') or 60)))
                except ValueError: retry = 60
            raise SnapError('snapchat_http_' + str(r.status), status=502, retry_after=retry)
        raw = r.read(512 * 1024 + 1)
        if len(raw) > 512 * 1024: raise SnapError('snapchat_response_too_large')
        try: result = json.loads(raw)
        except (ValueError, UnicodeError): raise SnapError('snapchat_response_not_json') from None
        if not isinstance(result, dict): raise SnapError('snapchat_response_invalid')
        if host == 'businessapi.snapchat.com' and result.get('request_status') != 'SUCCESS':
            raise SnapError('snapchat_request_not_successful')
        return result
    except SnapError: raise
    except Exception: raise SnapError('snapchat_transport_outcome_unknown') from None
    finally: conn.close()


def download(url, file, config, guard=lambda: None):
    source_locator(url, config.source_hosts)
    try: host, address, path = validate_url(url, config.source_hosts, all_addresses=True)
    except Exception: raise SnapError('source_network_address_rejected', status=400) from None
    conn = PinnedHTTPS(host, address, 20)
    total, sha = 0, hashlib.sha256()
    try:
        conn.request('GET', path, headers={'User-Agent': 'SH01-Snapchat/1', 'Accept-Encoding': 'identity'})
        response = conn.getresponse()
        if response.status != 200: raise SnapError('source_http_' + str(response.status), status=422)
        if response.getheader('Content-Encoding', 'identity') != 'identity': raise SnapError('source_encoding_rejected')
        length = response.getheader('Content-Length')
        if length is not None and (not length.isdigit() or int(length) > config.max_bytes): raise SnapError('source_size_limit')
        with open(file, 'wb') as target:
            while True:
                guard(); chunk = response.read(65536)
                if not chunk: break
                total += len(chunk)
                if total > config.max_bytes: raise SnapError('source_size_limit')
                sha.update(chunk); target.write(chunk)
        if not total or (length is not None and total != int(length)): raise SnapError('source_download_incomplete')
        return {'bytes': total, 'sha256': sha.hexdigest()}
    except SnapError: raise
    except Exception: raise SnapError('source_download_failed') from None
    finally: conn.close()


def probe(path):
    with open(path, 'rb') as f:
        if f.read(12)[4:8] != b'ftyp': raise SnapError('prepared_mp4_required', status=422)
    try:
        run = subprocess.run(['ffprobe', '-v', 'error', '-max_alloc', '16777216', '-threads', '1', '-protocol_whitelist', 'file,pipe',
            '-show_entries', 'format=duration:stream=codec_type,codec_name,width,height', '-of', 'json', str(path)],
            capture_output=True, timeout=20, check=True)
        data = json.loads(run.stdout)
        duration = float(data['format']['duration'])
        streams = [s for s in data['streams'] if s.get('codec_type') == 'video']
        if len(streams) != 1 or not math.isfinite(duration) or not 5 <= duration <= 300:
            raise ValueError()
        v = streams[0]
        if v.get('codec_name') != 'h264' or not 1 <= v.get('width', 0) <= 3840 or not 1 <= v.get('height', 0) <= 3840:
            raise ValueError()
        return {'duration_seconds': duration, 'width': v['width'], 'height': v['height'], 'codec': v['codec_name'],
                'spotlight_duration_at_least_30s': duration >= 30, 'monetization_eligibility': 'not_determined'}
    except Exception: raise SnapError('prepared_h264_mp4_5_to_300_seconds_required', status=422) from None


def encrypt_file(source, destination, guard=lambda: None):
    key, iv = os.urandom(32), os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    pad = padding.PKCS7(128).padder()
    with open(source, 'rb') as src, open(destination, 'wb') as dst:
        while True:
            guard(); chunk = src.read(65536)
            if not chunk: break
            dst.write(cipher.update(pad.update(chunk)))
        dst.write(cipher.update(pad.finalize()) + cipher.finalize())
    return base64.b64encode(key).decode(), base64.b64encode(iv).decode()


class SnapClient:
    def __init__(self, config, state, transport=request, clock=time.time):
        self.config, self.state, self.transport, self.clock = config, state, transport, clock

    def save_tokens(self, result, previous=None):
        refresh = result.get('refresh_token') or (previous or {}).get('refresh_token')
        access, expiry = result.get('access_token'), result.get('expires_in')
        if not isinstance(access, str) or not access or not isinstance(refresh, str) or not refresh or not isinstance(expiry, (int, float)) or not 0 < expiry <= 86400:
            raise SnapError('snapchat_token_response_invalid')
        scopes = result.get('scope', (previous or {}).get('scope', ''))
        if 'snapchat-profile-api' not in str(scopes).split(): raise SnapError('snapchat_profile_scope_missing')
        payload = {'access_token': access, 'refresh_token': refresh, 'scope': scopes, 'expires_at': self.clock() + expiry,
                   'client_id': self.config.client_id, 'profile_id': self.config.profile_id}
        self.state.put('oauth_tokens', {'encrypted': self.config.seal(payload), 'refresh_pending': False})
        return access

    def token(self):
        record = self.state.get('oauth_tokens')
        if not record: raise SnapError('snapchat_authorization_required', status=409)
        if record.get('refresh_pending'): raise SnapError('snapchat_refresh_unconfirmed_reconnect', status=409)
        value = self.config.unseal(record['encrypted'])
        if value.get('client_id') != self.config.client_id or value.get('profile_id') != self.config.profile_id:
            raise SnapError('snapchat_account_changed_reconnect', status=409)
        if value['expires_at'] > self.clock() + 120: return value['access_token']
        self.state.put('oauth_tokens', {**record, 'refresh_pending': True})
        result = self.transport(TOKEN, method='POST', form={'grant_type': 'refresh_token', 'refresh_token': value['refresh_token'],
            'client_id': self.config.client_id, 'client_secret': self.config.client_secret})
        return self.save_tokens(result, value)

    def api(self, path, **kwargs):
        self.state.guard()
        return self.transport(API + path, token=self.token(), **kwargs)

    def profile(self):
        result = self.api('/v1/public_profiles/' + self.config.profile_id)
        found = [r.get('public_profile', {}) for r in result.get('public_profiles', []) if r.get('sub_request_status') == 'SUCCESS']
        found = [r for r in found if r.get('id') == self.config.profile_id]
        if len(found) != 1: raise SnapError('snapchat_authorized_profile_not_confirmed', status=409)
        return {'id': self.config.profile_id, 'display_name': str(found[0].get('display_name', ''))[:180], 'checked_at': self.clock()}

    def upload(self, encrypted, key, iv, guard=lambda: None, progress=lambda stage, part=0: None):
        base = '/v1/public_profiles/' + self.config.profile_id
        created = self.api(base + '/media', method='POST', data={'type': 'VIDEO', 'name': 'SH01 prepared video', 'key': key, 'iv': iv})
        media = created.get('media_id', '')
        if not re.fullmatch(r'[a-zA-Z0-9-]{1,100}', media): raise SnapError('snapchat_media_id_invalid')
        paths = []
        for field in ('add_path', 'finalize_path'):
            path = created.get(field, '')
            if not isinstance(path, str) or not re.fullmatch(r'/(?:[a-z]{2}/)?v1/public_profiles/' + re.escape(self.config.profile_id) + '/media/' + re.escape(media) + '/multipart-upload', path):
                raise SnapError('snapchat_upload_path_rejected')
            paths.append(path)
        size = Path(encrypted).stat().st_size
        with open(encrypted, 'rb') as file:
            for number, offset in enumerate(range(0, size, CHUNK), 1):
                guard(); progress('uploading', number)
                payload = MultipartChunk(file, min(CHUNK, size - offset), number, guard)
                self.api(paths[0], method='POST', multipart=payload)
        guard(); progress('finalizing')
        self.api(paths[1], method='POST', multipart=MultipartFields({'action': 'FINALIZE'}))
        return media

    def publish(self, media, row):
        payload = {'media_id': media}
        suffix = 'stories'
        if row['destination'] == 'spotlight':
            suffix = 'spotlights'; payload.update(description=row['description'], locale=row['locale'])
        return self.api('/v1/public_profiles/' + self.config.profile_id + '/' + suffix, method='POST', data=payload)

    def verify(self, provider_id):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,200}', provider_id): raise SnapError('snapchat_post_id_invalid')
        result = self.api('/v1/public_profiles/' + self.config.profile_id + '/spotlights/' + provider_id)
        rows = [r.get('spotlight', {}) for r in result.get('spotlights', []) if r.get('sub_request_status') == 'SUCCESS']
        matches = [r for r in rows if r.get('id') == provider_id and r.get('profile_id') == self.config.profile_id]
        if len(matches) != 1: raise SnapError('snapchat_post_identity_unconfirmed')
        status = matches[0].get('status')
        return status if status in {'LIVE', 'PENDING', 'REJECTED', 'IN_REVIEW'} else 'UNKNOWN'

    def metrics(self):
        end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=28)
        fields = 'STORY_VIEW_TIME_MILLIS,SPOTLIGHT_VIEW_TIME_MILLIS,STORY_VIEWS,SPOTLIGHT_VIEWS'
        params = {'assetType': 'PROFILE', 'granularity': 'TOTAL', 'fields': fields,
                  'startTime': start.strftime('%Y-%m-%dT00:00:00'), 'endTime': end.strftime('%Y-%m-%dT00:00:00')}
        result = self.api('/v1/public_profiles/' + self.config.profile_id + '/stats?' + urlencode(params))
        subscribers = self.api('/v1/public_profiles/' + self.config.profile_id + '/stats?' + urlencode({'assetType': 'PROFILE', 'granularity': 'LIFETIME', 'fields': 'SUBSCRIBERS'}))
        values = {}
        for data in (result, subscribers):
            assets = data.get('assets', [])
            if len(assets) != 1 or assets[0].get('asset_id') != self.config.profile_id or assets[0].get('sub_request_status') != 'SUCCESS':
                raise SnapError('snapchat_metrics_incomplete')
            for series in assets[0].get('timeseries', []):
                for cell in series.get('fields', []):
                    field = cell.get('field', {}).get('field_name')
                    stats = cell.get('stats', [])
                    if field in (fields + ',SUBSCRIBERS').split(',') and len(stats) == 1:
                        value = stats[0].get('value')
                        try: value = float(value)
                        except (ValueError, TypeError): continue
                        if math.isfinite(value) and value >= 0: values[field] = value
        if any(name not in values for name in (fields + ',SUBSCRIBERS').split(',')):
            raise SnapError('snapchat_metrics_incomplete')
        return {'values': values, 'period_start': start.isoformat(), 'period_end': end.isoformat(), 'checked_at': self.clock(),
                'period_note': 'Previous 28 complete UTC days. Snapchat in-app eligibility may use a different window.',
                'monetization_eligibility': 'not_determined', 'revenue': None}
