"""Bounded Meta account discovery. Credentials stay in process memory only."""
import hashlib
import json
import os
import re
import threading
import time
from urllib.parse import urlencode

_lock = threading.Lock()
_cache = {}
TTL = 300

def credential():
    from social_inventory import SafeError
    value = os.getenv('META_SYSVAR', '').strip() or os.getenv('FB_PAGE_ACCESS_TOKEN', '').strip()
    if value.startswith('{'):
        try:
            obj = json.loads(value)
            value = next((obj.get(k) for k in ('access_token', 'token', 'page_access_token', 'user_access_token') if obj.get(k)), '')
        except Exception:
            raise SafeError('meta_credential_invalid') from None
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9._-]{20,}', value):
        raise SafeError('meta_credential_missing_or_invalid')
    return value

def cache_key():
    from social_inventory import SafeError
    version = os.getenv('META_GRAPH_VERSION', 'v24.0')
    if not re.fullmatch(r'v\d+\.\d+', version):
        raise SafeError('invalid_graph_version')
    token = credential()
    return hashlib.sha256((version + ':' + token).encode()).hexdigest(), token, version

def snapshot():
    """Status never makes provider requests or returns credentials."""
    try:
        key, _, _ = cache_key()
    except Exception:
        return {'meta_credential_configured': False, 'discovered_accounts': 0, 'discovery_checked_at': None, 'discovery_stale': True}
    value = _cache if _cache.get('key') == key else {}
    return {'meta_credential_configured': True, 'meta_credential_source': 'META_SYSVAR' if os.getenv('META_SYSVAR', '').strip() else 'FB_PAGE_ACCESS_TOKEN', 'discovered_accounts': len(value.get('accounts', {})),
            'discovery_checked_at': value.get('checked_at'), 'discovery_stale': time.monotonic() >= value.get('expires', 0)}

def discover(http=None):
    from social_inventory import SafeError, SafeHTTP, Budget, safe_text
    key, token, version = cache_key()
    with _lock:
        if _cache.get('key') == key and time.monotonic() < _cache.get('expires', 0):
            return _cache['accounts'], _cache['coverage']
        http = http or SafeHTTP(Budget(seconds=45, max_bytes=4 * 1024 * 1024))
        accounts, errors, completed = {}, [], []
        for edge in ('me/accounts', 'me/assigned_pages'):
            after, seen = None, set()
            try:
                for _ in range(20):
                    params = {'fields': 'id,name,access_token,instagram_business_account', 'limit': 100}
                    if after:
                        params['after'] = after
                    data = http.json('https://graph.facebook.com/' + version + '/' + edge + '?' + urlencode(params),
                                     ('graph.facebook.com',), headers={'Authorization': 'Bearer ' + token})
                    if data.get('error') or not isinstance(data.get('data'), list):
                        raise SafeError('meta_discovery_response_invalid')
                    for page in data['data']:
                        pid = str(page.get('id', ''))
                        if not re.fullmatch(r'\d+', pid):
                            raise SafeError('meta_discovery_account_invalid')
                        page_token = page.get('access_token') or token
                        if not isinstance(page_token, str) or not re.fullmatch(r'[A-Za-z0-9._-]{20,}', page_token):
                            raise SafeError('meta_discovery_credential_invalid')
                        common = {'handle': safe_text(page.get('name') or pid, 200), '_meta_token': page_token}
                        accounts[('facebook', pid)] = dict(common, account_id=pid, provider_id=pid, platform='facebook')
                        ig = page.get('instagram_business_account') or {}
                        if ig.get('id'):
                            iid = str(ig['id'])
                            if not re.fullmatch(r'\d+', iid):
                                raise SafeError('meta_discovery_account_invalid')
                            accounts[('instagram', iid)] = dict(common, account_id=iid, provider_id=iid, platform='instagram')
                    paging = data.get('paging') or {}
                    if not paging.get('next'):
                        completed.append(edge)
                        break
                    after = paging.get('cursors', {}).get('after')
                    if not isinstance(after, str) or not after or len(after) > 4096 or after in seen:
                        raise SafeError('meta_discovery_cursor_invalid')
                    seen.add(after)
                else:
                    raise SafeError('meta_discovery_page_limit')
            except SafeError as exc:
                errors.append({'edge': edge, 'code': str(exc)})
        coverage = {'completed_edges': completed, 'errors': errors, 'all_accessible_accounts_verified': False,
                    'cache_seconds': TTL, 'scope': 'Facebook Pages and their linked Instagram business accounts'}
        if not accounts and not completed:
            raise SafeError('meta_discovery_failed')
        # Cache partial results briefly to avoid repeated failed-edge calls on every post.
        expires = time.monotonic() + (60 if errors else TTL)
        from datetime import datetime, timezone
        _cache.clear()
        _cache.update(key=key, accounts=accounts, coverage=coverage, expires=expires,
                      checked_at=datetime.now(timezone.utc).isoformat())
        return accounts, coverage

def public_accounts():
    accounts, coverage = discover()
    return {'rows': [{k: cfg[k] for k in ('platform', 'account_id', 'handle')} for cfg in accounts.values()],
            'coverage': coverage}
