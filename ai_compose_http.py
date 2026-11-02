"""Stable project identity and process-wide, per-host HTTP cooldowns.

Never rotate identities to retry a throttled request. All callers share a
monotonic deadline; workers return a useful error instead of sleeping for minutes.
"""
from datetime import timezone
from email.utils import parsedate_to_datetime
import math
import threading
import time
from urllib.parse import urlsplit


USER_AGENT = 'SH01LocalEditor/2.0 (https://github.com/BarrieBretton/sh01-fastapi-server)'
_cooldowns = {}
_lock = threading.Lock()


class MediaHTTPBackoff(RuntimeError):
    def __init__(self, host, status, seconds):
        self.host = host
        self.http_status = status
        self.retry_after_seconds = seconds
        super().__init__(f'Media host {host} returned HTTP {status}; '
                         f'retry after {seconds}s (host cooldown active)')


def _host(url):
    return (urlsplit(str(url)).hostname or '').lower().rstrip('.')


def retry_after_seconds(value, *, default=600):
    try:
        text = str(value).strip()
        if text.isascii() and text.isdigit():
            return int(text)
        date = parsedate_to_datetime(text)
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        return max(0, math.ceil(date.timestamp() - time.time()))
    except (TypeError, ValueError, OverflowError):
        return default


def check_cooldown(url):
    host = _host(url)
    with _lock:
        entry = _cooldowns.get(host)
        if entry is None:
            return
        deadline, status = entry
        remaining = math.ceil(deadline - time.monotonic())
        if remaining <= 0:
            del _cooldowns[host]
            return
    raise MediaHTTPBackoff(host, status, remaining)


def observe_response(url, response):
    status = response.status_code
    value = response.headers.get('Retry-After')
    if status != 429 and not (status == 503 and value is not None):
        return
    host = _host(url)
    seconds = retry_after_seconds(value)
    deadline = time.monotonic() + seconds
    with _lock:
        previous = _cooldowns.get(host)
        if previous is None or previous[0] <= deadline:
            _cooldowns[host] = (deadline, status)
        else:
            seconds = max(0, math.ceil(previous[0] - time.monotonic()))
    raise MediaHTTPBackoff(host, status, seconds)


def failure_details(exc, url=None):
    """Do not expose URL queries, credentials, response bodies or headers."""
    response = getattr(exc, 'response', None)
    status = getattr(exc, 'http_status', None)
    if status is None and response is not None:
        status = getattr(response, 'status_code', None)
    details = {'error_type': type(exc).__name__, 'http_status': status}
    if url:
        details['host'] = _host(url)
    if isinstance(exc, MediaHTTPBackoff):
        details.update(host=exc.host, retry_after_seconds=exc.retry_after_seconds)
    elif response is not None and response.headers.get('Retry-After') is not None:
        details['retry_after_seconds'] = retry_after_seconds(response.headers['Retry-After'])
    return details
