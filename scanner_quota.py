"""Process-local scanner request pacing; durable progress belongs to the caller."""
import hashlib
import json
import math
import random
import threading
import time
from collections import OrderedDict
import social_inventory as core

_LOCK = threading.Lock()
_STATES = OrderedDict()

def scope(base, headers):
    return hashlib.sha256(json.dumps([base, sorted(headers.items())]).encode()).hexdigest()

def state_for(key):
    with _LOCK:
        if key not in _STATES:
            _STATES[key] = {'lock': threading.Lock(), 'next': 0., 'cooldown': 0.}
        _STATES.move_to_end(key)
        while len(_STATES) > 32:
            _STATES.popitem(last=False)
        return _STATES[key]

def deferred(seconds):
    exc = core.SafeError('scanner_storage_cooldown')
    exc.diagnostic = {'http_status': 503, 'dashboard_code': 'store_rate_limited',
                      'retry_after_seconds': max(1, math.ceil(seconds))}
    return exc

class QuotaHTTP:
    def __init__(self, http, key):
        self.inner, self.state = http, state_for(key)
        self.budget = http.budget

    def json(self, *args, **kwargs):
        # Serialize requests using this credential within one serving process.
        with self.state['lock']:
            for attempt in range(3):
                now = time.monotonic()
                if self.state['cooldown'] > now:
                    raise deferred(self.state['cooldown'] - now)
                delay = max(0., self.state['next'] - now)
                if self.budget.remaining() <= delay + 1:
                    raise core.SafeError('scan_time_budget')
                if delay: time.sleep(delay)
                try:
                    result = self.inner.json(*args, **kwargs)
                    self.state['next'] = time.monotonic() + 2
                    return result
                except core.SafeError as exc:
                    self.state['next'] = time.monotonic() + 2
                    diag = dict(getattr(exc, 'diagnostic', {}))
                    status, category = diag.get('http_status'), diag.get('dashboard_code')
                    diag['request_attempts'] = attempt + 1
                    exc.diagnostic = diag
                    if status == 429 or category == 'store_rate_limited':
                        delay = max(75., float(diag.get('retry_after_seconds', 0))) + random.uniform(0, 5)
                        self.state['cooldown'] = time.monotonic() + delay
                        diag['retry_after_seconds'] = math.ceil(delay)
                        # Do not occupy a Render request for an entire quota window.
                        raise
                    transient = category in ('store_network_unavailable', 'store_google_unavailable',
                                               'store_read_conflict_retry', 'store_response_not_json')
                    retry = transient or (not category and status in (500, 502, 503, 504)) or isinstance(exc, core.TransportError)
                    # Writes already have bounded replay in FingerprintSink.
                    if kwargs.get('method', 'GET') != 'GET' or not retry or attempt == 2:
                        raise
                    delay = max(float(diag.get('retry_after_seconds', 0)), 2 ** attempt) + random.uniform(0, 2 ** attempt)
                    if self.budget.remaining() <= delay + 1:
                        raise
                    time.sleep(delay)
