"""Paced, bounded, redacted dashboard GET client; never follows redirects."""
import json, time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.error import HTTPError, URLError

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs): return None

class VerificationError(RuntimeError):
    def __init__(self, status, category):
        self.status, self.category = status, category
        super().__init__(f'Read-only verification HTTP {status}; category={category}')

# Only known codes can reach output. Never echo arbitrary provider error bodies.
CODES = {'store_rate_limited','store_network_unavailable','store_google_unavailable',
 'store_google_oauth_rejected','store_google_permission_denied','store_google_scope_missing',
 'store_revision_conflict','store_read_conflict_retry','snapshot_changed_retry','snapshot_expired'}
TRANSIENT = {'store_rate_limited','store_network_unavailable','store_google_unavailable','store_read_conflict_retry'}
class ReadOnlyClient:
    def __init__(self, host, token, pace=5, *, opener=None, sleep=time.sleep, clock=time.monotonic):
        u=urlsplit(host)
        if u.scheme!='https' or not u.hostname or u.username or u.password or u.port not in (None,443) or u.query or u.fragment or u.path not in ('','/') or not token:
            raise ValueError('Set an HTTPS dashboard origin and token locally.')
        self.host,self.token,self.pace=host.rstrip('/'),token,max(1,pace)
        self.opener=opener or build_opener(NoRedirect());self.sleep,self.clock=sleep,clock;self.last=None
    def get(self,path):
        if not path.startswith('/api/') or path.startswith('//'):raise ValueError('Dashboard API path required')
        for attempt in range(4):
            if self.last is not None:self.sleep(max(0,self.pace-(self.clock()-self.last)))
            self.last=self.clock()
            try:
                req=Request(self.host+path,headers={'Authorization':'Bearer '+self.token,'User-Agent':'SH01-Operations-ReadOnly-QC/3.0','Cache-Control':'no-cache'})
                with self.opener.open(req,timeout=45) as r:return json.load(r)
            except HTTPError as e:
                code='unclassified_error'
                try:
                    raw=json.loads(e.read(4096));value=raw.get('error')
                    if value in CODES:code=value
                except Exception:pass
                if e.code==409:raise VerificationError(409,code) from None
                retry=e.code in (429,502,504) or e.code==503 and code in TRANSIENT
                if not retry or attempt==3:raise VerificationError(e.code,code) from None
            except (URLError,TimeoutError):
                if attempt==3:raise VerificationError(0,'network_unavailable') from None
            except (ValueError,UnicodeError):raise VerificationError(0,'invalid_json') from None
            self.sleep((20,40,60)[attempt])
