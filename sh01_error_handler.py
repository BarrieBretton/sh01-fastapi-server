"""Opt-in, bounded ERROR telemetry. Sends exception type/module, never log payloads."""
import hashlib,json,logging,os,queue,threading
from datetime import datetime,timezone
from urllib.parse import urlsplit
from urllib.request import Request,build_opener,HTTPRedirectHandler

class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self,*args,**kwargs):return None

class OpsHandler(logging.Handler):
    def __init__(self):
        super().__init__(logging.ERROR);self.events=queue.Queue(maxsize=100)
        threading.Thread(target=self._send,daemon=True,name='ops-error-telemetry').start()
    def emit(self,record):
        if record.levelno<logging.ERROR or not os.getenv('OPS_DASHBOARD_URL') or not os.getenv('OPS_INGEST_TOKEN'):return
        at=datetime.fromtimestamp(record.created,timezone.utc).isoformat()
        # Formatted messages and even templates can contain interpolated secrets. Only classes are sent.
        kind=record.exc_info[0].__name__ if record.exc_info and record.exc_info[0] else record.levelname
        message=f'{record.module}: {kind}'[:200]
        event={'source':'sh01','event_id':hashlib.sha256((at+record.name+message).encode()).hexdigest(),
               'happened_at':at,'service':os.getenv('RENDER_SERVICE_ID','sh01'),'message':message}
        try:self.events.put_nowait(event)
        except queue.Full:pass
    def _send(self):
        opener=build_opener(NoRedirect())
        while True:
            event=self.events.get()
            try:
                host=os.environ['OPS_DASHBOARD_URL'].rstrip('/');u=urlsplit(host)
                if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment or u.path:continue
                req=Request(host+'/api/ingest/event',json.dumps(event).encode(),method='POST',headers={'X-INGEST-KEY':os.environ['OPS_INGEST_TOKEN'],'Content-Type':'application/json','User-Agent':'SH01-Error-Telemetry/2'})
                with opener.open(req,timeout=5):pass
            except Exception:pass
            finally:self.events.task_done()

def install_ops_handler(logger):
    if not any(isinstance(h,OpsHandler) for h in logger.handlers):logger.addHandler(OpsHandler())
