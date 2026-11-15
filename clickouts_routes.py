"""Authenticated Clickouts endpoints and a sheet-driven daily scheduler."""
from __future__ import annotations

import asyncio
import hmac
import logging
import os

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from clickouts_state import ClickoutError, RedisState, google_store
from clickouts_worker import ClickoutsWorker

log = logging.getLogger(__name__)
router = APIRouter(prefix='/social/clickouts', tags=['Clickouts deletion'])
_scheduler = None


def authorize(x_api_key: str | None = Header(default=None, alias='X-API-KEY')):
    # Reuse the already deployed media credential on every instance. Retain an
    # existing social key as a compatible alternative, never a required setup step.
    expected = [value for name in ('MEDIA_CLIP_API_KEY', 'SOCIAL_INVENTORY_API_KEY')
                if (value := os.getenv(name, '').strip())]
    if not expected:
        raise HTTPException(status_code=503, detail='social_api_not_configured')
    if not x_api_key or not any(hmac.compare_digest(value.encode(), x_api_key.encode()) for value in expected):
        raise HTTPException(status_code=401, detail='invalid_api_key')


def worker():
    import redis
    workbook = os.getenv('SPREADSHEET_ID', '').strip()
    if not workbook:
        raise ClickoutError('spreadsheet_not_configured')
    redis_url = os.getenv('REDIS_URL', '').strip()
    if not redis_url:
        raise ClickoutError('shared_redis_not_configured')
    client = redis.from_url(redis_url, decode_responses=True, socket_connect_timeout=2,
                            socket_timeout=3, retry_on_timeout=False)
    return ClickoutsWorker(google_store(workbook), RedisState(client, workbook))


class RunRequest(BaseModel):
    dry_run: bool = True
    max_items: int = Field(default=3, ge=1, le=5)


async def invoke(callback):
    try:
        return await asyncio.to_thread(callback)
    except ClickoutError as exc:
        raise HTTPException(status_code=503 if exc.retry_after else 409, detail=str(exc),
            headers={'Retry-After': str(int(exc.retry_after) + 1)} if exc.retry_after else None) from None
    except Exception:
        raise HTTPException(status_code=503, detail='clickouts_unavailable_no_automatic_delete_retry') from None


@router.get('/status', dependencies=[Depends(authorize)])
async def get_status():
    def inspect():
        service = worker()
        return {**service.preview(), 'last_run': service.state.get('last_result'),
                'scheduler_enabled': os.getenv('SOCIAL_DELETE_ENABLED') == '1',
                'dashboard_receipts_configured': bool(os.getenv('OPS_INGEST_TOKEN') or os.getenv('INGEST_TOKEN'))}
    return await invoke(inspect)


@router.post('/preview', dependencies=[Depends(authorize)])
async def preview():
    return await invoke(lambda: worker().preview())


@router.post('/run', dependencies=[Depends(authorize)])
async def run(request: RunRequest):
    """dry_run defaults true. false executes one batch irrespective of daily timing."""
    return await invoke(lambda: worker().preview() if request.dry_run else worker().run(request.max_items))


@router.post('/tick', dependencies=[Depends(authorize)])
async def tick():
    """External scheduler entry point. Honors the sheet schedule and server opt-in."""
    return await invoke(lambda: worker().run(scheduled=True))


def scheduled_tick():
    if os.getenv('SOCIAL_DELETE_ENABLED') != '1':
        return
    try:
        result = worker().run(scheduled=True)
        log.info('clickouts_tick status=%s done=%s', result['status'], result.get('done', 0))
    except ClickoutError as exc:
        log.warning('clickouts_tick code=%s', str(exc))
    except Exception:
        # Never log Redis connection URLs, Google payloads, or token-bearing errors.
        log.warning('clickouts_tick code=service_unavailable')


@router.on_event('startup')
def start_clickouts_scheduler():
    global _scheduler
    if _scheduler is not None or os.getenv('SOCIAL_DELETE_ENABLED') != '1':
        return
    from apscheduler.schedulers.background import BackgroundScheduler
    _scheduler = BackgroundScheduler(timezone='Asia/Kolkata')
    _scheduler.add_job(scheduled_tick, 'interval', seconds=60, id='sheet_clickouts',
                       max_instances=1, coalesce=True, misfire_grace_time=60)
    _scheduler.start()


@router.on_event('shutdown')
def stop_clickouts_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
