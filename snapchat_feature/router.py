from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from clickouts_routes import authorize
from clickouts_state import ClickoutError
from .core import Config, SnapError
from .service import Service, safe_code

router = APIRouter(prefix='/snapchat', tags=['Snapchat publishing'])
log = logging.getLogger(__name__)
_scheduler = None
_process_lock = threading.Lock()


async def invoke(operation):
    try:
        value = await asyncio.to_thread(operation)
        return JSONResponse(value, headers={'Cache-Control': 'no-store'})
    except Exception as exc:
        status = exc.status if isinstance(exc, SnapError) else 429 if isinstance(exc, ClickoutError) and exc.retry_after else 503
        retry = getattr(exc, 'retry_after', 0)
        raise HTTPException(status, safe_code(exc), headers={'Retry-After': str(retry)} if retry else None) from None


async def process():
    if not Config.env().enabled or not _process_lock.acquire(blocking=False): return
    try:
        if not await asyncio.to_thread(lambda: Service().has_work()): return
        # Reuse global media admission: don't compete with editing jobs for the
        # limited Render memory. A busy admission leaves the durable job queued.
        from media_admission import heavy_request
        async with heavy_request('snapchat_publish', minimum_mb=96):
            await asyncio.to_thread(lambda: Service().process_one())
    except Exception as exc:
        log.warning('snapchat_worker code=%s', safe_code(exc))
    finally: _process_lock.release()


@router.get('/health', dependencies=[Depends(authorize)])
async def health():
    return await invoke(lambda: Service().health())


@router.post('/setup', dependencies=[Depends(authorize)])
async def setup():
    return await invoke(lambda: Service().book.setup())


@router.post('/connect', dependencies=[Depends(authorize)])
async def connect():
    return await invoke(lambda: Service().connect())


@router.post('/connection/check', dependencies=[Depends(authorize)])
async def connection_check():
    return await invoke(lambda: Service().check_connection())


@router.get('/oauth/callback')
async def callback(request: Request):
    # Do not reflect codes/provider error text into the response. Scrub ASGI
    # query_string before Uvicorn's response access log is emitted.
    query = request.query_params
    code, nonce = query.get('code', ''), query.get('state', '')
    request.scope['query_string'] = b''
    try:
        await asyncio.to_thread(lambda: Service().callback(code, nonce))
        text, status = 'Snapchat connected. Return to Operations desk and refresh the Snapchat panel.', 200
    except Exception as exc:
        text, status = 'Connection was not completed: ' + safe_code(exc) + '. Return to Operations desk to reconnect.', 400
    return HTMLResponse('<!doctype html><html lang="en"><meta charset="utf-8"><title>Snapchat connection</title><p>' + text + '</p></html>', status_code=status,
        headers={'Cache-Control': 'no-store', 'Referrer-Policy': 'no-referrer', 'Content-Security-Policy': "default-src 'none'; frame-ancestors 'none'"})


@router.post('/preview', dependencies=[Depends(authorize)])
async def preview():
    return await invoke(lambda: Service().preview())


class PublishRequest(BaseModel):
    item_id: str = Field(min_length=1, max_length=100, pattern=r'^[A-Za-z0-9_-]+$')
    fingerprint: str = Field(pattern=r'^[a-f0-9]{64}$')


@router.post('/run', dependencies=[Depends(authorize)], status_code=202)
async def run(body: PublishRequest, background: BackgroundTasks):
    response = await invoke(lambda: Service().run(body.item_id, body.fingerprint))
    response.status_code = 202
    background.add_task(process)
    return response


@router.post('/tick', dependencies=[Depends(authorize)])
async def tick(background: BackgroundTasks):
    response = await invoke(lambda: Service().tick())
    background.add_task(process)
    return response


@router.get('/jobs/{job_id}', dependencies=[Depends(authorize)])
async def job(job_id: str):
    if not re.fullmatch(r'[a-f0-9]{40}', job_id): raise HTTPException(404, 'snapchat_job_not_found')
    return await invoke(lambda: Service().job(job_id))


@router.get('/metrics', dependencies=[Depends(authorize)])
async def metrics():
    def read():
        service = Service(); service.state.load()
        return {'metrics': service.state.get('metrics:' + service.config.profile_id), 'monetization_eligibility': 'not_determined'}
    return await invoke(read)


@router.post('/metrics/refresh', dependencies=[Depends(authorize)])
async def refresh_metrics():
    return await invoke(lambda: Service().refresh_metrics())


async def scheduled_tick():
    if not Config.env().enabled: return
    try:
        await asyncio.to_thread(lambda: Service().tick())
        await process()
    except Exception as exc: log.warning('snapchat_tick code=%s', safe_code(exc))


@router.on_event('startup')
async def start_scheduler():
    global _scheduler
    if _scheduler is not None or not Config.env().enabled: return
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    _scheduler = AsyncIOScheduler(timezone='Asia/Kolkata')
    _scheduler.add_job(scheduled_tick, 'interval', seconds=60, id='snapchat_sheet_schedule', max_instances=1, coalesce=True, misfire_grace_time=60)
    _scheduler.start()


@router.on_event('shutdown')
async def stop_scheduler():
    global _scheduler
    if _scheduler is not None: _scheduler.shutdown(wait=False); _scheduler = None
