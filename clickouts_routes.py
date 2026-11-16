"""Authenticated Clickouts endpoints and a sheet-driven daily scheduler."""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
import re

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from clickouts_state import ClickoutError, google_store
from clickouts_sheet_state import SheetsState
from clickouts_worker import ClickoutsWorker, blocked_code, read_schedule

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


def configured_workbook():
    workbook = os.getenv('SPREADSHEET_ID', '').strip()
    if not workbook:
        raise ClickoutError('spreadsheet_not_configured')
    return workbook


def sheet_state(store):
    return SheetsState(store)


def worker(read_only=False):
    store = google_store(configured_workbook())
    return ClickoutsWorker(store, None if read_only else sheet_state(store))


def check_failure(error, fallback):
    """Only static codes and bounded exception categories, never exception text."""
    result = {'status': 'error', 'code': str(error) if isinstance(error, ClickoutError) else fallback}
    if isinstance(error, ClickoutError) and error.retry_after:
        result['retry_after'] = error.retry_after
    if not isinstance(error, ClickoutError):
        kind = type(error).__name__
        result['error_type'] = kind if kind in {'KeyError', 'TypeError', 'ValueError', 'RuntimeError',
            'ImportError', 'ModuleNotFoundError', 'TimeoutError', 'ConnectionError'} else 'Exception'
    return result


def last_run_summary(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ClickoutError('clickouts_state_invalid_shape')
    result = {}
    if value.get('status') in {'completed', 'more_pending', 'idle', 'disabled', 'busy', 'waiting', 'paused', 'cooldown'}:
        result['status'] = value['status']
    for field in ('done', 'remaining', 'blocked', 'invalid_rows', 'manual_required',
                  'unverified_receipts', 'acknowledgements', 'deferred', 'retry_after'):
        number = value.get(field)
        if type(number) is int and number >= 0:
            result[field] = number
    if isinstance(value.get('checked_at'), str):
        from datetime import datetime
        try:
            result['checked_at'] = datetime.fromisoformat(value['checked_at'].replace('Z', '+00:00')).isoformat()
        except ValueError:
            pass
    if isinstance(value.get('blocked_reasons'), dict):
        reasons = {}
        for code, count in value['blocked_reasons'].items():
            if type(count) is int and count >= 0:
                safe = blocked_code(code)
                reasons[safe] = reasons.get(safe, 0) + count
        result['blocked_reasons'] = reasons
    return result


def runtime_details():
    result = {}
    for field, name, pattern in (('commit', 'RENDER_GIT_COMMIT', r'[a-fA-F0-9]{40}'),
                                 ('service_id', 'RENDER_SERVICE_ID', r'srv-[a-zA-Z0-9_-]{1,100}'),
                                 ('instance_id', 'RENDER_INSTANCE_ID', r'[a-zA-Z0-9_-]{1,128}')):
        value = os.getenv(name, '')
        result[field] = value if re.fullmatch(pattern, value) else None
    return result


def failure_summary(value):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ClickoutError('clickouts_failure_state_invalid')
    code = value.get('code')
    result = {'code': code if isinstance(code, str) and re.fullmatch(r'[a-z][a-z0-9_]{0,95}', code)
              else 'execution_error'}
    if value.get('stage') in {'coordination', 'read_queue', 'prepare_columns', 'verify_row',
                            'verify_provider', 'mark_processing', 'delete_request', 'save_receipt', 'update_done'}:
        result['stage'] = value['stage']
    result.update(last_run_summary({'checked_at': value.get('checked_at')}))
    details = value.get('diagnostics')
    if isinstance(details, dict):
        result['diagnostics'] = {name: details[name] for name in ('matching_ranges', 'expected_columns',
            'returned_columns', 'first_mismatch_column') if type(details.get(name)) is int and 0 <= details[name] <= 10000}
    return result


def inspect_status():
    """Independent read-only checks. An unavailable dependency cannot erase other evidence."""
    result = {'diagnostics_version': 5, 'dry_run': True, 'enabled': None,
              'scheduler_enabled': None, 'state_backend': 'google_sheets',
              'scheduler_running': bool(_scheduler is not None and _scheduler.running),
              'dashboard_receipts_configured': bool(os.getenv('OPS_INGEST_TOKEN') or os.getenv('INGEST_TOKEN')),
              'provider_permissions': 'not_checked', 'rows': None, 'eligible': None,
              'status_counts': None, 'blocked_counts': None, 'schedule': None, 'last_run': None,
              'last_failure': None, 'checks': {}}
    # Render supplies these automatically; validate and expose only public identifiers.
    result['runtime'] = runtime_details()
    checks = result['checks']
    service = None
    try:
        service = worker(read_only=True)
        checks['google_client'] = {'status': 'ok'}
    except Exception as exc:
        checks['google_client'] = check_failure(exc, 'google_client_initialization_failed')
    if service is None:
        checks['queue'] = {'status': 'skipped', 'code': 'google_client_unavailable'}
        checks['schedule'] = {'status': 'skipped', 'code': 'google_client_unavailable'}
    else:
        try:
            preview = service.preview(include_schedule=False)
            for field in ('rows', 'eligible', 'status_counts', 'blocked_counts'):
                if field in preview:
                    result[field] = preview[field]
            checks['queue'] = {'status': 'ok'}
        except Exception as exc:
            checks['queue'] = check_failure(exc, 'queue_read_failed')
        if checks['queue'].get('retry_after'):
            checks['schedule'] = {'status': 'skipped', 'code': 'sheets_cooldown'}
        else:
            try:
                result['schedule'] = read_schedule(service.store.schedule_rows())
                checks['schedule'] = {'status': 'ok'}
                result['enabled'] = result['scheduler_enabled'] = result['schedule']['enabled']
            except Exception as exc:
                checks['schedule'] = check_failure(exc, 'schedule_read_failed')
    if service is None or checks['queue'].get('retry_after') or checks['schedule'].get('retry_after'):
        checks['state'] = {'status': 'skipped', 'code': 'google_client_unavailable' if service is None else 'sheets_cooldown'}
    else:
        try:
            state = sheet_state(service.store)
            initialized = state.load()
            result['last_run'] = last_run_summary(state.get('last_result')) if initialized else None
            result['last_failure'] = failure_summary(state.get('last_failure')) if initialized else None
            checks['state'] = {'status': 'ok', 'initialized': initialized}
        except Exception as exc:
            checks['state'] = check_failure(exc, 'clickouts_state_read_failed')
    dependencies_ok = all(check['status'] == 'ok' for check in checks.values())
    counts = result['status_counts'] or {}
    attention = bool(counts.get('blocked') or counts.get('manual_required'))
    result['ok'] = dependencies_ok and result['last_failure'] is None and not attention
    result['status'] = ('degraded' if not dependencies_ok else 'execution_failed' if result['last_failure']
                        else 'attention_required' if attention else 'ok')
    return result


class RunRequest(BaseModel):
    dry_run: bool = True
    max_items: int = Field(default=3, ge=1, le=5)


class PreviewRequest(BaseModel):
    check_provider: bool = False
    request_id: str | None = Field(default=None, min_length=1, max_length=160, pattern=r'^[A-Za-z0-9:_-]+$')
    expected_token_sha256: str | None = Field(default=None, pattern=r'^[a-f0-9]{64}$')


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
    return await invoke(inspect_status)


@router.post('/preview', dependencies=[Depends(authorize)])
async def preview(request: PreviewRequest | None = None):
    if request is not None and request.check_provider:
        if request.request_id is None:
            raise HTTPException(status_code=400, detail='request_id_required_for_access_check')
        result = await invoke(lambda: worker(read_only=True).check_access(
            request.request_id, request.expected_token_sha256))
        return {**result, 'diagnostics_version': 5, 'runtime': runtime_details()}
    if request is not None and (request.request_id is not None or request.expected_token_sha256 is not None):
        raise HTTPException(status_code=400, detail='check_provider_required_for_access_check')
    return await invoke(lambda: worker(read_only=True).preview())


@router.post('/run', dependencies=[Depends(authorize)])
async def run(request: RunRequest):
    """dry_run defaults true. false executes one batch irrespective of daily timing."""
    return await invoke(lambda: worker(read_only=True).preview() if request.dry_run else worker().run(request.max_items))


@router.post('/tick', dependencies=[Depends(authorize)])
async def tick():
    """External scheduler entry point. Honors the daily sheet schedule."""
    return await invoke(lambda: worker().run(scheduled=True))


def scheduled_tick():
    try:
        result = worker().run(scheduled=True)
        log.info('clickouts_tick status=%s done=%s blocked=%s manual_required=%s deferred=%s remaining=%s reason=%s',
                 result['status'], result.get('done', 0), result.get('blocked', 0),
                 result.get('manual_required', 0), result.get('deferred', 0),
                 result.get('remaining'), result.get('reason', 'none'))
    except ClickoutError as exc:
        log.warning('clickouts_tick code=%s', str(exc))
    except Exception:
        # Never log Google payloads or token-bearing errors.
        log.warning('clickouts_tick code=service_unavailable')


@router.on_event('startup')
def start_clickouts_scheduler():
    global _scheduler
    if _scheduler is not None or not os.getenv('SPREADSHEET_ID', '').strip() or not any(
            os.getenv(name, '').strip() for name in ('MEDIA_CLIP_API_KEY', 'SOCIAL_INVENTORY_API_KEY')):
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
