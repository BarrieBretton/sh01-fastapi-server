"""Kaggle-only application hosting the existing AI compose router.

Run with one worker, bound to 127.0.0.1. Production app.py and its queue,
caption routes, storage clients and schedulers retain their existing wiring.
"""
from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
import ai_compose
from ai_compose_local_store import NotebookJobStore


@asynccontextmanager
async def lifespan(app):
    if not os.getenv('MEDIA_CLIP_API_KEY', '').strip():
        raise RuntimeError('MEDIA_CLIP_API_KEY must be configured before startup')
    os.environ['AI_COMPOSE_LOCAL_FALLBACK'] = 'true'
    local = NotebookJobStore(Path(os.getenv('AI_COMPOSE_WORK_DIR', 'media_jobs')) / 'notebook-jobs.sqlite3')
    local.mark_interrupted()
    previous = ai_compose.store
    ai_compose.store = local
    try:
        yield
    finally:
        # Uvicorn shutdown must not leave tasks writing after store restoration.
        import asyncio
        tasks = list(ai_compose._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        ai_compose.store = previous


app = FastAPI(title='SH01 notebook editor', lifespan=lifespan)
app.include_router(ai_compose.router)
