"""Single-worker notebook persistence; not a replacement for production queues."""
from __future__ import annotations

import json
from contextlib import contextmanager
import sqlite3
import time
from pathlib import Path


class NotebookJobStore:
    """Keep receipts across server restarts using one SQLite database per notebook."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, body TEXT NOT NULL)')

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=15)
        try:
            with db:
                yield db
        finally:
            db.close()

    def put_job(self, job_id, kind, status, payload=None, result=None, error=None):
        row = dict(kind=kind, status=status, payload=payload, result=result, error=error,
                   updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        with self._connect() as db:
            db.execute('INSERT INTO jobs VALUES (?, ?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                       (job_id, json.dumps(row)))
        return row

    def get_job(self, job_id):
        with self._connect() as db:
            row = db.execute('SELECT body FROM jobs WHERE id=?', (job_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def queue_enqueue(self, *args):
        # The existing router starts a process-local task through its fallback.
        raise RuntimeError('Notebook jobs run in this single process')

    def queue_get(self, job_id):
        return None

    def mark_interrupted(self):
        """A stopped GPU process cannot resume transparently; report that honestly."""
        with self._connect() as db:
            for job_id, body in db.execute('SELECT id, body FROM jobs').fetchall():
                row = json.loads(body)
                if row['status'] in {'running', 'queued'}:
                    row['status'] = 'failed'
                    result = row.get('result') or {'job_id': job_id}
                    result.update(status='failed', stage='interrupted',
                                  error='Notebook server restarted; submit a new job to render again')
                    row['result'] = result
                    db.execute('UPDATE jobs SET body=? WHERE id=?', (json.dumps(row), job_id))
