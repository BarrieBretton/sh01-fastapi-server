import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const code = readFileSync(new URL('../control-plane-worker.js', import.meta.url), 'utf8');
const { default: worker } = await import('data:text/javascript,' + encodeURIComponent(code));

class FakeD1 {
  constructor() {
    this.row = { name: 'global', generation: 0, owner: null, token: null,
      lease_until: 0, active_render: 'render-b', active_postgres: 'supabase-b',
      active_b2: 'b2-a', updated_at: 0 };
  }
  prepare(sql) {
    const db = this;
    let args = [];
    return {
      bind(...values) { args = values; return this; },
      first() { return { ...db.row }; },
      run() {
        const r = db.row;
        if (sql.includes('generation=generation+1')) {
          if (r.lease_until > args[4] || r.generation !== args[5]) return { meta: { changes: 0 } };
          [r.owner, r.token, r.lease_until, r.updated_at] = args.slice(0, 4);
          r.generation++;
        } else {
          const [owner, token, generation, now] = args.slice(-4);
          if (r.owner !== owner || r.token !== token || r.generation !== generation || r.lease_until <= now)
            return { meta: { changes: 0 } };
          if (sql.includes('active_render=?')) {
            [r.active_render, r.active_postgres, r.active_b2, r.updated_at] = args.slice(0, 4);
          } else if (sql.includes('lease_until=0')) {
            r.owner = null; r.token = null; r.lease_until = 0; r.updated_at = args[0];
          } else {
            r.lease_until = args[0]; r.updated_at = args[1];
          }
        }
        return { meta: { changes: 1 } };
      },
    };
  }
}

const base = 'https://infra-control.vivojaymail.workers.dev/internal/infra-coordination/';
function request(action, key, body) {
  return new Request(base + action, { method: body ? 'POST' : 'GET',
    headers: { 'X-API-KEY': key }, body: body ? JSON.stringify(body) : undefined });
}

test('competing masters and stale generation cannot commit', async () => {
  const db = new FakeD1();
  const env = { CONTROL_PLANE_API_KEY: 'test-secret', MEDIA_QUEUE_DB: db };
  const active = { render: 'render-b', postgres: 'supabase-b', b2: 'b2-a' };
  assert.equal((await worker.fetch(request('status', 'wrong'), env)).status, 401);
  const first = await worker.fetch(request('acquire', 'test-secret',
    { owner: 'master_a', expected_active: active }), env);
  const leader = await first.json();
  assert.equal(first.status, 200);
  assert.equal((await worker.fetch(request('acquire', 'test-secret',
    { owner: 'master_b', expected_active: active }), env)).status, 409);
  assert.equal((await worker.fetch(request('commit', 'test-secret', {
    owner: 'master_a', token: 'stolen-token', generation: leader.generation,
    active: { ...active, postgres: 'supabase-a' },
  }), env)).status, 409);
  const committed = await worker.fetch(request('commit', 'test-secret', {
    owner: 'master_a', token: leader.token, generation: leader.generation,
    active: { ...active, postgres: 'supabase-a' },
  }), env);
  assert.equal(committed.status, 200);
  assert.equal(db.row.active_postgres, 'supabase-a');
  db.row.lease_until = Date.now() - 1;
  assert.equal((await worker.fetch(request('renew', 'test-secret', {
    owner: 'master_a', token: leader.token, generation: leader.generation,
  }), env)).status, 409);
});
