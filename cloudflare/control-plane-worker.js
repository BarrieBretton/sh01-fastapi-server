const STATE_KEY = "control-plane-routing-state-v1";
const RETRIABLE_STATUSES = new Set([502, 503, 504, 520, 521, 522, 523, 524]);
const HOP_HEADERS = new Set([
  "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
  "te", "trailer", "transfer-encoding", "upgrade", "content-length", "host",
]);

function json(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

// Shared queue implementation. Paste the completed control-plane-worker.js
// into the Cloudflare dashboard editor; this fragment is build input only.
const QUEUE_BASE = "/internal/media-jobs/queue/";
const COORD_BASE = "/internal/infra-coordination/";
let lastQueueCleanupAt = 0;

// D1 conditional writes serialize the two master processes. Never use KV as a
// distributed lock. No endpoint accepts a caller-supplied generation.
async function infraCoordination(request, env) {
  if (!env.CONTROL_PLANE_API_KEY || request.headers.get("X-API-KEY") !== env.CONTROL_PLANE_API_KEY)
    return json({ error: "unauthorized" }, 401);
  if (!env.MEDIA_QUEUE_DB) return json({ error: "coordinator_db_missing" }, 503);
  const db = env.MEDIA_QUEUE_DB;
  const path = new URL(request.url).pathname.slice(COORD_BASE.length);
  try {
    const read = () => db.prepare("SELECT * FROM infra_coordination WHERE name='global'").first();
    const current = await read();
    if (!current) return json({ error: "coordinator_unseeded" }, 503);
    const view = (row) => ({ generation: row.generation, active: {
      render: row.active_render, postgres: row.active_postgres, b2: row.active_b2,
    }, locked: row.lease_until > Date.now(), lease_until: row.lease_until });
    if (path === "status" && request.method === "GET") return json(view(current));
    if (request.method !== "POST" || !["acquire", "renew", "commit", "release"].includes(path))
      return json({ error: "not_found" }, 404);
    let input;
    try { input = await request.json(); } catch { return json({ error: "invalid_json" }, 400); }
    if (!input || typeof input !== "object" || typeof input.owner !== "string" ||
        !/^[a-zA-Z0-9_-]{1,128}$/.test(input.owner)) return json({ error: "invalid_owner" }, 400);
    const now = Date.now();
    const ttl = 180000;
    if (path === "acquire") {
      if (!input.expected_active || ["render", "postgres", "b2"].some(k =>
          input.expected_active[k] !== current[`active_${k}`]))
        return json({ error: "active_state_mismatch", state: view(current) }, 409);
      const token = crypto.randomUUID();
      const result = await db.prepare("UPDATE infra_coordination SET owner=?,token=?," +
        "generation=generation+1,lease_until=?,updated_at=? " +
        "WHERE name='global' AND lease_until<=? AND generation=?")
        .bind(input.owner, token, now + ttl, now, now, current.generation).run();
      if (result.meta.changes !== 1) return json({ error: "lease_busy_or_stale" }, 409);
      return json({ ...view(await read()), owner: input.owner, token });
    }
    if (typeof input.token !== "string" || typeof input.generation !== "number")
      return json({ error: "missing_lease" }, 400);
    const args = [input.owner, input.token, input.generation, now];
    if (path === "renew") {
      const result = await db.prepare("UPDATE infra_coordination SET lease_until=?,updated_at=? " +
        "WHERE name='global' AND owner=? AND token=? AND generation=? AND lease_until>?")
        .bind(now + ttl, now, ...args).run();
      return result.meta.changes === 1 ? json(view(await read())) : json({ error: "lease_lost" }, 409);
    }
    if (path === "commit") {
      const next = input.active;
      if (!next || ["render", "postgres", "b2"].some(k =>
          typeof next[k] !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(next[k])))
        return json({ error: "invalid_active" }, 400);
      const result = await db.prepare("UPDATE infra_coordination SET active_render=?,active_postgres=?," +
        "active_b2=?,updated_at=? WHERE name='global' AND owner=? AND token=? " +
        "AND generation=? AND lease_until>?")
        .bind(next.render, next.postgres, next.b2, now, ...args).run();
      return result.meta.changes === 1 ? json(view(await read())) : json({ error: "lease_lost" }, 409);
    }
    const result = await db.prepare("UPDATE infra_coordination SET owner=NULL,token=NULL," +
      "lease_until=0,updated_at=? WHERE name='global' AND owner=? AND token=? " +
      "AND generation=? AND lease_until>?").bind(now, ...args).run();
    return result.meta.changes === 1 ? json(view(await read())) : json({ error: "lease_lost" }, 409);
  } catch (err) {
    console.error("Infra coordinator D1 failure", err);
    return json({ error: "coordinator_unavailable" }, 503);
  }
}

class Sh01D1MediaQueue {
  constructor(db, env) { this.db = db; this.env = env; }
  async first(sql, ...params) {
    const stmt = this.db.prepare(sql);
    return (params.length ? stmt.bind(...params) : stmt).first();
  }
  async run(sql, ...params) {
    const stmt = this.db.prepare(sql);
    return (params.length ? stmt.bind(...params) : stmt).run();
  }
  config(name, fallback, minimum, maximum) {
    const n = Number(this.env[name]);
    return Number.isSafeInteger(n) && n >= minimum && n <= maximum ? n : fallback;
  }
  stable(value) {
    if (Array.isArray(value)) return `[${value.map(x => this.stable(x)).join(",")}]`;
    if (value && typeof value === "object")
      return `{${Object.keys(value).sort().map(k => `${JSON.stringify(k)}:${this.stable(value[k])}`).join(",")}}`;
    return JSON.stringify(value);
  }
  present(row) {
    if (!row) return null;
    const payload = JSON.parse(row.payload);
    return {
      id: row.id, kind: row.kind, status: row.status,
      request_id: payload.request_id ?? null,
      result: row.result ? JSON.parse(row.result) : null,
      error: row.error, attempts: row.attempts,
      created_at: new Date(row.created_at).toISOString(),
      updated_at: new Date(row.updated_at).toISOString(),
    };
  }
  async expired(now) {
    await this.run("UPDATE jobs SET status='interrupted',error='Worker lease expired',updated_at=? WHERE status='running' AND lease_until<=?", now, now);
    if (now - lastQueueCleanupAt > 86400000) {
      lastQueueCleanupAt = now;
      await this.run("DELETE FROM jobs WHERE status IN ('complete','failed','interrupted') AND updated_at<?", now - 30 * 86400000);
    }
  }
  async enqueue({ job_id, kind, payload }) {
    if (typeof job_id !== "string" || !/^[a-zA-Z0-9_-]{1,128}$/.test(job_id) ||
        !["auto_edit", "clip_caption", "audio_image_video"].includes(kind) ||
        !payload || typeof payload !== "object" || Array.isArray(payload) ||
        typeof payload.request_id !== "string" || !payload.request ||
        typeof payload.request !== "object" || Array.isArray(payload.request))
      return json({ detail: "Invalid queue admission" }, 400);
    const encoded = this.stable(payload);
    if (new TextEncoder().encode(encoded).byteLength > 128 * 1024)
      return json({ detail: "Queue request too large" }, 413);
    const existing = await this.first("SELECT * FROM jobs WHERE id=?", job_id);
    if (existing) return this.enqueueResult(existing, kind, encoded);
    const now = Date.now();
    const inserted = await this.run(
      "INSERT OR IGNORE INTO jobs(id,kind,payload,status,created_at,updated_at) " +
      "SELECT ?,?,?,'queued',?,? WHERE (SELECT COUNT(*) FROM jobs WHERE status='queued') < ?",
      job_id, kind, encoded, now, now, this.config("MEDIA_QUEUE_MAX_PENDING", 100, 1, 10000));
    const row = await this.first("SELECT * FROM jobs WHERE id=?", job_id);
    if (row) return this.enqueueResult(row, kind, encoded);
    if (inserted.meta.changes === 0) return json({ detail: "Media queue full" }, 429);
    return json({ detail: "Queue insertion unavailable" }, 503);
  }
  enqueueResult(row, kind, encoded) {
    if (row.kind !== kind || row.payload !== encoded)
      return json({ detail: "Job id reused with a different request" }, 409);
    return json({ job: this.present(row) });
  }
  async get(jobId) {
    await this.expired(Date.now());
    const row = await this.first("SELECT * FROM jobs WHERE id=?", jobId);
    return row ? json({ job: this.present(row) }) : json({ detail: "Unknown queue job" }, 404);
  }
  async claim(input) {
    const owner = input.owner;
    if (typeof owner !== "string" || owner.length < 1 || owner.length > 256)
      return json({ detail: "Invalid owner" }, 400);
    const now = Date.now();
    await this.expired(now);
    let running = await this.first("SELECT * FROM jobs WHERE status='running' LIMIT 1");
    if (running) return json({ job: running.owner === owner &&
      ["auto_edit", "clip_caption", "audio_image_video"].includes(running.kind)
        ? { ...this.present(running), payload: JSON.parse(running.payload), lease_token: running.lease_token }
        : null });
    const token = crypto.randomUUID();
    const until = now + this.config("MEDIA_QUEUE_LEASE_SECONDS", 300, 60, 3600) * 1000;
    // One conditional SQL write serializes all contenders on the D1 primary.
    const changed = await this.run(
      "UPDATE jobs SET status='running',owner=?,lease_token=?,lease_until=?,attempts=attempts+1,updated_at=? " +
      "WHERE id=(SELECT id FROM jobs WHERE status='queued' AND " +
      "(?=1 OR COALESCE(json_extract(payload,'$.request.unlimited_resources'),0)<>1) " +
      "ORDER BY created_at,id LIMIT 1) AND NOT EXISTS(SELECT 1 FROM jobs WHERE status='running')",
      owner, token, until, now, input.high_capable === true ? 1 : 0);
    if (changed.meta.changes === 0) return json({ job: null });
    running = await this.first("SELECT * FROM jobs WHERE lease_token=?", token);
    if (!running) return json({ detail: "Queue claim could not be confirmed" }, 503);
    return json({ job: { ...this.present(running), payload: JSON.parse(running.payload), lease_token: token } });
  }
  async heartbeat(input) {
    if (![input.job_id, input.owner, input.token].every(v => typeof v === "string" && v.length))
      return json({ alive: false }, 400);
    const now = Date.now();
    const changed = await this.run(
      "UPDATE jobs SET lease_until=?,updated_at=? WHERE id=? AND owner=? AND lease_token=? AND status='running' AND lease_until>?",
      now + this.config("MEDIA_QUEUE_LEASE_SECONDS", 300, 60, 3600) * 1000, now,
      input.job_id, input.owner, input.token, now);
    return json({ alive: changed.meta.changes === 1 });
  }
  async finish(input) {
    if (!["complete", "failed", "interrupted"].includes(input.status) ||
        !input.result || typeof input.result !== "object" || Array.isArray(input.result) ||
        typeof input.job_id !== "string" || typeof input.owner !== "string" || typeof input.token !== "string")
      return json({ detail: "Invalid queue finish" }, 400);
    const result = this.stable(input.result);
    if (new TextEncoder().encode(result).byteLength > 1024 * 1024)
      return json({ detail: "Queue result too large" }, 413);
    const now = Date.now();
    const changed = await this.run(
      "UPDATE jobs SET status=?,result=?,error=?,lease_until=NULL,updated_at=? " +
      "WHERE id=? AND owner=? AND lease_token=? AND status='running' AND lease_until>?",
      input.status, result, String(input.result.error ?? "").slice(0, 4000), now,
      input.job_id, input.owner, input.token, now);
    if (changed.meta.changes === 1) return json({ finished: true });
    const row = await this.first("SELECT status,result,owner,lease_token FROM jobs WHERE id=?", input.job_id);
    return json({ finished: Boolean(row && row.owner === input.owner && row.lease_token === input.token &&
      row.status === input.status && row.result === result) });
  }
  async acquireExternal(input) {
    if (![input.job_id, input.kind, input.owner].every(v => typeof v === "string" && v.length && v.length <= 256))
      return json({ detail: "Invalid external admission" }, 400);
    const now = Date.now();
    await this.expired(now);
    const existing = await this.first("SELECT status,owner,lease_token FROM jobs WHERE id=?", input.job_id);
    if (existing) return json({ token: existing.status === "running" && existing.owner === input.owner ? existing.lease_token : null });
    const token = crypto.randomUUID();
    const until = now + this.config("MEDIA_QUEUE_LEASE_SECONDS", 300, 60, 3600) * 1000;
    const changed = await this.run(
      "INSERT OR IGNORE INTO jobs(id,kind,payload,status,owner,lease_token,lease_until,attempts,created_at,updated_at) " +
      "SELECT ?,?,'{}','running',?,?,?,1,?,? WHERE NOT EXISTS(SELECT 1 FROM jobs WHERE status IN ('queued','running'))",
      input.job_id, input.kind, input.owner, token, until, now, now);
    return json({ token: changed.meta.changes === 1 ? token : null });
  }
  async fetch(request) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith(QUEUE_BASE)) return json({ detail: "Not found" }, 404);
    try {
      const action = url.pathname.slice(QUEUE_BASE.length);
      if (request.method === "GET" && action === "health") {
        const table = await this.first("SELECT name FROM sqlite_master WHERE type='table' AND name='jobs'");
        return table ? json({ ok: true, queue: "cloudflare-d1" }) : json({ detail: "Media queue schema missing" }, 503);
      }
      if (request.method === "GET" && /^[a-zA-Z0-9_-]{1,128}$/.test(action)) return this.get(action);
      if (request.method !== "POST" || !["enqueue", "claim", "heartbeat", "finish", "acquire-external"].includes(action))
        return json({ detail: "Not found" }, 404);
      const content = await request.text();
      if (new TextEncoder().encode(content).byteLength > 1536 * 1024)
        return json({ detail: "Queue request too large" }, 413);
      let input;
      try { input = JSON.parse(content); } catch { return json({ detail: "Invalid JSON" }, 400); }
      if (!input || typeof input !== "object" || Array.isArray(input)) return json({ detail: "Invalid request" }, 400);
      switch (action) {
        case "enqueue": return this.enqueue(input);
        case "claim": return this.claim(input);
        case "heartbeat": return this.heartbeat(input);
        case "finish": return this.finish(input);
        case "acquire-external": return this.acquireExternal(input);
      }
    } catch (error) {
      console.error("D1 media queue request failed", error);
      return json({ detail: "Media queue database unavailable" }, 503);
    }
  }
}

function origin(value) {
  try {
    const u = new URL(String(value || "").trim());
    if (u.protocol !== "https:" || u.username || u.password || u.pathname !== "/" || u.search || u.hash) return "";
    return u.origin;
  } catch { return ""; }
}

function backends(env, publicOrigin) {
  return [...new Set(String(env.BACKENDS || "").split(",").map(origin).filter(
    (value) => value && value !== publicOrigin && !value.endsWith(".workers.dev")
  ))];
}

function isAuthorized(request, env) {
  const expected = String(env.ROUTER_CONTROL_KEY || "");
  return Boolean(expected) && request.headers.get("X-ROUTER-KEY") === expected;
}

async function readState(env) {
  if (!env.ROUTER_STATE) return { preferred: null, maintenance: false, persistent: false };
  try {
    const value = await env.ROUTER_STATE.get(STATE_KEY, "json");
    return {
      preferred: origin(value?.preferred) || null,
      maintenance: Boolean(value?.maintenance),
      persistent: true,
    };
  } catch {
    return { preferred: null, maintenance: false, persistent: false };
  }
}

async function writeState(env, next) {
  if (!env.ROUTER_STATE) throw new Error("ROUTER_STATE KV binding is not configured");
  await env.ROUTER_STATE.put(STATE_KEY, JSON.stringify({
    preferred: next.preferred, maintenance: next.maintenance,
    updated_at: new Date().toISOString(),
  }));
}

function order(configured, preferred, fallback) {
  if (!configured.length) return [];
  const selected = configured.includes(preferred) ? preferred : configured[0];
  return fallback ? [selected, ...configured.filter((v) => v !== selected)] : [selected];
}

function timeoutMs(env, name, fallback) {
  const parsed = Number(env[name]);
  return Number.isFinite(parsed) && parsed >= 1000 && parsed <= 120000 ? parsed : fallback;
}

async function timedFetch(url, init, ms) {
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), ms);
  try { return await fetch(url, { ...init, signal: abort.signal }); }
  finally { clearTimeout(timer); }
}

async function isReady(target, env) {
  const key = String(env.CONTROL_PLANE_HEALTH_KEY || "");
  if (!key) return false;
  try {
    const response = await timedFetch(target + "/infra/health", {
      method: "GET", redirect: "manual",
      headers: { "X-API-KEY": key, "accept": "application/json" },
    }, timeoutMs(env, "HEALTH_TIMEOUT_MS", 10000));
    if (!response.ok || !response.headers.get("content-type")?.includes("application/json")) return false;
    const value = await response.json();
    return value?.ok === true && value?.service === "infra-control-plane";
  } catch { return false; }
}

function forwardHeaders(request) {
  const headers = new Headers();
  for (const [name, value] of request.headers) {
    const lower = name.toLowerCase();
    if (!HOP_HEADERS.has(lower) && lower !== "x-forwarded-host" && lower !== "x-forwarded-proto")
      headers.set(name, value);
  }
  headers.set("X-Forwarded-Host", new URL(request.url).host);
  headers.set("X-Forwarded-Proto", "https");
  return headers;
}

function responseHeaders(response, target, attempts, publicOrigin) {
  const headers = new Headers(response.headers);
  for (const header of HOP_HEADERS) headers.delete(header);
  headers.set("x-backend-used", target);
  headers.set("x-backend-status", String(response.status));
  headers.set("x-failover-attempts", attempts.join(", "));
  const location = headers.get("location");
  if (location) {
    try {
      const absolute = new URL(location, target);
      if (absolute.origin === target) {
        absolute.host = new URL(publicOrigin).host;
        headers.set("location", absolute.href);
      }
    } catch { /* preserve invalid upstream header */ }
  }
  return headers;
}

function safeToReplay(method, path) {
  return method === "GET" || method === "HEAD" ||
    (method === "PUT" && /^\/internal\/media-jobs\/[^/]+$/.test(path));
}

function suspended(text) {
  const lower = text.toLowerCase();
  return lower.includes("service suspended") ||
    lower.includes("site disabled") ||
    (lower.includes("render.com") && lower.includes("suspended"));
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname.startsWith(COORD_BASE)) return infraCoordination(request, env);
    if (url.pathname.startsWith("/internal/media-jobs/queue/")) {
      if (!env.MEDIA_JOB_STORE_API_KEY) return json({ detail: "Media queue key missing" }, 503);
      if (request.headers.get("X-MEDIA-JOB-KEY") !== env.MEDIA_JOB_STORE_API_KEY)
        return json({ detail: "Missing/invalid media queue key" }, 401);
      if (!env.MEDIA_QUEUE_DB) return json({ detail: "D1 media queue binding missing" }, 503);
      return new Sh01D1MediaQueue(env.MEDIA_QUEUE_DB, env).fetch(request);
    }
    const configured = backends(env, url.origin);
    const state = await readState(env);
    const fallback = String(env.ALLOW_FALLBACK || "true").toLowerCase() === "true";
    const targets = order(configured, state.preferred, fallback);

    if (url.pathname.startsWith("/__control/")) {
      if (!isAuthorized(request, env)) return json({ error: "unauthorized" }, 401);
      if (url.pathname === "/__control/status" && request.method === "GET") {
        return json({ ok: true, router: "control-plane", configured,
          preferred: state.preferred, effective_order: targets,
          allow_fallback: fallback, maintenance: state.maintenance,
          persistence_available: state.persistent });
      }
      if (url.pathname === "/__control/preferred" && request.method === "POST") {
        let body;
        try { body = await request.json(); } catch { return json({ error: "invalid_json" }, 400); }
        const preferred = origin(body?.backend);
        if (!configured.includes(preferred)) return json({ error: "backend_not_configured" }, 400);
        try {
          await writeState(env, { ...state, preferred });
          return json({ ok: true, preferred });
        } catch (e) { return json({ error: "state_persistence_failed", detail: String(e) }, 503); }
      }
      if (url.pathname === "/__control/maintenance" && request.method === "POST") {
        let body;
        try { body = await request.json(); } catch { return json({ error: "invalid_json" }, 400); }
        if (typeof body?.enabled !== "boolean") return json({ error: "enabled_must_be_boolean" }, 400);
        try {
          await writeState(env, { ...state, maintenance: body.enabled });
          return json({ ok: true, maintenance: body.enabled });
        } catch (e) { return json({ error: "state_persistence_failed", detail: String(e) }, 503); }
      }
      return json({ error: "not_found" }, 404);
    }

    if (state.maintenance) return json({ error: "control_plane_maintenance" }, 503);
    if (!configured.length || !env.CONTROL_PLANE_HEALTH_KEY)
      return json({ error: "router_not_configured" }, 503);

    // Select a working master before sending any mutating request. Never
    // replay a non-idempotent POST after it may have reached an origin.
    let body = null;
    if (request.method !== "GET" && request.method !== "HEAD")
      body = await request.arrayBuffer();
    const replayable = safeToReplay(request.method, url.pathname);
    const attempts = [];
    for (const target of targets) {
      if (!await isReady(target, env)) {
        attempts.push(target + ":unhealthy");
        continue;
      }
      try {
        const response = await timedFetch(target + url.pathname + url.search, {
          method: request.method,
          headers: forwardHeaders(request),
          body, redirect: "manual",
        }, timeoutMs(env, "REQUEST_TIMEOUT_MS", 90000));
        attempts.push(target + ":" + response.status);
        const headers = responseHeaders(response, target, attempts, url.origin);
        const isHtml = (response.headers.get("content-type") || "").includes("text/html");
        if (isHtml) {
          const html = await response.text();
          if (suspended(html) && replayable && fallback) continue;
          if (suspended(html)) return json({ error: "master_suspended" }, 503);
          return new Response(html, { status: response.status, headers });
        }
        if (RETRIABLE_STATUSES.has(response.status) && replayable && fallback) continue;
        return new Response(response.body, { status: response.status, headers });
      } catch (e) {
        attempts.push(target + ":transport_error");
        if (!replayable) return json({ error: "master_request_uncertain", retryable: false }, 502);
      }
    }
    return json({ error: "no_healthy_control_plane", attempts }, 503);
  },
};
