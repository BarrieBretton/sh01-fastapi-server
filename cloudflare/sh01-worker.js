const STATE_KEY = "routing-state-v2";

function normalizeOrigin(value) {
  try {
    const u = new URL(String(value || "").trim());
    if (u.protocol !== "https:") return "";
    return u.origin;
  } catch { return ""; }
}
function json(data, status = 200) {
  return new Response(JSON.stringify(data), { status, headers: { "content-type": "application/json; charset=utf-8" } });
}
function configuredBackends(env) {
  const xs = String(env?.BACKENDS || "").split(",").map(normalizeOrigin).filter(Boolean);
  return xs.length ? [...new Set(xs)] : ["https://sh01-fastapi-server.onrender.com", "https://sh01.onrender.com"];
}
function allowFallback(env) {
  return String(env?.ALLOW_FALLBACK || "false").trim().toLowerCase() === "true";
}
function authorized(request, env) {
  const expected = String(env?.ROUTER_CONTROL_KEY || "");
  const received = request.headers.get("X-ROUTER-KEY") || "";
  return Boolean(expected) && received === expected;
}
async function loadControlState(env) {
  if (!env?.ROUTER_STATE) return { preferred: null, maintenance: false, updated_at: null, persistence_available: false };
  try {
    const stored = await env.ROUTER_STATE.get(STATE_KEY, "json");
    return {
      preferred: normalizeOrigin(stored?.preferred) || null,
      maintenance: Boolean(stored?.maintenance),
      updated_at: stored?.updated_at || null,
      persistence_available: true,
    };
  } catch { return { preferred: null, maintenance: false, updated_at: null, persistence_available: false }; }
}
async function saveControlState(env, state) {
  if (!env?.ROUTER_STATE) throw new Error("ROUTER_STATE KV binding is not configured");
  const payload = { preferred: state.preferred || null, maintenance: Boolean(state.maintenance), updated_at: new Date().toISOString() };
  await env.ROUTER_STATE.put(STATE_KEY, JSON.stringify(payload));
  return payload;
}
function effectiveOrder(configured, preferred, fallback) {
  if (preferred && configured.includes(preferred)) return fallback ? [preferred, ...configured.filter((x) => x !== preferred)] : [preferred];
  if (!configured.length) return [];
  return fallback ? [...configured] : [configured[0]];
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const configured = configuredBackends(env);
    const control = await loadControlState(env);
    const fallback = allowFallback(env);
    const targets = effectiveOrder(configured, control.preferred, fallback);

    if (url.pathname === "/__control/status") {
      if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
      return json({ ok: true, router: "sh01", configured, preferred: control.preferred, effective_order: targets, allow_fallback: fallback, maintenance: control.maintenance, updated_at: control.updated_at, persistence_available: control.persistence_available });
    }
    if (url.pathname === "/__control/preferred" && request.method === "POST") {
      if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
      let body; try { body = await request.json(); } catch { return json({ error: "invalid_json" }, 400); }
      const preferred = normalizeOrigin(body?.backend);
      if (!preferred || !configured.includes(preferred)) return json({ error: "backend_not_configured", configured }, 400);
      try {
        const saved = await saveControlState(env, { ...control, preferred });
        return json({ ok: true, configured, preferred: saved.preferred, effective_order: effectiveOrder(configured, saved.preferred, fallback), allow_fallback: fallback, maintenance: saved.maintenance, updated_at: saved.updated_at });
      } catch (e) { return json({ error: "state_persistence_failed", detail: String(e?.message || e) }, 503); }
    }
    if (url.pathname === "/__control/maintenance" && request.method === "POST") {
      if (!authorized(request, env)) return json({ error: "unauthorized" }, 401);
      let body; try { body = await request.json(); } catch { return json({ error: "invalid_json" }, 400); }
      if (typeof body?.enabled !== "boolean") return json({ error: "enabled_must_be_boolean" }, 400);
      try {
        const saved = await saveControlState(env, { ...control, maintenance: body.enabled });
        return json({ ok: true, maintenance: saved.maintenance, updated_at: saved.updated_at });
      } catch (e) { return json({ error: "state_persistence_failed", detail: String(e?.message || e) }, 503); }
    }
    if (url.pathname.startsWith("/__control/")) return json({ error: "not_found" }, 404);
    if (control.maintenance) return new Response("SH01 maintenance in progress", { status: 503, headers: { "retry-after": "30", "x-router-maintenance": "1" } });

    let bodyBytes = null;
    if (!["GET", "HEAD"].includes(request.method)) bodyBytes = await request.clone().arrayBuffer();
    const retriableStatuses = new Set([502, 503, 504, 520, 521, 522, 523, 524]);
    const hopByHop = new Set(["connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade", "content-length", "host"]);
    const timeoutMs = Number(env?.TIMEOUT_MS || 90000);

    function looksLikeRenderSuspended(htmlText) {
      const t = htmlText.toLowerCase();
      return t.includes("service suspended") || t.includes("site disabled") || (t.includes("render.com") && t.includes("suspended"));
    }
    function buildForwardHeaders(originalHeaders) {
      const h = new Headers();
      for (const [k, v] of originalHeaders.entries()) if (!hopByHop.has(k.toLowerCase())) h.set(k, v);
      h.set("x-proxy-by", "cf-worker-ring-router");
      h.set("x-forwarded-host", new URL(request.url).host);
      return h;
    }
    async function fetchWithTimeout(targetUrl, init, ms) {
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort("timeout"), ms);
      try { return await fetch(targetUrl, { ...init, signal: ac.signal }); }
      finally { clearTimeout(timer); }
    }

    let lastFailure = "none";
    const attempts = [];
    for (const target of targets) {
      const targetUrl = new URL(target);
      const forwardUrl = targetUrl.origin + url.pathname + url.search;
      try {
        const resp = await fetchWithTimeout(forwardUrl, { method: request.method, headers: buildForwardHeaders(request.headers), body: bodyBytes, redirect: "manual" }, timeoutMs);
        attempts.push(`${targetUrl.origin}:${resp.status}`);
        const outHeaders = new Headers(resp.headers);
        outHeaders.set("x-backend-used", targetUrl.origin);
        outHeaders.set("x-backend-status", String(resp.status));
        outHeaders.set("x-failover-attempts", attempts.join(", "));
        if (retriableStatuses.has(resp.status)) {
          lastFailure = `retriable_status_${resp.status}`;
          if (fallback) continue;
        }
        const contentType = resp.headers.get("content-type") || "";
        if (contentType.includes("text/html")) {
          const text = await resp.clone().text();
          if (looksLikeRenderSuspended(text)) {
            lastFailure = "render_suspended_page";
            if (fallback) continue;
          }
          return new Response(text, { status: resp.status, headers: outHeaders });
        }
        return new Response(resp.body ?? null, { status: resp.status, headers: outHeaders });
      } catch (e) {
        const msg = String(e?.message || e);
        attempts.push(`${target}:EX_${msg}`);
        lastFailure = msg.includes("timeout") ? "exception_timeout" : `exception_${msg}`;
        if (!fallback) break;
      }
    }
    return new Response("Active backend unavailable", { status: 502, headers: { "x-failover-last-failure": lastFailure, "x-failover-attempts": attempts.join(", ") } });
  },
};
