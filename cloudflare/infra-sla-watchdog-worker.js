function json(data, status = 200) {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json; charset=utf-8" },
  });
}

async function triggerTick(env, dryRun = false) {
  const base = String(env?.CONTROL_PLANE_BASE_URL || "").trim().replace(/\/$/, "");
  const key = String(env?.CONTROL_PLANE_API_KEY || "").trim();
  if (!base) throw new Error("CONTROL_PLANE_BASE_URL is not configured");
  if (!key) throw new Error("CONTROL_PLANE_API_KEY is not configured");

  const response = await fetch(`${base}/infra/sla/tick`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "X-API-KEY": key,
    },
    body: JSON.stringify({ dry_run: Boolean(dryRun) }),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(`SLA tick failed HTTP ${response.status}: ${text.slice(0, 1000)}`);
  }
  try { return JSON.parse(text); }
  catch { return { raw: text }; }
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      triggerTick(env, false).catch((err) => {
        console.error("infra SLA watchdog tick failed", String(err?.message || err));
      })
    );
  },

  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/health") {
      return json({ ok: true, service: "infra-sla-watchdog" });
    }

    if (url.pathname === "/tick" && request.method === "POST") {
      const expected = String(env?.WATCHDOG_MANUAL_KEY || "");
      const received = request.headers.get("X-WATCHDOG-KEY") || "";
      if (!expected || received !== expected) return json({ error: "unauthorized" }, 401);
      try {
        return json(await triggerTick(env, false));
      } catch (err) {
        return json({ error: String(err?.message || err) }, 500);
      }
    }

    return json({ error: "not_found" }, 404);
  },
};
