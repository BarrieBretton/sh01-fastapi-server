"""D1-backed active-state compare-and-swap and failover lease.

Every control-plane instance shares the same Cloudflare endpoint. Lease loss
prevents further mutations; an expiry is never taken as authorization.
"""

import json
import os
import threading
import uuid
from contextlib import contextmanager
from functools import wraps
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_local = threading.local()


def enabled() -> bool:
    return os.getenv("INFRA_COORDINATION_ENABLED", "").lower() == "true"


def _call(action: str, payload: dict | None = None) -> dict:
    base = os.getenv("CONTROL_PLANE_BASE_URL", "").rstrip("/")
    key = os.getenv("CONTROL_PLANE_API_KEY", "")
    if not base.startswith("https://") or not key:
        raise RuntimeError("Infra coordinator URL/key missing; refusing mutation")
    request = Request(
        base + "/internal/infra-coordination/" + action,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        method="GET" if payload is None else "POST",
        headers={"X-API-KEY": key, "Content-Type": "application/json"},
    )
    try:
        with urlopen(request, timeout=12) as response:
            value = json.load(response)
    except (HTTPError, URLError, ValueError, TimeoutError) as exc:
        raise RuntimeError(f"Infra coordinator {action} unavailable or rejected") from exc
    if not isinstance(value, dict):
        raise RuntimeError("Infra coordinator returned invalid data")
    return value


def active() -> dict:
    value = _call("status")
    state = value.get("active")
    if not isinstance(state, dict) or any(not state.get(k) for k in ("render", "postgres", "b2")):
        raise RuntimeError("Infra coordinator active state has not been seeded")
    return dict(state)


def require_lease() -> None:
    if not enabled():
        raise RuntimeError("Infra coordinator is disabled; live mutations are blocked")
    lease = getattr(_local, "lease", None)
    if not lease or lease["lost"].is_set():
        raise RuntimeError("No live infrastructure lease")
    _call("renew", {k: lease[k] for k in ("owner", "token", "generation")})


def commit_active(next_state: dict) -> dict:
    require_lease()
    lease = _local.lease
    value = _call("commit", {
        **{k: lease[k] for k in ("owner", "token", "generation")},
        "active": next_state,
    })
    if value.get("active") != next_state:
        raise RuntimeError("Coordinator did not commit requested active state")
    return value


@contextmanager
def mutation_lease(expected_active: dict):
    if not enabled():
        raise RuntimeError("Enable and seed the infra coordinator before live failover")
    if getattr(_local, "lease", None):
        require_lease()
        yield
        return
    owner = str(uuid.uuid4())
    value = _call("acquire", {"owner": owner, "expected_active": expected_active})
    token, generation = value.get("token"), value.get("generation")
    if not token or not isinstance(generation, int):
        raise RuntimeError("Infra coordinator failed to grant lease")
    lease = {"owner": owner, "token": token, "generation": generation,
             "lost": threading.Event(), "stop": threading.Event()}
    _local.lease = lease

    def renew():
        while not lease["stop"].wait(45):
            try:
                _call("renew", {k: lease[k] for k in ("owner", "token", "generation")})
            except Exception:
                lease["lost"].set()
                return

    thread = threading.Thread(target=renew, daemon=True, name="infra-lease-renew")
    thread.start()
    try:
        yield
    finally:
        lease["stop"].set()
        thread.join(timeout=2)
        try:
            _call("release", {k: lease[k] for k in ("owner", "token", "generation")})
        except Exception:
            pass  # TTL expires. No other writer can acquire until it does.
        _local.lease = None


def coordinated(fn):
    """Acquire across both controller processes; dry runs never take a lease."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from .state import state
        expected = state.refresh_authoritative()
        if kwargs.get("dry_run") is True:
            return fn(*args, **kwargs)
        with mutation_lease(expected):
            return fn(*args, **kwargs)
    return wrapper
