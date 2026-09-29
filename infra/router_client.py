import logging
import os
from typing import Any

import httpx

logger = logging.getLogger("infra.router")

_ALLOWED_ROLES = {"n8n", "sh01"}


def _role(role: str) -> str:
    value = role.strip().lower()
    if value not in _ALLOWED_ROLES:
        raise RuntimeError(f"Unknown router role: {role}")
    return value


def _router_base(role: str) -> str:
    role = _role(role)
    env_name = f"{role.upper()}_ROUTER_CONTROL_BASE_URL"
    value = os.getenv(env_name, "").strip().rstrip("/")
    # Backward compatibility for the first package; only n8n may use it.
    if not value and role == "n8n":
        value = os.getenv("ROUTER_CONTROL_BASE_URL", "").strip().rstrip("/")
    if not value:
        raise RuntimeError(f"{env_name} is not configured")
    return value


def _router_key(role: str) -> str:
    role = _role(role)
    env_name = f"{role.upper()}_ROUTER_CONTROL_KEY"
    value = os.getenv(env_name, "").strip()
    if not value and role == "n8n":
        value = os.getenv("ROUTER_CONTROL_KEY", "").strip()
    if not value:
        raise RuntimeError(f"{env_name} is not configured")
    return value


def _headers(role: str) -> dict[str, str]:
    return {
        "X-ROUTER-KEY": _router_key(role),
        "Content-Type": "application/json",
        "Accept": "application/json",
    }


def _request(role: str, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    with httpx.Client(timeout=20.0) as client:
        response = client.request(
            method,
            _router_base(role) + path,
            headers=_headers(role),
            json=payload,
        )
    if response.status_code >= 300:
        raise RuntimeError(
            f"{role} router {method} {path} failed: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    return response.json()


def router_status(role: str = "n8n") -> dict[str, Any]:
    return _request(role, "GET", "/__control/status")


def router_set_preferred(role: str, backend: str) -> dict[str, Any]:
    backend = backend.rstrip("/")
    result = _request(
        role,
        "POST",
        "/__control/preferred",
        {"backend": backend},
    )
    logger.info("%s router preferred backend set to %s", role, backend)
    return result


def router_set_maintenance(role: str, enabled: bool) -> dict[str, Any]:
    result = _request(
        role,
        "POST",
        "/__control/maintenance",
        {"enabled": bool(enabled)},
    )
    logger.info("%s router maintenance=%s", role, enabled)
    return result
