import logging
import os
import time
from typing import Any

import httpx

from .registry import registry

logger = logging.getLogger("infra.render")
RENDER_API_BASE = "https://api.render.com/v1"


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _service(slot: str, role: str) -> dict[str, Any]:
    slot_cfg = registry.get("render", slot)
    svc = registry.render_service(slot, role)
    required = {"service_id_env", "base_url_env"}
    missing = required - set(svc)
    if missing:
        raise RuntimeError(
            f"Render slot '{slot}' role '{role}' is missing registry fields: "
            + ", ".join(sorted(missing))
        )

    api_key_env = svc.get("api_key_env") or slot_cfg.get("api_key_env") or "RENDER_API_KEY"
    default_health = "/healthz/readiness" if role == "n8n" else "/"
    return {
        "slot": slot,
        "role": role,
        "service_id": _env_value(svc["service_id_env"]),
        "api_key": _env_value(api_key_env),
        "base_url": _env_value(svc["base_url_env"]).rstrip("/"),
        "health_path": svc.get("health_path", default_health),
        "db_env_map": svc.get(
            "db_env_map",
            {
                "host": "DB_POSTGRESDB_HOST",
                "port": "DB_POSTGRESDB_PORT",
                "database": "DB_POSTGRESDB_DATABASE",
                "user": "DB_POSTGRESDB_USER",
                "password": "DB_POSTGRESDB_PASSWORD",
            },
        ),
        "b2_env_map": svc.get(
            "b2_env_map",
            {
                "key_id": "BACKBLAZE_KEY_ID",
                "application_key": "BACKBLAZE_APPLICATION_KEY",
                "bucket_name": "BACKBLAZE_BUCKET_NAME",
            },
        ),
    }


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def render_service_status(slot: str, role: str) -> dict[str, Any]:
    cfg = _service(slot, role)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code >= 300:
        raise RuntimeError(
            f"Render service status failed for {slot}/{role}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    payload = response.json()
    return {
        "slot": slot,
        "role": role,
        "suspended": payload.get("suspended"),
        "name": payload.get("name"),
        "raw": payload,
    }


def render_health(slot: str, role: str, timeout: float = 20.0) -> dict[str, Any]:
    cfg = _service(slot, role)
    url = cfg["base_url"] + cfg["health_path"]
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
        return {
            "slot": slot,
            "role": role,
            "healthy": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": url,
            "body": response.text[:500],
        }
    except Exception as exc:
        return {
            "slot": slot,
            "role": role,
            "healthy": False,
            "status_code": None,
            "url": url,
            "error": str(exc),
        }


def render_update_env(slot: str, role: str, values: dict[str, str]) -> dict[str, Any]:
    cfg = _service(slot, role)
    changed: list[str] = []
    with httpx.Client(timeout=30.0) as client:
        for key, value in values.items():
            response = client.put(
                f"{RENDER_API_BASE}/services/{cfg['service_id']}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
                json={"value": str(value)},
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env update failed for {slot}/{role}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            changed.append(key)
    logger.info("Updated Render env for %s/%s keys=%s", slot, role, changed)
    return {"slot": slot, "role": role, "updated": changed}


def render_runtime_env_keys(slot: str, role: str = "n8n") -> list[str]:
    cfg = _service(slot, role)
    keys = list(cfg["db_env_map"].values())
    keys.extend(cfg["b2_env_map"].values())
    return sorted(set(keys))


def render_get_env(slot: str, role: str, keys: list[str]) -> dict[str, str | None]:
    cfg = _service(slot, role)
    result: dict[str, str | None] = {}
    with httpx.Client(timeout=30.0) as client:
        for key in keys:
            response = client.get(
                f"{RENDER_API_BASE}/services/{cfg['service_id']}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
            )
            if response.status_code == 404:
                result[key] = None
                continue
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env read failed for {slot}/{role}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            payload = response.json()
            env_var = payload.get("envVar", payload)
            result[key] = env_var.get("value")
    return result


def render_restore_env(slot: str, role: str, values: dict[str, str | None]) -> dict[str, Any]:
    cfg = _service(slot, role)
    restored: list[str] = []
    deleted: list[str] = []
    with httpx.Client(timeout=30.0) as client:
        for key, value in values.items():
            if value is None:
                response = client.delete(
                    f"{RENDER_API_BASE}/services/{cfg['service_id']}/env-vars/{key}",
                    headers=_headers(cfg["api_key"]),
                )
                if response.status_code not in (204, 404):
                    raise RuntimeError(
                        f"Render env delete failed for {slot}/{role}:{key}: "
                        f"HTTP {response.status_code} {response.text[:1000]}"
                    )
                deleted.append(key)
                continue
            response = client.put(
                f"{RENDER_API_BASE}/services/{cfg['service_id']}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
                json={"value": str(value)},
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env restore failed for {slot}/{role}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            restored.append(key)
    return {"slot": slot, "role": role, "restored": restored, "deleted": deleted}


def render_suspend(slot: str, role: str) -> dict[str, Any]:
    cfg = _service(slot, role)
    before = render_service_status(slot, role)
    if before.get("suspended") == "suspended":
        return {"slot": slot, "role": role, "changed": False, "status": "already_suspended"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}/suspend",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code != 202:
        raise RuntimeError(
            f"Render suspend failed for {slot}/{role}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    deadline = time.time() + 180
    while time.time() < deadline:
        current = render_service_status(slot, role)
        if current.get("suspended") == "suspended":
            return {"slot": slot, "role": role, "changed": True, "status": "suspended"}
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Render service {slot}/{role} to suspend")


def render_resume(slot: str, role: str) -> dict[str, Any]:
    cfg = _service(slot, role)
    before = render_service_status(slot, role)
    if before.get("suspended") == "not_suspended":
        return {"slot": slot, "role": role, "changed": False, "status": "already_running"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}/resume",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code != 202:
        raise RuntimeError(
            f"Render resume failed for {slot}/{role}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    deadline = time.time() + 180
    while time.time() < deadline:
        current = render_service_status(slot, role)
        if current.get("suspended") == "not_suspended":
            return {"slot": slot, "role": role, "changed": True, "status": "running"}
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Render service {slot}/{role} to resume")


def render_trigger_deploy(slot: str, role: str) -> dict[str, Any]:
    cfg = _service(slot, role)
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}/deploys",
            headers=_headers(cfg["api_key"]),
            json={"deployMode": "deploy_only"},
        )
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"Render deploy trigger failed for {slot}/{role}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    payload = response.json()
    deploy_id = payload.get("id") or payload.get("deploy", {}).get("id")
    if not deploy_id:
        raise RuntimeError(f"Render deploy response did not contain deploy id: {payload}")
    return {"slot": slot, "role": role, "deploy_id": deploy_id, "raw": payload}


def render_wait_for_deploy(
    slot: str,
    role: str,
    deploy_id: str,
    timeout_seconds: int = 900,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    cfg = _service(slot, role)
    deadline = time.time() + timeout_seconds
    terminal_failure = {
        "deactivated",
        "build_failed",
        "update_failed",
        "canceled",
        "pre_deploy_failed",
    }
    with httpx.Client(timeout=30.0) as client:
        while time.time() < deadline:
            response = client.get(
                f"{RENDER_API_BASE}/services/{cfg['service_id']}/deploys/{deploy_id}",
                headers=_headers(cfg["api_key"]),
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render deploy status failed for {slot}/{role}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            payload = response.json()
            deploy = payload.get("deploy", payload)
            status = deploy.get("status")
            if status == "live":
                return {"slot": slot, "role": role, "deploy_id": deploy_id, "status": status, "raw": payload}
            if status in terminal_failure:
                raise RuntimeError(
                    f"Render deploy failed slot={slot}/{role} deploy_id={deploy_id} status={status}"
                )
            time.sleep(poll_seconds)
    raise TimeoutError(f"Timed out waiting for Render deploy slot={slot}/{role} deploy_id={deploy_id}")


def render_configure_n8n_runtime(
    slot: str,
    postgres_runtime: dict[str, str],
    b2_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    cfg = _service(slot, "n8n")
    env: dict[str, str] = {}
    for field, env_name in cfg["db_env_map"].items():
        if field in postgres_runtime:
            env[env_name] = str(postgres_runtime[field])
    if b2_runtime:
        for field, env_name in cfg["b2_env_map"].items():
            if field in b2_runtime:
                env[env_name] = str(b2_runtime[field])
    return render_update_env(slot, "n8n", env)


def render_configure_sh01_runtime(
    slot: str,
    b2_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    cfg = _service(slot, "sh01")

    env: dict[str, str] = {}

    if b2_runtime:
        for field, env_name in cfg["b2_env_map"].items():
            if field in b2_runtime:
                env[env_name] = str(b2_runtime[field])

    if not env:
        return {
            "slot": slot,
            "role": "sh01",
            "updated": [],
        }

    return render_update_env(
        slot,
        "sh01",
        env,
    )


def render_deploy_and_health(slot: str, role: str) -> dict[str, Any]:
    resume = render_resume(slot, role)
    deploy = render_trigger_deploy(slot, role)
    status = render_wait_for_deploy(slot, role, deploy["deploy_id"])
    health = None
    for _ in range(24):
        health = render_health(slot, role)
        if health.get("healthy"):
            return {"resume": resume, "deploy": status, "health": health}
        time.sleep(5)
    raise RuntimeError(f"Render service {slot}/{role} deployed but health check failed: {health}")


def render_base_url(slot: str, role: str) -> str:
    return _service(slot, role)["base_url"]
