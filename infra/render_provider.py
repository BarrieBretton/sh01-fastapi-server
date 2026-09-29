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


def _slot(slot: str) -> dict[str, Any]:
    cfg = registry.get("render", slot)
    required = {"service_id_env", "api_key_env", "base_url_env"}
    missing = required - set(cfg)
    if missing:
        raise RuntimeError(
            f"Render slot '{slot}' is missing registry fields: "
            + ", ".join(sorted(missing))
        )
    return {
        "role": str(cfg.get("role", "n8n")).strip().lower(),
        "service_id": _env_value(cfg["service_id_env"]),
        "api_key": _env_value(cfg["api_key_env"]),
        "base_url": _env_value(cfg["base_url_env"]).rstrip("/"),
        "health_path": cfg.get(
            "health_path",
            "/healthz/readiness" if str(cfg.get("role", "n8n")).lower() == "n8n" else "/",
        ),
        "db_env_map": cfg.get(
            "db_env_map",
            {
                "host": "DB_POSTGRESDB_HOST",
                "port": "DB_POSTGRESDB_PORT",
                "database": "DB_POSTGRESDB_DATABASE",
                "user": "DB_POSTGRESDB_USER",
                "password": "DB_POSTGRESDB_PASSWORD",
            },
        ),
        "b2_env_map": cfg.get(
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


def render_service_status(slot: str) -> dict[str, Any]:
    cfg = _slot(slot)
    with httpx.Client(timeout=30.0) as client:
        response = client.get(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code >= 300:
        raise RuntimeError(
            f"Render service status failed for {slot}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    payload = response.json()
    return {
        "slot": slot,
        "role": cfg["role"],
        "suspended": payload.get("suspended"),
        "name": payload.get("name"),
        "raw": payload,
    }


def render_health(slot: str, timeout: float = 20.0) -> dict[str, Any]:
    cfg = _slot(slot)
    url = cfg["base_url"] + cfg["health_path"]
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
        return {
            "slot": slot,
            "role": cfg["role"],
            "healthy": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "url": url,
            "body": response.text[:500],
        }
    except Exception as exc:
        return {
            "slot": slot,
            "role": cfg["role"],
            "healthy": False,
            "status_code": None,
            "url": url,
            "error": str(exc),
        }


def render_update_env(slot: str, values: dict[str, str]) -> dict[str, Any]:
    cfg = _slot(slot)
    service_id = cfg["service_id"]
    changed: list[str] = []

    with httpx.Client(timeout=30.0) as client:
        for key, value in values.items():
            response = client.put(
                f"{RENDER_API_BASE}/services/{service_id}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
                json={"value": str(value)},
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env update failed for {slot}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            changed.append(key)

    logger.info("Updated Render env for %s keys=%s", slot, changed)
    return {"slot": slot, "updated": changed}

def render_runtime_env_keys(slot: str) -> list[str]:
    cfg = _slot(slot)

    keys = list(cfg["db_env_map"].values())
    keys.extend(cfg["b2_env_map"].values())

    return sorted(set(keys))

def render_get_env(slot: str, keys: list[str]) -> dict[str, str | None]:
    cfg = _slot(slot)
    service_id = cfg["service_id"]
    result: dict[str, str | None] = {}

    with httpx.Client(timeout=30.0) as client:
        for key in keys:
            response = client.get(
                f"{RENDER_API_BASE}/services/{service_id}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
            )

            if response.status_code == 404:
                result[key] = None
                continue

            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env read failed for {slot}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )

            payload = response.json()
            env_var = payload.get("envVar", payload)
            result[key] = env_var.get("value")

    return result


def render_restore_env(slot: str, values: dict[str, str | None]) -> dict[str, Any]:
    cfg = _slot(slot)
    service_id = cfg["service_id"]

    restored: list[str] = []
    deleted: list[str] = []

    with httpx.Client(timeout=30.0) as client:
        for key, value in values.items():
            if value is None:
                response = client.delete(
                    f"{RENDER_API_BASE}/services/{service_id}/env-vars/{key}",
                    headers=_headers(cfg["api_key"]),
                )

                if response.status_code not in (204, 404):
                    raise RuntimeError(
                        f"Render env delete failed for {slot}:{key}: "
                        f"HTTP {response.status_code} {response.text[:1000]}"
                    )

                deleted.append(key)
                continue

            response = client.put(
                f"{RENDER_API_BASE}/services/{service_id}/env-vars/{key}",
                headers=_headers(cfg["api_key"]),
                json={"value": str(value)},
            )

            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render env restore failed for {slot}:{key}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )

            restored.append(key)

    return {
        "slot": slot,
        "restored": restored,
        "deleted": deleted,
    }

def render_suspend(slot: str) -> dict[str, Any]:
    cfg = _slot(slot)
    before = render_service_status(slot)
    if before.get("suspended") == "suspended":
        return {"slot": slot, "changed": False, "status": "already_suspended"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}/suspend",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code != 202:
        raise RuntimeError(
            f"Render suspend failed for {slot}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    logger.info("Suspension requested for Render service %s", slot)
    deadline = time.time() + 180
    while time.time() < deadline:
        current = render_service_status(slot)
        if current.get("suspended") == "suspended":
            return {"slot": slot, "changed": True, "status": "suspended"}
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Render service {slot} to suspend")


def render_resume(slot: str) -> dict[str, Any]:
    cfg = _slot(slot)
    before = render_service_status(slot)
    if before.get("suspended") == "not_suspended":
        return {"slot": slot, "changed": False, "status": "already_running"}
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{cfg['service_id']}/resume",
            headers=_headers(cfg["api_key"]),
        )
    if response.status_code != 202:
        raise RuntimeError(
            f"Render resume failed for {slot}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    logger.info("Resume requested for Render service %s", slot)
    deadline = time.time() + 180
    while time.time() < deadline:
        current = render_service_status(slot)
        if current.get("suspended") == "not_suspended":
            return {"slot": slot, "changed": True, "status": "running"}
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for Render service {slot} to resume")


def render_trigger_deploy(slot: str) -> dict[str, Any]:
    cfg = _slot(slot)
    service_id = cfg["service_id"]
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{RENDER_API_BASE}/services/{service_id}/deploys",
            headers=_headers(cfg["api_key"]),
            # json={"clearCache": "do_not_clear", "deployMode": "deploy_only"},
            json={"deployMode": "deploy_only"},
        )
    if response.status_code not in (200, 201, 202):
        raise RuntimeError(
            f"Render deploy trigger failed for {slot}: "
            f"HTTP {response.status_code} {response.text[:1000]}"
        )
    payload = response.json()
    deploy_id = payload.get("id") or payload.get("deploy", {}).get("id")
    if not deploy_id:
        raise RuntimeError(f"Render deploy response did not contain deploy id: {payload}")
    logger.info("Triggered Render deploy slot=%s deploy_id=%s", slot, deploy_id)
    return {"slot": slot, "deploy_id": deploy_id, "raw": payload}


def render_wait_for_deploy(
    slot: str,
    deploy_id: str,
    timeout_seconds: int = 900,
    poll_seconds: int = 5,
) -> dict[str, Any]:
    cfg = _slot(slot)
    service_id = cfg["service_id"]
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
                f"{RENDER_API_BASE}/services/{service_id}/deploys/{deploy_id}",
                headers=_headers(cfg["api_key"]),
            )
            if response.status_code >= 300:
                raise RuntimeError(
                    f"Render deploy status failed for {slot}: "
                    f"HTTP {response.status_code} {response.text[:1000]}"
                )
            payload = response.json()
            deploy = payload.get("deploy", payload)
            status = deploy.get("status")
            logger.info("Render deploy %s status=%s", deploy_id, status)
            if status == "live":
                return {"slot": slot, "deploy_id": deploy_id, "status": status, "raw": payload}
            if status in terminal_failure:
                raise RuntimeError(
                    f"Render deploy failed slot={slot} deploy_id={deploy_id} status={status}"
                )
            time.sleep(poll_seconds)

    raise TimeoutError(f"Timed out waiting for Render deploy slot={slot} deploy_id={deploy_id}")


def render_configure_runtime(
    slot: str,
    postgres_runtime: dict[str, str],
    b2_runtime: dict[str, str] | None = None,
) -> dict[str, Any]:
    cfg = _slot(slot)
    if cfg["role"] != "n8n":
        raise RuntimeError(
            f"Refusing to write n8n DB/B2 runtime variables to non-n8n Render slot {slot}"
        )
    env: dict[str, str] = {}
    for field, env_name in cfg["db_env_map"].items():
        if field in postgres_runtime:
            env[env_name] = str(postgres_runtime[field])
    if b2_runtime:
        for field, env_name in cfg["b2_env_map"].items():
            if field in b2_runtime:
                env[env_name] = str(b2_runtime[field])
    return render_update_env(slot, env)


def render_deploy_and_health(slot: str) -> dict[str, Any]:
    resume = render_resume(slot)
    deploy = render_trigger_deploy(slot)
    status = render_wait_for_deploy(slot, deploy["deploy_id"])
    health = None
    for _ in range(24):
        health = render_health(slot)
        if health.get("healthy"):
            return {"resume": resume, "deploy": status, "health": health}
        time.sleep(5)
    raise RuntimeError(f"Render service {slot} deployed but health check failed: {health}")


def render_base_url(slot: str) -> str:
    return _slot(slot)["base_url"]


def render_role(slot: str) -> str:
    return _slot(slot)["role"]
