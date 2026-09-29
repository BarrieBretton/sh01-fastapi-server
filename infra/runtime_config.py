import os
from typing import Any

from .postgres import get_postgres_config
from .registry import registry


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def postgres_runtime_config(slot: str) -> dict[str, str]:
    base = get_postgres_config(slot)
    cfg = registry.get("postgres", slot)

    runtime_host = base.host
    if cfg.get("runtime_host_env"):
        runtime_host = _env_value(cfg["runtime_host_env"])

    runtime_port = "6543"
    if cfg.get("runtime_port_env"):
        runtime_port = _env_value(cfg["runtime_port_env"])
    elif cfg.get("runtime_port") is not None:
        runtime_port = str(cfg["runtime_port"])

    return {
        "host": runtime_host,
        "port": runtime_port,
        "database": base.database,
        "user": base.user,
        "password": base.password,
    }


def safe_runtime_summary(slot: str) -> dict[str, Any]:
    cfg = postgres_runtime_config(slot)
    return {
        "slot": slot,
        "host": cfg["host"],
        "port": cfg["port"],
        "database": cfg["database"],
        "user": cfg["user"],
    }
