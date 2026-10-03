import base64
import json
import logging
import os
import subprocess
from typing import Any

from .registry import registry

logger = logging.getLogger("infra.persistence")


_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS control_plane;

CREATE TABLE IF NOT EXISTS control_plane.state (
    key TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    revision BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS control_plane.jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB,
    result JSONB,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def _slot_config(slot: str) -> dict[str, Any]:
    cfg = registry.get("postgres", slot)
    required = {
        "host_env",
        "port_env",
        "user_env",
        "database_env",
        "password_env",
    }
    missing = required - set(cfg)
    if missing:
        raise RuntimeError(
            f"Postgres slot '{slot}' is missing registry fields: "
            + ", ".join(sorted(missing))
        )
    return {
        "host": _env_value(cfg["host_env"]),
        "port": int(_env_value(cfg["port_env"])),
        "user": _env_value(cfg["user_env"]),
        "database": _env_value(cfg["database_env"]),
        "password": _env_value(cfg["password_env"]),
    }


def _run_psql(slot: str, sql: str, timeout: int = 30) -> subprocess.CompletedProcess[str]:
    db = _slot_config(slot)
    env = {**os.environ, "PGPASSWORD": db["password"]}
    return subprocess.run(
        [
            "psql",
            "--host", db["host"],
            "--port", str(db["port"]),
            "--username", db["user"],
            "--dbname", db["database"],
            "--no-password",
            "--tuples-only",
            "--no-align",
            "--command", sql,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _json_sql(value: Any) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"convert_from(decode('{encoded}','base64'),'UTF8')::jsonb"


class DurableStore:
    def __init__(self) -> None:
        self._initialized: set[str] = set()

    def _healthy_slots(self) -> list[str]:
        healthy: list[str] = []
        for slot in registry.slots("postgres"):
            try:
                result = _run_psql(slot, "SELECT 1;", timeout=10)
                if result.returncode == 0:
                    healthy.append(slot)
            except Exception as exc:
                logger.warning("Durable store health check failed for %s: %s", slot, exc)
        return healthy

    def ensure(self, slot: str) -> None:
        if slot in self._initialized:
            return
        result = _run_psql(slot, _SCHEMA_SQL, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Could not initialize durable control-plane store on {slot}: "
                f"{result.stderr.strip()}"
            )
        self._initialized.add(slot)

    def load_state(self, key: str) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        for slot in self._healthy_slots():
            try:
                self.ensure(slot)
                safe_key = key.replace("'", "''")
                result = _run_psql(
                    slot,
                    (
                        "SELECT json_build_object(" 
                        "'value', value, 'revision', revision, 'updated_at', updated_at) "
                        f"FROM control_plane.state WHERE key='{safe_key}';"
                    ),
                    timeout=20,
                )
                if result.returncode != 0 or not result.stdout.strip():
                    continue
                payload = json.loads(result.stdout.strip())
                if best is None or int(payload["revision"]) > int(best["revision"]):
                    best = payload
            except Exception as exc:
                logger.warning("Failed loading state %s from %s: %s", key, slot, exc)
        return best

    def save_state(self, key: str, value: Any, revision: int) -> dict[str, Any]:
        safe_key = key.replace("'", "''")
        value_sql = _json_sql(value)
        successes: list[str] = []
        failures: dict[str, str] = {}

        for slot in self._healthy_slots():
            try:
                self.ensure(slot)
                sql = f"""
                INSERT INTO control_plane.state(key, value, revision, updated_at)
                VALUES ('{safe_key}', {value_sql}, {int(revision)}, NOW())
                ON CONFLICT (key) DO UPDATE SET
                    value = EXCLUDED.value,
                    revision = EXCLUDED.revision,
                    updated_at = NOW()
                WHERE control_plane.state.revision <= EXCLUDED.revision;
                """
                result = _run_psql(slot, sql, timeout=30)
                if result.returncode == 0:
                    successes.append(slot)
                else:
                    failures[slot] = result.stderr.strip()
            except Exception as exc:
                failures[slot] = str(exc)

        if not successes:
            raise RuntimeError(
                "Could not persist control-plane state to any healthy Postgres slot. "
                + json.dumps(failures, sort_keys=True)
            )

        return {"persisted_to": successes, "failures": failures}

    def put_job(
        self,
        job_id: str,
        kind: str,
        status: str,
        payload: Any = None,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        safe_id = job_id.replace("'", "''")
        safe_kind = kind.replace("'", "''")
        safe_status = status.replace("'", "''")
        safe_error = (error or "").replace("'", "''")
        payload_sql = "NULL" if payload is None else _json_sql(payload)
        result_sql = "NULL" if result is None else _json_sql(result)
        error_sql = "NULL" if error is None else f"'{safe_error}'"

        sql = f"""
        INSERT INTO control_plane.jobs(id, kind, status, payload, result, error, created_at, updated_at)
        VALUES ('{safe_id}', '{safe_kind}', '{safe_status}', {payload_sql}, {result_sql}, {error_sql}, NOW(), NOW())
        ON CONFLICT (id) DO UPDATE SET
            kind = EXCLUDED.kind,
            status = EXCLUDED.status,
            payload = COALESCE(EXCLUDED.payload, control_plane.jobs.payload),
            result = EXCLUDED.result,
            error = EXCLUDED.error,
            updated_at = NOW();
        """

        written = False
        for slot in self._healthy_slots():
            try:
                self.ensure(slot)
                r = _run_psql(slot, sql, timeout=30)
                if r.returncode == 0:
                    written = True
                else:
                    logger.warning("Failed persisting job %s to %s: %s", job_id, slot, r.stderr.strip())
            except Exception as exc:
                logger.warning("Failed persisting job %s to %s: %s", job_id, slot, exc)

        if not written:
            message = f"Job {job_id} could not be persisted to any Postgres slot"
            logger.error(message)
            raise RuntimeError(message)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        safe_id = job_id.replace("'", "''")
        for slot in self._healthy_slots():
            try:
                self.ensure(slot)
                r = _run_psql(
                    slot,
                    (
                        "SELECT json_build_object(" 
                        "'id',id,'kind',kind,'status',status,'payload',payload,'result',result," 
                        "'error',error,'created_at',created_at,'updated_at',updated_at) "
                        f"FROM control_plane.jobs WHERE id='{safe_id}';"
                    ),
                    timeout=20,
                )
                if r.returncode == 0 and r.stdout.strip():
                    return json.loads(r.stdout.strip())
            except Exception as exc:
                logger.warning("Failed reading job %s from %s: %s", job_id, slot, exc)
        return None


store = DurableStore()
