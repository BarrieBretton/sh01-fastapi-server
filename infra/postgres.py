import os
import subprocess
from dataclasses import dataclass
from typing import Any

from .registry import registry


@dataclass
class PostgresConfig:
    host: str
    port: int
    user: str
    database: str
    password: str


def _env_value(name: str) -> str:
    value = os.getenv(name, "").strip()

    if not value:
        raise RuntimeError(
            f"Required environment variable is missing: {name}"
        )

    return value


def get_postgres_config(
    slot: str,
) -> PostgresConfig:
    config = registry.get(
        "postgres",
        slot,
    )

    required_keys = {
        "host_env",
        "port_env",
        "user_env",
        "database_env",
        "password_env",
    }

    missing = required_keys - set(config)

    if missing:
        raise RuntimeError(
            f"Postgres slot '{slot}' is missing registry fields: "
            + ", ".join(sorted(missing))
        )

    return PostgresConfig(
        host=_env_value(
            config["host_env"]
        ),
        port=int(
            _env_value(
                config["port_env"]
            )
        ),
        user=_env_value(
            config["user_env"]
        ),
        database=_env_value(
            config["database_env"]
        ),
        password=_env_value(
            config["password_env"]
        ),
    )


def postgres_health(
    slot: str,
) -> dict[str, Any]:
    db = get_postgres_config(slot)

    env = {
        **os.environ,
        "PGPASSWORD": db.password,
    }

    result = subprocess.run(
        [
            "psql",
            "--host", db.host,
            "--port", str(db.port),
            "--username", db.user,
            "--dbname", db.database,
            "--no-password",
            "--tuples-only",
            "--command", "SELECT 1;",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    return {
        "slot": slot,
        "healthy": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def postgres_verify(
    slot: str,
) -> dict[str, Any]:
    db = get_postgres_config(slot)

    env = {
        **os.environ,
        "PGPASSWORD": db.password,
    }

    result = subprocess.run(
        [
            "psql",
            "--host", db.host,
            "--port", str(db.port),
            "--username", db.user,
            "--dbname", db.database,
            "--no-password",
            "--tuples-only",
            "--command",
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public';
            """,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Postgres verification failed for {slot}: "
            f"{result.stderr.strip()}"
        )

    return {
        "slot": slot,
        "public_table_count": int(
            result.stdout.strip() or 0
        ),
    }


def postgres_migrate(
    source_slot: str,
    destination_slot: str,
) -> dict[str, Any]:
    if source_slot == destination_slot:
        raise ValueError(
            "Source and destination Postgres slots must differ"
        )

    source = get_postgres_config(
        source_slot
    )

    destination = get_postgres_config(
        destination_slot
    )

    source_env = {
        **os.environ,
        "PGPASSWORD": source.password,
    }

    destination_env = {
        **os.environ,
        "PGPASSWORD": destination.password,
    }

    dump = subprocess.Popen(
        [
            "pg_dump",
            "--host", source.host,
            "--port", str(source.port),
            "--username", source.user,
            "--dbname", source.database,
            "--format", "custom",
            "--no-owner",
            "--no-privileges",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=source_env,
    )

    assert dump.stdout is not None

    restore = subprocess.Popen(
        [
            "pg_restore",
            "--host", destination.host,
            "--port", str(destination.port),
            "--username", destination.user,
            "--dbname", destination.database,
            "--clean",
            "--if-exists",
            "--no-owner",
            "--no-privileges",
        ],
        stdin=dump.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=destination_env,
    )

    dump.stdout.close()

    restore_stdout, restore_stderr = (
        restore.communicate()
    )

    dump_stderr = dump.stderr.read() if dump.stderr else b""
    dump_returncode = dump.wait()

    if dump_returncode != 0:
        raise RuntimeError(
            "pg_dump failed: "
            + dump_stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    if restore.returncode != 0:
        raise RuntimeError(
            "pg_restore failed: "
            + restore_stderr.decode(
                "utf-8",
                errors="replace",
            )
        )

    return {
        "source": source_slot,
        "destination": destination_slot,
        "dump_returncode": dump_returncode,
        "restore_returncode": restore.returncode,
        "restore_stdout": restore_stdout.decode(
            "utf-8",
            errors="replace",
        ).strip(),
        "restore_stderr": restore_stderr.decode(
            "utf-8",
            errors="replace",
        ).strip(),
    }