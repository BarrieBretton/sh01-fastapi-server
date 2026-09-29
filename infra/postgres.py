import os
import logging
import subprocess
from dataclasses import dataclass
from typing import Any

from .registry import registry

logger = logging.getLogger("infra.postgres")

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

    sql = r"""
    WITH public_tables AS (
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
          AND table_type = 'BASE TABLE'
    ),
    key_tables AS (
        SELECT
            t.table_name,
            CASE
                WHEN to_regclass('public."' || t.table_name || '"') IS NOT NULL
                    THEN true
                ELSE false
            END AS exists
        FROM (
            VALUES
                ('workflow_entity'),
                ('credentials_entity'),
                ('execution_entity')
        ) AS t(table_name)
    )
    SELECT json_build_object(
        'public_table_count',
        (SELECT COUNT(*) FROM public_tables),
        'key_tables',
        (
            SELECT json_object_agg(table_name, exists)
            FROM key_tables
        )
    );
    """

    result = subprocess.run(
        [
            "psql",
            "--host", db.host,
            "--port", str(db.port),
            "--username", db.user,
            "--dbname", db.database,
            "--no-password",
            "--tuples-only",
            "--no-align",
            "--command", sql,
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

    import json

    raw = result.stdout.strip()

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Could not parse verification output for {slot}: {raw}"
        ) from exc

    return {
        "slot": slot,
        **payload,
    }


def postgres_public_table_count(
    slot: str,
) -> int:
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
            "--no-align",
            "--command",
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'public'
              AND table_type = 'BASE TABLE';
            """,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect destination database {slot}: "
            f"{result.stderr.strip()}"
        )

    return int(result.stdout.strip() or 0)

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

    logger.info(
        "Starting postgres migration: %s -> %s",
        source_slot,
        destination_slot,
    )

    dump = subprocess.Popen(
        [
            "pg_dump",
            "--host", source.host,
            "--port", str(source.port),
            "--username", source.user,
            "--dbname", source.database,
            "--format", "custom",
            "--schema", "public",
            "--no-owner",
            "--no-privileges",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=source_env,
    )

    logger.info(
        "pg_dump started pid=%s",
        dump.pid,
    )

    assert dump.stdout is not None

    destination_table_count = postgres_public_table_count(
        destination_slot
    )

    logger.info(
        "Destination %s currently has %s public tables",
        destination_slot,
        destination_table_count,
    )

    restore_command = [
        "pg_restore",
        "--host", destination.host,
        "--port", str(destination.port),
        "--username", destination.user,
        "--dbname", destination.database,
        "--no-owner",
        "--no-privileges",
        "--exit-on-error",
        "--verbose",
    ]

    # An empty destination must not use --clean because archive cleanup
    # can reference parent relations that do not exist yet.
    if destination_table_count > 0:
        restore_command.extend([
            "--clean",
            "--if-exists",
        ])

    logger.info(
        "pg_restore clean mode=%s",
        destination_table_count > 0,
    )

    restore = subprocess.Popen(
        restore_command,
        stdin=dump.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=destination_env,
    )

    logger.info(
        "pg_restore started pid=%s",
        restore.pid,
    )

    dump.stdout.close()

    logger.info("Waiting for pg_restore to finish")

    restore_stdout, restore_stderr = restore.communicate()

    logger.info(
        "pg_restore finished returncode=%s",
        restore.returncode,
    )

    dump_stderr = dump.stderr.read() if dump.stderr else b""
    dump_returncode = dump.wait()

    logger.info(
        "pg_dump finished returncode=%s",
        dump_returncode,
    )

    restore_stdout_text = restore_stdout.decode(
        "utf-8",
        errors="replace",
    ).strip()

    restore_stderr_text = restore_stderr.decode(
        "utf-8",
        errors="replace",
    ).strip()

    dump_stderr_text = dump_stderr.decode(
        "utf-8",
        errors="replace",
    ).strip()

    # IMPORTANT:
    # If pg_restore exits early, pg_dump may subsequently fail because
    # the stdout pipe was closed. Therefore report the restore error first.
    if restore.returncode != 0:
        logger.error(
            "Migration failed: "
            "restore_rc=%s restore_stderr=%r "
            "dump_rc=%s dump_stderr=%r",
            restore.returncode,
            restore_stderr_text,
            dump_returncode,
            dump_stderr_text,
        )

        raise RuntimeError(
            "Postgres migration failed. "
            f"pg_restore returncode={restore.returncode}; "
            f"pg_restore stderr={restore_stderr_text!r}; "
            f"pg_dump returncode={dump_returncode}; "
            f"pg_dump stderr={dump_stderr_text!r}"
        )

    if dump_returncode != 0:
        logger.error(
            "Migration failed: "
            "dump_rc=%s dump_stderr=%r "
            "restore_rc=%s restore_stderr=%r",
            dump_returncode,
            dump_stderr_text,
            restore.returncode,
            restore_stderr_text,
        )

        raise RuntimeError(
            "Postgres migration failed. "
            f"pg_dump returncode={dump_returncode}; "
            f"pg_dump stderr={dump_stderr_text!r}; "
            f"pg_restore returncode={restore.returncode}; "
            f"pg_restore stderr={restore_stderr_text!r}"
        )

    logger.info(
        "Postgres migration completed successfully: %s -> %s",
        source_slot,
        destination_slot,
    )

    return {
        "source": source_slot,
        "destination": destination_slot,
        "dump_returncode": dump_returncode,
        "restore_returncode": restore.returncode,
        "restore_stdout": restore_stdout_text,
        "restore_stderr": restore_stderr_text,
        "dump_stderr": dump_stderr_text,
    }
