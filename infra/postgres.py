import json
import logging
import os
import subprocess
import tempfile
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


def postgres_public_extensions(
    slot: str,
) -> list[str]:
    """
    Return extensions whose extension schema is public.

    The migration owns/replaces the destination public schema.
    If an extension is installed directly into public, dropping
    public could destroy extension-managed objects, so migration
    is refused in that case.
    """

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
            SELECT e.extname
            FROM pg_extension e
            JOIN pg_namespace n
              ON n.oid = e.extnamespace
            WHERE n.nspname = 'public'
            ORDER BY e.extname;
            """,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not inspect public-schema extensions for {slot}: "
            f"{result.stderr.strip()}"
        )

    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


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
        "Starting transactional postgres migration: %s -> %s",
        source_slot,
        destination_slot,
    )

    # ---------------------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------------------
    #
    # The ring owns/replaces the n8n application objects in
    # public. Do not destroy extension-managed objects that may
    # happen to have been installed directly into public.
    #
    destination_extensions = postgres_public_extensions(
        destination_slot
    )

    if destination_extensions:
        raise RuntimeError(
            "Destination public schema contains extension-managed "
            "objects. Refusing destructive replacement. Extensions: "
            + ", ".join(destination_extensions)
        )

    logger.info(
        "Destination public schema extension check passed"
    )

    # ---------------------------------------------------------
    # DUMP + RESTORE
    # ---------------------------------------------------------
    #
    # Temporary dump files live only for the duration of this
    # migration and are automatically deleted.
    #
    # We intentionally use separate schema and data dumps so the
    # destination can be recreated deterministically and restored
    # inside one destination transaction.
    #
    with tempfile.TemporaryDirectory(
        prefix="postgres-ring-migration-"
    ) as tmpdir:

        schema_file = os.path.join(
            tmpdir,
            "schema.sql",
        )

        data_file = os.path.join(
            tmpdir,
            "data.sql",
        )

        # -----------------------------------------------------
        # PHASE 1: SOURCE SCHEMA DUMP
        # -----------------------------------------------------

        logger.info(
            "Dumping source public schema"
        )

        schema_dump = subprocess.run(
            [
                "pg_dump",
                "--host", source.host,
                "--port", str(source.port),
                "--username", source.user,
                "--dbname", source.database,
                "--no-password",

                "--format", "plain",
                "--schema", "public",
                "--schema-only",

                "--no-owner",
                "--no-privileges",

                "--file", schema_file,
            ],
            env=source_env,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if schema_dump.returncode != 0:
            logger.error(
                "Schema dump failed rc=%s stderr=%r",
                schema_dump.returncode,
                schema_dump.stderr,
            )

            raise RuntimeError(
                "Source schema dump failed: "
                + schema_dump.stderr.strip()
            )

        schema_size = os.path.getsize(
            schema_file
        )

        logger.info(
            "Source schema dump completed: %s bytes",
            schema_size,
        )

        # -----------------------------------------------------
        # PHASE 2: SOURCE DATA DUMP
        # -----------------------------------------------------

        logger.info(
            "Dumping source public data"
        )

        data_dump = subprocess.run(
            [
                "pg_dump",
                "--host", source.host,
                "--port", str(source.port),
                "--username", source.user,
                "--dbname", source.database,
                "--no-password",

                "--format", "plain",
                "--schema", "public",
                "--data-only",

                "--no-owner",
                "--no-privileges",

                "--file", data_file,
            ],
            env=source_env,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if data_dump.returncode != 0:
            logger.error(
                "Data dump failed rc=%s stderr=%r",
                data_dump.returncode,
                data_dump.stderr,
            )

            raise RuntimeError(
                "Source data dump failed: "
                + data_dump.stderr.strip()
            )

        data_size = os.path.getsize(
            data_file
        )

        logger.info(
            "Source data dump completed: %s bytes",
            data_size,
        )

        # -----------------------------------------------------
        # PHASE 3: TRANSACTIONAL DESTINATION REPLACEMENT
        # -----------------------------------------------------
        #
        # psql --single-transaction wraps all command/file
        # operations below in one transaction.
        #
        # If schema or data restoration fails, ON_ERROR_STOP
        # causes psql to fail and PostgreSQL rolls the transaction
        # back instead of leaving the destination half-restored.
        #
        # session_replication_role=replica suppresses trigger/FK
        # execution while inserting the already-consistent source
        # data.
        #

        logger.info(
            "Beginning transactional restore into %s",
            destination_slot,
        )

        restore = subprocess.run(
            [
                "psql",
                "--host", destination.host,
                "--port", str(destination.port),
                "--username", destination.user,
                "--dbname", destination.database,
                "--no-password",

                "--single-transaction",
                "--set", "ON_ERROR_STOP=1",

                "--command",
                "DROP SCHEMA IF EXISTS public CASCADE;",

                "--file",
                schema_file,

                "--command",
                "SET session_replication_role = replica;",

                "--file",
                data_file,
            ],
            env=destination_env,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if restore.returncode != 0:
            logger.error(
                "Transactional restore failed: "
                "rc=%s stdout=%r stderr=%r",
                restore.returncode,
                restore.stdout,
                restore.stderr,
            )

            raise RuntimeError(
                "Transactional destination restore failed. "
                f"returncode={restore.returncode}; "
                f"stderr={restore.stderr.strip()!r}; "
                f"stdout={restore.stdout.strip()!r}"
            )

        logger.info(
            "Transactional restore completed successfully"
        )

    # Temporary files have now been automatically deleted.

    verification = postgres_verify(
        destination_slot
    )

    logger.info(
        "Migration verification completed: %s",
        verification,
    )

    logger.info(
        "Postgres migration completed successfully: %s -> %s",
        source_slot,
        destination_slot,
    )

    return {
        "source": source_slot,
        "destination": destination_slot,
        "schema_dump_bytes": schema_size,
        "data_dump_bytes": data_size,
        "restore_returncode": restore.returncode,
        "verification": verification,
    }