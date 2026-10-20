"""Prepare a *new empty* Supabase standby for one-way n8n logical replication.

Run with --check first. --apply creates only in the chosen source and EMPTY
destination; never use an existing project with data to preserve. Both projects
must be backed up before use. Run from a host that reaches both database ports;
the subscriber must itself be able to reach the publisher directly.
"""

import argparse
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from infra.pg_replica import _name  # noqa: E402
from infra.postgres import get_postgres_config  # noqa: E402


def sql_literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def conninfo_value(value):
    # PostgreSQL libpq single-quoted conninfo field, *then* SQL literal.
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def execute(cfg, sql, *, timeout=60):
    result = subprocess.run(
        ["psql", "-X", "-q", "-A", "-t", "-v", "ON_ERROR_STOP=1",
         "--no-password", "-h", cfg.host, "-p", str(cfg.port),
         "-U", cfg.user, "-d", cfg.database],
        input=sql, text=True, capture_output=True, timeout=timeout,
        env={**os.environ, "PGPASSWORD": cfg.password},
    )
    if result.returncode:
        # psql may echo conninfo (including its password) on errors.
        raise RuntimeError("Database operation failed; inspect local protected logs without sharing secrets")
    return result.stdout.strip()


def run(source_slot, target_slot, apply):
    if source_slot == target_slot:
        raise RuntimeError("Source and target must be different registered projects")
    source, target = get_postgres_config(source_slot), get_postgres_config(target_slot)
    publication = _name("PG_REPLICA_PUBLICATION", "sh01_public")
    subscription = _name("PG_REPLICA_SUBSCRIPTION", "sh01_standby")
    slot = _name("PG_REPLICA_SLOT", "sh01_standby")
    if any(c.port != 5432 for c in (source, target)):
        raise RuntimeError("Use direct/session connections on port 5432 for replication")
    source_info = execute(source, """SELECT count(*) FROM pg_class c JOIN pg_namespace n
        ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p');""")
    target_info = execute(target, """SELECT count(*) FROM pg_class c JOIN pg_namespace n
        ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p');""")
    ext = execute(target, "SELECT count(*) FROM pg_extension WHERE extnamespace='public'::regnamespace;")
    if not int(source_info) or int(target_info) or int(ext):
        raise RuntimeError("Source must have public tables and target public schema must be empty and extension-free")
    if execute(source, "SELECT current_setting('wal_level');") != "logical":
        raise RuntimeError("Source wal_level is not logical")
    unidentified = execute(source, """SELECT count(*) FROM pg_class c JOIN pg_namespace n
        ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p')
        AND c.relpersistence='p' AND c.relreplident='d' AND NOT EXISTS
        (SELECT 1 FROM pg_index i WHERE i.indrelid=c.oid AND i.indisprimary);""")
    if int(unidentified):
        raise RuntimeError(f"{unidentified} source tables need a primary key or replica identity before replication")
    if execute(source, f"SELECT count(*) FROM pg_publication WHERE pubname={sql_literal(publication)};") != "0":
        raise RuntimeError("Publication already exists; inspect it before rerunning")
    if execute(target, f"SELECT count(*) FROM pg_subscription WHERE subname={sql_literal(subscription)};") != "0":
        raise RuntimeError("Subscription already exists; inspect it before rerunning")
    print(f"Source {source_slot}: {source_info} public tables; destination {target_slot}: empty")
    if not apply:
        print("Read-only check passed. Apply only after confirming backups and target disposal.")
        return

    execute(source, """CREATE TABLE IF NOT EXISTS public.sh01_replication_barrier
        (id integer PRIMARY KEY CHECK (id=1), marker uuid NOT NULL,
         written_at timestamptz NOT NULL);""")
    cmd = ["pg_dump", "--schema-only", "--schema=public", "--no-owner", "--no-privileges",
           "--no-password", "-h", source.host, "-p", str(source.port),
           "-U", source.user, "-d", source.database]
    dump = subprocess.run(cmd, text=True, capture_output=True, timeout=300,
                          env={**os.environ, "PGPASSWORD": source.password})
    if dump.returncode:
        raise RuntimeError("Public schema dump failed; inspect protected local logs")
    schema = re.sub(r"(?m)^CREATE SCHEMA public;$", "CREATE SCHEMA IF NOT EXISTS public;", dump.stdout)
    # Abort if the dump contained unexpected schema alterations or a \connect.
    if re.search(r"(?im)^\\connect\b|^DROP\b", schema):
        raise RuntimeError("Schema dump contains unsupported commands; aborting")
    execute(target, schema, timeout=300)
    execute(source, f"""CREATE PUBLICATION {publication};
        DO $do$ DECLARE tbl record; BEGIN
          FOR tbl IN SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname='public' AND c.relkind IN ('r','p') AND c.relpersistence='p'
          LOOP EXECUTE format('ALTER PUBLICATION %I ADD TABLE public.%I','{publication}',tbl.relname);
          END LOOP;
        END $do$;""")
    # This connection secret is stored in pg_subscription by PostgreSQL.
    # Supply SQL on stdin, never in process arguments or printed output.
    fields = {"host": source.host, "port": str(source.port), "user": source.user,
              "password": source.password, "dbname": source.database,
              "sslmode": "require"}
    conninfo = " ".join(k + "=" + conninfo_value(v) for k, v in fields.items())
    execute(target, f"CREATE SUBSCRIPTION {subscription} CONNECTION {sql_literal(conninfo)} "
            f"PUBLICATION {publication} WITH (copy_data=true,create_slot=true,slot_name={sql_literal(slot)});",
            timeout=120)
    print("Subscription created; wait until /infra/postgres/replica-status reports ready."
          " Confirm schema parity and rehearsal before enabling switching.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="registered source slot")
    parser.add_argument("--destination", required=True, help="registered EMPTY target slot")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        run(args.source, args.destination, args.apply)
    except Exception as exc:
        print(f"Standby setup stopped: {exc}", file=sys.stderr)
        sys.exit(1)
