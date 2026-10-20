"""Readiness gates for a user-provisioned, one-way PostgreSQL subscription.

No subscription is created by an HTTP request. A replica is promotable only
after the operator has bootstrapped a matching schema and a live subscription.
"""

import json
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone

from .postgres import get_postgres_config

_NAME = re.compile(r"[a-z_][a-z0-9_]{0,62}\Z")


def _name(key: str, default: str) -> str:
    value = os.getenv(key, default)
    if not _NAME.fullmatch(value):
        raise RuntimeError(f"{key} must be a lowercase PostgreSQL identifier")
    return value


def _query(slot: str, sql: str) -> dict:
    cfg = get_postgres_config(slot)
    result = subprocess.run(
        ["psql", "-X", "-q", "-v", "ON_ERROR_STOP=1", "--no-password", "-A", "-t",
         "--host", cfg.host, "--port", str(cfg.port), "--username", cfg.user,
         "--dbname", cfg.database, "--command", sql],
        env={**os.environ, "PGPASSWORD": cfg.password}, capture_output=True,
        text=True, timeout=20,
    )
    if result.returncode:
        # psql stderr can include a connection string. Never return it via API.
        raise RuntimeError(f"Database inspection failed on {slot}")
    try:
        value = json.loads(result.stdout.strip())
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid database inspection response on {slot}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Invalid database inspection response on {slot}")
    return value


def inspect_replica(source: str, destination: str) -> dict:
    """Check publication coverage, slot, subscriber, copy state and WAL lag.

    A failed/unreachable source returns no proof of zero data loss. Emergency
    promotion is therefore never authorized by this method.
    """
    publication = _name("PG_REPLICA_PUBLICATION", "sh01_public")
    subscription = _name("PG_REPLICA_SUBSCRIPTION", "sh01_standby")
    slot = _name("PG_REPLICA_SLOT", "sh01_standby")
    out = {"source": source, "destination": destination, "ready": False, "reasons": []}
    if source == destination:
        out["reasons"].append("same_slot")
        return out
    try:
        dst = _query(destination, f"""
        SELECT json_build_object(
          'in_recovery', pg_is_in_recovery(),
          'subscription', (SELECT json_build_object('enabled', subenabled,
             'slot', subslotname, 'publications', subpublications)
             FROM pg_subscription WHERE subname = '{subscription}'),
          'apply', (SELECT json_build_object('pid', pid, 'received', received_lsn::text,
             'latest', latest_end_lsn::text, 'last_message', last_msg_receipt_time,
             'latest_time', latest_end_time)
             FROM pg_stat_subscription WHERE subname = '{subscription}' AND relid IS NULL LIMIT 1),
          'tables', (SELECT count(*) FROM pg_subscription_rel r JOIN pg_subscription s
             ON s.oid = r.srsubid WHERE s.subname = '{subscription}'),
          'unready', (SELECT count(*) FROM pg_subscription_rel r JOIN pg_subscription s
             ON s.oid = r.srsubid WHERE s.subname = '{subscription}' AND r.srsubstate <> 'r')
        );""")
    except Exception:
        out["reasons"].append("destination_unavailable_or_inspection_denied")
        return out
    sub = dst.get("subscription") or {}
    apply = dst.get("apply") or {}
    if not sub.get("enabled") or sub.get("slot") != slot or publication not in (sub.get("publications") or []):
        out["reasons"].append("subscription_missing_disabled_or_mismatched")
    if not apply.get("pid") or not apply.get("latest"):
        out["reasons"].append("subscription_apply_worker_not_running")
    if not dst.get("tables") or dst.get("unready") != 0:
        out["reasons"].append("initial_table_copy_incomplete")
    out["subscriber"] = dst
    try:
        src = _query(source, f"""
        SELECT json_build_object(
          'wal_level', current_setting('wal_level'),
          'current_lsn', pg_current_wal_lsn()::text,
          'slot', (SELECT json_build_object('active', active, 'plugin', plugin,
              'database', database, 'confirmed', confirmed_flush_lsn::text)
              FROM pg_replication_slots WHERE slot_name = '{slot}'),
          'publication', (SELECT pubname FROM pg_publication WHERE pubname = '{publication}'),
          'published_names', (SELECT array_agg(tablename ORDER BY tablename)
              FROM pg_publication_tables WHERE pubname = '{publication}' AND schemaname = 'public'),
          'app_names', (SELECT array_agg(c.relname ORDER BY c.relname)
              FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
              WHERE n.nspname = 'public' AND c.relkind IN ('r','p') AND c.relpersistence='p'),
          'published_tables', (SELECT count(*) FROM pg_publication_tables
              WHERE pubname = '{publication}' AND schemaname = 'public'),
          'app_tables', (SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
              WHERE n.nspname = 'public' AND c.relkind IN ('r','p')
                AND c.relpersistence = 'p'),
          'unidentified_tables', (SELECT count(*) FROM pg_class c JOIN pg_namespace n
              ON n.oid=c.relnamespace WHERE n.nspname='public' AND c.relkind IN ('r','p')
                AND c.relpersistence='p' AND c.relreplident='d' AND NOT EXISTS
                (SELECT 1 FROM pg_index i WHERE i.indrelid=c.oid AND i.indisprimary)),
          'lag_bytes', (SELECT pg_wal_lsn_diff(pg_current_wal_lsn(), confirmed_flush_lsn)::bigint
              FROM pg_replication_slots WHERE slot_name = '{slot}')
        );""")
        out["publisher"] = src
    except Exception:
        out["reasons"].append("source_unavailable_or_inspection_denied")
        return out
    source_slot = src.get("slot") or {}
    if src.get("wal_level") != "logical" or not source_slot.get("active") or source_slot.get("plugin") != "pgoutput":
        out["reasons"].append("logical_slot_missing_or_inactive")
    if (src.get("publication") != publication or
            src.get("published_names") != src.get("app_names") or
            not src.get("app_tables")):
        out["reasons"].append("publication_does_not_cover_all_public_tables")
    if src.get("unidentified_tables") != 0:
        out["reasons"].append("tables_missing_replica_identity")
    # Readiness is only a hint until the post-quiesce write barrier is applied.
    # Latest-end and confirmed-flush LSNs are reported at different stages and
    # should not be compared as strings for exact equality.
    if src.get("lag_bytes") != 0:
        out["reasons"].append("replication_not_caught_up")
    try:
        last = datetime.fromisoformat(str(apply["latest_time"]).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - last).total_seconds()
        out["last_reply_age_seconds"] = round(age, 1)
        if age < 0 or age > int(os.getenv("PG_REPLICA_MAX_REPLY_AGE_SECONDS", "120")):
            out["reasons"].append("subscription_last_reply_stale")
    except (ValueError, KeyError, TypeError):
        out["reasons"].append("subscription_last_reply_unknown")
    out["ready"] = not out["reasons"]
    return out


def assert_replica_ready(source: str, destination: str) -> dict:
    status = inspect_replica(source, destination)
    if not status["ready"]:
        raise RuntimeError("Postgres replica cannot be promoted: " + ", ".join(status["reasons"]))
    return status


def assert_matching_schema(source: str, destination: str) -> None:
    sql = """
    SELECT json_build_object('definition', md5(coalesce(string_agg(definition,E'\\n' ORDER BY definition),'')))
    FROM (
      SELECT format('table:%s:%s:%s:%s:%s',c.relname,a.attname,
         format_type(a.atttypid,a.atttypmod),a.attnotnull,
         coalesce(pg_get_expr(d.adbin,d.adrelid),'')) AS definition
        FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
        JOIN pg_attribute a ON a.attrelid=c.oid AND a.attnum>0 AND NOT a.attisdropped
        LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
        WHERE n.nspname='public' AND c.relkind IN ('r','p')
      UNION ALL
      SELECT 'index:'||pg_get_indexdef(i.indexrelid) FROM pg_index i
        JOIN pg_class c ON c.oid=i.indrelid JOIN pg_namespace n ON n.oid=c.relnamespace
        WHERE n.nspname='public'
      UNION ALL
      SELECT 'constraint:'||pg_get_constraintdef(k.oid) FROM pg_constraint k
        JOIN pg_namespace n ON n.oid=k.connamespace WHERE n.nspname='public'
    ) definitions;
    """
    if _query(source, sql) != _query(destination, sql):
        raise RuntimeError("Standby public schema differs from source; sync DDL first")


def replicate_barrier(source: str, destination: str, timeout_seconds: int = 90) -> dict:
    """Wait until subscriber has applied a source commit after writers stopped.

    Provision public.sh01_replication_barrier on both projects and include it in
    the publication before enabling this. A marker includes all preceding
    public-table writes in this subscription's commit order.
    """
    marker = str(uuid.uuid4())
    _query(source, f"""INSERT INTO public.sh01_replication_barrier(id, marker, written_at)
      VALUES (1, '{marker}'::uuid, clock_timestamp())
      ON CONFLICT (id) DO UPDATE SET marker=EXCLUDED.marker,
         written_at=EXCLUDED.written_at
      RETURNING json_build_object('marker',marker::text);""")
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        replica = inspect_replica(source, destination)
        if not replica["ready"] and any(reason not in {"replication_not_caught_up"}
                                        for reason in replica["reasons"]):
            raise RuntimeError("Standby failed readiness checks: " + ", ".join(replica["reasons"]))
        seen = _query(destination, "SELECT json_build_object('marker', "
                      "(SELECT marker::text FROM public.sh01_replication_barrier WHERE id=1));")
        if seen.get("marker") == marker:
            return {"marker_applied": True, "replica": replica}
        time.sleep(2)
    raise RuntimeError("Standby has not applied the write barrier; retaining old writer")


def prepare_standby_for_writes(destination: str) -> None:
    """Irreversible boundary: stop incoming replication, advance owned sequences.

    Apply only after a barrier and before any application process is resumed.
    Standby schema must be identical to the source (DDL is not replicated).
    """
    subscription = _name("PG_REPLICA_SUBSCRIPTION", "sh01_standby")
    _query(destination, f"ALTER SUBSCRIPTION {subscription} DISABLE; "
          "SELECT json_build_object('disabled',true);")
    _query(destination, """
    DO $do$
    DECLARE r record; highest bigint;
    BEGIN
      FOR r IN SELECT ns.nspname AS schema_name, seq.relname AS sequence_name,
                      tbl.relname AS table_name, att.attname AS column_name
        FROM pg_class seq JOIN pg_namespace ns ON ns.oid=seq.relnamespace
        JOIN pg_depend dep ON dep.objid=seq.oid AND dep.deptype IN ('a','i')
        JOIN pg_class tbl ON tbl.oid=dep.refobjid
        JOIN pg_attribute att ON att.attrelid=tbl.oid AND att.attnum=dep.refobjsubid
        WHERE seq.relkind='S' AND tbl.relnamespace='public'::regnamespace
      LOOP
        EXECUTE format('SELECT max(%I) FROM %I.%I',r.column_name,r.schema_name,r.table_name)
          INTO highest;
        -- Use table maxima; the old sequence's cache/gaps need not be copied.
        EXECUTE format('SELECT setval(%L, greatest(1,%s), true)',
          format('%I.%I',r.schema_name,r.sequence_name),coalesce(highest,0));
      END LOOP;
    END $do$;
    SELECT json_build_object('sequences_advanced',true);
    """)
