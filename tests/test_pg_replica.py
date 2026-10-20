import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from infra import pg_replica


class ReplicaGateTests(unittest.TestCase):
    def inspect(self, *, lag=0, unready=0, source_fails=False):
        def query(slot, sql):
            if slot == "supabase-b":
                return {"subscription": {"enabled": True, "slot": "sh01_standby",
                         "publications": ["sh01_public"]},
                        "apply": {"pid": 1, "latest": "0/5", "latest_time":
                          datetime.now(timezone.utc).isoformat()},
                        "tables": 4, "unready": unready}
            if source_fails:
                raise RuntimeError("database down")
            return {"wal_level": "logical", "current_lsn": "0/5",
                    "slot": {"active": True, "plugin": "pgoutput", "confirmed": "0/5"},
                    "publication": "sh01_public", "published_tables": 4,
                    "app_tables": 4, "published_names": ["a", "b", "c", "d"],
                    "app_names": ["a", "b", "c", "d"], "lag_bytes": lag,
                    "unidentified_tables": 0}
        with patch.object(pg_replica, "_query", side_effect=query):
            return pg_replica.inspect_replica("supabase-a", "supabase-b")

    def test_readiness_requires_all_tables_copied_and_source_online(self):
        self.assertTrue(self.inspect()["ready"])
        self.assertIn("initial_table_copy_incomplete", self.inspect(unready=1)["reasons"])
        self.assertIn("replication_not_caught_up", self.inspect(lag=4)["reasons"])
        self.assertIn("source_unavailable_or_inspection_denied", self.inspect(source_fails=True)["reasons"])

    def test_barrier_never_promotes_without_source_commit(self):
        with patch.object(pg_replica, "_query", side_effect=RuntimeError("down")):
            with self.assertRaises(RuntimeError):
                pg_replica.replicate_barrier("supabase-a", "supabase-b")


if __name__ == "__main__":
    unittest.main()
