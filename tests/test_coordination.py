import os
import unittest
from unittest.mock import patch

from infra import coordination


class CoordinatorTests(unittest.TestCase):
    def test_rejected_acquire_cannot_run_mutation(self):
        with patch.dict(os.environ, {"INFRA_COORDINATION_ENABLED": "true"}), \
             patch.object(coordination, "_call", side_effect=RuntimeError("busy")):
            with self.assertRaisesRegex(RuntimeError, "busy"):
                with coordination.mutation_lease({"render": "render-b", "postgres": "supabase-a", "b2": "b2-a"}):
                    self.fail("must not run")

    def test_lost_lease_rejects_mutations(self):
        with patch.dict(os.environ, {"INFRA_COORDINATION_ENABLED": "true"}), \
             patch.object(coordination, "_call", side_effect=lambda action, data=None:
               {"token": "token", "generation": 8} if action == "acquire" else
               (_ for _ in ()).throw(RuntimeError("lost")) if action == "renew" else {}):
            with coordination.mutation_lease({"render": "render-b", "postgres": "supabase-a", "b2": "b2-a"}):
                with self.assertRaisesRegex(RuntimeError, "lost"):
                    coordination.require_lease()


if __name__ == "__main__":
    unittest.main()
