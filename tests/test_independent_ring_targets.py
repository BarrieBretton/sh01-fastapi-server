"""Keep a Render failover from silently advancing independent storage rings."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_orchestrator():
    root = Path(__file__).resolve().parents[1]
    infra = types.ModuleType("infra")
    infra.__path__ = [str(root / "infra")]
    modules = {"infra": infra}
    dependencies = {
        "coordination": ("coordinated", "enabled"),
        "pg_replica": ("assert_matching_schema", "inspect_replica", "prepare_standby_for_writes", "replicate_barrier"),
        "b2ring": ("b2_compare", "b2_health", "b2_migrate", "b2_runtime_config"),
        "postgres": ("postgres_compare", "postgres_health", "postgres_migrate"),
        "registry": ("registry",),
        "render_provider": (
            "render_base_url", "render_configure_n8n_runtime",
            "render_configure_sh01_runtime", "render_deploy_and_health",
            "render_get_env", "render_health", "render_restore_env",
            "render_runtime_env_keys", "render_service_status", "render_suspend",
            "render_resume", "render_resume_blocker",
        ),
        "router_client": ("router_set_maintenance", "router_set_preferred", "router_status"),
        "runtime_config": ("postgres_runtime_config", "safe_runtime_summary"),
        "state": ("state",),
    }
    for name, fields in dependencies.items():
        module = types.ModuleType("infra." + name)
        for field in fields:
            setattr(module, field, (lambda fn: fn) if field == "coordinated" else Mock(name=field))
        modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location(
            "infra.orchestrator", root / "infra/orchestrator.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["infra.orchestrator"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            del sys.modules["infra.orchestrator"]
    module.state.all.return_value = {
        "render": "render-b", "postgres": "supabase-b", "b2": "b2-a",
    }
    module.state.refresh_authoritative.side_effect = module.state.all
    module.registry.exists.return_value = True
    return module


class IndependentRingTargetTests(unittest.TestCase):
    def test_render_target_does_not_implicitly_change_storage(self):
        engine = load_orchestrator()
        target = engine.build_failover_plan(target_render="render-a")["target"]
        self.assertEqual(target, {
            "render": "render-a", "postgres": "supabase-b", "b2": "b2-a",
        })

    def test_explicit_storage_target_does_not_implicitly_change_render(self):
        engine = load_orchestrator()
        target = engine.build_failover_plan(
            target_b2="b2-b", target_postgres="supabase-a")["target"]
        self.assertEqual(target, {
            "render": "render-b", "postgres": "supabase-a", "b2": "b2-b",
        })

    def test_no_target_means_no_ring_advances(self):
        engine = load_orchestrator()
        self.assertEqual(engine.build_failover_plan()["target"], engine.state.all())

    def test_live_empty_request_does_not_quiesce_serving_pair(self):
        engine = load_orchestrator()
        with self.assertRaisesRegex(RuntimeError, "explicit target"):
            engine.run_failover()
        engine.registry.render_service.assert_not_called()

    def test_explicit_same_slot_allows_deliberate_reconfigure(self):
        engine = load_orchestrator()
        plan = engine.build_failover_plan(target_render="render-b")
        self.assertEqual(plan["target"], engine.state.all())

    def test_lost_source_database_cannot_be_copied_by_planned_failover(self):
        engine = load_orchestrator()
        engine.postgres_health.side_effect = lambda slot: {"healthy": slot != "supabase-b"}
        with self.assertRaisesRegex(RuntimeError, "Source Postgres unhealthy"):
            engine.run_failover(target_postgres="supabase-a", target_render="render-a")
        engine.postgres_migrate.assert_not_called()
        engine.render_suspend.assert_not_called()

    def test_unknown_render_state_cannot_count_as_quiesced(self):
        engine = load_orchestrator()
        with self.assertRaisesRegex(RuntimeError, "Cannot prove"):
            engine._suspend_all_running("n8n", {"render-b": {"suspended": "unknown"}})
        engine.render_suspend.assert_not_called()

    def test_b2_preflight_blocks_before_any_database_promotion(self):
        engine = load_orchestrator()
        engine.postgres_health.return_value = {"healthy": True}
        engine.coordination.enabled.return_value = True
        engine.inspect_replica.return_value = {"reasons": []}
        engine.b2_health.return_value = {"healthy": False}
        with self.assertRaisesRegex(RuntimeError, "Target B2 unhealthy"):
            engine.run_failover(target_postgres="supabase-a")
        engine.prepare_standby_for_writes.assert_not_called()
        engine.router_set_maintenance.assert_not_called()

    def test_failed_promotion_does_not_reopen_old_database(self):
        engine = load_orchestrator()
        engine.coordination.enabled.return_value = True
        engine.registry.render_slots.return_value = ["render-b"]
        engine._suspend_all_running = Mock(return_value={})
        engine.postgres_health.return_value = {"healthy": True}
        engine.render_service_status.return_value = {"suspended": "not_suspended"}
        engine.render_health.return_value = {"healthy": True}
        engine.render_resume_blocker.return_value = None
        engine.router_status.return_value = {"maintenance": False}
        engine.inspect_replica.return_value = {"reasons": []}
        engine.replicate_barrier.return_value = {"marker_applied": True}
        engine.b2_health.return_value = {"healthy": True}
        engine.postgres_runtime_config.return_value = {}
        engine.render_configure_n8n_runtime.side_effect = RuntimeError("new env unavailable")
        with self.assertRaisesRegex(RuntimeError, "new env unavailable"):
            engine.run_failover(target_postgres="supabase-a")
        engine.prepare_standby_for_writes.assert_called_once_with("supabase-a")
        engine.state.set_all.assert_not_called()
        engine.router_set_preferred.assert_not_called()
        engine.render_resume.assert_not_called()
        # No call should open ingress after a one-way database promotion.
        self.assertFalse(any(call.args[1] is False
          for call in engine.router_set_maintenance.call_args_list))


if __name__ == "__main__":
    unittest.main()
