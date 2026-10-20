"""Regression checks for the October 17 Render hard-stop incident."""

import importlib.util
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def load_sla():
    """Import the real SLA module with its provider adapters replaced by stubs."""
    root = Path(__file__).resolve().parents[1]
    infra = types.ModuleType("infra")
    infra.__path__ = [str(root / "infra")]
    modules = {"infra": infra}
    attributes = {
        "coordination": ("coordinated", "enabled"),
        "b2ring": ("b2_runtime_config",),
        "orchestrator": ("run_failover",),
        "persistence": ("store",),
        "registry": ("registry",),
        "render_provider": (
            "render_base_url", "render_get_env", "render_health",
            "render_service_status", "render_suspend", "render_resume_blocker",
        ),
        "router_client": (
            "router_set_maintenance", "router_set_preferred", "router_status",
        ),
        "runtime_config": ("postgres_runtime_config",),
        "state": ("state",),
    }
    for name, fields in attributes.items():
        module = types.ModuleType("infra." + name)
        for field in fields:
            setattr(module, field, (lambda fn: fn) if field == "coordinated" else Mock(name=field))
        modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("infra.sla", root / "infra/sla.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["infra.sla"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            del sys.modules["infra.sla"]
    return module


def load_orchestrator():
    """Load failover code with provider adapters, keeping the real preflight."""
    root = Path(__file__).resolve().parents[1]
    infra = types.ModuleType("infra")
    infra.__path__ = [str(root / "infra")]
    modules = {"infra": infra}
    attributes = {
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
    for name, fields in attributes.items():
        module = types.ModuleType("infra." + name)
        for field in fields:
            setattr(module, field, (lambda fn: fn) if field == "coordinated" else Mock(name=field))
        modules[module.__name__] = module
    with patch.dict(sys.modules, modules):
        spec = importlib.util.spec_from_file_location("infra.orchestrator", root / "infra/orchestrator.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["infra.orchestrator"] = module
        try:
            spec.loader.exec_module(module)
        finally:
            del sys.modules["infra.orchestrator"]
    return module


class SlaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.sla = load_sla()
        self.sla.registry.exists.return_value = True
        self.sla.registry.render_slots.return_value = ["render-a", "render-b"]
        self.sla.state.get_active.return_value = "render-b"
        self.sla.state.all.return_value = {"render": "render-b", "postgres": "supabase-b"}
        self.sla._routers_in_maintenance = Mock(return_value={"maintenance": True})
        self.sla._load_monitor_state = Mock(return_value=({"hard_stop": True}, 23))
        self.sla._save_monitor_state = Mock()
        self.sla.render_resume_blocker.return_value = None

    def test_hard_stop_never_suspends_a_running_service(self):
        result = self.sla._hard_stop_render_ring("active_pair_healthcheck_failed")
        self.assertEqual(result["render"]["action"], "left_running_for_recovery")
        self.sla.render_suspend.assert_not_called()
        self.assertEqual(self.sla.router_set_maintenance.call_count, 2)

    def test_healthy_active_pair_reopens_both_routers_and_resets_hard_stop(self):
        self.sla._active_pair_state = Mock(return_value={"healthy": True})
        events = []
        self.sla._align_routers = Mock(side_effect=lambda _: events.append("align"))
        self.sla.router_set_maintenance.side_effect = lambda *args: events.append(args)
        result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "recovered_hard_stop")
        self.assertEqual(events, ["align", ("n8n", False), ("sh01", False)])
        self.assertFalse(self.sla._save_monitor_state.call_args.args[0]["hard_stop"])

    def test_unhealthy_pair_keeps_maintenance_and_rechecks_next_tick(self):
        self.sla._active_pair_state = Mock(side_effect=lambda slot: {
            "healthy": False, "slot": slot,
        })
        result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "recovery_pending")
        self.assertFalse(result["ok"])
        self.assertTrue(self.sla._save_monitor_state.call_args.args[0]["hard_stop"])
        self.sla.router_set_maintenance.assert_not_called()
        self.sla.render_suspend.assert_not_called()

    def test_healthy_configured_standby_can_recover_a_hard_stop(self):
        self.sla._active_pair_state = Mock(side_effect=lambda slot: {
            "healthy": slot == "render-a", "slot": slot,
        })
        self.sla._candidate_runtime_matches = Mock(return_value={"match": True})
        self.sla._promote_live_candidate = Mock(return_value={"changed": True})
        result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "recovered_hard_stop")
        self.assertEqual(result["active_render"], "render-a")
        self.sla._promote_live_candidate.assert_called_once_with("render-b", "render-a")
        self.assertFalse(self.sla._save_monitor_state.call_args.args[0]["hard_stop"])

    def test_failed_router_reopen_restores_both_to_maintenance(self):
        self.sla._active_pair_state = Mock(return_value={"healthy": True})
        self.sla._align_routers = Mock()
        self.sla.router_set_maintenance.side_effect = [
            {}, RuntimeError("sh01 router unavailable"), {}, {},
        ]
        with self.assertLogs("infra.sla", logging.ERROR):
            result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "recovery_failed")
        self.assertEqual(self.sla.router_set_maintenance.call_args_list[-2].args,
                         ("n8n", True))
        self.assertEqual(self.sla.router_set_maintenance.call_args_list[-1].args,
                         ("sh01", True))
        self.sla._save_monitor_state.assert_not_called()

    def test_dry_run_does_not_mutate_hard_stop(self):
        self.sla._active_pair_state = Mock(return_value={"healthy": True})
        result = self.sla.run_sla_tick(dry_run=True)
        self.assertEqual(result["action"], "would_recover_hard_stop")
        self.sla._save_monitor_state.assert_not_called()
        self.sla.router_set_maintenance.assert_not_called()

    def test_failed_promotion_restores_preexisting_maintenance(self):
        self.sla.render_service_status.return_value = {"suspended": "suspended"}
        self.sla.render_base_url.side_effect = lambda slot, role: f"https://{slot}-{role}.example"
        self.sla.router_status.return_value = {"maintenance": True}
        self.sla.router_set_maintenance.side_effect = [
            {}, {}, {}, RuntimeError("could not reopen sh01"), {}, {},
        ]
        self.sla.state.set_active = Mock()
        with self.assertRaisesRegex(RuntimeError, "Fast promotion failed"):
            self.sla._promote_live_candidate("render-b", "render-a")
        self.assertEqual(self.sla.router_set_maintenance.call_args_list[-2].args,
                         ("n8n", True))
        self.assertEqual(self.sla.router_set_maintenance.call_args_list[-1].args,
                         ("sh01", True))

    def test_fast_promotion_rejects_running_old_writer(self):
        self.sla.render_service_status.return_value = {"suspended": "not_suspended"}
        with self.assertRaisesRegex(RuntimeError, "not confirmed suspended"):
            self.sla._promote_live_candidate("render-b", "render-a")
        self.sla.router_set_preferred.assert_not_called()
        self.sla.state.set_active.assert_not_called()

    def test_transient_active_recovery_avoids_hard_stop_after_failed_failover(self):
        self.sla._load_monitor_state.return_value = ({"consecutive_failures": 1}, 23)
        self.sla._routers_in_maintenance.return_value = {"maintenance": False}
        self.sla._active_pair_state = Mock(side_effect=[
            {"healthy": False, "hard_down": False},
            {"healthy": False, "hard_down": False,
             "n8n": {"status": {"suspended": "suspended"}},
             "sh01": {"status": {"suspended": "suspended"}}},
            {"healthy": True, "hard_down": False},
        ])
        self.sla.run_failover.side_effect = RuntimeError("Render refused standby resume")
        self.sla._hard_stop_render_ring = Mock()
        self.sla._align_routers = Mock()
        result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "active_pair_recovered_after_failed_failover")
        self.sla._hard_stop_render_ring.assert_not_called()
        self.sla.render_suspend.assert_not_called()

    def test_unresumable_candidate_is_skipped_before_full_failover(self):
        self.sla._load_monitor_state.return_value = ({"consecutive_failures": 1}, 23)
        self.sla._routers_in_maintenance.return_value = {"maintenance": False}
        self.sla._active_pair_state = Mock(side_effect=[
            {"healthy": False, "hard_down": True},
            {"healthy": False, "hard_down": True,
             "n8n": {"status": {"suspended": "suspended"}},
             "sh01": {"status": {"suspended": "suspended"}}},
            {"healthy": False, "hard_down": True},
        ])
        self.sla.render_resume_blocker.side_effect = lambda status: "suspenders=['admin']"
        result = self.sla.run_sla_tick()
        self.assertEqual(result["action"], "render_ring_failed_completely")
        self.assertIn("cannot resume", result["errors"]["render-a"])
        self.sla.run_failover.assert_not_called()
        self.sla.render_suspend.assert_not_called()


class FailoverPreflightTests(unittest.TestCase):
    def test_render_suspend_reasons_are_checked_before_attempting_resume(self):
        root = Path(__file__).resolve().parents[1]
        infra = types.ModuleType("infra")
        infra.__path__ = [str(root / "infra")]
        registry = types.ModuleType("infra.registry")
        registry.registry = Mock()
        with patch.dict(sys.modules, {"infra": infra, "infra.registry": registry,
                                     "httpx": types.ModuleType("httpx")}):
            spec = importlib.util.spec_from_file_location(
                "infra.render_provider", root / "infra/render_provider.py")
            provider = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(provider)
        self.assertIsNone(provider.render_resume_blocker({"suspended": "not_suspended"}))
        self.assertIsNone(provider.render_resume_blocker(
            {"suspended": "suspended", "raw": {"suspenders": ["user"]}}))
        for suspenders in ([], ["admin"], ["billing"], ["user", "admin"]):
            self.assertIsNotNone(provider.render_resume_blocker(
                {"suspended": "suspended", "raw": {"suspenders": suspenders}}))

    def test_unresumable_target_does_not_quiesce_source_during_preflight(self):
        engine = load_orchestrator()
        engine.build_failover_plan = Mock(return_value={
            "current": {"render": "render-b", "postgres": "db"},
            "target": {"render": "render-a", "postgres": "db", "b2": None},
            "switch_router": True, "quiesce_source": True,
        })
        engine.registry.render_slots.return_value = ["render-a", "render-b"]
        engine.state.all.return_value = {"render": "render-b", "postgres": "db"}
        engine.postgres_health.return_value = {"healthy": True}
        engine.render_service_status.side_effect = lambda slot, role: {
            "suspended": "not_suspended" if slot == "render-b" else "suspended",
            "raw": {"suspenders": []},
        }
        engine.render_resume_blocker.side_effect = lambda status: (
            "suspenders=[]" if status["suspended"] == "suspended" else None)
        engine._suspend_all_running = Mock()
        with self.assertLogs("infra.orchestrator", logging.WARNING):
            with self.assertRaisesRegex(RuntimeError, "not resumable"):
                engine.run_failover(target_render="render-a")
        engine.render_suspend.assert_not_called()
        engine.router_set_maintenance.assert_not_called()
        engine._suspend_all_running.assert_not_called()


if __name__ == "__main__":
    unittest.main()
