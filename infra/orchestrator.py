import logging
import threading
from typing import Any

from .b2ring import b2_compare, b2_health, b2_migrate, b2_runtime_config
from .postgres import postgres_compare, postgres_health, postgres_migrate
from .registry import registry
from .render_provider import (
    render_base_url,
    render_configure_runtime,
    render_deploy_and_health,
    render_get_env,
    render_health,
    render_restore_env,
    render_role,
    render_runtime_env_keys,
    render_resume,
    render_service_status,
    render_suspend,
)
from .router_client import (
    router_set_maintenance,
    router_set_preferred,
    router_status,
)
from .runtime_config import postgres_runtime_config, safe_runtime_summary
from .state import state

logger = logging.getLogger("infra.orchestrator")
_FAILOVER_LOCK = threading.Lock()


def _next_slot(kind: str, current: str | None, render_role_name: str | None = None) -> str | None:
    if kind == "render":
        slots = registry.render_slots(render_role_name)
    else:
        slots = registry.slots(kind)
    if not slots:
        return None
    if current not in slots:
        return slots[0]
    if len(slots) == 1:
        return current
    return slots[(slots.index(current) + 1) % len(slots)]


def _n8n_service_snapshot() -> dict[str, dict[str, Any]]:
    """Return the Render service status for every registered n8n slot."""
    snapshot: dict[str, dict[str, Any]] = {}
    for slot in registry.render_slots("n8n"):
        snapshot[slot] = render_service_status(slot)
    return snapshot


def _running_slots(snapshot: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        slot
        for slot, status in snapshot.items()
        if status.get("suspended") == "not_suspended"
    )


def _suspend_all_running_n8n(snapshot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Quiesce every n8n Render writer before taking a Postgres snapshot."""
    running_before = _running_slots(snapshot)
    results: dict[str, Any] = {}
    for slot in running_before:
        results[slot] = render_suspend(slot)

    # Verify the invariant we actually care about: no registered n8n service is running.
    after = _n8n_service_snapshot()
    still_running = _running_slots(after)
    if still_running:
        raise RuntimeError(
            "Failed to quiesce all n8n Render slots; still running: "
            + ", ".join(still_running)
        )

    return {
        "running_before": running_before,
        "suspend_results": results,
        "service_status_after": after,
    }


def _restore_n8n_running_set(running_before: list[str]) -> dict[str, Any]:
    """Restore the exact pre-failover running/suspended topology on rollback."""
    expected_running = set(running_before)
    current = _n8n_service_snapshot()
    actions: dict[str, Any] = {}

    for slot in registry.render_slots("n8n"):
        is_running = current[slot].get("suspended") == "not_suspended"
        should_run = slot in expected_running
        if should_run and not is_running:
            actions[slot] = {"action": "resume", "result": render_resume(slot)}
        elif not should_run and is_running:
            actions[slot] = {"action": "suspend", "result": render_suspend(slot)}
        else:
            actions[slot] = {
                "action": "none",
                "status": "running" if is_running else "suspended",
            }

    final = _n8n_service_snapshot()
    final_running = set(_running_slots(final))
    if final_running != expected_running:
        raise RuntimeError(
            "Rollback could not restore original n8n running set; "
            f"expected={sorted(expected_running)} actual={sorted(final_running)}"
        )

    return {
        "expected_running": sorted(expected_running),
        "actions": actions,
        "service_status_after": final,
    }


def build_failover_plan(
    target_postgres: str | None = None,
    target_render: str | None = None,
    target_b2: str | None = None,
    sync_b2: bool = False,
    prune_b2_extra: bool = False,
    switch_router: bool = True,
    quiesce_source: bool = True,
) -> dict[str, Any]:
    current = state.all()
    current_n8n_render = current["render"].get("n8n")
    current_postgres = current.get("postgres")


    target_postgres = target_postgres or _next_slot("postgres", current.get("postgres"))
    target_render = target_render or _next_slot("render", current_n8n_render, "n8n")
    target_b2 = target_b2 or _next_slot("b2", current.get("b2"))

    if (
        current_postgres
        and target_postgres != current_postgres
        and not quiesce_source
    ):
        raise RuntimeError(
            "Changing Postgres slots requires quiesce_source=true"
        )

    if not target_postgres:
        raise RuntimeError("No Postgres target is available")
    if not target_render:
        raise RuntimeError("No n8n Render target is available")

    if not registry.exists("postgres", target_postgres):
        raise RuntimeError(f"Unknown postgres slot: {target_postgres}")
    if not registry.exists("render", target_render):
        raise RuntimeError(f"Unknown render slot: {target_render}")
    if registry.render_role(target_render) != "n8n":
        raise RuntimeError(f"Render target {target_render} is not role=n8n")
    if target_b2 and not registry.exists("b2", target_b2):
        raise RuntimeError(f"Unknown b2 slot: {target_b2}")
    if quiesce_source and not switch_router:
        raise RuntimeError(
            "quiesce_source=true requires switch_router=true so ingress can be placed in maintenance mode"
        )

    steps = ["preflight"]
    if switch_router:
        steps.append("router_maintenance_on")
    if quiesce_source:
        steps.append("suspend_all_running_n8n")
    steps.extend([
        "postgres_migrate_and_compare",
        "b2_sync_or_compare" if target_b2 else "b2_skipped",
        "configure_target_n8n",
        "resume_deploy_and_healthcheck_target_n8n",
    ])
    if switch_router:
        steps.extend(["set_n8n_router_preferred", "router_maintenance_off"])
    steps.append("persist_active_state")

    return {
        "current": current,
        "target": {
            "postgres": target_postgres,
            "render": {"n8n": target_render},
            "b2": target_b2,
        },
        "sync_b2": sync_b2,
        "prune_b2_extra": prune_b2_extra,
        "switch_router": switch_router,
        "quiesce_source": quiesce_source,
        "steps": steps,
        "postgres_runtime": safe_runtime_summary(target_postgres),
    }


def run_failover(
    target_postgres: str | None = None,
    target_render: str | None = None,
    target_b2: str | None = None,
    sync_b2: bool = False,
    prune_b2_extra: bool = False,
    switch_router: bool = True,
    quiesce_source: bool = True,
) -> dict[str, Any]:
    if not _FAILOVER_LOCK.acquire(blocking=False):
        raise RuntimeError("Another infrastructure failover is already running")

    try:
        plan = build_failover_plan(
            target_postgres=target_postgres,
            target_render=target_render,
            target_b2=target_b2,
            sync_b2=sync_b2,
            prune_b2_extra=prune_b2_extra,
            switch_router=switch_router,
            quiesce_source=quiesce_source,
        )
    except Exception:
        _FAILOVER_LOCK.release()
        raise

    current = plan["current"]
    target = plan["target"]
    source_postgres = current.get("postgres")
    source_b2 = current.get("b2")
    previous_render = current["render"].get("n8n")
    target_render = target["render"]["n8n"]

    if not source_postgres:
        _FAILOVER_LOCK.release()
        raise RuntimeError("No active Postgres slot is configured")

    result: dict[str, Any] = {"plan": plan, "phases": {}, "rollback": {}}
    router_maintenance_enabled = False
    n8n_snapshot_before: dict[str, dict[str, Any]] | None = None
    running_before: list[str] = []
    target_env_before: dict[str, str | None] | None = None
    target_env_changed = False

    try:
        # PRE-FLIGHT
        source_pg_health = postgres_health(source_postgres)
        target_pg_health = postgres_health(target["postgres"])
        if not source_pg_health.get("healthy"):
            raise RuntimeError(f"Source Postgres unhealthy: {source_pg_health}")
        if not target_pg_health.get("healthy"):
            raise RuntimeError(f"Target Postgres unhealthy: {target_pg_health}")

        if render_role(target_render) != "n8n":
            raise RuntimeError(f"Target Render slot {target_render} is not an n8n slot")

        n8n_snapshot_before = _n8n_service_snapshot()
        running_before = _running_slots(n8n_snapshot_before)
        target_env_before = render_get_env(
            target_render,
            render_runtime_env_keys(target_render),
        )

        preflight: dict[str, Any] = {
            "source_postgres": source_pg_health,
            "target_postgres": target_pg_health,
            "target_render_service": n8n_snapshot_before[target_render],
            "target_render_health_before": render_health(target_render),
            "n8n_render_services": n8n_snapshot_before,
            "n8n_running_before": running_before,
        }
        if previous_render and previous_render in n8n_snapshot_before:
            preflight["source_render_service"] = n8n_snapshot_before[previous_render]
        if plan["switch_router"]:
            preflight["n8n_router"] = router_status("n8n")
        result["phases"]["preflight"] = preflight

        # Freeze public n8n ingress before quiescing every registered writer.
        if plan["switch_router"]:
            result["phases"]["router_maintenance_on"] = router_set_maintenance("n8n", True)
            router_maintenance_enabled = True

        if plan["quiesce_source"]:
            result["phases"]["n8n_quiesce"] = _suspend_all_running_n8n(n8n_snapshot_before)

        # POSTGRES -- this begins only after every registered n8n writer is confirmed suspended.
        if source_postgres != target["postgres"]:
            migration = postgres_migrate(source_postgres, target["postgres"])
            comparison = postgres_compare(source_postgres, target["postgres"])
            if not comparison.get("match"):
                raise RuntimeError(f"Postgres comparison failed after migration: {comparison}")
        else:
            migration = {"skipped": True, "reason": "source equals target"}
            comparison = {"match": True, "skipped": True, "reason": "source equals target"}
        result["phases"]["postgres"] = {"migration": migration, "comparison": comparison}

        # B2
        target_b2 = target.get("b2")
        if target_b2:
            target_b2_health = b2_health(target_b2)
            if not target_b2_health.get("healthy"):
                raise RuntimeError(f"Target B2 unhealthy: {target_b2_health}")

            if source_b2 and source_b2 != target_b2:
                source_b2_health = b2_health(source_b2)
                if not source_b2_health.get("healthy"):
                    raise RuntimeError(f"Source B2 unhealthy: {source_b2_health}")
                if plan["sync_b2"]:
                    b2_result = b2_migrate(
                        source_b2,
                        target_b2,
                        prune_extra=plan["prune_b2_extra"],
                    )
                else:
                    comparison_b2 = b2_compare(source_b2, target_b2)
                    if not comparison_b2.get("match"):
                        raise RuntimeError(
                            "B2 slots do not match and sync_b2=false. "
                            "Run with sync_b2=true to mirror the source first."
                        )
                    b2_result = {"comparison": comparison_b2}
            else:
                b2_result = {"skipped": True, "reason": "source equals target or no source"}
            result["phases"]["b2"] = b2_result

        # CONFIGURE/START TARGET N8N. All other n8n slots remain suspended.
        pg_runtime = postgres_runtime_config(target["postgres"])
        b2_runtime = b2_runtime_config(target_b2) if target_b2 else None
        target_env_changed = True
        result["phases"]["render_config"] = render_configure_runtime(
            target_render,
            pg_runtime,
            b2_runtime,
        )
        result["phases"]["render_deploy"] = render_deploy_and_health(target_render)

        # ROUTER PREPARE -- still under maintenance.
        if plan["switch_router"]:
            result["phases"]["router_preferred"] = router_set_preferred(
                "n8n",
                render_base_url(target_render),
            )

        # Persist the new control-plane state while ingress is still frozen.
        next_state = state.all()
        next_state["render"]["n8n"] = target_render
        next_state["postgres"] = target["postgres"]
        if target_b2:
            next_state["b2"] = target_b2

        result["phases"]["state_commit"] = state.set_all(
            next_state,
            persist=True,
        )

        # Open public ingress LAST.
        if plan["switch_router"]:
            result["phases"]["router_maintenance_off"] = router_set_maintenance(
                "n8n",
                False,
            )
            router_maintenance_enabled = False

        result["ok"] = True
        return result

    except Exception:
        logger.exception("Infrastructure failover failed; attempting rollback")

        # Put ingress back into maintenance even if the failure happened after
        # we had attempted to reopen it. The Render suspension invariant below
        # remains the hard write-safety barrier.
        if plan["switch_router"]:
            try:
                result["rollback"]["router_maintenance_on"] = (
                    router_set_maintenance("n8n", True)
                )
                router_maintenance_enabled = True
            except Exception as exc:
                result["rollback"]["router_maintenance_on_error"] = str(exc)

        # Stop every currently-running n8n service before changing runtime
        # configuration during rollback.
        if n8n_snapshot_before is not None:
            try:
                rollback_snapshot = _n8n_service_snapshot()
                result["rollback"]["quiesce_before_restore"] = (
                    _suspend_all_running_n8n(rollback_snapshot)
                )
            except Exception as exc:
                result["rollback"]["quiesce_before_restore_error"] = str(exc)

        # Restore the target's original direct Render env vars.
        if target_env_changed and target_env_before is not None:
            try:
                result["rollback"]["restore_target_env"] = render_restore_env(
                    target_render,
                    target_env_before,
                )

                # Render env changes require a deploy before they affect runtime.
                # If this service was running before the failed failover, bring it
                # back using its restored configuration.
                if target_render in running_before:
                    result["rollback"]["redeploy_restored_target"] = (
                        render_deploy_and_health(target_render)
                    )
            except Exception as exc:
                result["rollback"]["restore_target_env_error"] = str(exc)

        # Restore the complete original running/suspended topology.
        if n8n_snapshot_before is not None:
            try:
                result["rollback"]["restore_n8n_running_set"] = (
                    _restore_n8n_running_set(running_before)
                )
            except Exception as exc:
                result["rollback"]["restore_n8n_running_set_error"] = str(exc)

        # Restore the original router preference.
        if plan["switch_router"] and previous_render:
            try:
                result["rollback"]["restore_router_preferred"] = (
                    router_set_preferred(
                        "n8n",
                        render_base_url(previous_render),
                    )
                )
            except Exception as exc:
                result["rollback"]["restore_router_preferred_error"] = str(exc)

        # Restore old durable control-plane state if the new state was already
        # persisted before the later failure.
        try:
            if state.all() != current:
                result["rollback"]["restore_state"] = state.set_all(
                    current,
                    persist=True,
                )
        except Exception as exc:
            result["rollback"]["restore_state_error"] = str(exc)

        # Reopen ingress only after rollback has completed.
        if plan["switch_router"]:
            try:
                result["rollback"]["router_maintenance_off"] = (
                    router_set_maintenance("n8n", False)
                )
                router_maintenance_enabled = False
            except Exception as exc:
                result["rollback"]["router_maintenance_off_error"] = str(exc)

        raise

    finally:
        _FAILOVER_LOCK.release()
