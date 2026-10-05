import logging
import threading
from typing import Any

from .b2ring import b2_compare, b2_health, b2_migrate, b2_runtime_config
from .postgres import postgres_compare, postgres_health, postgres_migrate
from .registry import registry
from .render_provider import (
    render_base_url,
    render_configure_n8n_runtime,
    render_configure_sh01_runtime,
    render_deploy_and_health,
    render_get_env,
    render_health,
    render_restore_env,
    render_runtime_env_keys,
    render_service_status,
    render_suspend,
    render_resume,
)
from .router_client import router_set_maintenance, router_set_preferred, router_status
from .runtime_config import postgres_runtime_config, safe_runtime_summary
from .state import state

logger = logging.getLogger("infra.orchestrator")
_FAILOVER_LOCK = threading.Lock()


def _next_slot(kind: str, current: str | None) -> str | None:
    slots = registry.render_slots() if kind == "render" else registry.slots(kind)
    if not slots:
        return None
    if current not in slots:
        return slots[0]
    if len(slots) == 1:
        return current
    return slots[(slots.index(current) + 1) % len(slots)]


def _service_snapshot(role: str) -> dict[str, dict[str, Any]]:
    return {slot: render_service_status(slot, role) for slot in registry.render_slots()}


def _running_slots(snapshot: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        slot
        for slot, status in snapshot.items()
        if status.get("suspended") == "not_suspended"
    )


def _suspend_all_running(role: str, snapshot: dict[str, dict[str, Any]]) -> dict[str, Any]:
    running_before = _running_slots(snapshot)
    results: dict[str, Any] = {}
    for slot in running_before:
        results[slot] = render_suspend(slot, role)
    after = _service_snapshot(role)
    still_running = _running_slots(after)
    if still_running:
        raise RuntimeError(
            f"Failed to suspend all {role} Render services; still running: "
            + ", ".join(still_running)
        )
    return {
        "running_before": running_before,
        "suspend_results": results,
        "service_status_after": after,
    }


def _restore_running_set(role: str, running_before: list[str]) -> dict[str, Any]:
    expected_running = set(running_before)
    current = _service_snapshot(role)
    actions: dict[str, Any] = {}
    for slot in registry.render_slots():
        is_running = current[slot].get("suspended") == "not_suspended"
        should_run = slot in expected_running
        if should_run and not is_running:
            actions[slot] = {"action": "resume", "result": render_resume(slot, role)}
        elif not should_run and is_running:
            actions[slot] = {"action": "suspend", "result": render_suspend(slot, role)}
        else:
            actions[slot] = {
                "action": "none",
                "status": "running" if is_running else "suspended",
            }
    final = _service_snapshot(role)
    final_running = set(_running_slots(final))
    if final_running != expected_running:
        raise RuntimeError(
            f"Rollback could not restore original {role} running set; "
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
    target_postgres = target_postgres or _next_slot("postgres", current.get("postgres"))
    target_render = target_render or _next_slot("render", current.get("render"))
    target_b2 = target_b2 or _next_slot("b2", current.get("b2"))
    current_render = current.get("render")
    current_postgres = current.get("postgres")

    if (
        current_render
        and target_render != current_render
        and not switch_router
    ):
        raise RuntimeError(
            "Changing paired Render slots requires switch_router=true"
        )

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
        raise RuntimeError("No paired Render target is available")
    if not registry.exists("postgres", target_postgres):
        raise RuntimeError(f"Unknown postgres slot: {target_postgres}")
    if not registry.exists("render", target_render):
        raise RuntimeError(f"Unknown render slot: {target_render}")
    registry.render_service(target_render, "n8n")
    registry.render_service(target_render, "sh01")
    if target_b2 and not registry.exists("b2", target_b2):
        raise RuntimeError(f"Unknown b2 slot: {target_b2}")
    if quiesce_source and not switch_router:
        raise RuntimeError(
            "quiesce_source=true requires switch_router=true so ingress can be placed in maintenance mode"
        )

    steps = ["preflight"]
    if switch_router:
        steps.extend(["n8n_router_maintenance_on", "sh01_router_maintenance_on"])
    if quiesce_source:
        steps.extend([
            "suspend_all_running_n8n",
            "suspend_all_running_sh01",
        ])
    steps.extend([
        "postgres_migrate_and_compare",
        "b2_sync_or_compare" if target_b2 else "b2_skipped",
        "configure_target_n8n",
        "configure_target_sh01",
        "resume_deploy_healthcheck_target_n8n",
        "resume_deploy_healthcheck_target_sh01",
    ])
    if switch_router:
        steps.extend([
            "set_n8n_router_preferred",
            "set_sh01_router_preferred",
            "persist_active_state",
            "n8n_router_maintenance_off",
            "sh01_router_maintenance_off",
        ])
    else:
        steps.append("persist_active_state")

    return {
        "current": current,
        "target": {
            "postgres": target_postgres,
            "render": target_render,
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
    previous_render = current.get("render")
    target_render = target["render"]

    if not source_postgres:
        _FAILOVER_LOCK.release()
        raise RuntimeError("No active Postgres slot is configured")

    result: dict[str, Any] = {"plan": plan, "phases": {}, "rollback": {}}
    maintenance_enabled = {"n8n": False, "sh01": False}
    snapshots: dict[str, dict[str, dict[str, Any]]] = {}
    running_before: dict[str, list[str]] = {"n8n": [], "sh01": []}
    target_env_before: dict[str, dict[str, str | None] | None] = {
        "n8n": None,
        "sh01": None,
    }

    target_env_may_have_changed = {
        "n8n": False,
        "sh01": False,
    }

    try:
        source_pg_health = postgres_health(source_postgres)
        target_pg_health = postgres_health(target["postgres"])
        if not source_pg_health.get("healthy"):
            raise RuntimeError(f"Source Postgres unhealthy: {source_pg_health}")
        if not target_pg_health.get("healthy"):
            raise RuntimeError(f"Target Postgres unhealthy: {target_pg_health}")

        for role in ("n8n", "sh01"):
            snapshots[role] = _service_snapshot(role)
            running_before[role] = _running_slots(snapshots[role])

        target_env_before["n8n"] = render_get_env(
            target_render,
            "n8n",
            render_runtime_env_keys(target_render, "n8n"),
        )

        target_env_before["sh01"] = render_get_env(
            target_render,
            "sh01",
            render_runtime_env_keys(target_render, "sh01"),
        )

        preflight: dict[str, Any] = {
            "source_postgres": source_pg_health,
            "target_postgres": target_pg_health,
            "render_services": snapshots,
            "running_before": running_before,
            "target_n8n_health_before": render_health(target_render, "n8n"),
            "target_sh01_health_before": render_health(target_render, "sh01"),
        }

        if plan["switch_router"]:
            preflight["n8n_router"] = router_status("n8n")
            preflight["sh01_router"] = router_status("sh01")

            for role in ("n8n", "sh01"):
                router_state = preflight[f"{role}_router"]

                if router_state.get("maintenance"):
                    raise RuntimeError(
                        f"{role} router is already in maintenance mode; "
                        "refusing to begin paired failover"
                    )

        result["phases"]["preflight"] = preflight

        if plan["switch_router"]:
            result["phases"]["n8n_router_maintenance_on"] = router_set_maintenance("n8n", True)
            maintenance_enabled["n8n"] = True
            result["phases"]["sh01_router_maintenance_on"] = router_set_maintenance("sh01", True)
            maintenance_enabled["sh01"] = True

        if plan["quiesce_source"]:
            result["phases"]["n8n_quiesce"] = _suspend_all_running(
                "n8n",
                snapshots["n8n"],
            )

            result["phases"]["sh01_quiesce"] = _suspend_all_running(
                "sh01",
                snapshots["sh01"],
            )

        if source_postgres != target["postgres"]:
            migration = postgres_migrate(source_postgres, target["postgres"])
            comparison = postgres_compare(source_postgres, target["postgres"])
            if not comparison.get("match"):
                raise RuntimeError(f"Postgres comparison failed after migration: {comparison}")
        else:
            migration = {"skipped": True, "reason": "source equals target"}
            comparison = {"match": True, "skipped": True, "reason": "source equals target"}
        result["phases"]["postgres"] = {"migration": migration, "comparison": comparison}

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
                            "B2 slots do not match and sync_b2=false. Run with sync_b2=true to mirror the source first."
                        )
                    b2_result = {"comparison": comparison_b2}
            else:
                b2_result = {"skipped": True, "reason": "source equals target or no source"}
            result["phases"]["b2"] = b2_result

        pg_runtime = postgres_runtime_config(target["postgres"])
        b2_runtime = b2_runtime_config(target_b2) if target_b2 else None

        target_env_may_have_changed["n8n"] = True
        result["phases"]["render_config_n8n"] = render_configure_n8n_runtime(
            target_render,
            pg_runtime,
            b2_runtime,
        )

        target_env_may_have_changed["sh01"] = True
        result["phases"]["render_config_sh01"] = render_configure_sh01_runtime(
            target_render,
            pg_runtime,
            b2_runtime,
        )

        result["phases"]["render_deploy_n8n"] = render_deploy_and_health(
            target_render,
            "n8n",
        )

        result["phases"]["render_deploy_sh01"] = render_deploy_and_health(
            target_render,
            "sh01",
        )

        if plan["switch_router"]:
            result["phases"]["n8n_router_preferred"] = router_set_preferred(
                "n8n", render_base_url(target_render, "n8n")
            )
            result["phases"]["sh01_router_preferred"] = router_set_preferred(
                "sh01", render_base_url(target_render, "sh01")
            )

        next_state = state.all()
        next_state["render"] = target_render
        next_state["postgres"] = target["postgres"]
        if target_b2:
            next_state["b2"] = target_b2
        result["phases"]["state_commit"] = state.set_all(next_state, persist=True)

        if plan["switch_router"]:
            result["phases"]["n8n_router_maintenance_off"] = router_set_maintenance("n8n", False)
            maintenance_enabled["n8n"] = False
            result["phases"]["sh01_router_maintenance_off"] = router_set_maintenance("sh01", False)
            maintenance_enabled["sh01"] = False

        result["ok"] = True
        return result

    except Exception:
        logger.exception("Infrastructure failover failed; attempting rollback")

        # Freeze both public entry points before manipulating services/config.
        if plan["switch_router"]:
            for role in ("n8n", "sh01"):
                try:
                    result["rollback"][f"{role}_router_maintenance_on"] = (
                        router_set_maintenance(role, True)
                    )
                    maintenance_enabled[role] = True
                except Exception as exc:
                    result["rollback"][
                        f"{role}_router_maintenance_on_error"
                    ] = str(exc)

        # Stop both halves of every paired Render slot before restoring env.
        for role in ("n8n", "sh01"):
            if snapshots.get(role) is not None:
                try:
                    result["rollback"][f"{role}_quiesce_before_restore"] = (
                        _suspend_all_running(
                            role,
                            _service_snapshot(role),
                        )
                    )
                except Exception as exc:
                    result["rollback"][
                        f"{role}_quiesce_before_restore_error"
                    ] = str(exc)

        # Restore runtime env for each target service that may have been changed.
        for role in ("n8n", "sh01"):
            previous_env = target_env_before.get(role)

            if (
                target_env_may_have_changed.get(role)
                and previous_env is not None
            ):
                try:
                    result["rollback"][f"restore_target_{role}_env"] = (
                        render_restore_env(
                            target_render,
                            role,
                            previous_env,
                        )
                    )

                    # A service that was originally running must come back with
                    # its restored environment actually deployed.
                    if target_render in running_before[role]:
                        result["rollback"][
                            f"redeploy_restored_target_{role}"
                        ] = render_deploy_and_health(
                            target_render,
                            role,
                        )

                except Exception as exc:
                    result["rollback"][
                        f"restore_target_{role}_env_error"
                    ] = str(exc)

        # Restore exactly which n8n and SH01 services were running beforehand.
        for role in ("n8n", "sh01"):
            if snapshots.get(role) is not None:
                try:
                    result["rollback"][f"restore_{role}_running_set"] = (
                        _restore_running_set(
                            role,
                            running_before[role],
                        )
                    )
                except Exception as exc:
                    result["rollback"][
                        f"restore_{role}_running_set_error"
                    ] = str(exc)

        # Restore both Workers to the previous paired Render slot.
        if plan["switch_router"] and previous_render:
            for role in ("n8n", "sh01"):
                try:
                    result["rollback"][
                        f"restore_{role}_router_preferred"
                    ] = router_set_preferred(
                        role,
                        render_base_url(
                            previous_render,
                            role,
                        ),
                    )
                except Exception as exc:
                    result["rollback"][
                        f"restore_{role}_router_preferred_error"
                    ] = str(exc)

        # Restore durable control-plane state if it had already committed.
        try:
            if state.all() != current:
                result["rollback"]["restore_state"] = state.set_all(
                    current,
                    persist=True,
                )
        except Exception as exc:
            result["rollback"]["restore_state_error"] = str(exc)

        # Reopen both public entry points only after rollback is complete.
        if plan["switch_router"]:
            for role in ("n8n", "sh01"):
                try:
                    result["rollback"][
                        f"{role}_router_maintenance_off"
                    ] = router_set_maintenance(
                        role,
                        False,
                    )
                    maintenance_enabled[role] = False
                except Exception as exc:
                    result["rollback"][
                        f"{role}_router_maintenance_off_error"
                    ] = str(exc)

        raise
    finally:
        _FAILOVER_LOCK.release()
