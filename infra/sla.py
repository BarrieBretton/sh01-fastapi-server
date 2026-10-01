import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from .b2ring import b2_runtime_config
from .orchestrator import run_failover
from .persistence import store
from .registry import registry
from .render_provider import (
    render_base_url,
    render_get_env,
    render_health,
    render_service_status,
    render_suspend,
)
from .router_client import (
    router_set_maintenance,
    router_set_preferred,
    router_status,
)
from .runtime_config import postgres_runtime_config
from .state import state

logger = logging.getLogger("infra.sla")
_SLA_LOCK = threading.Lock()
_STATE_KEY = "infra_sla_monitor"

_DEFAULT_DB_ENV_MAP = {
    "host": "DB_POSTGRESDB_HOST",
    "port": "DB_POSTGRESDB_PORT",
    "database": "DB_POSTGRESDB_DATABASE",
    "user": "DB_POSTGRESDB_USER",
    "password": "DB_POSTGRESDB_PASSWORD",
}

_DEFAULT_B2_ENV_MAP = {
    "key_id": "BACKBLAZE_KEY_ID",
    "application_key": "BACKBLAZE_APPLICATION_KEY",
    "bucket_name": "BACKBLAZE_BUCKET_NAME",
}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, "").strip()
    try:
        value = int(raw) if raw else default
    except ValueError:
        value = default
    return max(minimum, value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_monitor_state() -> tuple[dict[str, Any], int]:
    payload = store.load_state(_STATE_KEY)
    if not payload or not isinstance(payload.get("value"), dict):
        return {
            "consecutive_failures": 0,
            "last_reason": None,
            "last_checked_at": None,
            "last_failover_at": None,
            "last_failover_from": None,
            "last_failover_to": None,
            "last_failover_mode": None,
        }, 0
    return dict(payload["value"]), int(payload.get("revision", 0))


def _save_monitor_state(value: dict[str, Any], revision: int) -> dict[str, Any]:
    return store.save_state(_STATE_KEY, value, revision)


def _render_slots_in_rotation(current: str) -> list[str]:
    slots = registry.render_slots()
    if current not in slots:
        return slots
    idx = slots.index(current)
    return slots[idx + 1 :] + slots[:idx]


def _service_state(slot: str, role: str) -> dict[str, Any]:
    try:
        status = render_service_status(slot, role)
    except Exception as exc:
        # Losing Render API visibility/control of the active node is itself a
        # hard failure from the controller's perspective. Do not let one API
        # exception prevent traversal to the next registered ring member.
        return {
            "slot": slot,
            "role": role,
            "suspended": "unknown",
            "healthy": False,
            "health": None,
            "status": None,
            "inspection_error": str(exc),
        }

    suspended = status.get("suspended")
    health = None
    if suspended == "not_suspended":
        health = render_health(slot, role)
    return {
        "slot": slot,
        "role": role,
        "suspended": suspended,
        "healthy": bool(health and health.get("healthy")),
        "health": health,
        "status": status,
    }


def _active_pair_state(slot: str) -> dict[str, Any]:
    n8n = _service_state(slot, "n8n")
    sh01 = _service_state(slot, "sh01")
    hard_down = (
        n8n["suspended"] != "not_suspended"
        or sh01["suspended"] != "not_suspended"
    )
    healthy = n8n["healthy"] and sh01["healthy"]
    return {
        "slot": slot,
        "healthy": healthy,
        "hard_down": hard_down,
        "n8n": n8n,
        "sh01": sh01,
    }


def _routers_in_maintenance() -> dict[str, Any]:
    n8n = router_status("n8n")
    sh01 = router_status("sh01")
    return {
        "n8n": n8n,
        "sh01": sh01,
        "maintenance": bool(n8n.get("maintenance") or sh01.get("maintenance")),
    }


def _align_routers(active_slot: str) -> dict[str, Any]:
    actions: dict[str, Any] = {}
    for role in ("n8n", "sh01"):
        desired = render_base_url(active_slot, role)
        current = router_status(role)
        if current.get("preferred") != desired:
            actions[role] = router_set_preferred(role, desired)
        else:
            actions[role] = {
                "changed": False,
                "preferred": desired,
            }
    return actions


def _suspend_running_standbys(active_slot: str) -> dict[str, Any]:
    """Best-effort standby enforcement.

    One broken/deleted/unreachable standby must not make a healthy active ring fail
    its SLA tick. Each slot/role is inspected independently and errors are reported
    in the result instead of aborting the tick.
    """
    actions: dict[str, Any] = {}
    for slot in registry.render_slots():
        if slot == active_slot:
            continue
        slot_actions: dict[str, Any] = {}
        for role in ("n8n", "sh01"):
            try:
                status = render_service_status(slot, role)
                if status.get("suspended") == "not_suspended":
                    slot_actions[role] = render_suspend(slot, role)
                else:
                    slot_actions[role] = {
                        "changed": False,
                        "status": "already_suspended",
                    }
            except Exception as exc:
                logger.warning(
                    "Could not enforce standby suspension for %s/%s: %s",
                    slot,
                    role,
                    exc,
                )
                slot_actions[role] = {
                    "changed": False,
                    "status": "standby_enforcement_failed",
                    "error": str(exc),
                }
        actions[slot] = slot_actions
    return actions


def _service_env_maps(slot: str, role: str) -> tuple[dict[str, str], dict[str, str]]:
    svc = registry.render_service(slot, role)
    db_map = svc.get("db_env_map", _DEFAULT_DB_ENV_MAP)
    b2_map = svc.get("b2_env_map", _DEFAULT_B2_ENV_MAP)
    return dict(db_map), dict(b2_map)


def _expected_runtime_env(slot: str, role: str) -> dict[str, str]:
    """Build the runtime env that the candidate should already have.

    Values are used only for in-process comparison. They are never returned from
    this module, so DB/B2 secrets are not exposed by SLA responses.
    """
    active = state.all()
    expected: dict[str, str] = {}
    db_map, b2_map = _service_env_maps(slot, role)

    if role == "n8n":
        postgres_slot = active.get("postgres")
        if not postgres_slot:
            raise RuntimeError("No active Postgres slot is configured")
        pg = postgres_runtime_config(postgres_slot)
        for field, env_name in db_map.items():
            if field in pg:
                expected[env_name] = str(pg[field])

    b2_slot = active.get("b2")
    if b2_slot:
        b2 = b2_runtime_config(b2_slot)
        for field, env_name in b2_map.items():
            if field in b2:
                expected[env_name] = str(b2[field])

    return expected


def _candidate_runtime_matches(slot: str) -> dict[str, Any]:
    """Verify that an already-live candidate points at current Postgres/B2.

    This is the gate for zero-redeploy promotion. If any runtime value differs,
    the controller falls back to the existing full run_failover() path, which
    rewrites target runtime configuration before serving traffic.
    """
    roles: dict[str, Any] = {}
    overall = True

    for role in ("n8n", "sh01"):
        expected = _expected_runtime_env(slot, role)
        if not expected:
            roles[role] = {
                "match": True,
                "checked_keys": [],
                "mismatched_keys": [],
            }
            continue

        actual = render_get_env(slot, role, sorted(expected))
        mismatched = sorted(
            key
            for key, expected_value in expected.items()
            if actual.get(key) != expected_value
        )
        match = not mismatched
        overall = overall and match
        roles[role] = {
            "match": match,
            "checked_keys": sorted(expected),
            "mismatched_keys": mismatched,
        }

    return {
        "slot": slot,
        "match": overall,
        "roles": roles,
    }


def _promote_live_candidate(previous_active: str, candidate: str) -> dict[str, Any]:
    """Promote an already-running, healthy, correctly-configured Render pair.

    Unlike run_failover(), this path intentionally does NOT suspend or redeploy
    the candidate. It freezes ingress briefly, switches both routers, persists
    leadership, reopens ingress, then best-effort suspends all non-active slots.
    """
    if previous_active == candidate:
        return {
            "mode": "fast_promote_live_candidate",
            "from": previous_active,
            "to": candidate,
            "changed": False,
        }

    previous_urls = {
        role: render_base_url(previous_active, role)
        for role in ("n8n", "sh01")
    }
    maintenance_on = {"n8n": False, "sh01": False}
    state_changed = False
    phases: dict[str, Any] = {}

    try:
        phases["n8n_maintenance_on"] = router_set_maintenance("n8n", True)
        maintenance_on["n8n"] = True
        phases["sh01_maintenance_on"] = router_set_maintenance("sh01", True)
        maintenance_on["sh01"] = True

        phases["n8n_preferred"] = router_set_preferred(
            "n8n",
            render_base_url(candidate, "n8n"),
        )
        phases["sh01_preferred"] = router_set_preferred(
            "sh01",
            render_base_url(candidate, "sh01"),
        )

        phases["persist_active_render"] = state.set_active(
            "render",
            candidate,
            persist=True,
        )
        state_changed = True

        phases["n8n_maintenance_off"] = router_set_maintenance("n8n", False)
        maintenance_on["n8n"] = False
        phases["sh01_maintenance_off"] = router_set_maintenance("sh01", False)
        maintenance_on["sh01"] = False

        phases["standby_enforcement"] = _suspend_running_standbys(candidate)

        return {
            "mode": "fast_promote_live_candidate",
            "from": previous_active,
            "to": candidate,
            "changed": True,
            "phases": phases,
        }

    except Exception:
        logger.exception(
            "Fast promotion from %s to %s failed; attempting rollback",
            previous_active,
            candidate,
        )

        rollback: dict[str, Any] = {}

        if state_changed:
            try:
                rollback["state"] = state.set_active(
                    "render",
                    previous_active,
                    persist=True,
                )
            except Exception as exc:
                rollback["state_error"] = str(exc)

        for role in ("n8n", "sh01"):
            try:
                rollback[f"{role}_preferred"] = router_set_preferred(
                    role,
                    previous_urls[role],
                )
            except Exception as exc:
                rollback[f"{role}_preferred_error"] = str(exc)

        for role in ("n8n", "sh01"):
            if maintenance_on[role]:
                try:
                    rollback[f"{role}_maintenance_off"] = router_set_maintenance(
                        role,
                        False,
                    )
                except Exception as exc:
                    rollback[f"{role}_maintenance_off_error"] = str(exc)

        raise RuntimeError(
            f"Fast promotion failed; rollback attempted: {rollback}"
        )




def _hard_stop_render_ring(reason: str) -> dict[str, Any]:
    """Put the public stack into maintenance after the Render ring is exhausted.

    This is deliberately terminal for automatic operation. Once every registered
    Render candidate has failed, a human has to add/fix a ring member and then
    clear router maintenance before retrying an SLA tick. That is appropriate
    because creating a new provider account/service is inherently an operator
    action.
    """
    actions: dict[str, Any] = {"routers": {}, "render": {}}

    for role in ("n8n", "sh01"):
        try:
            actions["routers"][role] = router_set_maintenance(role, True)
        except Exception as exc:
            actions["routers"][role] = {"error": str(exc)}

    for slot in registry.render_slots():
        slot_actions: dict[str, Any] = {}
        for role in ("n8n", "sh01"):
            try:
                status = render_service_status(slot, role)
                if status.get("suspended") == "not_suspended":
                    slot_actions[role] = render_suspend(slot, role)
                else:
                    slot_actions[role] = {
                        "changed": False,
                        "status": "already_not_running",
                    }
            except Exception as exc:
                slot_actions[role] = {
                    "changed": False,
                    "status": "hard_stop_suspend_failed",
                    "error": str(exc),
                }
        actions["render"][slot] = slot_actions

    actions["reason"] = reason
    return actions

def sla_status() -> dict[str, Any]:
    monitor, revision = _load_monitor_state()
    active = state.get_active("render")
    return {
        "active_render": active,
        "registered_render_slots": registry.render_slots(),
        "failure_threshold": _env_int("INFRA_SLA_FAILURE_THRESHOLD", 2),
        "monitor": monitor,
        "monitor_revision": revision,
    }


def run_sla_tick(dry_run: bool = False) -> dict[str, Any]:
    if not _SLA_LOCK.acquire(blocking=False):
        return {
            "ok": True,
            "skipped": True,
            "reason": "another_sla_tick_is_running",
        }

    try:
        active = state.get_active("render")
        if not active or not registry.exists("render", active):
            raise RuntimeError(f"Active Render slot is missing or unknown: {active}")

        routers = _routers_in_maintenance()
        if routers["maintenance"]:
            return {
                "ok": True,
                "skipped": True,
                "reason": "router_maintenance_active",
                "routers": routers,
            }

        pair = _active_pair_state(active)
        monitor, revision = _load_monitor_state()
        threshold = _env_int("INFRA_SLA_FAILURE_THRESHOLD", 2)

        if pair["healthy"]:
            next_monitor = {
                **monitor,
                "consecutive_failures": 0,
                "last_reason": None,
                "last_checked_at": _now(),
                "hard_stop": False,
            }
            if not dry_run:
                _save_monitor_state(next_monitor, revision + 1)
                router_actions = _align_routers(active)
                standby_actions = _suspend_running_standbys(active)
            else:
                router_actions = {"would_align_to": active}
                standby_actions = {"would_suspend_standbys": True}

            return {
                "ok": True,
                "action": "healthy_keep_active",
                "active_render": active,
                "pair": pair,
                "router_actions": router_actions,
                "standby_actions": standby_actions,
                "dry_run": dry_run,
            }

        reason = (
            "active_pair_suspended_or_not_running"
            if pair["hard_down"]
            else "active_pair_healthcheck_failed"
        )
        failures = int(monitor.get("consecutive_failures", 0)) + 1
        should_failover = pair["hard_down"] or failures >= threshold

        next_monitor = {
            **monitor,
            "consecutive_failures": failures,
            "last_reason": reason,
            "last_checked_at": _now(),
        }

        if not dry_run:
            _save_monitor_state(next_monitor, revision + 1)

        if not should_failover:
            return {
                "ok": False,
                "action": "wait_for_failure_threshold",
                "active_render": active,
                "pair": pair,
                "reason": reason,
                "consecutive_failures": failures,
                "failure_threshold": threshold,
                "dry_run": dry_run,
            }

        candidates = _render_slots_in_rotation(active)
        if not candidates:
            return {
                "ok": False,
                "action": "no_failover_candidate",
                "active_render": active,
                "pair": pair,
                "reason": reason,
            }

        candidate_diagnostics: dict[str, Any] = {}

        if dry_run:
            for candidate in candidates:
                try:
                    candidate_pair = _active_pair_state(candidate)
                    runtime = None
                    fast_promotable = False
                    if candidate_pair["healthy"]:
                        runtime = _candidate_runtime_matches(candidate)
                        fast_promotable = bool(runtime.get("match"))
                    candidate_diagnostics[candidate] = {
                        "pair": candidate_pair,
                        "runtime": runtime,
                        "fast_promotable": fast_promotable,
                    }
                except Exception as exc:
                    candidate_diagnostics[candidate] = {
                        "inspection_error": str(exc),
                        "fast_promotable": False,
                    }

            return {
                "ok": False,
                "action": "would_failover",
                "active_render": active,
                "candidate_order": candidates,
                "candidate_diagnostics": candidate_diagnostics,
                "pair": pair,
                "reason": reason,
                "dry_run": True,
            }

        current = state.all()
        errors: dict[str, str] = {}

        for candidate in candidates:
            try:
                candidate_pair = _active_pair_state(candidate)
                runtime = None

                if candidate_pair["healthy"]:
                    runtime = _candidate_runtime_matches(candidate)
                    if runtime.get("match"):
                        result = _promote_live_candidate(active, candidate)

                        monitor_after, revision_after = _load_monitor_state()
                        committed = {
                            **monitor_after,
                            "consecutive_failures": 0,
                            "last_reason": None,
                            "last_checked_at": _now(),
                            "last_failover_at": _now(),
                            "last_failover_from": active,
                            "last_failover_to": candidate,
                            "last_failover_mode": "fast_promote_live_candidate",
                            "hard_stop": False,
                        }
                        _save_monitor_state(committed, revision_after + 1)

                        return {
                            "ok": True,
                            "action": "automatic_failover",
                            "mode": "fast_promote_live_candidate",
                            "from": active,
                            "to": candidate,
                            "reason": reason,
                            "candidate_pair": candidate_pair,
                            "runtime_check": runtime,
                            "result": result,
                        }

                # Candidate is not already healthy+correctly-configured.
                # Use the existing full failover engine. It will configure,
                # resume/deploy, verify, route, persist, and roll back on failure.
                result = run_failover(
                    target_postgres=current.get("postgres"),
                    target_render=candidate,
                    target_b2=current.get("b2"),
                    sync_b2=False,
                    prune_b2_extra=False,
                    switch_router=True,
                    quiesce_source=True,
                )

                monitor_after, revision_after = _load_monitor_state()
                committed = {
                    **monitor_after,
                    "consecutive_failures": 0,
                    "last_reason": None,
                    "last_checked_at": _now(),
                    "last_failover_at": _now(),
                    "last_failover_from": active,
                    "last_failover_to": candidate,
                    "last_failover_mode": "full_failover",
                    "hard_stop": False,
                }
                _save_monitor_state(committed, revision_after + 1)
                _suspend_running_standbys(candidate)
                _align_routers(candidate)

                return {
                    "ok": True,
                    "action": "automatic_failover",
                    "mode": "full_failover",
                    "from": active,
                    "to": candidate,
                    "reason": reason,
                    "candidate_pair": candidate_pair,
                    "runtime_check": runtime,
                    "result": result,
                }

            except Exception as exc:
                logger.exception(
                    "Automatic SLA failover from %s to %s failed",
                    active,
                    candidate,
                )
                errors[candidate] = str(exc)

        # Every registered alternative failed. The Render ring is exhausted.
        # Freeze public ingress and best-effort suspend every controllable Render
        # service. Recovery now requires an operator to repair/add a ring member,
        # clear router maintenance, and trigger another SLA tick.
        hard_stop_actions = _hard_stop_render_ring(reason)
        monitor_after, revision_after = _load_monitor_state()
        halted = {
            **monitor_after,
            "last_checked_at": _now(),
            "last_reason": "render_ring_failed_completely",
            "hard_stop": True,
        }
        try:
            _save_monitor_state(halted, revision_after + 1)
        except Exception:
            logger.exception("Could not persist SLA hard-stop monitor state")

        return {
            "ok": False,
            "action": "render_ring_failed_completely",
            "active_render": active,
            "reason": reason,
            "errors": errors,
            "hard_stop_actions": hard_stop_actions,
        }

    finally:
        _SLA_LOCK.release()
