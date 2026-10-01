import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

from .orchestrator import run_failover
from .persistence import store
from .registry import registry
from .render_provider import (
    render_base_url,
    render_health,
    render_service_status,
    render_suspend,
)
from .router_client import router_set_preferred, router_status
from .state import state

logger = logging.getLogger("infra.sla")
_SLA_LOCK = threading.Lock()
_STATE_KEY = "infra_sla_monitor"


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
    status = render_service_status(slot, role)
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
    actions: dict[str, Any] = {}
    for slot in registry.render_slots():
        if slot == active_slot:
            continue
        slot_actions: dict[str, Any] = {}
        for role in ("n8n", "sh01"):
            status = render_service_status(slot, role)
            if status.get("suspended") == "not_suspended":
                slot_actions[role] = render_suspend(slot, role)
            else:
                slot_actions[role] = {
                    "changed": False,
                    "status": "already_suspended",
                }
        actions[slot] = slot_actions
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

        if dry_run:
            return {
                "ok": False,
                "action": "would_failover",
                "active_render": active,
                "candidate_order": candidates,
                "pair": pair,
                "reason": reason,
                "dry_run": True,
            }

        current = state.all()
        errors: dict[str, str] = {}
        for candidate in candidates:
            try:
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
                }
                _save_monitor_state(committed, revision_after + 1)
                _suspend_running_standbys(candidate)
                _align_routers(candidate)
                return {
                    "ok": True,
                    "action": "automatic_failover",
                    "from": active,
                    "to": candidate,
                    "reason": reason,
                    "result": result,
                }
            except Exception as exc:
                logger.exception(
                    "Automatic SLA failover from %s to %s failed",
                    active,
                    candidate,
                )
                errors[candidate] = str(exc)

        return {
            "ok": False,
            "action": "all_failover_candidates_failed",
            "active_render": active,
            "reason": reason,
            "errors": errors,
        }
    finally:
        _SLA_LOCK.release()
