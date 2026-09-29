import logging
import os
import threading
from typing import Any

from .persistence import store

logger = logging.getLogger("infra.state")


class InfraState:
    """Durable active-slot state.

    Render has independent roles because the n8n and SH01 Render rings are
    separate routing domains. ACTIVE_RENDER_SLOT remains a backward-compatible
    startup fallback for the n8n role.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._revision = 0
        self._active: dict[str, Any] = {
            "render": {
                "n8n": os.getenv("ACTIVE_N8N_RENDER_SLOT") or os.getenv("ACTIVE_RENDER_SLOT") or None,
                "sh01": os.getenv("ACTIVE_SH01_RENDER_SLOT") or None,
            },
            "postgres": os.getenv("ACTIVE_POSTGRES_SLOT") or None,
            "b2": os.getenv("ACTIVE_B2_SLOT") or None,
        }
        self._load_persisted()

    @staticmethod
    def _normalize(value: dict[str, Any]) -> dict[str, Any]:
        normalized: dict[str, Any] = {
            "render": {"n8n": None, "sh01": None},
            "postgres": value.get("postgres"),
            "b2": value.get("b2"),
        }
        render = value.get("render")
        if isinstance(render, dict):
            normalized["render"]["n8n"] = render.get("n8n")
            normalized["render"]["sh01"] = render.get("sh01")
        elif isinstance(render, str):
            # Backward compatibility with revision 1 of the control plane.
            normalized["render"]["n8n"] = render
        return normalized

    def _load_persisted(self) -> None:
        try:
            payload = store.load_state("active_infrastructure")
            if payload and isinstance(payload.get("value"), dict):
                persisted = self._normalize(payload["value"])
                for role in ("n8n", "sh01"):
                    if persisted["render"].get(role) is not None:
                        self._active["render"][role] = persisted["render"][role]
                for key in ("postgres", "b2"):
                    if persisted.get(key) is not None:
                        self._active[key] = persisted[key]
                self._revision = int(payload.get("revision", 0))
                logger.info(
                    "Loaded durable infrastructure state revision=%s active=%s",
                    self._revision,
                    self._active,
                )
        except Exception as exc:
            logger.warning("Could not load durable state; using env defaults: %s", exc)

    def get_active(self, infra_type: str, role: str | None = None) -> str | None:
        with self._lock:
            if infra_type == "render":
                return self._active["render"].get(role or "n8n")
            return self._active.get(infra_type)

    def set_active(
        self,
        infra_type: str,
        slot: str,
        persist: bool = True,
        role: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            next_state = self.all()
            if infra_type == "render":
                selected_role = role or "n8n"
                previous = next_state["render"].get(selected_role)
                next_state["render"][selected_role] = slot
            else:
                selected_role = None
                previous = next_state.get(infra_type)
                next_state[infra_type] = slot

            next_revision = self._revision + 1
            persistence = None
            if persist:
                persistence = store.save_state(
                    "active_infrastructure",
                    next_state,
                    next_revision,
                )

            self._active = next_state
            self._revision = next_revision
            return {
                "infra_type": infra_type,
                "role": selected_role,
                "previous": previous,
                "target": slot,
                "revision": self._revision,
                "persistence": persistence,
            }

    def set_all(self, active: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        with self._lock:
            next_state = self.all()
            incoming = self._normalize(active)
            if isinstance(active.get("render"), (dict, str)):
                if isinstance(active.get("render"), dict):
                    for role in ("n8n", "sh01"):
                        if role in active["render"]:
                            next_state["render"][role] = incoming["render"][role]
                else:
                    next_state["render"]["n8n"] = incoming["render"]["n8n"]
            for key in ("postgres", "b2"):
                if key in active:
                    next_state[key] = incoming[key]

            next_revision = self._revision + 1
            persistence = None
            if persist:
                persistence = store.save_state(
                    "active_infrastructure",
                    next_state,
                    next_revision,
                )

            previous = self.all()
            self._active = next_state
            self._revision = next_revision
            return {
                "previous": previous,
                "active": self.all(),
                "revision": self._revision,
                "persistence": persistence,
            }

    def all(self) -> dict[str, Any]:
        with self._lock:
            return {
                "render": dict(self._active["render"]),
                "postgres": self._active.get("postgres"),
                "b2": self._active.get("b2"),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"active": self.all(), "revision": self._revision}


state = InfraState()
