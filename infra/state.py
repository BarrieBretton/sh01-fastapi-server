import logging
import os
import threading
from typing import Any

from .persistence import store

logger = logging.getLogger("infra.state")


def _legacy_render_slot(value: Any) -> str | None:
    if isinstance(value, str):
        if value.startswith("sh01-"):
            return "render-" + value.split("-", 1)[1]
        return value
    if isinstance(value, dict):
        n8n = value.get("n8n")
        if isinstance(n8n, str):
            return _legacy_render_slot(n8n)
        sh01 = value.get("sh01")
        if isinstance(sh01, str):
            return _legacy_render_slot(sh01)
    return None


class InfraState:
    """Durable active-slot state for paired Render slots.

    A Render slot is one paired infrastructure unit, e.g. render-a contains
    n8n-a + sh01-a. Legacy ACTIVE_N8N_RENDER_SLOT / ACTIVE_SH01_RENDER_SLOT and
    legacy persisted nested render state are accepted only for bootstrap.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._revision = 0
        legacy_bootstrap = (
            os.getenv("ACTIVE_N8N_RENDER_SLOT")
            or os.getenv("ACTIVE_SH01_RENDER_SLOT")
            or None
        )
        self._active: dict[str, Any] = {
            "render": os.getenv("ACTIVE_RENDER_SLOT") or _legacy_render_slot(legacy_bootstrap),
            "postgres": os.getenv("ACTIVE_POSTGRES_SLOT") or None,
            "b2": os.getenv("ACTIVE_B2_SLOT") or None,
        }
        self._load_persisted()

    @staticmethod
    def _normalize(value: dict[str, Any]) -> dict[str, Any]:
        return {
            "render": _legacy_render_slot(value.get("render")),
            "postgres": value.get("postgres"),
            "b2": value.get("b2"),
        }

    def _load_persisted(self) -> None:
        try:
            payload = store.load_state("active_infrastructure")
            if payload and isinstance(payload.get("value"), dict):
                persisted = self._normalize(payload["value"])
                for key in ("render", "postgres", "b2"):
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

    def get_active(self, infra_type: str) -> str | None:
        with self._lock:
            return self._active.get(infra_type)

    def set_active(self, infra_type: str, slot: str, persist: bool = True) -> dict[str, Any]:
        with self._lock:
            next_state = self.all()
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
                "previous": previous,
                "target": slot,
                "revision": self._revision,
                "persistence": persistence,
            }

    def set_all(self, active: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        with self._lock:
            next_state = self.all()
            incoming = self._normalize(active)
            for key in ("render", "postgres", "b2"):
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
                "render": self._active.get("render"),
                "postgres": self._active.get("postgres"),
                "b2": self._active.get("b2"),
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"active": self.all(), "revision": self._revision}


state = InfraState()
