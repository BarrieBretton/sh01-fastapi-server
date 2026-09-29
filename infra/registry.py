# infra/registry.py

import json
import os
from typing import Any


class InfraRegistry:
    def __init__(self) -> None:
        raw = os.getenv("INFRA_REGISTRY_JSON", "{}").strip()
        try:
            self._registry: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("INFRA_REGISTRY_JSON contains invalid JSON") from exc

        for section in ("render", "postgres", "b2"):
            self._registry.setdefault(section, {})

    def all(self) -> dict[str, Any]:
        return self._registry

    def slots(self, infra_type: str) -> list[str]:
        section = self._registry.get(infra_type, {})
        return sorted(section.keys())

    def render_slots(self, role: str | None = None) -> list[str]:
        slots = self.slots("render")
        if role is None:
            return slots
        return [
            slot
            for slot in slots
            if str(self.get("render", slot).get("role", "n8n")).strip().lower() == role.lower()
        ]

    def render_role(self, slot: str) -> str:
        return str(self.get("render", slot).get("role", "n8n")).strip().lower()

    def exists(self, infra_type: str, slot: str) -> bool:
        return slot in self._registry.get(infra_type, {})

    def get(self, infra_type: str, slot: str) -> dict[str, Any]:
        if not self.exists(infra_type, slot):
            raise KeyError(f"Unknown {infra_type} slot: {slot}")
        value = self._registry[infra_type][slot]
        if not isinstance(value, dict):
            raise RuntimeError(f"Registry entry {infra_type}.{slot} must be an object")
        return value


registry = InfraRegistry()
