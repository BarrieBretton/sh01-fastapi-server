# infra/state.py

import os


class InfraState:
    def __init__(self) -> None:
        self._active = {
            "render": (
                os.getenv("ACTIVE_RENDER_SLOT")
                or None
            ),
            "postgres": (
                os.getenv("ACTIVE_POSTGRES_SLOT")
                or None
            ),
            "b2": (
                os.getenv("ACTIVE_B2_SLOT")
                or None
            ),
        }

    def get_active(
        self,
        infra_type: str,
    ) -> str | None:
        return self._active.get(
            infra_type
        )

    def set_active(
        self,
        infra_type: str,
        slot: str,
    ) -> None:
        self._active[
            infra_type
        ] = slot

    def all(self) -> dict[str, str | None]:
        return dict(
            self._active
        )


state = InfraState()