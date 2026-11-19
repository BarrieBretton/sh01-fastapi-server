# infra/auth.py

import os

from fastapi import Header, HTTPException


CONTROL_PLANE_API_KEY = os.getenv(
    "CONTROL_PLANE_API_KEY",
    "",
).strip()


def require_control_plane_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-KEY",
    ),
) -> None:
    if os.getenv("SH01_ROLE", "").strip().lower() != "controller":
        raise HTTPException(status_code=403, detail="Controller role required")

    if not CONTROL_PLANE_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="CONTROL_PLANE_API_KEY is not configured",
        )

    if not x_api_key or x_api_key != CONTROL_PLANE_API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Missing/invalid X-API-KEY",
        )