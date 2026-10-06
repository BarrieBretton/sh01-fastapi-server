from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote_plus


def normalize_account(value: str) -> str:
    return str(value or "").strip().lstrip("@").lower()


def _prefix(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def load_account_prefixes() -> dict[str, str]:
    """
    X_ACCOUNTS_JSON accepts a JSON object mapping X handle -> credential prefix.

    Example:
      {"erika.devereux":"ERIKA_DEVEREUX","cyootstuff":"CYOOTSTUFF"}

    The prefix ERIKA_DEVEREUX resolves these env vars:
      X_API_KEY_ERIKA_DEVEREUX
      X_API_KEY_SECRET_ERIKA_DEVEREUX
      X_ACCESS_TOKEN_ERIKA_DEVEREUX
      X_ACCESS_TOKEN_SECRET_ERIKA_DEVEREUX

    For backward compatibility, X_DEFAULT_ACCOUNT maps the existing legacy vars:
      API_KEY, API_KEY_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET
    """
    result: dict[str, str] = {}

    raw = os.getenv("X_ACCOUNTS_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("X_ACCOUNTS_JSON must contain valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("X_ACCOUNTS_JSON must be a JSON object")
        for account, prefix in parsed.items():
            key = normalize_account(account)
            pfx = _prefix(str(prefix or key))
            if not key or not pfx:
                raise RuntimeError("X_ACCOUNTS_JSON contains an empty account/prefix")
            result[key] = pfx

    legacy = normalize_account(os.getenv("X_DEFAULT_ACCOUNT", ""))
    if legacy:
        result.setdefault(legacy, "__LEGACY__")

    return result


ACCOUNT_PREFIXES: Final[dict[str, str]] = load_account_prefixes()


@dataclass(frozen=True)
class XCredentials:
    account: str
    api_key: str
    api_key_secret: str
    access_token: str
    access_token_secret: str


@dataclass(frozen=True)
class Settings:
    database_url: str
    internal_api_key: str
    api_base: str = "https://api.x.com"
    upload_base: str = "https://upload.twitter.com/1.1/media/upload.json"
    media_max_bytes: int = 512 * 1024 * 1024
    video_chunk_bytes: int = 4 * 1024 * 1024
    video_poll_interval_seconds: float = 5.0
    video_poll_timeout_seconds: int = 600
    stale_job_seconds: int = 900

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.getenv("DB_POSTGRESDB_HOST", "").strip()
        port = os.getenv("DB_POSTGRESDB_PORT", "6543").strip()
        database = os.getenv("DB_POSTGRESDB_DATABASE", "postgres").strip()
        user = os.getenv("DB_POSTGRESDB_USER", "").strip()
        password = os.getenv("DB_POSTGRESDB_PASSWORD", "").strip()

        # By default X and Threads can share the same internal SH01 caller key.
        # Set X_INTERNAL_API_KEY to split them later without code changes.
        internal_api_key = (
            os.getenv("X_INTERNAL_API_KEY")
            or os.getenv("SOCIAL_INTERNAL_API_KEY")
            or os.getenv("THREADS_INTERNAL_API_KEY")
            or ""
        ).strip()

        missing = [
            name for name, value in (
                ("DB_POSTGRESDB_HOST", host),
                ("DB_POSTGRESDB_PORT", port),
                ("DB_POSTGRESDB_DATABASE", database),
                ("DB_POSTGRESDB_USER", user),
                ("DB_POSTGRESDB_PASSWORD", password),
                ("X_INTERNAL_API_KEY/SOCIAL_INTERNAL_API_KEY/THREADS_INTERNAL_API_KEY", internal_api_key),
            )
            if not value
        ]
        if missing:
            raise RuntimeError("Missing required X feature environment variables: " + ", ".join(missing))

        try:
            port_number = int(port)
        except ValueError as exc:
            raise RuntimeError(f"DB_POSTGRESDB_PORT must be an integer; got {port!r}") from exc

        database_url = (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port_number}/{quote_plus(database)}?sslmode=require"
        )

        return cls(
            database_url=database_url,
            internal_api_key=internal_api_key,
            api_base=os.getenv("X_API_BASE", "https://api.x.com").rstrip("/"),
            upload_base=os.getenv("X_UPLOAD_BASE", "https://upload.twitter.com/1.1/media/upload.json"),
            media_max_bytes=max(1_048_576, int(os.getenv("X_MEDIA_MAX_BYTES", str(512 * 1024 * 1024)))),
            video_chunk_bytes=max(1_048_576, int(os.getenv("X_VIDEO_CHUNK_BYTES", str(4 * 1024 * 1024)))),
            video_poll_interval_seconds=max(1.0, float(os.getenv("X_VIDEO_POLL_INTERVAL_SECONDS", "5"))),
            video_poll_timeout_seconds=max(30, int(os.getenv("X_VIDEO_POLL_TIMEOUT_SECONDS", "600"))),
            stale_job_seconds=max(60, int(os.getenv("X_STALE_JOB_SECONDS", "900"))),
        )


def credentials_for(account: str) -> XCredentials:
    key = normalize_account(account)
    prefix = ACCOUNT_PREFIXES.get(key)
    if not prefix:
        known = ", ".join(sorted(ACCOUNT_PREFIXES)) or "(none configured)"
        raise ValueError(f"Unknown X account @{key}; configured accounts: {known}")

    if prefix == "__LEGACY__":
        names = {
            "api_key": "API_KEY",
            "api_key_secret": "API_KEY_SECRET",
            "access_token": "ACCESS_TOKEN",
            "access_token_secret": "ACCESS_TOKEN_SECRET",
        }
    else:
        names = {
            "api_key": f"X_API_KEY_{prefix}",
            "api_key_secret": f"X_API_KEY_SECRET_{prefix}",
            "access_token": f"X_ACCESS_TOKEN_{prefix}",
            "access_token_secret": f"X_ACCESS_TOKEN_SECRET_{prefix}",
        }

    values = {field: os.getenv(env_name, "").strip() for field, env_name in names.items()}
    missing = [names[field] for field, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"X credentials for @{key} are incomplete; missing: {', '.join(missing)}")

    return XCredentials(account=key, **values)
