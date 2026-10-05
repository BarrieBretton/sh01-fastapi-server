from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Final
from urllib.parse import quote_plus


DEFAULT_THREADS_ACCOUNTS: Final[dict[str, str]] = {
    "erika.devereux": "25447546524875159",
    "vlvt.ave": "25962539886681226",
    "cyootstuff": "25634868322816151",
}


def _normalize_account(value: str) -> str:
    return value.strip().lstrip("@").lower()


def load_account_map() -> dict[str, str]:
    """
    Optional THREADS_ACCOUNTS_JSON may override/extend the built-in mapping.

    Example:
    {"vlvt.ave":"259...","new.account":"123..."}
    """
    mapping = dict(DEFAULT_THREADS_ACCOUNTS)

    raw = os.getenv("THREADS_ACCOUNTS_JSON", "").strip()
    if not raw:
        return mapping

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("THREADS_ACCOUNTS_JSON must contain valid JSON") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("THREADS_ACCOUNTS_JSON must be a JSON object")

    for key, value in parsed.items():
        account = _normalize_account(str(key))
        user_id = str(value).strip()

        if not account or not user_id:
            raise RuntimeError(
                "THREADS_ACCOUNTS_JSON contains an empty account or Threads user id"
            )

        mapping[account] = user_id

    return mapping


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False

    raise RuntimeError(
        f"{name} must be one of true/false, 1/0, yes/no, on/off; got {raw!r}"
    )


@dataclass(frozen=True)
class Settings:
    database_url: str
    encryption_key: str
    internal_api_key: str

    api_base: str = "https://graph.threads.net"
    refresh_threshold_days: int = 21
    assumed_long_lived_lifetime_seconds: int = 5_184_000
    poll_interval_seconds: float = 5.0
    poll_timeout_seconds: int = 300

    @classmethod
    def from_env(cls) -> "Settings":
        # Reuse the SAME effective Postgres runtime settings as n8n.
        host = os.getenv("DB_POSTGRESDB_HOST", "").strip()
        port = os.getenv("DB_POSTGRESDB_PORT", "6543").strip()
        database = os.getenv("DB_POSTGRESDB_DATABASE", "postgres").strip()
        user = os.getenv("DB_POSTGRESDB_USER", "").strip()
        password = os.getenv("DB_POSTGRESDB_PASSWORD", "").strip()

        encryption_key = os.getenv("THREADS_TOKEN_ENCRYPTION_KEY", "").strip()
        internal_api_key = (
            os.getenv("THREADS_INTERNAL_API_KEY")
            or os.getenv("X_API_KEY")
            or ""
        ).strip()

        missing: list[str] = []

        if not host:
            missing.append("DB_POSTGRESDB_HOST")
        if not port:
            missing.append("DB_POSTGRESDB_PORT")
        if not database:
            missing.append("DB_POSTGRESDB_DATABASE")
        if not user:
            missing.append("DB_POSTGRESDB_USER")
        if not password:
            missing.append("DB_POSTGRESDB_PASSWORD")
        if not encryption_key:
            missing.append("THREADS_TOKEN_ENCRYPTION_KEY")
        if not internal_api_key:
            missing.append("THREADS_INTERNAL_API_KEY (or X_API_KEY)")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )

        try:
            port_number = int(port)
        except ValueError as exc:
            raise RuntimeError(
                f"DB_POSTGRESDB_PORT must be an integer; got {port!r}"
            ) from exc

        if not 1 <= port_number <= 65535:
            raise RuntimeError(
                f"DB_POSTGRESDB_PORT is out of range: {port_number}"
            )

        # n8n normally has DB_POSTGRESDB_SSL_ENABLED=true for Supabase.
        # Default to true because this feature is explicitly using the
        # existing Supabase runtime database.
        ssl_enabled = _env_bool("DB_POSTGRESDB_SSL_ENABLED", True)

        database_url = (
            f"postgresql://{quote_plus(user)}:{quote_plus(password)}"
            f"@{host}:{port_number}/{quote_plus(database)}"
        )

        if ssl_enabled:
            database_url += "?sslmode=require"

        return cls(
            database_url=database_url,
            encryption_key=encryption_key,
            internal_api_key=internal_api_key,
            api_base=os.getenv(
                "THREADS_API_BASE",
                "https://graph.threads.net",
            ).rstrip("/"),
            refresh_threshold_days=int(
                os.getenv("THREADS_REFRESH_THRESHOLD_DAYS", "21")
            ),
            assumed_long_lived_lifetime_seconds=int(
                os.getenv(
                    "THREADS_LONG_LIVED_LIFETIME_SECONDS",
                    "5184000",
                )
            ),
            poll_interval_seconds=float(
                os.getenv("THREADS_POLL_INTERVAL_SECONDS", "5")
            ),
            poll_timeout_seconds=int(
                os.getenv("THREADS_POLL_TIMEOUT_SECONDS", "300")
            ),
        )


ACCOUNT_MAP = load_account_map()
