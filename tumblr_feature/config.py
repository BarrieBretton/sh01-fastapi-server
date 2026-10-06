from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from urllib.parse import quote_plus


def normalize_account(value: str) -> str:
    return str(value or "").strip().lstrip("@").lower()


def _prefix(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


LEGACY_PREFIXES = {
    "erika.devereux": "ERIKA_DEVEREUX",
    "vlvt.ave": "VLVT_AVE",
    "cyootstuff": "CYOOTSTUFF",
}


def load_account_prefixes() -> dict[str, str]:
    result: dict[str, str] = {}
    for account, prefix in LEGACY_PREFIXES.items():
        if os.getenv(f"TUMBLR_BLOG_IDENTIFIER_{prefix}", "").strip():
            result[account] = prefix

    raw = os.getenv("TUMBLR_ACCOUNTS_JSON", "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("TUMBLR_ACCOUNTS_JSON must contain valid JSON") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("TUMBLR_ACCOUNTS_JSON must be a JSON object")
        for account, prefix in parsed.items():
            key = normalize_account(account)
            pfx = _prefix(str(prefix or key))
            if not key or not pfx:
                raise RuntimeError("TUMBLR_ACCOUNTS_JSON contains an empty account/prefix")
            result[key] = pfx
    return result


ACCOUNT_PREFIXES = load_account_prefixes()


@dataclass(frozen=True)
class TumblrCredentials:
    account: str
    consumer_key: str
    consumer_secret: str
    token: str
    token_secret: str
    blog_identifier: str


@dataclass(frozen=True)
class Settings:
    database_url: str
    internal_api_key: str
    api_base: str = "https://api.tumblr.com"
    media_max_bytes: int = 500 * 1024 * 1024
    user_agent: str = "SH01-Tumblr-Publisher/1.0"

    @classmethod
    def from_env(cls) -> "Settings":
        host = os.getenv("DB_POSTGRESDB_HOST", "").strip()
        port = os.getenv("DB_POSTGRESDB_PORT", "6543").strip()
        database = os.getenv("DB_POSTGRESDB_DATABASE", "postgres").strip()
        user = os.getenv("DB_POSTGRESDB_USER", "").strip()
        password = os.getenv("DB_POSTGRESDB_PASSWORD", "").strip()
        internal_api_key = (
            os.getenv("TUMBLR_INTERNAL_API_KEY")
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
                ("TUMBLR_INTERNAL_API_KEY/SOCIAL_INTERNAL_API_KEY/THREADS_INTERNAL_API_KEY", internal_api_key),
            ) if not value
        ]
        if missing:
            raise RuntimeError("Missing required Tumblr feature environment variables: " + ", ".join(missing))
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
            api_base=os.getenv("TUMBLR_API_BASE", "https://api.tumblr.com").rstrip("/"),
            media_max_bytes=max(1_048_576, int(os.getenv("TUMBLR_MEDIA_MAX_BYTES", str(500 * 1024 * 1024)))),
            user_agent=os.getenv("TUMBLR_USER_AGENT", "SH01-Tumblr-Publisher/1.0").strip() or "SH01-Tumblr-Publisher/1.0",
        )


def credentials_for(account: str) -> TumblrCredentials:
    key = normalize_account(account)
    prefix = ACCOUNT_PREFIXES.get(key)
    if not prefix:
        known = ", ".join(sorted(ACCOUNT_PREFIXES)) or "(none configured)"
        raise ValueError(f"Unknown Tumblr account @{key}; configured accounts: {known}")
    envs = {
        "consumer_key": f"TUMBLR_CONSUMER_KEY_{prefix}",
        "consumer_secret": f"TUMBLR_CONSUMER_SECRET_{prefix}",
        "token": f"TUMBLR_TOKEN_{prefix}",
        "token_secret": f"TUMBLR_TOKEN_SECRET_{prefix}",
        "blog_identifier": f"TUMBLR_BLOG_IDENTIFIER_{prefix}",
    }
    values = {field: os.getenv(name, "").strip() for field, name in envs.items()}
    missing = [envs[field] for field, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"Tumblr credentials for @{key} are incomplete; missing: {', '.join(missing)}")
    return TumblrCredentials(account=key, **values)
