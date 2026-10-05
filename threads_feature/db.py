from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import asyncpg

from .config import Settings


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self.pool is not None:
            return

        self.pool = await asyncpg.create_pool(
            dsn=self.settings.database_url,
            min_size=1,
            max_size=5,
            command_timeout=30,
            # Required/recommended for PgBouncer-style transaction pooling
            # such as the Supabase transaction pooler on port 6543.
            statement_cache_size=0,
        )

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[asyncpg.Connection]:
        if self.pool is None:
            await self.connect()

        assert self.pool is not None

        async with self.pool.acquire() as conn:
            yield conn
