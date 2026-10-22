from __future__ import annotations

import asyncio
import os
import shutil
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlparse

import pytest

from sentientai.config import Settings, load_settings
from sentientai.storage.database import Database
from sentientai.storage.schema import metadata

POSTGRES_URL = os.environ.get(
    "SENTIENT_TEST_POSTGRES_URL", "postgresql+asyncpg://sentient:sentient@127.0.0.1:5432/sentientai"
)
REDIS_URL = os.environ.get("SENTIENT_TEST_REDIS_URL", "redis://127.0.0.1:6379/15")


def _reachable(url: str) -> bool:
    parsed = urlparse(url.replace("+asyncpg", ""))
    try:
        with socket.create_connection((parsed.hostname or "127.0.0.1", parsed.port or 0), 0.5):
            return True
    except OSError:
        return False


def postgres_available() -> bool:
    return _reachable(POSTGRES_URL)


def redis_available() -> bool:
    return _reachable(REDIS_URL)


def docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    import subprocess

    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=10, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("SENTIENT_CONFIG", raising=False)
    s = load_settings(
        storage={"data_dir": tmp_path / "data"},
        safety={"workspace_roots": [tmp_path / "workspace"], "audit_fsync": False},
        sandbox={"backend": "disabled"},
        web={"search_backend": "disabled"},
        models={"embedding": {"provider": "hashing", "dimensions": 256}},
    )
    s.ensure_directories()
    return s


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.integration)])
async def db(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[Database]:
    if request.param == "postgres":
        if not postgres_available():
            pytest.skip("PostgreSQL not reachable")
        database = Database(POSTGRES_URL)
        async with database.engine.begin() as conn:
            await conn.run_sync(metadata.drop_all)
    else:
        database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await database.create_all()
    try:
        yield database
    finally:
        await database.close()


@pytest.fixture
async def sqlite_db(tmp_path: Path) -> AsyncIterator[Database]:
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await database.create_all()
    try:
        yield database
    finally:
        await database.close()


async def wait_for(predicate, timeout: float = 5.0, interval: float = 0.02) -> None:  # type: ignore[no-untyped-def]
    """Poll an (a)sync predicate until truthy."""

    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        result = predicate()
        if asyncio.iscoroutine(result):
            result = await result
        if result:
            return
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met before timeout")
        await asyncio.sleep(interval)
