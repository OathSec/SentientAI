"""``sentientai users``: the way to make the first administrator, and to recover accounts."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sentientai.accounts.passwords import verify_password
from sentientai.cli import app
from sentientai.config import load_settings
from sentientai.storage.database import Database
from sentientai.storage.records import UserRole, UserStatus

GOOD = "correct horse battery staple"


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.delenv("SENTIENT_CONFIG", raising=False)
    monkeypatch.setenv("SENTIENT_STORAGE__DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SENTIENT_SAFETY__WORKSPACE_ROOTS", f'["{tmp_path / "workspace"}"]')
    runner = CliRunner()

    def run(*args: str, stdin: str | None = None):  # type: ignore[no-untyped-def]
        return runner.invoke(app, ["users", *args], input=stdin)

    return run


def users(tmp_path: Path) -> dict[str, object]:
    async def go() -> dict[str, object]:
        settings = load_settings(storage={"data_dir": tmp_path / "data"})
        db = Database(settings.storage.resolved_database_url())
        try:
            return {u.username: u for u in await db.users.list()}
        finally:
            await db.close()

    return asyncio.run(go())


def test_first_admin_and_everyday_management(cli, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    empty = cli("list")
    assert empty.exit_code == 0 and "users create NAME --admin" in empty.output

    made = cli("create", "root", "--admin", "--name", "Root", "--password-stdin", stdin=GOOD + "\n")
    assert made.exit_code == 0 and "created admin root" in made.output
    prompted = cli("create", "dana", "--teach", stdin=f"{GOOD}\n{GOOD}\n")
    assert prompted.exit_code == 0

    found = users(tmp_path)
    root, dana = found["root"], found["dana"]
    assert root.role is UserRole.ADMIN and root.display_name == "Root"  # type: ignore[attr-defined]
    assert dana.can_teach and dana.role is UserRole.USER  # type: ignore[attr-defined]
    assert verify_password(GOOD, root.password_hash)  # type: ignore[attr-defined]

    weak = cli("create", "eve", "--password-stdin", stdin="short\n")
    assert weak.exit_code == 1 and "at least 12 characters" in weak.output
    taken = cli("create", "ROOT", "--password-stdin", stdin=GOOD + "\n")
    assert taken.exit_code == 1 and "taken" in taken.output
    assert "eve" not in users(tmp_path)

    listing = cli("list").output
    assert "root" in listing and "admin" in listing and "teaches" in listing

    assert cli("disable", "dana").exit_code == 0
    assert users(tmp_path)["dana"].status is UserStatus.DISABLED  # type: ignore[attr-defined]
    assert cli("enable", "dana").exit_code == 0
    assert users(tmp_path)["dana"].status is UserStatus.ACTIVE  # type: ignore[attr-defined]
    assert cli("set-role", "dana", "admin").exit_code == 0
    assert users(tmp_path)["dana"].role is UserRole.ADMIN  # type: ignore[attr-defined]
    assert cli("set-role", "dana", "wizard").exit_code == 1
    missing = cli("disable", "nobody")
    assert missing.exit_code == 1 and "No such user" in missing.output


def test_recovering_accounts(cli, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cli("create", "dana", "--password-stdin", stdin=GOOD + "\n")

    async def lock_and_enrol() -> None:
        settings = load_settings(storage={"data_dir": tmp_path / "data"})
        db = Database(settings.storage.resolved_database_url())
        try:
            user = (await db.users.list())[0]
            user.failed_logins, user.totp_secret = 3, "JBSWY3DPEHPK3PXP"
            from datetime import UTC, datetime, timedelta

            user.locked_until = datetime.now(UTC) + timedelta(hours=1)
            await db.users.save(user)
        finally:
            await db.close()

    asyncio.run(lock_and_enrol())
    assert cli("unlock", "dana").exit_code == 0
    assert users(tmp_path)["dana"].locked_until is None  # type: ignore[attr-defined]
    assert cli("reset-2fa", "dana").exit_code == 0
    assert users(tmp_path)["dana"].totp_secret is None  # type: ignore[attr-defined]

    new = "a completely different phrase"
    bad = cli("passwd", "dana", "--password-stdin", stdin="weak\n")
    assert bad.exit_code == 1
    assert cli("passwd", "dana", "--password-stdin", stdin=new + "\n").exit_code == 0
    assert verify_password(new, users(tmp_path)["dana"].password_hash)  # type: ignore[attr-defined]


def test_invites_are_single_use_codes(cli, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    first = cli("invite", "--teach", "--days", "2")
    second = cli("invite")
    assert first.exit_code == 0 and second.exit_code == 0
    a, b = first.stdout.strip(), second.stdout.strip()
    assert len(a) >= 20 and a != b

    async def stored() -> list[object]:
        settings = load_settings(storage={"data_dir": tmp_path / "data"})
        db = Database(settings.storage.resolved_database_url())
        try:
            return list(await db.invites.list())
        finally:
            await db.close()

    invites = asyncio.run(stored())
    assert len(invites) == 2 and all(a not in repr(i) and b not in repr(i) for i in invites)
