"""The browser scripts must at least parse (node is optional on CI)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

STATIC = Path(__file__).parent.parent / "src" / "sentientai" / "web" / "static"
SCRIPTS = sorted(p for p in STATIC.glob("*.js"))


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses(script: Path) -> None:
    result = subprocess.run(
        ["node", "--check", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_every_page_loads_only_existing_scripts() -> None:
    pages = (STATIC.parent / "pages").glob("*.html")
    for page in pages:
        for line in page.read_text().splitlines():
            if 'src="/static/' in line:
                name = line.split('src="/static/')[1].split('"')[0]
                assert (STATIC / name).is_file(), f"{page.name} loads missing {name}"
