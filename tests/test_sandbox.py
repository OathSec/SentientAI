from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from sentientai.config import RiskLevel, SandboxConfig
from sentientai.tools.base import ToolContext
from sentientai.tools.sandbox import DockerSandbox, LocalUnsafeSandbox, SandboxPython, SandboxShell
from tests.conftest import docker_available

IMAGE = "python:3.12-slim"

pytestmark = []


def _image_present() -> bool:
    import subprocess

    return (
        subprocess.run(
            ["docker", "image", "inspect", IMAGE], capture_output=True, check=False
        ).returncode
        == 0
    )


docker = pytest.mark.skipif(
    not (docker_available() and _image_present()), reason=f"docker or image {IMAGE} unavailable"
)


def test_docker_command_is_locked_down(tmp_path: Path) -> None:
    sandbox = DockerSandbox.__new__(DockerSandbox)
    sandbox.config = SandboxConfig()
    sandbox.workspace = tmp_path
    cmd = sandbox.build_command("n", ["python", "-V"], tmp_path / "scratch", "t1")
    joined = " ".join(cmd)
    for flag in [
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "--security-opt no-new-privileges",
        "--user 65534:65534",
        "--pids-limit 128",
        "--memory 512m",
        f"{tmp_path}:/workspace:ro",
    ]:
        assert flag in joined, flag
    assert "sentientai.sandbox.task=t1" in joined


@pytest.mark.integration
@docker
async def test_docker_python_isolation(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "data.txt").write_text("42")
    config = SandboxConfig(image=IMAGE, timeout_s=30)
    backend = DockerSandbox(config, workspace)
    tool = SandboxPython(backend, config, tmp_path / "scratch")
    ctx = ToolContext(task_id="t1")
    code = """
import os, socket, pathlib
print("uid", os.getuid())
print("data", pathlib.Path('/workspace/data.txt').read_text())
pathlib.Path('/work/result.txt').write_text('ok')
for target, label in [('/workspace/x', 'workspace'), ('/usr/evil', 'rootfs')]:
    try:
        open(target, 'w').write('x'); print(label, 'WRITABLE')
    except OSError:
        print(label, 'readonly')
try:
    socket.create_connection(('1.1.1.1', 53), timeout=2); print('net OPEN')
except OSError:
    print('net blocked')
"""
    result = await tool.run(tool.args_model(code=code), ctx)
    assert result.ok, result.output
    assert "uid 65534" in result.output and "data 42" in result.output
    assert "workspace readonly" in result.output and "rootfs readonly" in result.output
    assert "net blocked" in result.output
    assert (
        tmp_path / "scratch" / "t1" / "result.txt"
    ).read_text() == "ok"  # per-task scratch persists

    shell = SandboxShell(backend, config, tmp_path / "scratch")
    listing = await shell.run(shell.args_model(command="cat /work/result.txt && exit 3"), ctx)
    assert not listing.ok and "ok" in listing.output and "exit_code: 3" in listing.output


@pytest.mark.integration
@docker
async def test_docker_timeout_and_kill_all(tmp_path: Path) -> None:
    config = SandboxConfig(image=IMAGE, timeout_s=2)
    backend = DockerSandbox(config)
    result = await backend.run(["sleep", "30"], scratch=tmp_path / "s", timeout_s=2, task_id="t2")
    assert result.timed_out

    task = asyncio.create_task(
        backend.run(["sleep", "30"], scratch=tmp_path / "s", timeout_s=60, task_id="t3")
    )
    for _ in range(50):
        await asyncio.sleep(0.2)
        if await backend.kill_all():
            break
    else:
        pytest.fail("container never appeared")
    result = await asyncio.wait_for(task, 20)
    assert result.exit_code != 0


async def test_local_unsafe_requires_approval(tmp_path: Path) -> None:
    config = SandboxConfig(backend="local_unsafe", timeout_s=5)
    backend = LocalUnsafeSandbox(config)
    tool = SandboxPython(backend, config, tmp_path)
    args = tool.args_model(code="print(6 * 7)")
    risk, reasons = tool.assess(args, ToolContext())
    assert risk is RiskLevel.HIGH and "host" in reasons[0]
    result = await tool.run(args, ToolContext(task_id="t"))
    assert result.ok and "42" in result.output
    slow = await tool.run(
        tool.args_model(code="import time; time.sleep(10)", timeout_s=0.5), ToolContext()
    )
    assert not slow.ok and "timed out" in slow.output
