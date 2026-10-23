"""Isolated code execution.

:class:`DockerSandbox` runs each command in a fresh, locked-down container:
no network (by default), read-only root filesystem, all Linux capabilities
dropped, no privilege escalation, an unprivileged user, and memory / CPU /
PID limits. A per-task scratch directory is mounted read-write at ``/work``
so multi-step work can build on earlier files; the workspace is mounted
read-only at ``/workspace``. Containers are labelled so the kill switch can
terminate all of them at once.

:class:`LocalUnsafeSandbox` runs directly on the host. It exists for
development machines without Docker, and every call through it is escalated
to HIGH risk so it always requires operator approval.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from sentientai.config import RiskLevel, SandboxConfig
from sentientai.core.clock import new_id
from sentientai.core.logging import get_logger
from sentientai.errors import SandboxError
from sentientai.tools.base import Tool, ToolArgs, ToolContext, ToolResult

log = get_logger(__name__)

LABEL = "sentientai.sandbox"


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float

    def render(self) -> str:
        parts = [f"exit_code: {self.exit_code}" + (" (timed out)" if self.timed_out else "")]
        if self.stdout:
            parts.append(f"stdout:\n{self.stdout}")
        if self.stderr:
            parts.append(f"stderr:\n{self.stderr}")
        return "\n".join(parts)


class SandboxBackend(Protocol):
    name: str
    isolated: bool

    async def run(
        self, command: list[str], *, scratch: Path, timeout_s: float, task_id: str | None
    ) -> ExecResult: ...

    async def kill_all(self) -> int: ...


async def _collect(
    proc: asyncio.subprocess.Process, timeout_s: float, max_bytes: int
) -> tuple[bytes, bytes, bool]:
    async def read(stream: asyncio.StreamReader | None) -> bytes:
        if stream is None:
            return b""
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            if size < max_bytes:
                chunks.append(chunk[: max_bytes - size])
            size += len(chunk)
        data = b"".join(chunks)
        if size > max_bytes:
            data += f"\n[... {size - max_bytes} bytes truncated]".encode()
        return data

    reader = asyncio.gather(read(proc.stdout), read(proc.stderr), proc.wait())
    try:
        out, err, _ = await asyncio.wait_for(reader, timeout=timeout_s)
        return out, err, False
    except TimeoutError:
        return b"", b"", True


class DockerSandbox:
    name = "docker"
    isolated = True

    def __init__(self, config: SandboxConfig, workspace: Path | None = None) -> None:
        self.config = config
        self.workspace = workspace
        if shutil.which("docker") is None:
            raise SandboxError("docker CLI not found; install Docker or set sandbox.backend")

    def build_command(
        self, name: str, command: list[str], scratch: Path, task_id: str | None
    ) -> list[str]:
        cfg = self.config
        cmd = [
            "docker",
            "run",
            "--rm",
            "--name",
            name,
            "--label",
            f"{LABEL}=1",
            "--label",
            f"{LABEL}.task={task_id or 'none'}",
            "--network",
            "bridge" if cfg.network else "none",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,size=64m",  # noqa: S108 - container path
            "--memory",
            cfg.memory,
            "--memory-swap",
            cfg.memory,
            "--cpus",
            str(cfg.cpus),
            "--pids-limit",
            str(cfg.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--user",
            cfg.user,
            "--workdir",
            "/work",
            "--env",
            "HOME=/tmp",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            "PYTHONUNBUFFERED=1",
            "-v",
            f"{scratch}:/work:rw",
        ]
        if self.workspace is not None and cfg.mount_workspace_readonly:
            cmd += ["-v", f"{self.workspace.resolve()}:/workspace:ro"]
        return [*cmd, cfg.image, *command]

    async def run(
        self, command: list[str], *, scratch: Path, timeout_s: float, task_id: str | None
    ) -> ExecResult:
        scratch.mkdir(parents=True, exist_ok=True)
        os.chmod(scratch, 0o777)  # noqa: S103 - the unprivileged container user must write here
        name = new_id("sentient-sbx").replace("_", "-")
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *self.build_command(name, command, scratch, task_id),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err, timed_out = await _collect(proc, timeout_s, self.config.max_output_bytes)
        if timed_out:
            await self._kill(name)
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.wait()
        return ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=out.decode("utf-8", errors="replace"),
            stderr=err.decode("utf-8", errors="replace"),
            timed_out=timed_out,
            duration_s=time.monotonic() - started,
        )

    async def _kill(self, name: str) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "kill",
            name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()

    async def kill_all(self) -> int:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "ps",
            "-q",
            "--filter",
            f"label={LABEL}=1",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        ids = out.decode().split()
        for container in ids:
            await self._kill(container)
        return len(ids)


class LocalUnsafeSandbox:
    name = "local_unsafe"
    isolated = False

    def __init__(self, config: SandboxConfig) -> None:
        self.config = config
        self._procs: set[asyncio.subprocess.Process] = set()

    async def run(
        self, command: list[str], *, scratch: Path, timeout_s: float, task_id: str | None
    ) -> ExecResult:
        scratch.mkdir(parents=True, exist_ok=True)
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(scratch),
            "PYTHONUNBUFFERED": "1",
        }
        started = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=scratch,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        self._procs.add(proc)
        try:
            out, err, timed_out = await _collect(proc, timeout_s, self.config.max_output_bytes)
            if timed_out:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, 9)
                await proc.wait()
        finally:
            self._procs.discard(proc)
        return ExecResult(
            proc.returncode if proc.returncode is not None else -1,
            out.decode("utf-8", errors="replace"),
            err.decode("utf-8", errors="replace"),
            timed_out,
            time.monotonic() - started,
        )

    async def kill_all(self) -> int:
        count = 0
        for proc in list(self._procs):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, 9)
                count += 1
        return count


# ------------------------------------------------------------------- tools
class _SandboxTool(Tool[Any]):
    def __init__(self, backend: SandboxBackend, config: SandboxConfig, scratch_root: Path) -> None:
        self.backend = backend
        self.config = config
        self.scratch_root = scratch_root
        self.timeout_s = config.timeout_s + 15
        if config.network:  # tell Fuzzy the truth about what its code can reach
            self.description = self.description.replace("no network", "internet access")

    def _scratch(self, ctx: ToolContext) -> Path:
        return ctx.scratch_dir or self.scratch_root / (ctx.task_id or "adhoc")

    def assess(self, args: Any, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        if not self.backend.isolated:
            return RiskLevel.HIGH, ["code runs directly on the host (local_unsafe sandbox)"]
        if self.config.network:
            return RiskLevel.MEDIUM, ["sandbox has network access"]
        return self.risk, []


class PythonArgs(ToolArgs):
    code: str = Field(description="Python 3 source to execute. Print results to stdout.")
    timeout_s: float | None = Field(default=None, gt=0, le=600)


class SandboxPython(_SandboxTool):
    name = "sandbox_python"
    description = (
        "Execute Python in an isolated sandbox (no network, /work is a persistent per-task "
        "scratch dir, /workspace is read-only). Use print() to return results."
    )
    args_model = PythonArgs
    risk = RiskLevel.LOW
    tags = ("execution",)

    async def run(self, args: PythonArgs, ctx: ToolContext) -> ToolResult:
        scratch = self._scratch(ctx)
        scratch.mkdir(parents=True, exist_ok=True)
        script = f"snippet_{new_id('py').split('_', 1)[1]}.py"
        (scratch / script).write_text(args.code, encoding="utf-8")
        path = f"/work/{script}" if self.backend.isolated else str(scratch / script)
        result = await self.backend.run(
            ["python", path],
            scratch=scratch,
            timeout_s=args.timeout_s or self.config.timeout_s,
            task_id=ctx.task_id,
        )
        ok = result.exit_code == 0 and not result.timed_out
        output = result.render()
        return ToolResult(
            ok=ok,
            output=output,
            error=None if ok else f"exit code {result.exit_code}",
            data={"exit_code": result.exit_code, "duration_s": round(result.duration_s, 3)},
        )


class ShellArgs(ToolArgs):
    command: str = Field(description="POSIX shell command, run with sh -c in the sandbox.")
    timeout_s: float | None = Field(default=None, gt=0, le=600)


class SandboxShell(_SandboxTool):
    name = "sandbox_shell"
    description = "Run a shell command in an isolated sandbox container (working dir /work)."
    args_model = ShellArgs
    risk = RiskLevel.LOW
    tags = ("execution",)

    async def run(self, args: ShellArgs, ctx: ToolContext) -> ToolResult:
        result = await self.backend.run(
            ["sh", "-c", args.command],
            scratch=self._scratch(ctx),
            timeout_s=args.timeout_s or self.config.timeout_s,
            task_id=ctx.task_id,
        )
        ok = result.exit_code == 0 and not result.timed_out
        return ToolResult(
            ok=ok,
            output=result.render(),
            error=None if ok else f"exit code {result.exit_code}",
            data={"exit_code": result.exit_code},
        )


def build_sandbox(config: SandboxConfig, workspace: Path | None) -> SandboxBackend | None:
    if config.backend == "disabled":
        return None
    if config.backend == "local_unsafe":
        log.warning("sandbox backend is local_unsafe: every execution will require approval")
        return LocalUnsafeSandbox(config)
    return DockerSandbox(config, workspace)


def sandbox_tools(
    backend: SandboxBackend, config: SandboxConfig, scratch_root: Path
) -> list[Tool[Any]]:
    return [
        SandboxPython(backend, config, scratch_root),
        SandboxShell(backend, config, scratch_root),
    ]
