"""Filesystem tools, jailed to the configured workspace roots."""

from __future__ import annotations

import difflib
import fnmatch
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sentientai.config import RiskLevel
from sentientai.errors import ToolError
from sentientai.tools.base import A, Tool, ToolArgs, ToolContext, ToolResult

SKIP_DIRS = {
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".tox",
    ".idea",
    ".vscode",
    "target",
}
MAX_READ_BYTES = 2_000_000


class Workspace:
    """Resolves model-supplied paths and refuses anything outside the roots."""

    def __init__(self, roots: list[Path], protected: list[str] | None = None) -> None:
        if not roots:
            raise ValueError("at least one workspace root is required")
        self.roots = [r.expanduser().resolve() for r in roots]
        self.protected = protected or []

    @property
    def primary(self) -> Path:
        return self.roots[0]

    def resolve(self, path: str, *, must_exist: bool = False) -> Path:
        raw = Path(path).expanduser()
        candidate = raw if raw.is_absolute() else self.primary / raw
        resolved = candidate.resolve()  # follows symlinks, so links cannot escape the jail
        if not any(resolved == root or resolved.is_relative_to(root) for root in self.roots):
            raise ToolError(
                f"path {path!r} is outside the workspace ({', '.join(map(str, self.roots))})"
            )
        if must_exist and not resolved.exists():
            raise ToolError(f"path {path!r} does not exist")
        return resolved

    def relative(self, path: Path) -> str:
        for root in self.roots:
            if path == root or path.is_relative_to(root):
                rel = path.relative_to(root)
                return str(rel) if str(rel) != "." else "."
        return str(path)

    def is_protected(self, path: Path) -> bool:
        rel = self.relative(path)
        parts = Path(rel).parts
        for pattern in self.protected:
            if fnmatch.fnmatch(rel, pattern) or any(fnmatch.fnmatch(p, pattern) for p in parts):
                return True
        return False


def _is_binary(data: bytes) -> bool:
    return b"\x00" in data[:8192]


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024:
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}TB"


class _FsTool(Tool[A]):
    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace


# ------------------------------------------------------------------- list
class FsListArgs(ToolArgs):
    path: str = Field(default=".", description="Directory relative to the workspace root.")
    depth: int = Field(default=2, ge=1, le=6)
    pattern: str | None = Field(default=None, description="Optional glob filter, e.g. '*.py'.")
    include_hidden: bool = False


class FsList(_FsTool[FsListArgs]):
    name = "fs_list"
    description = "List a directory tree in the workspace with file sizes."
    args_model = FsListArgs
    tags = ("filesystem", "read")

    async def run(self, args: FsListArgs, ctx: ToolContext) -> ToolResult:
        root = self.ws.resolve(args.path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"{args.path!r} is not a directory")
        lines: list[str] = []
        limit = 400

        def walk(directory: Path, level: int) -> None:
            try:
                entries = sorted(
                    directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())
                )
            except PermissionError:
                return
            for entry in entries:
                if len(lines) >= limit:
                    return
                if not args.include_hidden and entry.name.startswith("."):
                    continue
                indent = "  " * level
                if entry.is_dir():
                    if entry.name in SKIP_DIRS:
                        lines.append(f"{indent}{entry.name}/ (skipped)")
                        continue
                    lines.append(f"{indent}{entry.name}/")
                    if level + 1 < args.depth:
                        walk(entry, level + 1)
                elif args.pattern is None or fnmatch.fnmatch(entry.name, args.pattern):
                    try:
                        size = entry.stat().st_size
                    except OSError:
                        size = 0
                    lines.append(f"{indent}{entry.name} ({_human(size)})")

        walk(root, 0)
        if len(lines) >= limit:
            lines.append(f"... truncated at {limit} entries")
        header = f"{self.ws.relative(root)}/"
        return ToolResult.success("\n".join([header, *lines]) if lines else f"{header} (empty)")


# ------------------------------------------------------------------- read
class FsReadArgs(ToolArgs):
    path: str
    start_line: int = Field(default=1, ge=1)
    max_lines: int = Field(default=400, ge=1, le=4000)


class FsRead(_FsTool[FsReadArgs]):
    name = "fs_read"
    description = "Read a text file from the workspace (with line numbers)."
    args_model = FsReadArgs
    tags = ("filesystem", "read")

    def assess(self, args: FsReadArgs, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        try:
            path = self.ws.resolve(args.path)
        except ToolError:
            return self.risk, []
        if self.ws.is_protected(path):
            return RiskLevel.HIGH, [f"reads protected path {args.path}"]
        return self.risk, []

    async def run(self, args: FsReadArgs, ctx: ToolContext) -> ToolResult:
        path = self.ws.resolve(args.path, must_exist=True)
        if path.is_dir():
            raise ToolError(f"{args.path!r} is a directory; use fs_list")
        size = path.stat().st_size
        with path.open("rb") as fh:
            data = fh.read(MAX_READ_BYTES)
        if _is_binary(data):
            return ToolResult.success(f"{args.path}: binary file ({_human(size)}); not displayed")
        lines = data.decode("utf-8", errors="replace").splitlines()
        start = args.start_line - 1
        window = lines[start : start + args.max_lines]
        width = len(str(start + len(window)))
        body = "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(window, start + 1))
        footer = ""
        if start + len(window) < len(lines):
            footer = f"\n[showing lines {start + 1}-{start + len(window)} of {len(lines)}]"
        if size > MAX_READ_BYTES:
            footer += (
                f"\n[file is {_human(size)}; only the first {_human(MAX_READ_BYTES)} were read]"
            )
        return ToolResult.success(body + footer, data={"lines": len(lines)})


# ----------------------------------------------------------------- search
class FsSearchArgs(ToolArgs):
    pattern: str = Field(description="Regular expression (or literal text when regex=false).")
    path: str = "."
    glob: str = Field(default="*", description="Filename glob, e.g. '*.py'.")
    regex: bool = True
    ignore_case: bool = True
    max_results: int = Field(default=60, ge=1, le=500)


class FsSearch(_FsTool[FsSearchArgs]):
    name = "fs_search"
    description = "Search file contents in the workspace (like grep -rn)."
    args_model = FsSearchArgs
    tags = ("filesystem", "read")

    async def run(self, args: FsSearchArgs, ctx: ToolContext) -> ToolResult:
        root = self.ws.resolve(args.path, must_exist=True)
        flags = re.IGNORECASE if args.ignore_case else 0
        try:
            rx = re.compile(args.pattern if args.regex else re.escape(args.pattern), flags)
        except re.error as exc:
            raise ToolError(f"invalid regex: {exc}") from exc
        results: list[str] = []
        files_scanned = 0
        candidates = [root] if root.is_file() else _iter_files(root)
        for file in candidates:
            if not fnmatch.fnmatch(file.name, args.glob) or self.ws.is_protected(file):
                continue
            try:
                if file.stat().st_size > MAX_READ_BYTES:
                    continue
                data = file.read_bytes()
            except OSError:
                continue
            if _is_binary(data):
                continue
            files_scanned += 1
            for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if rx.search(line):
                    results.append(f"{self.ws.relative(file)}:{lineno}: {line.strip()[:240]}")
                    if len(results) >= args.max_results:
                        break
            if len(results) >= args.max_results:
                results.append(f"[stopped at {args.max_results} matches]")
                break
        if not results:
            return ToolResult.success(f"no matches in {files_scanned} files")
        return ToolResult.success("\n".join(results), data={"files_scanned": files_scanned})


def _iter_files(root: Path) -> Iterator[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            yield Path(dirpath) / name


# ------------------------------------------------------------------ write
class FsWriteArgs(ToolArgs):
    path: str
    content: str
    mode: Literal["overwrite", "append", "create"] = Field(
        default="create", description="'create' fails if the file exists."
    )


class FsWrite(_FsTool[FsWriteArgs]):
    name = "fs_write"
    description = "Write a text file in the workspace (create, overwrite or append)."
    args_model = FsWriteArgs
    risk = RiskLevel.MEDIUM
    tags = ("filesystem", "write")

    def assess(self, args: FsWriteArgs, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        try:
            path = self.ws.resolve(args.path)
        except ToolError:
            return self.risk, []
        if self.ws.is_protected(path):
            return RiskLevel.CRITICAL, [f"writes protected path {args.path}"]
        return self.risk, []

    async def run(self, args: FsWriteArgs, ctx: ToolContext) -> ToolResult:
        path = self.ws.resolve(args.path)
        if path.is_dir():
            raise ToolError(f"{args.path!r} is a directory")
        if args.mode == "create" and path.exists():
            raise ToolError(f"{args.path!r} already exists; use mode='overwrite' or fs_edit")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a" if args.mode == "append" else "w", encoding="utf-8") as fh:
            fh.write(args.content)
        return ToolResult.success(
            f"wrote {len(args.content)} chars to {self.ws.relative(path)} ({args.mode})"
        )


# ------------------------------------------------------------------- edit
class FsEditArgs(ToolArgs):
    path: str
    old: str = Field(description="Exact text to replace (must match exactly).")
    new: str
    expected_occurrences: int = Field(default=1, ge=1)


class FsEdit(_FsTool[FsEditArgs]):
    name = "fs_edit"
    description = "Replace exact text in a workspace file; returns a unified diff."
    args_model = FsEditArgs
    risk = RiskLevel.MEDIUM
    tags = ("filesystem", "write")

    def assess(self, args: FsEditArgs, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        try:
            path = self.ws.resolve(args.path)
        except ToolError:
            return self.risk, []
        if self.ws.is_protected(path):
            return RiskLevel.CRITICAL, [f"edits protected path {args.path}"]
        return self.risk, []

    async def run(self, args: FsEditArgs, ctx: ToolContext) -> ToolResult:
        path = self.ws.resolve(args.path, must_exist=True)
        original = path.read_text(encoding="utf-8")
        count = original.count(args.old)
        if count != args.expected_occurrences:
            raise ToolError(
                f"expected {args.expected_occurrences} occurrence(s) of the old text, found {count}"
            )
        updated = original.replace(args.old, args.new)
        path.write_text(updated, encoding="utf-8")
        rel = self.ws.relative(path)
        diff = "".join(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                updated.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=2,
            )
        )
        return ToolResult.success(diff or "no change")


# ----------------------------------------------------------------- delete
class FsDeleteArgs(ToolArgs):
    path: str


class FsDelete(_FsTool[FsDeleteArgs]):
    name = "fs_delete"
    description = "Delete a single file in the workspace (not directories)."
    args_model = FsDeleteArgs
    risk = RiskLevel.HIGH
    tags = ("filesystem", "write")

    def assess(self, args: FsDeleteArgs, ctx: ToolContext) -> tuple[RiskLevel, list[str]]:
        try:
            path = self.ws.resolve(args.path)
        except ToolError:
            return self.risk, []
        if self.ws.is_protected(path):
            return RiskLevel.CRITICAL, [f"deletes protected path {args.path}"]
        return self.risk, ["irreversible file deletion"]

    async def run(self, args: FsDeleteArgs, ctx: ToolContext) -> ToolResult:
        path = self.ws.resolve(args.path, must_exist=True)
        if path.is_dir():
            raise ToolError("refusing to delete a directory")
        path.unlink()
        return ToolResult.success(f"deleted {self.ws.relative(path)}")


def filesystem_tools(workspace: Workspace) -> list[Tool[Any]]:
    return [
        FsList(workspace),
        FsRead(workspace),
        FsSearch(workspace),
        FsWrite(workspace),
        FsEdit(workspace),
        FsDelete(workspace),
    ]
