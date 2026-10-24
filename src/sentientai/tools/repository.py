"""Codebase and repository inspection tools."""

from __future__ import annotations

import ast
import asyncio
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from sentientai.config import RiskLevel
from sentientai.errors import ToolError
from sentientai.tools.base import A, Tool, ToolArgs, ToolContext, ToolResult
from sentientai.tools.filesystem import SKIP_DIRS, Workspace

LANGUAGES = {
    ".py": "Python",
    ".rs": "Rust",
    ".go": "Go",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".c": "C",
    ".h": "C",
    ".cpp": "C++",
    ".hpp": "C++",
    ".cc": "C++",
    ".cs": "C#",
    ".rb": "Ruby",
    ".php": "PHP",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".md": "Markdown",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".toml": "TOML",
    ".json": "JSON",
    ".html": "HTML",
    ".css": "CSS",
    ".lua": "Lua",
    ".zig": "Zig",
    ".proto": "Protobuf",
}
KEY_FILES = (
    "README.md",
    "README.rst",
    "pyproject.toml",
    "setup.py",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    "CLAUDE.md",
    "AGENTS.md",
)
_FORBIDDEN_GIT_ARGS = re.compile(
    r"^(-c|--config|--exec|--upload-pack|--receive-pack|--output|-o$|--ext-diff|--git-dir|--work-tree)"
)


async def run_git(repo: Path, *args: str, timeout: float = 30.0) -> tuple[int, str]:
    env = {**os.environ, "GIT_PAGER": "cat", "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1"}
    proc = await asyncio.create_subprocess_exec(
        "git",
        "--no-pager",
        "-C",
        str(repo),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        raise ToolError(f"git {' '.join(args[:2])} timed out") from None
    return proc.returncode or 0, out.decode("utf-8", errors="replace")


class _RepoTool(Tool[A]):
    def __init__(self, workspace: Workspace) -> None:
        self.ws = workspace


class RepoOverviewArgs(ToolArgs):
    path: str = "."


class RepoOverview(_RepoTool[RepoOverviewArgs]):
    name = "repo_overview"
    description = "Summarize a codebase: languages, size, key files, top-level layout, git state."
    args_model = RepoOverviewArgs
    tags = ("repository", "read")
    timeout_s = 120.0

    async def run(self, args: RepoOverviewArgs, ctx: ToolContext) -> ToolResult:
        root = self.ws.resolve(args.path, must_exist=True)
        if not root.is_dir():
            raise ToolError(f"{args.path!r} is not a directory")

        def scan() -> tuple[Counter[str], Counter[str], int]:
            files: Counter[str] = Counter()
            lines: Counter[str] = Counter()
            total = 0
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for name in filenames:
                    total += 1
                    lang = LANGUAGES.get(Path(name).suffix.lower())
                    if lang is None:
                        continue
                    files[lang] += 1
                    path = Path(dirpath) / name
                    try:
                        if path.stat().st_size < 1_000_000:
                            with path.open("rb") as fh:
                                lines[lang] += sum(1 for _ in fh)
                    except OSError:
                        continue
                    if total > 50_000:
                        return files, lines, total
            return files, lines, total

        files, lines, total = await asyncio.to_thread(scan)
        parts = [f"Repository: {self.ws.relative(root)}  ({total} files)"]
        if files:
            parts.append("Languages (files / lines):")
            parts += [
                f"  {lang}: {files[lang]} / {lines[lang]}" for lang, _ in lines.most_common(12)
            ]
        present = [k for k in KEY_FILES if (root / k).exists()]
        if present:
            parts.append("Key files: " + ", ".join(present))
        top = sorted(
            p.name + ("/" if p.is_dir() else "")
            for p in root.iterdir()
            if not p.name.startswith(".")
        )
        parts.append("Top level: " + ", ".join(top[:60]))
        if (root / ".git").exists():
            _, branch = await run_git(root, "rev-parse", "--abbrev-ref", "HEAD")
            _, log = await run_git(root, "log", "--oneline", "-5")
            _, status = await run_git(root, "status", "--short")
            changed = len([line for line in status.splitlines() if line.strip()])
            parts.append(f"Git: branch {branch.strip()}, {changed} changed files")
            parts.append(f"Recent commits:\n{log.strip()}")
        return ToolResult.success("\n".join(parts))


class RepoGitArgs(ToolArgs):
    command: Literal["status", "log", "diff", "show", "branch", "blame", "ls-files", "shortlog"]
    args: list[str] = Field(default_factory=list, description="Extra arguments, e.g. ['-n', '10'].")
    path: str = "."


class RepoGit(_RepoTool[RepoGitArgs]):
    name = "repo_git"
    description = "Run a read-only git command (status, log, diff, show, branch, blame, ls-files)."
    args_model = RepoGitArgs
    tags = ("repository", "read")

    async def run(self, args: RepoGitArgs, ctx: ToolContext) -> ToolResult:
        for arg in args.args:
            if _FORBIDDEN_GIT_ARGS.match(arg):
                raise ToolError(f"git option {arg!r} is not allowed")
        if args.command == "branch" and any(
            not a.startswith("-") or a in {"-d", "-D", "-m", "-M"} for a in args.args
        ):
            raise ToolError("repo_git branch is read-only (listing only)")
        repo = self.ws.resolve(args.path, must_exist=True)
        code, out = await run_git(repo, args.command, *args.args)
        if code != 0:
            return ToolResult.failure(f"git {args.command} exited {code}", out)
        return ToolResult.success(out or "(no output)")


class RepoSymbolsArgs(ToolArgs):
    path: str = Field(description="A source file or directory.")
    max_files: int = Field(default=40, ge=1, le=400)


def python_outline(source: str) -> list[str]:
    tree = ast.parse(source)
    out: list[str] = []

    def doc(node: Any) -> str:
        text = ast.get_docstring(node) or ""
        return f"  - {text.splitlines()[0][:100]}" if text else ""

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            out.append(f"L{node.lineno} class {node.name}{doc(node)}")
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    prefix = "async def" if isinstance(item, ast.AsyncFunctionDef) else "def"
                    out.append(f"  L{item.lineno} {prefix} {item.name}({ast.unparse(item.args)})")
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
            out.append(f"L{node.lineno} {prefix} {node.name}({ast.unparse(node.args)}){doc(node)}")
    return out


_GENERIC_DECL = re.compile(
    r"^\s*(pub\s+)?(export\s+)?(async\s+)?(fn|func|function|class|struct|enum|trait|interface|impl|type)\s+([A-Za-z_]\w*)"
)


class RepoSymbols(_RepoTool[RepoSymbolsArgs]):
    name = "repo_symbols"
    description = (
        "Outline classes and functions in source files (Python via AST, others heuristically)."
    )
    args_model = RepoSymbolsArgs
    tags = ("repository", "read")

    async def run(self, args: RepoSymbolsArgs, ctx: ToolContext) -> ToolResult:
        target = self.ws.resolve(args.path, must_exist=True)
        if target.is_file():
            files = [target]
        else:
            files = []
            for dirpath, dirnames, filenames in os.walk(target):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                files += [
                    Path(dirpath) / f for f in sorted(filenames) if Path(f).suffix in LANGUAGES
                ]
                if len(files) >= args.max_files:
                    break
        sections = []
        for file in files[: args.max_files]:
            if file.suffix not in LANGUAGES or file.suffix in {
                ".md",
                ".json",
                ".yaml",
                ".yml",
                ".toml",
            }:
                continue
            try:
                source = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if file.suffix == ".py":
                try:
                    outline = python_outline(source)
                except SyntaxError as exc:
                    outline = [f"(syntax error: {exc.msg} at line {exc.lineno})"]
            else:
                outline = [
                    f"L{i} {m.group(4)} {m.group(5)}"
                    for i, line in enumerate(source.splitlines(), 1)
                    if (m := _GENERIC_DECL.match(line))
                ]
            if outline:
                sections.append(f"## {self.ws.relative(file)}\n" + "\n".join(outline))
        return ToolResult.success("\n\n".join(sections) or "no symbols found")


class RepoCommitArgs(ToolArgs):
    message: str = Field(min_length=3)
    files: list[str] | None = Field(
        default=None, description="Files to stage; omit to stage tracked changes."
    )
    path: str = "."


class RepoCommit(_RepoTool[RepoCommitArgs]):
    name = "repo_commit"
    description = "Create a git commit (requires operator approval). Never pushes."
    args_model = RepoCommitArgs
    risk = RiskLevel.HIGH
    tags = ("repository", "write")

    async def run(self, args: RepoCommitArgs, ctx: ToolContext) -> ToolResult:
        repo = self.ws.resolve(args.path, must_exist=True)
        if args.files:
            paths = [str(self.ws.resolve(f, must_exist=True)) for f in args.files]
            code, out = await run_git(repo, "add", "--", *paths)
        else:
            code, out = await run_git(repo, "add", "--update")
        if code != 0:
            return ToolResult.failure("git add failed", out)
        code, out = await run_git(
            repo,
            "-c",
            "user.name=SentientAI",
            "-c",
            "user.email=sentientai@localhost",
            "commit",
            "-m",
            args.message,
        )
        if code != 0:
            return ToolResult.failure("git commit failed", out)
        return ToolResult.success(out.strip())


def repository_tools(workspace: Workspace) -> list[Tool[Any]]:
    return [
        RepoOverview(workspace),
        RepoGit(workspace),
        RepoSymbols(workspace),
        RepoCommit(workspace),
    ]
