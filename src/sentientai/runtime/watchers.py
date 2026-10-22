"""Folder watchers: react to new or changed files (portable polling, no inotify needed)."""

from __future__ import annotations

import asyncio
import fnmatch
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from sentientai.config import WatchConfig
from sentientai.core.logging import get_logger
from sentientai.ingestion.pipeline import IngestionPipeline
from sentientai.storage.database import StateRepository

log = get_logger(__name__)

TaskSubmitFn = Callable[[WatchConfig, Path], Awaitable[object]]


class FolderWatcher:
    def __init__(
        self,
        config: WatchConfig,
        *,
        state: StateRepository,
        pipeline: IngestionPipeline,
        submit_task: TaskSubmitFn,
    ) -> None:
        self.config = config
        self.state = state
        self.pipeline = pipeline
        self.submit_task = submit_task
        self.key = f"watcher:{config.name}"

    def _scan(self) -> dict[str, list[float]]:
        root = self.config.path
        found: dict[str, list[float]] = {}
        if not root.is_dir():
            return found
        walker = os.walk(root) if self.config.recursive else [(str(root), [], os.listdir(root))]
        for dirpath, dirnames, filenames in walker:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                if name.startswith(".") or not fnmatch.fnmatch(name, self.config.pattern):
                    continue
                path = Path(dirpath) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                found[str(path)] = [stat.st_mtime, float(stat.st_size)]
        return found

    async def poll(self) -> list[Path]:
        """Process new/changed files once; returns the paths that were handled."""

        known_raw = await self.state.get(self.key)
        first_run = known_raw is None
        known: dict[str, list[float]] = known_raw or {}
        current = await asyncio.to_thread(self._scan)
        changed = [Path(p) for p, sig in current.items() if known.get(p) != sig]
        handled: list[Path] = []
        if first_run and self.config.action == "task":
            # Don't flood the queue with every pre-existing file the first time we look.
            await self.state.set(self.key, current)
            return handled
        for path in sorted(changed):
            try:
                if self.config.action == "ingest":
                    result = await self.pipeline.ingest_path(path)
                    if result.reason and not result.skipped:
                        log.warning(
                            "watcher ingest failed",
                            extra={"path": str(path), "reason": result.reason},
                        )
                else:
                    await self.submit_task(self.config, path)
                handled.append(path)
            except Exception:
                log.exception("watcher action failed", extra={"path": str(path)})
                current.pop(str(path), None)  # retry next poll
        await self.state.set(self.key, current)
        return handled

    async def run(self) -> None:
        while True:
            try:
                await self.poll()
            except Exception:
                log.exception("watcher poll failed", extra={"watcher": self.config.name})
            await asyncio.sleep(self.config.interval_s)
