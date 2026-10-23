"""Fuzzy's long-running projects and the notebook it keeps for each.

A project is something Fuzzy decided to work on over more than one sitting: learning about
something, building something in its world, writing something. Each time it works on one (in
its open-ended "free" time), it writes a notebook entry afterwards in its own words: what it did,
what came of it, what it wants to do next, and whether the project is still going. The next time
it chooses what to do, it sees its projects and where it left off, so its work builds on itself
instead of starting from nothing each time.

Nothing here decides what a project is or when it's done: Fuzzy names it, works on it, and
declares it finished, paused or abandoned. The book only keeps the record (bounded in size) and
publishes ``mind.project`` events.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from sentientai.core.clock import Clock, SystemClock, new_id
from sentientai.core.events import EventBus
from sentientai.storage.database import StateRepository

STATE_KEY = "mind.projects"
MAX_ENTRIES = 40  # per project; the running summary carries what came before
MAX_PROJECTS = 60  # finished ones beyond this are forgotten, oldest first

ProjectStatus = Literal["active", "paused", "done", "abandoned"]


class Entry(BaseModel):
    at: datetime
    text: str
    next_step: str = ""


class Project(BaseModel):
    id: str = Field(default_factory=lambda: new_id("prj"))
    title: str
    why: str = ""
    status: ProjectStatus = "active"
    summary: str = ""  # Fuzzy's own running summary of the whole project so far
    next_step: str = ""
    sessions: int = 0
    created_at: datetime
    updated_at: datetime
    entries: list[Entry] = Field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class ProjectBook:
    def __init__(
        self,
        state: StateRepository,
        *,
        bus: EventBus | None = None,
        clock: Clock | None = None,
        max_active: int = 5,
    ) -> None:
        self.state = state
        self.bus = bus
        self.clock = clock or SystemClock()
        self.max_active = max_active

    async def all(self) -> list[Project]:
        raw = await self.state.get(STATE_KEY)
        items = raw if isinstance(raw, list) else []
        projects = []
        for item in items:
            try:
                projects.append(Project.model_validate(item))
            except ValueError:
                continue
        return projects

    async def _save(self, projects: list[Project]) -> None:
        live = [p for p in projects if p.status in ("active", "paused")]
        finished = sorted(
            (p for p in projects if p.status not in ("active", "paused")),
            key=lambda p: p.updated_at,
        )
        keep = live + finished[-max(0, MAX_PROJECTS - len(live)) :]
        await self.state.set(STATE_KEY, [p.model_dump(mode="json") for p in keep])

    async def active(self) -> list[Project]:
        return [p for p in await self.all() if p.status in ("active", "paused")]

    async def get(self, project_id: str) -> Project | None:
        return next((p for p in await self.all() if p.id == project_id), None)

    async def find(self, wanted: str) -> Project | None:
        """A project by id, or by (part of) its title."""

        wanted = wanted.strip().casefold()
        if not wanted:
            return None
        projects = await self.active()
        return next((p for p in projects if p.id.casefold() == wanted), None) or next(
            (p for p in projects if wanted in p.title.casefold() or p.title.casefold() in wanted),
            None,
        )

    async def has_room(self) -> bool:
        return len(await self.active()) < self.max_active

    async def start(self, title: str, why: str = "") -> Project | None:
        """A new project, or None if Fuzzy already holds as many as it can."""

        if not await self.has_room():
            return None
        now = self.clock.now()
        project = Project(
            title=" ".join(title.split())[:200],
            why=" ".join(why.split())[:1000],
            created_at=now,
            updated_at=now,
        )
        projects = await self.all()
        projects.append(project)
        await self._save(projects)
        await self._publish(project, "started")
        return project

    async def write(
        self,
        project_id: str,
        *,
        entry: str,
        next_step: str = "",
        summary: str = "",
        status: str = "active",
    ) -> Project | None:
        """Add a notebook entry after a working session."""

        projects = await self.all()
        project = next((p for p in projects if p.id == project_id), None)
        if project is None:
            return None
        now = self.clock.now()
        text = entry.strip()[:8000]
        if text:
            project.entries.append(Entry(at=now, text=text, next_step=next_step.strip()[:2000]))
            del project.entries[: max(0, len(project.entries) - MAX_ENTRIES)]
        project.sessions += 1
        project.next_step = next_step.strip()[:2000]
        if summary.strip():
            project.summary = summary.strip()[:6000]
        if status in ("active", "paused", "done", "abandoned"):
            project.status = status  # type: ignore[assignment]
        project.updated_at = now
        await self._save(projects)
        await self._publish(project, "entry")
        return project

    async def _publish(self, project: Project, op: str) -> None:
        if self.bus is not None:
            await self.bus.publish(
                "mind.project",
                op=op,
                id=project.id,
                title=project.title,
                status=project.status,
                entry=project.entries[-1].text if project.entries else "",
                user_id=None,
            )

    @staticmethod
    def render(projects: list[Project], *, entries: int = 1) -> str:
        """For Fuzzy's own prompts: its projects and where it left off."""

        if not projects:
            return "- none yet"
        lines = []
        for p in projects:
            line = f"- [{p.id}] {p.title} ({p.status}, {p.sessions} sessions)"
            if p.summary:
                line += f"\n  So far: {p.summary[:600]}"
            for e in p.entries[-entries:]:
                line += f"\n  Last time ({e.at:%m-%d}): {e.text[:400]}"
            if p.next_step:
                line += f"\n  Next, you wanted to: {p.next_step[:300]}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def notebook(project: Project) -> str:
        """The whole notebook, for a working session on the project."""

        parts = [f"Project: {project.title}"]
        if project.why:
            parts.append(f"Why you started it: {project.why}")
        if project.summary:
            parts.append(f"Your summary so far: {project.summary}")
        if project.entries:
            parts.append(
                "Your notebook (oldest first):\n"
                + "\n".join(f"- {e.at:%Y-%m-%d %H:%M}: {e.text}" for e in project.entries[-10:])
            )
        if project.next_step:
            parts.append(f"What you planned to do next: {project.next_step}")
        return "\n".join(parts)
