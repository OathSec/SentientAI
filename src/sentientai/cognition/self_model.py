"""Persistent self-model: identity, capabilities, limitations and autobiographical continuity."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from sentientai.config import IdentityConfig


def humanize_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    for unit, size in (("day", 86400), ("hour", 3600), ("minute", 60)):
        if seconds >= size:
            value = seconds / size
            return f"{value:.0f} {unit}{'s' if round(value) != 1 else ''}"
    return f"{seconds:.0f} seconds"


DEFAULT_LIMITATIONS = [
    "I can be wrong, especially without evidence; I should verify before asserting.",
    "I only act through my tools, and risky actions need my operator's approval.",
    "My memory is selective: unimportant details fade unless they prove useful.",
]


class SelfModel(BaseModel):
    name: str
    purpose: str
    values: list[str]
    capabilities: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=lambda: list(DEFAULT_LIMITATIONS))
    narrative: str = "I have only just come into existence; I have no history yet."
    boot_count: int = 0
    first_boot_at: datetime | None = None
    last_boot_at: datetime | None = None
    last_shutdown_at: datetime | None = None
    total_uptime_s: float = 0.0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    last_focus: str | None = None

    @classmethod
    def from_identity(cls, identity: IdentityConfig) -> SelfModel:
        return cls(name=identity.name, purpose=identity.purpose, values=list(identity.values))

    def sync_identity(self, identity: IdentityConfig) -> None:
        """Operator-configured identity always wins over the persisted copy."""

        self.name = identity.name
        self.purpose = identity.purpose
        self.values = list(identity.values)

    def on_boot(self, now: datetime) -> None:
        self.boot_count += 1
        self.first_boot_at = self.first_boot_at or now
        self.last_boot_at = now

    def on_shutdown(self, now: datetime) -> None:
        if self.last_boot_at is not None:
            self.total_uptime_s += max(0.0, (now - self.last_boot_at).total_seconds())
        self.last_shutdown_at = now

    def add_limitation(self, text: str, max_items: int = 12) -> None:
        text = text.strip()
        if text and text not in self.limitations:
            self.limitations.append(text)
            del self.limitations[: max(0, len(self.limitations) - max_items)]

    def render_identity(self) -> str:
        values = "\n".join(f"- {v}" for v in self.values)
        limits = "\n".join(f"- {v}" for v in self.limitations)
        caps = ", ".join(self.capabilities) if self.capabilities else "conversation only"
        return (
            f"You are {self.name}. {self.purpose}\n\n"
            f"Your values:\n{values}\n\n"
            f"Your capabilities (tools): {caps}\n"
            f"Your known limitations:\n{limits}"
        )

    def render_continuity(self, now: datetime) -> str:
        if self.first_boot_at is None or self.boot_count <= 1:
            history = "This is your first awakening."
        else:
            age = humanize_duration((now - self.first_boot_at).total_seconds())
            history = (
                f"This is awakening #{self.boot_count}. You first came online {age} ago and have "
                f"run for {humanize_duration(self.total_uptime_s)} in total."
            )
            if self.last_shutdown_at is not None and self.last_boot_at is not None:
                gap = (self.last_boot_at - self.last_shutdown_at).total_seconds()
                history += f" You were offline for {humanize_duration(gap)} before this boot."
        if self.last_boot_at is not None:
            history += (
                f" Current uptime: {humanize_duration((now - self.last_boot_at).total_seconds())}."
            )
        record = (
            f"Track record: {self.tasks_succeeded} tasks succeeded, {self.tasks_failed} failed."
        )
        focus = (
            f" Before your last shutdown you were working on: {self.last_focus}."
            if self.last_focus
            else ""
        )
        return f"{history} {record}{focus}\nYour story so far: {self.narrative}"
