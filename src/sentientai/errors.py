"""Exception hierarchy. Every error raised by the runtime derives from SentientError."""

from __future__ import annotations


class SentientError(Exception):
    """Base class for all runtime errors."""


class ConfigError(SentientError):
    pass


class ModelError(SentientError):
    """A model endpoint returned an error or malformed output."""


class ModelUnavailableError(ModelError):
    """No model tier could serve the request (down, circuit open, or rate limited)."""


class ToolError(SentientError):
    """A tool failed in a way that should be reported back to the agent."""


class ToolNotFoundError(ToolError):
    pass


class ToolValidationError(ToolError):
    pass


class PolicyViolationError(SentientError):
    """An action was denied by the safety policy."""

    def __init__(self, message: str, *, reasons: list[str] | None = None) -> None:
        super().__init__(message)
        self.reasons = reasons or []


class ApprovalDeniedError(PolicyViolationError):
    """The operator (or approval timeout) denied an action."""


class KillSwitchEngagedError(SentientError):
    """The kill switch is engaged; all autonomous execution must stop."""


class LoopBreakError(SentientError):
    """A deterministic loop breaker tripped (budget, repetition, failures, time)."""

    def __init__(self, reason: str, *, limit: str) -> None:
        super().__init__(reason)
        self.limit = limit


class SandboxError(ToolError):
    pass


class AuditIntegrityError(SentientError):
    """The audit chain failed verification."""


class StorageError(SentientError):
    pass
