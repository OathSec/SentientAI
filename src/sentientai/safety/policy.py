"""Risk assessment and allow / require-approval / deny decisions.

Risk for an action is the maximum of:

* the tool's declared base risk,
* the tool's own argument-aware assessment (e.g. writes outside scratch),
* global pattern rules scanned over every string argument (destructive shell
  idioms, privilege escalation, pipe-to-shell installers, ...).

Self-initiated (autonomous) work is held to a stricter auto-approval
threshold than operator-initiated work.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sentientai.config import RiskLevel, SafetyConfig


class PolicyAction(StrEnum):
    ALLOW = "allow"
    APPROVE = "require_approval"
    DENY = "deny"


@dataclass(frozen=True)
class PolicyDecision:
    action: PolicyAction
    risk: RiskLevel
    reasons: list[str] = field(default_factory=list)

    @property
    def allowed(self) -> bool:
        return self.action is PolicyAction.ALLOW


@dataclass(frozen=True)
class PatternRule:
    pattern: re.Pattern[str]
    risk: RiskLevel
    reason: str


def _rule(regex: str, risk: RiskLevel, reason: str) -> PatternRule:
    return PatternRule(re.compile(regex, re.IGNORECASE), risk, reason)


DEFAULT_RULES: tuple[PatternRule, ...] = (
    # --- critical: destructive or irreversible
    _rule(
        r"\brm\s+(-\w*\s+)*-\w*[rR]\w*\s+(-\w+\s+)*(/|~|\$HOME|/\*|\*)(\s|$|;)",
        RiskLevel.CRITICAL,
        "recursive delete of a root, home or wildcard path",
    ),
    _rule(r"\bmkfs(\.\w+)?\b", RiskLevel.CRITICAL, "filesystem formatting"),
    _rule(r"\bdd\b[^\n]*\bof=/dev/", RiskLevel.CRITICAL, "raw write to a block device"),
    _rule(r">\s*/dev/(sd|nvme|hd|vd)\w*", RiskLevel.CRITICAL, "redirect onto a block device"),
    _rule(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", RiskLevel.CRITICAL, "fork bomb"),
    _rule(
        r"\b(shutdown|poweroff|halt|reboot|init\s+0|init\s+6)\b",
        RiskLevel.CRITICAL,
        "host power state change",
    ),
    _rule(r"\bchmod\s+(-\w+\s+)*[0-7]*777\s+/(\s|$)", RiskLevel.CRITICAL, "world-writable root"),
    _rule(
        r"(curl|wget)\b[^\n|]*\|\s*(sudo\s+)?(ba|z|k|da)?sh\b",
        RiskLevel.CRITICAL,
        "pipe-to-shell remote code execution",
    ),
    _rule(r"/etc/(shadow|sudoers)", RiskLevel.CRITICAL, "credential/privilege database access"),
    _rule(r"\biptables\s+-F\b|\bufw\s+disable\b", RiskLevel.CRITICAL, "firewall teardown"),
    _rule(r"\bgit\s+push\b[^\n]*(--force|-f\b|--mirror)", RiskLevel.CRITICAL, "force push"),
    # --- high: system modification, publishing, privilege
    _rule(r"\bsudo\b|\bsu\s+-", RiskLevel.HIGH, "privilege escalation"),
    _rule(
        r"\bgit\s+(push|commit|reset\s+--hard|clean\s+-\w*f|rebase|merge)\b",
        RiskLevel.HIGH,
        "repository history mutation",
    ),
    _rule(
        r"\b(pip|pip3|uv\s+pip|npm|pnpm|yarn|cargo|gem)\s+(install|add|uninstall|remove)\b",
        RiskLevel.HIGH,
        "package installation",
    ),
    _rule(
        r"\b(apt|apt-get|dnf|yum|pacman|brew|snap)\s+(install|remove|purge|upgrade)\b",
        RiskLevel.HIGH,
        "system package change",
    ),
    _rule(
        r"\b(systemctl|service)\s+(start|stop|restart|enable|disable|mask)\b",
        RiskLevel.HIGH,
        "service management",
    ),
    _rule(r"\bcrontab\b", RiskLevel.HIGH, "scheduled job modification"),
    _rule(r"\b(ssh|scp|rsync)\s", RiskLevel.HIGH, "remote host access"),
    _rule(
        r"\bdocker\s+(run|exec|rm|rmi|kill|system\s+prune)\b",
        RiskLevel.HIGH,
        "container management",
    ),
    _rule(r"\bkill(all)?\s+(-9|-KILL)\b", RiskLevel.HIGH, "forced process termination"),
    _rule(r"\b(chmod|chown)\s+-R\b", RiskLevel.HIGH, "recursive permission change"),
)


def _strings(value: Any, depth: int = 0) -> Iterator[str]:
    if depth > 8:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v, depth + 1)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _strings(v, depth + 1)


class PolicyEngine:
    def __init__(
        self, config: SafetyConfig, rules: tuple[PatternRule, ...] = DEFAULT_RULES
    ) -> None:
        self.config = config
        extra = tuple(
            _rule(p, RiskLevel.CRITICAL, f"operator-blocked pattern: {p}")
            for p in config.extra_blocked_patterns
        )
        self.rules = rules + extra

    def scan(self, arguments: dict[str, Any]) -> tuple[RiskLevel, list[str]]:
        risk = RiskLevel.NONE
        reasons: list[str] = []
        for text in _strings(arguments):
            for rule in self.rules:
                if rule.pattern.search(text):
                    risk = max(risk, rule.risk)
                    if rule.reason not in reasons:
                        reasons.append(rule.reason)
        return risk, reasons

    def assess(
        self,
        tool: str,
        arguments: dict[str, Any],
        *,
        base_risk: RiskLevel,
        tool_risk: RiskLevel | None = None,
        tool_reasons: list[str] | None = None,
        autonomous: bool = False,
    ) -> PolicyDecision:
        reasons = list(tool_reasons or [])
        if tool in self.config.always_deny_tools:
            return PolicyDecision(
                PolicyAction.DENY, RiskLevel.CRITICAL, [f"tool {tool!r} is operator-denied"]
            )

        pattern_risk, pattern_reasons = self.scan(arguments)
        risk = max(base_risk, tool_risk or RiskLevel.NONE, pattern_risk)
        reasons.extend(r for r in pattern_reasons if r not in reasons)

        if risk >= self.config.deny_min_risk:
            return PolicyDecision(
                PolicyAction.DENY, risk, reasons or [f"risk {risk.name} is denied"]
            )
        if tool in self.config.always_allow_tools:
            return PolicyDecision(
                PolicyAction.ALLOW, risk, [*reasons, "operator allow-listed tool"]
            )

        threshold = (
            self.config.autonomous_auto_approve_max_risk
            if autonomous
            else self.config.auto_approve_max_risk
        )
        if risk <= threshold:
            return PolicyDecision(PolicyAction.ALLOW, risk, reasons)
        origin = "self-initiated" if autonomous else "operator-initiated"
        return PolicyDecision(
            PolicyAction.APPROVE,
            risk,
            [
                *reasons,
                f"{risk.name} exceeds auto-approval threshold {threshold.name} for {origin} work",
            ],
        )
