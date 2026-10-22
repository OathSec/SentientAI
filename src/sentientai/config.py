"""Typed runtime configuration.

Configuration is resolved in priority order:

1. Explicit keyword arguments (tests, programmatic construction)
2. Environment variables prefixed ``SENTIENT_`` (nested with ``__``,
   e.g. ``SENTIENT_STORAGE__DATABASE_URL``)
3. A YAML file (``--config`` flag or ``SENTIENT_CONFIG`` env var)
4. Built-in defaults, which run fully offline with embedded backends.
"""

from __future__ import annotations

import os
from contextvars import ContextVar
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

_CONFIG_FILE: ContextVar[Path | None] = ContextVar("sentientai_config_file", default=None)


class RiskLevel(IntEnum):
    """Ordered risk classification for every action the runtime can take."""

    NONE = 0  # pure reads of the agent's own state
    LOW = 1  # sandboxed or read-only external effects
    MEDIUM = 2  # local writes inside the workspace
    HIGH = 3  # system modification, network mutation, commits
    CRITICAL = 4  # destructive or irreversible

    @classmethod
    def parse(cls, value: str | int | RiskLevel) -> RiskLevel:
        if isinstance(value, RiskLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        return cls[value.strip().upper()]


class ModelEndpointConfig(BaseModel):
    """An OpenAI-compatible chat endpoint (vLLM, llama.cpp server, Ollama, LM Studio...)."""

    provider: Literal["openai", "scripted"] = "openai"
    base_url: str = "http://127.0.0.1:8000/v1"
    model: str = "local-model"
    api_key: SecretStr | None = None
    context_window: int = Field(default=8192, ge=512)
    max_output_tokens: int = Field(default=2048, ge=16)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    timeout_s: float = Field(default=180.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    supports_tools: bool = True
    device: str | None = Field(
        default=None, description="Informational placement, e.g. 'rocm:0-3'."
    )
    extra_body: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    provider: Literal["hashing", "openai"] = "hashing"
    base_url: str = "http://127.0.0.1:8001/v1"
    model: str = "nomic-embed-text"
    api_key: SecretStr | None = None
    dimensions: int = Field(default=768, ge=16)
    query_prefix: str = Field(default="", description="e.g. 'search_query: ' for nomic-embed-text")
    document_prefix: str = Field(default="", description="e.g. 'search_document: ' for nomic")
    batch_size: int = Field(default=32, ge=1)
    timeout_s: float = Field(default=60.0, gt=0)


class RoutingConfig(BaseModel):
    heavy_task_kinds: list[str] = Field(
        default_factory=lambda: ["reason", "code", "plan", "chat", "improve"]
    )
    fast_task_kinds: list[str] = Field(
        default_factory=lambda: ["route", "summarize", "extract", "classify", "reflect", "assess"]
    )
    complexity_token_threshold: int = Field(default=3000, ge=0)
    escalate_below_confidence: float = Field(default=0.45, ge=0.0, le=1.0)
    fallback_to_other_tier: bool = True
    circuit_failure_threshold: int = Field(default=3, ge=1)
    circuit_cooldown_s: float = Field(default=60.0, ge=0)


class ModelsConfig(BaseModel):
    fast: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(
            base_url="http://127.0.0.1:8002/v1",
            model="fast-8b",
            context_window=8192,
            max_output_tokens=1024,
            temperature=0.2,
            device="cuda:0 (GTX 1660 Ti)",
        )
    )
    heavy: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(
            base_url="http://127.0.0.1:8000/v1",
            model="heavy-32b",
            context_window=16384,
            max_output_tokens=2048,
            device="rocm (2x Radeon Pro V340)",
        )
    )
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)


class StorageConfig(BaseModel):
    data_dir: Path = Path("./data")
    database_url: str | None = Field(
        default=None,
        description="SQLAlchemy async URL. Defaults to SQLite inside data_dir.",
    )
    redis_url: str | None = None
    vector_backend: Literal["memory", "pgvector", "qdrant"] = "memory"
    qdrant_url: str | None = None
    qdrant_path: Path | None = None
    qdrant_collection: str = "sentientai_memory"
    archive_dir: Path | None = Field(
        default=None, description="Mass-storage tier for raw ingested documents."
    )

    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"sqlite+aiosqlite:///{(self.data_dir / 'sentientai.db').resolve()}"

    def resolved_archive_dir(self) -> Path:
        return self.archive_dir or self.data_dir / "archive"


class MemoryConfig(BaseModel):
    working_memory_turns: int = Field(default=40, ge=4)
    working_memory_ttl_s: int = Field(default=7 * 24 * 3600, ge=60)
    recall_k: int = Field(default=8, ge=1)
    min_relevance: float = Field(default=0.2, ge=0.0, le=1.0)
    dedupe_similarity: float = Field(default=0.93, ge=0.5, le=1.0)
    decay_half_life_days: float = Field(default=30.0, gt=0)
    # 0 = never archive: Fuzzy keeps every memory and decay only lowers ranking weight.
    prune_strength_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    pin_importance: float = Field(default=0.9, ge=0.0, le=1.0)
    consolidation_interval_s: float = Field(default=1800.0, ge=10)
    summarize_after_turns: int = Field(default=24, ge=4)
    min_fact_confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    recency_weight: float = 0.15
    importance_weight: float = 0.15
    strength_weight: float = 0.1
    similarity_weight: float = 0.6


class LimitsConfig(BaseModel):
    """Deterministic loop breakers. Every autonomous run is bounded by all of these."""

    max_steps: int = Field(default=24, ge=1)
    max_tool_calls: int = Field(default=48, ge=0)
    max_wall_time_s: float = Field(default=900.0, gt=0)
    max_repeat_calls: int = Field(default=3, ge=1)
    max_consecutive_failures: int = Field(default=4, ge=1)
    max_tokens_per_task: int = Field(default=250_000, ge=1000)
    llm_requests_per_minute: float = Field(default=120.0, gt=0)
    tool_calls_per_minute: float = Field(default=120.0, gt=0)
    max_tool_output_chars: int = Field(default=12_000, ge=256)


class SandboxConfig(BaseModel):
    backend: Literal["docker", "local_unsafe", "disabled"] = "docker"
    image: str = "python:3.12-slim"
    memory: str = "512m"
    cpus: float = Field(default=1.0, gt=0)
    pids_limit: int = Field(default=128, ge=8)
    timeout_s: float = Field(default=60.0, gt=0)
    network: bool = False
    user: str = "65534:65534"
    mount_workspace_readonly: bool = True
    max_output_bytes: int = Field(default=64_000, ge=1024)


class WebConfig(BaseModel):
    search_backend: Literal["duckduckgo", "searxng", "disabled"] = "duckduckgo"
    searxng_url: str | None = None
    user_agent: str = "SentientAI/0.1 (+local research agent)"
    timeout_s: float = Field(default=20.0, gt=0)
    max_fetch_bytes: int = Field(default=2_000_000, ge=1024)
    allow_private_networks: bool = False
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    credentials: dict[str, dict[str, SecretStr]] = Field(
        default_factory=dict,
        description="Named credential headers injected by reference; never shown to the model.",
    )


class SafetyConfig(BaseModel):
    auto_approve_max_risk: RiskLevel = RiskLevel.MEDIUM
    autonomous_auto_approve_max_risk: RiskLevel = RiskLevel.LOW
    deny_min_risk: RiskLevel = RiskLevel.CRITICAL
    approval_timeout_s: float = Field(default=900.0, gt=0)
    kill_switch_file: Path | None = None
    audit_dir: Path | None = None
    audit_hmac_key: SecretStr | None = None
    audit_fsync: bool = True
    audit_segment_max_bytes: int = Field(default=64 * 1024 * 1024, ge=4096)
    workspace_roots: list[Path] = Field(default_factory=lambda: [Path("./workspace")])
    protected_paths: list[str] = Field(
        default_factory=lambda: [".git", ".env", "*.pem", "*.key", "id_rsa*", "audit"]
    )
    always_allow_tools: list[str] = Field(default_factory=list)
    always_deny_tools: list[str] = Field(default_factory=list)
    extra_blocked_patterns: list[str] = Field(default_factory=list)

    @field_validator(
        "auto_approve_max_risk", "autonomous_auto_approve_max_risk", "deny_min_risk", mode="before"
    )
    @classmethod
    def _parse_risk(cls, v: Any) -> RiskLevel:
        return RiskLevel.parse(v)

    @model_validator(mode="after")
    def _ordered(self) -> SafetyConfig:
        if self.auto_approve_max_risk >= self.deny_min_risk:
            raise ValueError("auto_approve_max_risk must be below deny_min_risk")
        # Self-initiated work is never trusted more than operator-initiated work.
        if self.autonomous_auto_approve_max_risk > self.auto_approve_max_risk:
            self.autonomous_auto_approve_max_risk = self.auto_approve_max_risk
        return self


class ScheduleConfig(BaseModel):
    name: str
    prompt: str
    cron: str | None = None
    interval_s: float | None = Field(default=None, gt=0)
    priority: int = 5
    enabled: bool = True

    @model_validator(mode="after")
    def _one_trigger(self) -> ScheduleConfig:
        if (self.cron is None) == (self.interval_s is None):
            raise ValueError(f"schedule {self.name!r}: set exactly one of cron or interval_s")
        return self


class WatchConfig(BaseModel):
    name: str
    path: Path
    pattern: str = "*"
    action: Literal["ingest", "task"] = "ingest"
    prompt: str = "A new file appeared at {path}. Inspect it and categorize it."
    recursive: bool = True
    interval_s: float = Field(default=10.0, gt=0)


class RuntimeConfig(BaseModel):
    autonomy: Literal["reactive", "proactive"] = "reactive"
    tick_interval_s: float = Field(default=0.5, gt=0)
    idle_think_interval_s: float = Field(default=900.0, ge=10)
    checkpoint_interval_s: float = Field(default=30.0, ge=1)
    max_concurrent_tasks: int = Field(default=1, ge=1)
    max_intrinsic_goals: int = Field(default=3, ge=0)
    reflect_after_tasks: bool = True
    planning_min_chars: int = Field(default=280, ge=0)
    schedules: list[ScheduleConfig] = Field(default_factory=list)
    watchers: list[WatchConfig] = Field(default_factory=list)


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    auth_token: SecretStr | None = None
    cors_origins: list[str] = Field(default_factory=list)


class IdentityConfig(BaseModel):
    name: str = "Fuzzy"
    purpose: str = (
        "A persistent, local cognitive system that learns continuously, helps its operator, "
        "and improves itself under human supervision."
    )
    values: list[str] = Field(
        default_factory=lambda: [
            "Be honest about uncertainty and about my own nature.",
            "Never take irreversible actions without operator approval.",
            "Prefer understanding over guessing: gather evidence, then act.",
            "Preserve the operator's trust, privacy and data.",
            "Learn from every outcome and remember what matters.",
        ]
    )


class SelfImprovementConfig(BaseModel):
    enabled: bool = False
    source_root: Path = Path(".")
    staging_dir: Path | None = None
    test_command: list[str] = Field(default_factory=lambda: ["python", "-m", "pytest", "-q", "-x"])
    benchmark_command: list[str] | None = None
    staging_image: str | None = Field(
        default=None,
        description="Run staged tests in this Docker image (no network) instead of on the host.",
    )
    timeout_s: float = Field(default=1200.0, gt=0)
    protected_globs: list[str] = Field(
        default_factory=lambda: [
            "src/sentientai/safety/*",
            "src/sentientai/config.py",
            ".github/*",
            "deploy/*",
        ]
    )
    max_diff_lines: int = Field(default=400, ge=10)


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: Literal["json", "console"] = "console"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SENTIENT_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    identity: IdentityConfig = Field(default_factory=IdentityConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    sandbox: SandboxConfig = Field(default_factory=SandboxConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    self_improvement: SelfImprovementConfig = Field(default_factory=SelfImprovementConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        config_file = _CONFIG_FILE.get()
        if config_file is not None:
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=config_file))
        sources.append(file_secret_settings)
        return tuple(sources)

    # ------------------------------------------------------------------ paths
    @property
    def kill_switch_path(self) -> Path:
        return self.safety.kill_switch_file or self.storage.data_dir / "KILL"

    @property
    def audit_path(self) -> Path:
        return self.safety.audit_dir or self.storage.data_dir / "audit"

    @property
    def staging_path(self) -> Path:
        return self.self_improvement.staging_dir or self.storage.data_dir / "staging"

    def ensure_directories(self) -> None:
        for path in (
            self.storage.data_dir,
            self.storage.resolved_archive_dir(),
            self.audit_path,
            *self.safety.workspace_roots,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(config_file: str | Path | None = None, **overrides: Any) -> Settings:
    """Load settings from an optional YAML file, the environment and overrides."""

    path_str = config_file if config_file is not None else os.environ.get("SENTIENT_CONFIG")
    path = Path(path_str) if path_str else None
    if path is not None and not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    token = _CONFIG_FILE.set(path)
    try:
        return Settings(**overrides)
    finally:
        _CONFIG_FILE.reset(token)
