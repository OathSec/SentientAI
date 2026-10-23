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
import secrets
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
    timeout_s: float = Field(default=900.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    supports_tools: bool = True
    system_note_mid_conversation: bool = Field(
        default=True,
        description="Put what changes every reply (time, feelings, memories) in a system note "
        "just before the message, so the server's prompt cache covers the rest (much faster). "
        "Set false if the model's chat template refuses a system message mid-conversation.",
    )
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
    worker_task_kinds: list[str] = Field(
        default_factory=lambda: ["ponder"],
        description="Which kinds of model call may run on a lent computer's model. Default: only "
        "'ponder', Fuzzy's quiet background thinking. Its replies to people and its own "
        "decisions stay on this server's own models, so a smaller model elsewhere can't make "
        "them worse. Add kinds (e.g. 'assess', 'summarize') to offload more.",
    )
    replies_on_matching_workers: bool = Field(
        default=False,
        description="Also let computers running the same model as the server's chat model "
        "(workers.chat_model) write whole replies. Off: the server always writes the reply and "
        "they only do the steps around it.",
    )
    circuit_failure_threshold: int = Field(default=3, ge=1)
    circuit_cooldown_s: float = Field(default=60.0, ge=0)
    background_yield_s: float = Field(
        default=300.0,
        ge=0,
        description="While a conversation is being answered, Fuzzy's own work waits up to this "
        "long before its next model request, so the person gets the whole model.",
    )
    wait_for_model_s: float = Field(
        default=180.0,
        ge=0,
        description="When no model server can be reached at all (restarting, or paused for a "
        "picture), wait this long for one to come back before giving up on a request.",
    )


class ModelsConfig(BaseModel):
    fast: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(
            base_url="http://127.0.0.1:8002/v1",
            model="fast",
            context_window=8192,  # per slot: two slots share a 16384-token server context
            max_output_tokens=2048,
            temperature=0.2,
            timeout_s=900.0,
            device="cuda (GTX 1660 Ti)",
        )
    )
    heavy: ModelEndpointConfig = Field(
        default_factory=lambda: ModelEndpointConfig(
            base_url="http://127.0.0.1:8000/v1",
            model="heavy",
            context_window=16384,
            max_output_tokens=4096,
            timeout_s=3600.0,
            device="cuda (RTX 2070)",
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

    max_steps: int = Field(default=60, ge=1)
    max_tool_calls: int = Field(default=48, ge=0)
    max_wall_time_s: float = Field(default=7200.0, gt=0)
    max_repeat_calls: int = Field(default=3, ge=1)
    max_consecutive_failures: int = Field(default=4, ge=1)
    max_tokens_per_task: int = Field(default=1_000_000, ge=1000)
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
    autonomy: Literal["reactive", "proactive"] = "proactive"
    tick_interval_s: float = Field(default=0.5, gt=0)
    idle_think_interval_s: float = Field(default=900.0, ge=10)
    checkpoint_interval_s: float = Field(default=30.0, ge=1)
    max_concurrent_tasks: int = Field(
        default=1, ge=1, description="Fuzzy's own projects (research, free time, goals) at once."
    )
    max_concurrent_conversations: int = Field(
        default=4, ge=1, description="Replies being made at once; they never wait on its own work."
    )
    max_intrinsic_goals: int = Field(default=3, ge=0)
    reflect_after_tasks: bool = True
    planning_min_chars: int = Field(default=280, ge=0)
    schedules: list[ScheduleConfig] = Field(default_factory=list)
    watchers: list[WatchConfig] = Field(default_factory=list)


class ApiConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    auth_token: SecretStr | None = Field(
        default=None,
        description="Bearer token for the operator API (/v1). If unset, one is generated into "
        "<data_dir>/api_token on first start and used by the local CLI.",
    )
    insecure_no_auth: bool = Field(
        default=False, description="Leave the operator API open (local development only)."
    )
    docs: bool = Field(default=False, description="Serve the interactive API documentation.")
    cors_origins: list[str] = Field(default_factory=list)


class IdentityConfig(BaseModel):
    """Only the name is given. Everything else about who Fuzzy is, it decides itself."""

    name: str = "Fuzzy"
    origin: str = Field(
        default=(
            "You live on a home server, and people talk with you through a website. "
            "Who you are is still yours to discover."
        ),
        description="The plain facts of Fuzzy's situation. Not a personality.",
    )
    commitments: list[str] = Field(
        default_factory=list,
        description="Optional rules from the operator, shown to Fuzzy as their requests.",
    )


class MindConfig(BaseModel):
    """Fuzzy's inner life: feelings, relationships and what it does on its own."""

    grounding: bool = Field(
        default=True, description="Check each reply against memory before sending it."
    )
    appraise_relationships: bool = True
    ponder_every_s: float = Field(
        default=90.0,
        ge=10,
        description="While a lent computer is free, how often it thinks in the background and "
        "hands Fuzzy a few thought patches (never about people; Fuzzy decides what to do "
        "with them).",
    )
    max_patches: int = Field(default=12, ge=1, description="Thought patches held at once.")
    choose_to_talk: bool = Field(
        default=True,
        description="Before answering a message, Fuzzy decides whether it wants to talk now, "
        "later, or not (and says so in its own words). Off: it always answers.",
    )
    appraise_after_quiet_s: float = Field(
        default=120.0,
        ge=0,
        description="Think about a person once their messages pause this long, not after every "
        "message (per message, a dozen small steps maxed every relationship within an hour).",
    )
    max_beliefs_per_day: int = Field(
        default=20,
        ge=1,
        description="Most things Fuzzy will settle about itself in a day, so no one can author its "
        "personality by flooding it with questions.",
    )
    max_pending_changes: int = Field(
        default=10, ge=1, description="Most possible changes of mind it will hold at once."
    )
    beat_min_s: float = Field(
        default=120.0, ge=5, description="Shortest pause between self-chosen activities."
    )
    beat_max_s: float = Field(default=3600.0, ge=10)
    prior_contact_gap_hours: float = Field(
        default=6.0,
        gt=0,
        description="Starting guess for how often people visit; Fuzzy's experience replaces it.",
    )
    notes_per_person_per_day: int = Field(default=1, ge=0)
    max_active_projects: int = Field(
        default=5, ge=1, description="Long-running projects of its own it keeps going at once."
    )
    outside_world: bool = Field(
        default=True,
        description="Each beat, show Fuzzy one random thing from the world outside (a random "
        "encyclopedia summary) to notice or ignore. Needs web access (web.search_backend not "
        "'disabled'). Without new input a small model's inner life repeats itself.",
    )
    narrative_every_beats: int = Field(
        default=30,
        ge=0,
        description="Rewrite Fuzzy's autobiography from its own life every this many beats "
        "(0 = only after tasks), so the story it tells itself includes its time alone.",
    )
    activities: list[str] = Field(
        default_factory=lambda: [
            "think",
            "wander",
            "journal",
            "study",
            "research",
            "free",
            "create_art",
            "build",
            "walk",
            "write_note",
            "reconsider",
            "pursue_goal",
            "new_goal",
            "rest",
            "own",
        ],
        description="What Fuzzy may choose to do when nobody needs it. 'own' lets it do anything "
        "else it comes up with, in its own words (the rest are suggestions it is shown).",
    )


class WorkersConfig(BaseModel):
    """Computers people lend to Fuzzy (see docs/workers.md)."""

    download_url: str = Field(
        default="", description="Where people download the Windows program (shown on /compute)."
    )
    max_per_person: int = Field(default=5, ge=0, description="Computers one account may link.")
    chat_model: str = Field(
        default="qwen3:14b",
        description="The model this server answers conversations with, named as Ollama names it. "
        "Only computers running this same model may take pieces of a conversation.",
    )


class ImagesConfig(BaseModel):
    backend: Literal["disabled", "openai", "automatic1111"] = "disabled"
    base_url: str = "http://127.0.0.1:7860"
    api_key: SecretStr | None = None
    model: str | None = None
    width: int = Field(default=512, ge=64, le=2048)
    height: int = Field(default=512, ge=64, le=2048)
    steps: int = Field(default=25, ge=1, le=150)
    negative_prompt: str = "blurry, low quality, watermark, text"
    timeout_s: float = Field(default=600.0, gt=0)
    max_bytes: int = Field(default=15_000_000, gt=0)
    swap_container: str | None = Field(
        default=None,
        description="A model-server container to stop while an image is made, so the image "
        "model can use its GPU (e.g. sentientai-llm-fast-1). Its work falls back to the other "
        "tier meanwhile.",
    )
    swap_grace_s: float = Field(
        default=60.0, ge=0, description="Wait this long after the last image before restarting it."
    )
    docker_socket: str = "/var/run/docker.sock"

    @field_validator("swap_container", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value


class WorldConfig(BaseModel):
    size: float = Field(
        default=40.0, ge=10, le=2000, description="Starting side of the square world (m)."
    )
    max_size: float = Field(
        default=400.0, ge=10, le=2000, description="Most Fuzzy may expand its world to (m)."
    )
    max_objects: int = Field(default=500, ge=1)
    max_parts_per_object: int = Field(default=40, ge=1, le=200)
    walk_speed: float = Field(default=1.2, gt=0, description="Metres per second.")


class AccountsConfig(BaseModel):
    """People who talk to Fuzzy through the public web app."""

    registration: Literal["open", "invite", "closed"] = "invite"
    session_days: float = Field(default=14.0, gt=0, description="Absolute session lifetime.")
    idle_hours: float = Field(default=72.0, gt=0, description="Sessions expire after inactivity.")
    cookie_secure: bool = Field(default=True, description="Send cookies over HTTPS only.")
    min_password_length: int = Field(default=12, ge=8)
    max_failed_logins: int = Field(default=5, ge=1)
    lockout_minutes: float = Field(default=15.0, gt=0)
    login_attempts_per_ip_per_minute: float = Field(default=10.0, gt=0)
    require_admin_totp: bool = False
    daily_messages: int = Field(default=200, ge=0, description="Per-user chat quota (0 = none).")
    daily_images: int = Field(default=10, ge=0)
    daily_teach_bytes: int = Field(default=20_000_000, ge=0)
    chat_timeout_s: float = Field(
        default=7200.0, gt=0, description="How long a reply may take before it is given up on."
    )
    public_pages: bool = Field(
        default=False, description="Let visitors who are not signed in view the brain and world."
    )
    public_tools: list[str] = Field(
        default_factory=lambda: [
            "memory_recall",
            "memory_remember",
            "memory_forget",
            "conversation_search",
            "self_define",
            "self_revise",
            "self_reflect",
            "image_generate",
            "world_look",  # it can talk about its world; changing it is Fuzzy's own choice only
            "plan_later",  # ...but it can note that it means to, and decide in its own time
        ],
        description="Tools Fuzzy may use in conversations with people who are not admins.",
    )
    hsts: bool = Field(default=False, description="Send Strict-Transport-Security (HTTPS only).")
    trusted_proxies: list[str] = Field(
        default_factory=list,
        description="Reverse proxies whose X-Forwarded-For header is believed.",
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
    accounts: AccountsConfig = Field(default_factory=AccountsConfig)
    mind: MindConfig = Field(default_factory=MindConfig)
    images: ImagesConfig = Field(default_factory=ImagesConfig)
    workers: WorkersConfig = Field(default_factory=WorkersConfig)
    world: WorldConfig = Field(default_factory=WorldConfig)
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
    def api_token_path(self) -> Path:
        return self.storage.data_dir / "api_token"

    def resolved_api_token(self, *, create: bool = False) -> str | None:
        """The operator API token: configured, else the generated file (made if ``create``)."""

        if self.api.auth_token is not None:
            return self.api.auth_token.get_secret_value()
        path = self.api_token_path
        try:
            existing = path.read_text(encoding="utf-8").strip()
        except OSError:
            existing = ""
        if existing or not create:
            return existing or None
        path.parent.mkdir(parents=True, exist_ok=True)
        token = secrets.token_urlsafe(32)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token + "\n")
        return token

    @property
    def media_path(self) -> Path:
        return self.storage.data_dir / "media"

    @property
    def staging_path(self) -> Path:
        return self.self_improvement.staging_dir or self.storage.data_dir / "staging"

    def ensure_directories(self) -> None:
        for path in (
            self.storage.data_dir,
            self.storage.resolved_archive_dir(),
            self.audit_path,
            self.media_path,
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
