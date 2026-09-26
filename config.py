"""Runtime configuration for the Freight Exception Triage service.

Every knob lives here and is overridable by environment variable (prefix `TRIAGE_`) or a
local `.env` file, so the same image runs locally, in CI and in production without a code
change. Nothing in this module touches the network or the filesystem at import time.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from schemas import TriagePolicy


class Settings(BaseSettings):
    """Process-wide settings; construct once in the composition root."""

    model_config = SettingsConfigDict(
        env_prefix="TRIAGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # Provider credentials are declared with explicit aliases (DEEPSEEK_API_KEY, not
        # TRIAGE_DEEPSEEK_API_KEY) because that is the name every SDK and every operator's
        # shell already uses. populate_by_name keeps Settings(deepseek_api_key=...) working
        # in tests.
        populate_by_name=True,
    )

    app_name: str = "freight-exception-triage"
    environment: Literal["local", "test", "staging", "production"] = "local"
    log_level: str = "INFO"

    # -- persistence ------------------------------------------------------------------
    database_path: Path = Field(
        default=Path("var/triage.db"),
        description="SQLite file. The directory is created on startup if missing.",
    )
    db_pool_size: int = Field(
        default=4,
        ge=1,
        le=32,
        description="One aiosqlite connection (and thread) per pool slot; WAL keeps readers "
        "from blocking on the single writer.",
    )
    db_busy_timeout_ms: int = Field(default=5_000, ge=0)
    seed_demo_slas: bool = Field(
        default=True,
        description="Insert the demo SLA rows on startup when the SLA table is empty.",
    )

    # -- worker pool ------------------------------------------------------------------
    worker_count: int = Field(default=4, ge=1, le=32, description="Concurrent triage workers.")
    worker_poll_interval_seconds: float = Field(
        default=5.0,
        gt=0,
        description="Safety-net wake-up. Workers are normally woken by an event, not by this.",
    )
    job_lease_seconds: int = Field(
        default=60,
        ge=5,
        description="Visibility timeout. A job whose lease expires is reclaimed by another "
        "worker, which is what makes a mid-triage crash recoverable.",
    )
    job_max_attempts: int = Field(default=3, ge=1)
    job_backoff_base_seconds: float = Field(default=2.0, gt=0)
    job_backoff_max_seconds: float = Field(default=60.0, gt=0)

    # -- backpressure and shutdown ----------------------------------------------------
    max_pending_jobs: int = Field(
        default=500,
        ge=1,
        description="Ingestion returns 503 above this backlog instead of accepting work it "
        "cannot drain.",
    )
    shutdown_grace_seconds: float = Field(default=10.0, gt=0)

    # -- triage engine ----------------------------------------------------------------
    triage_engine: Literal["rule_based", "llm"] = Field(
        default="rule_based",
        # Without the alias the prefix would make this TRIAGE_TRIAGE_ENGINE.
        validation_alias=AliasChoices("TRIAGE_ENGINE", "TRIAGE_TRIAGE_ENGINE"),
        description="rule_based needs no credentials and is the deterministic baseline; llm "
        "adds judgement and prose on top of the same arithmetic.",
    )
    prompt_version: str = "triage-prompt-v2"
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_max_attempts: int = Field(
        default=3,
        ge=1,
        description="tenacity transport attempts per provider (rate limit / connection / 5xx).",
    )
    llm_max_validation_retries: int = Field(
        default=2,
        ge=0,
        description="instructor schema repairs: how many times a malformed proposal is "
        "re-prompted with its own validation error before the provider is abandoned.",
    )
    llm_fallback_enabled: bool = Field(
        default=True, description="Try the next configured provider when one fails."
    )

    # DeepSeek is primary: cheapest per token of the two and strong at this kind of
    # constrained extraction. OpenAI is the fallback for provider outages.
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "TRIAGE_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"
    deepseek_mode: str = Field(
        default="TOOLS",
        description="instructor mode. DeepSeek does not implement strict json_schema outputs, "
        "so the proposal is enforced through a tool definition.",
    )

    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "TRIAGE_OPENAI_API_KEY"),
    )
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    openai_mode: str = Field(
        default="JSON_SCHEMA",
        description="OpenAI supports the real structured-outputs feature, so the schema is "
        "enforced server-side rather than by prompt.",
    )

    deepseek_enforcement: Literal["instructor", "sdk_parse"] = Field(
        default="instructor",
        description="DeepSeek has no strict json_schema output, so the proposal is enforced "
        "through a tool definition plus validation and repair.",
    )
    openai_enforcement: Literal["instructor", "sdk_parse"] = Field(
        default="sdk_parse",
        description="sdk_parse uses OpenAI's structured outputs (strict: true, every property "
        "required), which is enforced during generation rather than after it.",
    )

    log_json: bool = Field(default=False, description="Emit JSON log lines for a log shipper.")

    # -- policy -----------------------------------------------------------------------
    policy_version: str = "1.0"
    min_confidence_for_auto_email: float = Field(default=0.80, ge=0.0, le=1.0)
    max_penalty_for_auto_email: Decimal = Decimal("250.00")
    always_escalate_tiers: frozenset[str] = frozenset({"VIP"})

    def to_policy(self) -> TriagePolicy:
        """Materialise the guardrail policy that every decision is filtered through."""
        return TriagePolicy(
            policy_version=self.policy_version,
            min_confidence_for_auto_email=self.min_confidence_for_auto_email,
            max_penalty_for_auto_email=self.max_penalty_for_auto_email,
            always_escalate_tiers=frozenset(self.always_escalate_tiers),
        )

    # -- LLM provider chain -----------------------------------------------------------

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.deepseek_api_key or self.openai_api_key)

    def to_provider_chain(self) -> list[Any]:
        """Ordered providers: DeepSeek first, OpenAI second, unconfigured ones skipped.

        Imported lazily so that Settings stays usable without the LLM stack installed.
        """
        from llm import Enforcement, ProviderSpec

        chain: list[ProviderSpec] = []
        if self.deepseek_api_key:
            chain.append(
                ProviderSpec(
                    name="deepseek",
                    model=self.deepseek_model,
                    api_key=self.deepseek_api_key,
                    base_url=self.deepseek_base_url,
                    mode=self.deepseek_mode,
                    enforcement=Enforcement(self.deepseek_enforcement),
                )
            )
        if self.openai_api_key and (self.llm_fallback_enabled or not chain):
            chain.append(
                ProviderSpec(
                    name="openai",
                    model=self.openai_model,
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url,
                    mode=self.openai_mode,
                    enforcement=Enforcement(self.openai_enforcement),
                )
            )
        return chain

    @property
    def db_is_memory(self) -> bool:
        return str(self.database_path) in {":memory:", "file::memory:"}
