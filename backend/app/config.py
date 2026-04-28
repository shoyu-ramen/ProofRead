from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    database_url: str = (
        "postgresql+asyncpg://proofread:proofread@localhost:5432/proofread"
    )

    storage_backend: str = "local"
    storage_local_path: str = "./.storage"
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"

    ocr_provider: str = "mock"
    google_vision_credentials_path: str | None = None

    rule_definitions_path: str = "app/rules/definitions"
    canonical_texts_path: str = "app/canonical"

    auth_mode: str = "stub"
    auth0_domain: str | None = None
    auth0_audience: str | None = None

    health_warning_max_edit_distance: int = 0

    anthropic_api_key: str | None = None
    # Sonnet 4.6 is the right size for the verify-path OCR/extraction job:
    # it transcribes label fields as accurately as Opus 4.7 on structured
    # output but typically returns 2–3× faster, which dominates the user-
    # facing latency budget the iterative-design workflow runs against.
    # Opus is reserved for code paths that genuinely benefit from deeper
    # reasoning; reading printed text on a label is not one of them.
    anthropic_model: str = "claude-sonnet-4-6"
    # SPEC §0.5 mandates a redundant read of the Government Warning. Haiku
    # 4.5 is the cheapest, fastest model with strong vision OCR — perfect
    # for a single-paragraph re-read whose only job is to produce a second
    # independent transcription for the cross-check to reconcile against
    # the primary read. Override only when explicitly tuning.
    anthropic_health_warning_model: str = "claude-haiku-4-5-20251001"
    vision_extractor: str = "claude"  # "claude" | "mock"

    # Toggle the health-warning redundant second-pass. Default ON because
    # SPEC §0.5 mandates two independent reads of the warning paragraph,
    # and the second call runs concurrently with the primary so the
    # accuracy gain is essentially free in wall-clock terms. Set to false
    # only to debug the primary extractor in isolation.
    enable_health_warning_second_pass: bool = True

    # In-process LRU on /v1/verify results. A hit returns the prior
    # cold-path verdict in <50 ms, which is what the iterative-design
    # workflow (re-submitting the same artwork export) needs. 0 disables
    # the cache outright.
    verify_cache_max_entries: int = 1024

    # Local Qwen3-VL fallback. When `enable_qwen_fallback` is true and
    # `qwen_vl_base_url` points at an OpenAI-compatible chat-completions
    # endpoint (vLLM, Ollama, LM Studio, llama.cpp --api), the verify and
    # scan extractors fall back to Qwen3-VL if the Anthropic call raises
    # ExtractorUnavailable. Disabled by default so the prod deploy keeps
    # the same code path until the env vars are explicitly set.
    enable_qwen_fallback: bool = False
    qwen_vl_base_url: str | None = None
    qwen_vl_model: str = "qwen3-vl"
    qwen_vl_api_key: str | None = None


settings = Settings()
