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

    # Field-level confidence below this threshold downgrades a required rule
    # from pass/fail to ADVISORY. SPEC §0.5: "every check has an explicit
    # confidence threshold below which it downgrades from required to
    # advisory, with the reason surfaced to the user." Single source of
    # truth — extractors and the rule engine import from here so a future
    # tuning change cannot drift between the read side and the judge side.
    low_confidence_threshold: float = 0.6

    # Verify-path upload cap. SPEC v1 acceptance criteria assume ~1 MB
    # photos; 15 MiB is generous headroom for raw 12 MP captures while
    # bounding the worst-case memory hit on a 256 MB Fly/Railway machine
    # (raw + base64 ~= 40 MB at the cap, vs unbounded today).
    max_image_bytes: int = 15 * 1024 * 1024

    # Hard ceiling on wall-clock for one /v1/verify call. The orchestrator
    # already times out the underlying Anthropic SDK call individually, but
    # with retries a flaky upstream can chain together to 60+ s. This caps
    # the request from the FastAPI side so a small batch of stuck calls
    # cannot eat every uvicorn worker.
    verify_request_timeout_s: float = 30.0

    anthropic_api_key: str | None = None
    # Haiku 4.5 is the right size for the verify-path OCR/extraction job:
    # it transcribes label fields character-for-character as accurately as
    # Sonnet 4.6 on the bundled samples (verified manually) but typically
    # returns ~2× faster — measured 3.8 s vs 7.5 s on a real beer label —
    # which is the dominant share of the user-facing latency budget the
    # iterative-design workflow runs against. Override to `claude-sonnet-4-6`
    # (or Opus) only when tuning a specific accuracy regression.
    anthropic_model: str = "claude-haiku-4-5-20251001"
    # SPEC §0.5 mandates *two independent reads* of the Government Warning.
    # Routing the second pass to a different model family from the primary
    # is the load-bearing piece of that redundancy: two reads from the
    # same Haiku version are correlated and tend to agree on the same
    # mistake (e.g. misreading "(1)" as "(I)"). Sonnet 4.6 here keeps the
    # two reads genuinely independent while still finishing inside the
    # second-pass timeout (8 s; a one-paragraph transcription on Sonnet
    # lands well under that). Override only when explicitly tuning.
    anthropic_health_warning_model: str = "claude-sonnet-4-6"
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

    # Perceptual-hash reverse-image lookup sits underneath the byte-
    # exact verify cache and serves any visually-equivalent re-submission
    # — different JPEG quality, PNG-vs-JPEG of the same artwork, the
    # same physical bottle re-photographed. A hit reuses the prior
    # cold-path's vision extraction (skipping the VLM call) but still
    # re-runs the rule engine with the current request's container size,
    # imported flag, claim, and rule fingerprint. 0 disables the
    # reverse-lookup outright.
    reverse_lookup_max_entries: int = 4096
    # Hamming-distance threshold over a 64-bit dhash. ≤6 bit-flips out
    # of 64 keeps the false-positive rate well below 1 % on the
    # imagehash benchmarks; loosening past ~10 starts mixing visually-
    # similar-but-distinct labels (two beers from the same brand line
    # often sit ~12-15 bits apart on dhash).
    reverse_lookup_hamming_threshold: int = 6

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
    # 30 s suits hosted endpoints (OpenRouter, DashScope) where the model
    # is already loaded. Local Ollama with a quantised Qwen3-VL on a Mac
    # M-series can take 60–180 s on cold load + first inference, so this
    # is overrideable per environment rather than a hard-coded module
    # constant.
    qwen_vl_timeout_s: float = 30.0

    # AI-generated rule explanations. Run on the cold path post-verdict
    # to add a one-sentence plain-language gloss to each failed rule,
    # contextualised to THIS scan's extracted values and image quality.
    # The explanation is purely additive — generation failures never
    # block the verdict; the response simply omits the field. Keep on
    # by default since the upside (clearer fix-it UX) is unconditional
    # and the downside is bounded by the per-call timeout.
    explanation_enabled: bool = True
    # Empty string means "use anthropic_model" (Haiku 4.5 by default —
    # right size for one-sentence per-rule explanations). Override only
    # to A/B against Sonnet for tuning explanation quality.
    explanation_model: str = ""
    # Six-second cap is a balance: Haiku 4.5 typically returns the JSON
    # batch in 1.5–3 s, but a P99 spike under contention could hit 5 s+.
    # The verify response is already on the cold path here, so we can
    # afford the wait; what we cannot afford is a stuck call eating a
    # uvicorn worker. asyncio.wait_for trips at this value and we silently
    # drop the explanations rather than re-raising.
    explanation_timeout_s: float = 6.0
    # Cap how many failed rules we batch into one prompt. Six is enough
    # to cover the long-tail "label fails on 4–5 things" case while
    # keeping the prompt + response token count predictable.
    explanation_max_rules: int = 6

    # External-source tier — TTB COLA public search adapter. Default
    # OFF: the parser is keyed off the live HTML form, and operators
    # should validate it against their own TTB load before trusting
    # results in /v1/verify hits. The user-agent string identifies
    # ProofRead so the TTB can block us cleanly if our query rate ever
    # becomes a problem; the timeout caps a single HTTP round-trip,
    # not the rate-limit-aware queue around it.
    ttb_cola_lookup_enabled: bool = False
    ttb_cola_timeout_s: float = 4.0
    ttb_cola_user_agent: str = (
        "ProofRead/1.0 (+https://github.com/shoyu-ramen/ProofRead)"
    )

    # L3 perceptual cache backed by the `label_cache` Postgres table.
    # Survives process restart and accumulates a corpus of verified
    # labels across deploys. Default OFF until the operator has run the
    # schema migration (`Base.metadata.create_all`-equivalent for the
    # new `LabelCacheEntry` model). When OFF, the in-process L1+L2
    # caches still serve their normal hit rates; only the durable tier
    # is skipped.
    persisted_label_cache_enabled: bool = False
    # Hamming threshold for the L3 lookup. Mirrors the in-process L2
    # default (6 bit-flips out of 64 keeps the false-positive rate well
    # under 1 % per imagehash benchmarks). Tunable independently of L2
    # because the L3 corpus tends to grow much larger and operators may
    # want to tighten the threshold to compensate.
    persisted_label_cache_hamming_threshold: int = 6

    # Admin endpoint auth. The admin cache-health endpoint
    # (``GET /v1/admin/cache-health``) requires this token in the
    # ``X-Admin-Token`` request header. When unset the endpoint
    # returns 503 (admin disabled) so a stray production deploy
    # without the token configured cannot accidentally expose cache
    # internals to anonymous callers.
    admin_api_token: str | None = None

    # Telemetry. All three are optional — missing values mean the
    # corresponding init is a no-op and local dev / CI never sees a
    # remote callout. `environment` and `release` tag every event so
    # Sentry / Honeycomb dashboards can scope by deploy.
    sentry_dsn: str | None = None
    honeycomb_api_key: str | None = None
    otel_exporter_otlp_endpoint: str | None = None
    deploy_environment: str = "dev"
    deploy_release: str | None = None


settings = Settings()
