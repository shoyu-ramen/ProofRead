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
    anthropic_model: str = "claude-opus-4-7"
    vision_extractor: str = "claude"  # "claude" | "mock"


settings = Settings()
