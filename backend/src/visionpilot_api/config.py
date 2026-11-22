"""Backend settings from environment variables and the repo's .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parents[2]  # backend/
REPO_ROOT = PACKAGE_ROOT.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", PACKAGE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-3.5-flash-lite"
    gemini_timeout_s: float = 60.0
    gemini_max_retries: int = Field(default=3, ge=0, le=8)

    mcp_url: str = "http://127.0.0.1:8765/mcp"
    mcp_call_timeout_s: float = 240.0

    agent_max_steps: int = Field(default=8, ge=1, le=20)

    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    log_level: str = "INFO"

    data_dir: Path = Path("../data")
    samples_dir: Path = Path("../samples")
    max_upload_mb: int = Field(default=100, ge=1, le=1000)

    @field_validator("data_dir", "samples_dir")
    @classmethod
    def _resolve(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (PACKAGE_ROOT / value).resolve()

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
