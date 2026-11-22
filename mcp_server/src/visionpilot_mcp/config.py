"""Runtime settings, read from environment variables and the repo's .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# src/visionpilot_mcp/config.py -> parents[2] is the mcp_server/ folder.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PACKAGE_ROOT.parent


class Settings(BaseSettings):
    """All knobs for the MCP server. Every field can be overridden with an env var of the same name."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", PACKAGE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Where ONNX exports and OCR weights live (gitignored). Relative paths resolve against mcp_server/.
    model_dir: Path = Field(default=Path("models"))
    log_dir: Path = Field(default=Path("logs"))
    log_level: str = "INFO"

    # 8 GB RAM budget: keep at most this many models in memory, evict least recently used.
    max_loaded_models: int = Field(default=2, ge=1, le=8)
    result_cache_size: int = Field(default=128, ge=0)
    cpu_threads: int = Field(default=4, ge=1, le=64)

    # Transport for `visionpilot-mcp` (stdio for Claude Desktop, streamable-http for the agent backend).
    mcp_transport: Literal["stdio", "streamable-http"] = "stdio"
    mcp_http_host: str = "127.0.0.1"
    mcp_http_port: int = 8765

    allow_remote_urls: bool = True

    yolo_imgsz: int = 640
    florence_model_id: str = "florence-community/Florence-2-base-ft"
    florence_num_beams: int = Field(default=3, ge=1, le=5)
    florence_max_new_tokens: int = Field(default=256, ge=16, le=1024)

    @field_validator("model_dir", "log_dir")
    @classmethod
    def _resolve_relative(cls, value: Path) -> Path:
        value = value.expanduser()
        return value if value.is_absolute() else (PACKAGE_ROOT / value).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
