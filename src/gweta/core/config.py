"""Configuration management for Gweta.

This module provides the GwetaSettings class for managing all
configuration options, supporting both environment variables and
configuration files.
"""

from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GwetaSettings(BaseSettings):
    """Global configuration for Gweta.

    Settings can be configured via:
    - Environment variables (prefixed with GWETA_)
    - .env file
    - Direct instantiation

    Example:
        >>> settings = GwetaSettings(min_quality_score=0.7)
        >>> # Or via environment: GWETA_MIN_QUALITY_SCORE=0.7
    """

    # Quality thresholds
    min_quality_score: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum quality score for a chunk to pass validation",
    )
    min_density_score: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum information density score",
    )
    max_duplicate_similarity: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Maximum similarity threshold for duplicate detection",
    )

    # Extraction settings
    min_text_length: int = Field(
        default=50,
        ge=0,
        description="Minimum text length for a valid chunk",
    )
    max_gibberish_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum ratio of gibberish characters allowed",
    )

    # Chunking defaults
    default_chunk_size: int = Field(
        default=500,
        ge=50,
        description="Default chunk size in characters",
    )
    default_chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Default overlap between chunks",
    )

    # Authority settings
    authority_registry_path: Path | None = Field(
        default=None,
        description="Path to the source authority registry YAML file",
    )
    default_authority_tier: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Default authority tier for unknown sources",
    )
    default_freshness_days: int = Field(
        default=90,
        ge=1,
        description="Default freshness window in days",
    )

    # Crawling settings
    crawl_timeout: int = Field(
        default=30,
        ge=1,
        description="Timeout for crawl requests in seconds",
    )
    max_crawl_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum crawl depth",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to respect robots.txt",
    )

    # Database settings
    db_read_only: bool = Field(
        default=True,
        description="Enforce read-only database queries",
    )
    db_query_timeout: int = Field(
        default=30,
        ge=1,
        description="Database query timeout in seconds",
    )
    db_max_rows: int = Field(
        default=10000,
        ge=1,
        description="Maximum rows to return from database queries",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    log_format: str = Field(
        default="structured",
        description="Log format: 'structured' or 'plain'",
    )

    # MCP settings
    mcp_transport: str = Field(
        default="stdio",
        description="MCP transport: 'stdio' or 'http'",
    )
    mcp_http_port: int = Field(
        default=8080,
        ge=1,
        le=65535,
        description="HTTP port for MCP server",
    )

    model_config = SettingsConfigDict(
        env_prefix="GWETA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()


# Global settings instance (lazy-loaded)
_settings: GwetaSettings | None = None


def get_settings() -> GwetaSettings:
    """Get the global settings instance.

    Returns:
        The global GwetaSettings instance, creating it if needed.
    """
    global _settings
    if _settings is None:
        _settings = GwetaSettings()
    return _settings


def configure(**kwargs: Any) -> GwetaSettings:
    """Configure global settings.

    Args:
        **kwargs: Settings to override

    Returns:
        The updated global settings instance

    Example:
        >>> configure(min_quality_score=0.8, log_level="DEBUG")
    """
    global _settings
    _settings = GwetaSettings(**kwargs)
    return _settings
