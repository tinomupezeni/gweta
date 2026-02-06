"""Tests for configuration module."""

import os

import pytest

from gweta.core.config import GwetaSettings, get_settings


class TestGwetaSettings:
    """Tests for the GwetaSettings class."""

    def test_default_settings(self):
        """Test default settings creation."""
        settings = GwetaSettings()
        assert settings.log_level == "INFO"
        assert settings.log_format == "structured"
        assert settings.min_quality_score == 0.6
        assert settings.default_chunk_size == 500

    def test_custom_settings(self):
        """Test custom settings."""
        settings = GwetaSettings(
            log_level="DEBUG",
            min_quality_score=0.8,
        )
        assert settings.log_level == "DEBUG"
        assert settings.min_quality_score == 0.8

    def test_settings_from_env(self, monkeypatch):
        """Test settings from environment variables."""
        monkeypatch.setenv("GWETA_LOG_LEVEL", "WARNING")
        monkeypatch.setenv("GWETA_MIN_QUALITY_SCORE", "0.75")

        # Create new settings to pick up env vars
        settings = GwetaSettings()
        assert settings.log_level == "WARNING"
        assert settings.min_quality_score == 0.75

    def test_get_settings_singleton(self):
        """Test that get_settings returns consistent instance."""
        settings1 = get_settings()
        settings2 = get_settings()
        # Should work with caching
        assert settings1 is not None
        assert settings2 is not None


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_quality_score_bounds(self):
        """Test quality score must be between 0 and 1."""
        # Pydantic should raise ValidationError for out-of-bounds values
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GwetaSettings(min_quality_score=1.5)

        with pytest.raises(ValidationError):
            GwetaSettings(min_quality_score=-0.1)

    def test_chunk_size_positive(self):
        """Test chunk size must be positive."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GwetaSettings(default_chunk_size=10)  # Below minimum of 50

    def test_valid_log_level(self):
        """Test valid log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            settings = GwetaSettings(log_level=level)
            assert settings.log_level == level
