"""Source authority registry for tracking trusted data sources.

This module provides the SourceAuthorityRegistry class for managing
trusted sources, their authority tiers, and freshness requirements.
"""

import fnmatch
import re
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from gweta.core.config import get_settings
from gweta.core.exceptions import ConfigurationError
from gweta.core.types import Source


@dataclass
class SourcePattern:
    """Pattern for matching sources.

    Supports exact domain matching, wildcard patterns (*.gov.zw),
    and regex patterns.
    """
    pattern: str
    name: str
    authority_tier: int
    freshness_days: int
    is_regex: bool = False

    def matches(self, url: str) -> bool:
        """Check if URL matches this pattern."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = url.lower()

        if self.is_regex:
            return bool(re.match(self.pattern, domain))
        else:
            # Support wildcard patterns like *.gov.zw
            return fnmatch.fnmatch(domain, self.pattern.lower())


@dataclass
class SourceAuthorityRegistry:
    """Registry of trusted sources with authority tiers.

    Manages a collection of source patterns that define:
    - Which domains/URLs are allowed
    - Authority tier for each source (1-5)
    - Freshness requirements (how often content should be re-crawled)

    Example YAML format:
        sources:
          - domain: "zimra.co.zw"
            name: "ZIMRA Official"
            authority: 5
            freshness_days: 30
          - domain: "*.gov.zw"
            name: "Zimbabwe Government"
            authority: 4
            freshness_days: 90
        blocked:
          - "spam-site.com"
    """

    sources: list[SourcePattern] = field(default_factory=list)
    blocked_domains: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "SourceAuthorityRegistry":
        """Load registry from YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            Configured SourceAuthorityRegistry

        Raises:
            ConfigurationError: If file is invalid or missing
        """
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(
                f"Authority registry file not found: {path}",
                setting_name="authority_registry_path",
                setting_value=str(path),
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in authority registry: {e}",
                setting_name="authority_registry_path",
                setting_value=str(path),
            ) from e

        return cls.from_dict(data or {})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceAuthorityRegistry":
        """Create registry from dictionary.

        Args:
            data: Dictionary with 'sources' and optional 'blocked' keys

        Returns:
            Configured SourceAuthorityRegistry
        """
        sources = []
        for source_data in data.get("sources", []):
            domain = source_data.get("domain", source_data.get("pattern", ""))
            sources.append(
                SourcePattern(
                    pattern=domain,
                    name=source_data.get("name", domain),
                    authority_tier=source_data.get("authority", 3),
                    freshness_days=source_data.get("freshness_days", 90),
                    is_regex=source_data.get("is_regex", False),
                )
            )

        blocked = data.get("blocked", [])

        return cls(sources=sources, blocked_domains=blocked)

    @classmethod
    def get_default(cls) -> "SourceAuthorityRegistry":
        """Get the default registry from settings or empty registry."""
        settings = get_settings()
        if settings.authority_registry_path:
            return cls.from_yaml(settings.authority_registry_path)
        return cls()

    def is_allowed(self, url: str) -> bool:
        """Check if URL is allowed (not blocked).

        Args:
            url: URL to check

        Returns:
            True if URL is not in blocked list
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
        except Exception:
            domain = url.lower()

        for blocked in self.blocked_domains:
            if fnmatch.fnmatch(domain, blocked.lower()):
                return False

        return True

    def is_trusted(self, url: str) -> bool:
        """Check if URL is from a trusted (registered) source.

        Args:
            url: URL to check

        Returns:
            True if URL matches a registered source pattern
        """
        if not self.is_allowed(url):
            return False

        return any(source.matches(url) for source in self.sources)

    def get_authority(self, url: str) -> int:
        """Get authority tier for URL.

        Args:
            url: URL to look up

        Returns:
            Authority tier (1-5), or default from settings if not found
        """
        for source in self.sources:
            if source.matches(url):
                return source.authority_tier

        return get_settings().default_authority_tier

    def get_freshness_window(self, url: str) -> timedelta:
        """Get freshness window for URL.

        Args:
            url: URL to look up

        Returns:
            Timedelta representing freshness window
        """
        for source in self.sources:
            if source.matches(url):
                return timedelta(days=source.freshness_days)

        return timedelta(days=get_settings().default_freshness_days)

    def get_source(self, url: str) -> Source:
        """Get Source object for URL.

        Args:
            url: URL to look up

        Returns:
            Source object with authority and freshness info
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
        except Exception:
            domain = url

        for source_pattern in self.sources:
            if source_pattern.matches(url):
                return Source(
                    id=domain,
                    name=source_pattern.name,
                    url=url,
                    authority_tier=source_pattern.authority_tier,
                    freshness_days=source_pattern.freshness_days,
                )

        # Return default source
        settings = get_settings()
        return Source(
            id=domain,
            name=domain,
            url=url,
            authority_tier=settings.default_authority_tier,
            freshness_days=settings.default_freshness_days,
        )

    def get_all(self) -> list[Source]:
        """Get all registered sources as Source objects."""
        return [
            Source(
                id=pattern.pattern,
                name=pattern.name,
                authority_tier=pattern.authority_tier,
                freshness_days=pattern.freshness_days,
            )
            for pattern in self.sources
        ]

    def add_source(
        self,
        domain: str,
        name: str | None = None,
        authority: int = 3,
        freshness_days: int = 90,
    ) -> None:
        """Add a source pattern to the registry.

        Args:
            domain: Domain pattern (supports wildcards like *.gov.zw)
            name: Human-readable name
            authority: Authority tier (1-5)
            freshness_days: Freshness window in days
        """
        self.sources.append(
            SourcePattern(
                pattern=domain,
                name=name or domain,
                authority_tier=authority,
                freshness_days=freshness_days,
            )
        )

    def block_domain(self, domain: str) -> None:
        """Add a domain to the blocked list.

        Args:
            domain: Domain pattern to block
        """
        if domain not in self.blocked_domains:
            self.blocked_domains.append(domain)

    def to_yaml(self) -> str:
        """Export registry to YAML string."""
        data = {
            "sources": [
                {
                    "domain": s.pattern,
                    "name": s.name,
                    "authority": s.authority_tier,
                    "freshness_days": s.freshness_days,
                }
                for s in self.sources
            ],
            "blocked": self.blocked_domains,
        }
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
