"""MCP resources for Gweta.

This module defines the MCP resources that expose
Gweta's data to AI agents.
"""

from typing import Any

import yaml

from gweta.core.logging import get_logger

logger = get_logger(__name__)


def register_resources(mcp: Any) -> None:
    """Register all MCP resources.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.resource("gweta://sources")
    async def list_sources() -> str:
        """List all registered data sources with their authority tiers,
        last crawl dates, and freshness status.

        Returns:
            YAML-formatted list of sources
        """
        from gweta.core.registry import SourceAuthorityRegistry

        registry = SourceAuthorityRegistry.get_default()
        sources = registry.get_all()

        return yaml.dump(
            [s.to_dict() for s in sources],
            default_flow_style=False,
        )

    @mcp.resource("gweta://quality/{collection}")
    async def quality_report(collection: str) -> str:
        """Get the latest quality report for a collection.

        Args:
            collection: Collection name

        Returns:
            JSON-formatted quality report
        """
        import json

        from gweta.ingest.stores.chroma import ChromaStore
        from gweta.validate.health import HealthChecker

        store = ChromaStore(collection_name=collection)
        checker = HealthChecker(store)
        report = await checker.full_health_check()

        return json.dumps(
            {
                "collection": report.collection,
                "total_chunks": report.total_chunks,
                "avg_quality_score": report.avg_quality_score,
                "recommendations": report.recommendations,
            },
            indent=2,
        )

    @mcp.resource("gweta://rules/{domain}")
    async def domain_rules(domain: str) -> str:
        """Get the validation rules for a specific domain.

        Args:
            domain: Domain name

        Returns:
            YAML-formatted rule definitions
        """
        from pathlib import Path

        from gweta.validate.rules import DomainRuleEngine

        # Try to load rules from standard location
        rules_path = Path(f"rules/{domain}.yaml")
        if rules_path.exists():
            engine = DomainRuleEngine.from_yaml(rules_path)
            return engine.to_yaml()
        else:
            return f"# No rules found for domain: {domain}\nrules: []\nknown_facts: []\n"

    @mcp.resource("gweta://config")
    async def current_config() -> str:
        """Get current Gweta configuration.

        Returns:
            JSON-formatted configuration
        """
        import json

        from gweta.core.config import get_settings

        settings = get_settings()
        return json.dumps(settings.to_dict(), indent=2)

    logger.info("Registered MCP resources")
