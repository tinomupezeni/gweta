"""MCP prompts for Gweta.

This module defines the MCP prompts that provide
interaction templates for AI agents.
"""

from typing import Any

from gweta.core.logging import get_logger

logger = get_logger(__name__)


def register_prompts(mcp: Any) -> None:
    """Register all MCP prompts.

    Args:
        mcp: FastMCP server instance
    """

    @mcp.prompt()
    async def plan_ingestion(
        sources: str,
        target: str,
    ) -> str:
        """Generate an ingestion plan for the given sources and target.

        Analyzes source types, recommends crawl strategies, and
        estimates quality thresholds.

        Args:
            sources: Description of sources to ingest
            target: Target collection name
        """
        return f"""
You are planning a data ingestion job for Gweta, a RAG data quality framework.

**Sources to ingest:**
{sources}

**Target collection:** {target}

Please analyze the sources and create a plan covering:
1. Source type identification (web, PDF, database, API)
2. Recommended crawl depth for web sources
3. Authority tier assignment for each source (1=blog, 5=legislation)
4. Suggested quality threshold (0.0-1.0)
5. Estimated chunk count
6. Potential quality issues to watch for

Use the gweta tools to execute the plan once approved:
- `crawl_and_ingest` for websites
- `ingest_from_database` for databases
- `check_health` to verify quality after ingestion
"""

    @mcp.prompt()
    async def quality_review(
        collection: str,
    ) -> str:
        """Review the quality of a knowledge base collection.

        Args:
            collection: Collection name to review
        """
        return f"""
You are reviewing the quality of the "{collection}" knowledge base.

Use the `check_health` tool to get the current quality report, then:
1. Identify sources with low quality scores
2. Find duplicate or near-duplicate chunks
3. Detect stale content that needs refreshing
4. Recommend specific actions to improve quality

Provide a summary with actionable recommendations.
"""

    @mcp.prompt()
    async def troubleshoot_rag(
        issue: str,
        collection: str,
    ) -> str:
        """Troubleshoot RAG quality issues.

        Args:
            issue: Description of the quality issue
            collection: Collection experiencing issues
        """
        return f"""
You are troubleshooting a RAG quality issue in the "{collection}" knowledge base.

**Reported Issue:**
{issue}

Use Gweta tools to investigate:
1. Run `check_health` to get overall quality metrics
2. Check for specific chunk issues
3. Review the source authority and freshness
4. Identify potential causes

Provide a diagnosis and recommended fixes using Gweta tools.
"""

    logger.info("Registered MCP prompts")
