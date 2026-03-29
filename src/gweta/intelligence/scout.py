"""Intelligence Scout for Goal-Driven Data Acquisition.

This module provides the IntelligenceScout class, which uses LLMs
to autonomously discover, navigate, and extract content based on
natural language goals or system intent.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

from gweta.acquire.crawler import GwetaCrawler
from gweta.core.logging import get_logger
from gweta.core.types import Chunk
from gweta.intelligence.intent import SystemIntent
from gweta.intelligence.llm import LLMClient

logger = get_logger(__name__)


@dataclass
class ScoutResult:
    """Result of a scouting operation.

    Attributes:
        goal: The original goal
        queries: Search queries generated
        urls_discovered: List of URLs found via search
        pages_visited: Count of pages actually crawled
        chunks_extracted: Total chunks created
        extracted_data: List of structured data objects (if requested)
    """
    goal: str
    queries: List[str] = field(default_factory=list)
    urls_discovered: List[str] = field(default_factory=list)
    pages_visited: int = 0
    chunks_extracted: int = 0
    extracted_data: List[Dict[str, Any]] = field(default_factory=list)
    raw_chunks: list[Chunk] = field(default_factory=list, repr=False)


class IntelligenceScout:
    """AI-powered scout for goal-driven web discovery and extraction.

    Uses LLMs to:
    1. Generate search queries from a natural language goal.
    2. Rank and select links on a page (agentic navigation).
    3. Extract structured data from pages (smart extraction).

    Example:
        >>> scout = IntelligenceScout(model="gpt-4o")
        >>> result = await scout.scout(
        ...     goal="Find recent business registration fees in Zimbabwe",
        ...     max_pages=5
        ... )
        >>> print(f"Found {len(result.urls_discovered)} URLs")
    """

    def __init__(
        self,
        intent: Optional[SystemIntent] = None,
        llm: Optional[LLMClient] = None,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize the scout.

        Args:
            intent: Optional system intent to guide the scout.
            llm: Optional pre-configured LLMClient.
            model: Model name for LLM operations.
            api_key: Optional API key for LLM.
        """
        self.intent = intent
        self.llm = llm or LLMClient(model=model, api_key=api_key)
        self.crawler = GwetaCrawler()
        self._visited_urls: Set[str] = set()

    async def scout(
        self,
        goal: str,
        max_pages: int = 10,
        max_depth: int = 2,
        extract_schema: Optional[Dict[str, Any]] = None,
    ) -> ScoutResult:
        """Perform a scouting operation based on a goal.

        Args:
            goal: Natural language goal or description.
            max_pages: Maximum number of pages to visit.
            max_depth: Maximum link depth from search results.
            extract_schema: Optional JSON schema for structured extraction.

        Returns:
            ScoutResult with discovery and extraction details.
        """
        logger.info(f"Starting scouting operation for goal: '{goal}'")
        result = ScoutResult(goal=goal)

        # 1. Generate search queries
        queries = await self.generate_search_queries(goal)
        result.queries = queries
        logger.info(f"Generated {len(queries)} search queries: {queries}")

        # 2. Discover URLs via search
        discovery_urls = await self.discover_urls(queries, limit=max_pages * 2)
        result.urls_discovered = discovery_urls
        logger.info(f"Discovered {len(discovery_urls)} potential URLs")

        # 3. Crawl and navigate
        to_visit = discovery_urls[:max_pages]
        pending_depth = {url: 0 for url in to_visit}

        while to_visit and result.pages_visited < max_pages:
            url = to_visit.pop(0)
            if url in self._visited_urls:
                continue

            current_depth = pending_depth.get(url, 0)
            if current_depth > max_depth:
                continue

            logger.info(f"Scouting page ({result.pages_visited + 1}/{max_pages}): {url}")
            try:
                page_result = await self.crawler.crawl(url, depth=1)
                self._visited_urls.add(url)
                result.pages_visited += 1

                if not page_result.chunks:
                    continue

                # Smart Extraction
                if extract_schema:
                    combined_text = "\n\n".join(c.text for c in page_result.chunks[:5])
                    extracted = await self.llm.extract_json(
                        text=combined_text,
                        goal=goal,
                        schema=extract_schema
                    )
                    if "error" not in extracted:
                        result.extracted_data.append(extracted)

                # Collect chunks
                result.raw_chunks.extend(page_result.chunks)
                result.chunks_extracted += len(page_result.chunks)

                # Agentic Navigation (only if depth permits)
                if current_depth < max_depth:
                    # Extract links from page (assuming crawler or BeautifulSoup can get them)
                    # For now, we'll look at the 'links' metadata if available or just use crawler's next depth
                    # But the requirement is agentic navigation, so we should see the links and pick.
                    pass # TODO: Implement link ranking

            except Exception as e:
                logger.error(f"Failed to scout {url}: {e}")

        return result

    async def generate_search_queries(self, goal: str, count: int = 3) -> List[str]:
        """Use LLM to generate effective search queries for a goal.

        Args:
            goal: The goal description.
            count: Number of queries to generate.

        Returns:
            List of search query strings.
        """
        prompt = (
            f"Given the following goal for a data gathering task, generate {count} "
            "effective search queries for a search engine like Google or DuckDuckGo. "
            "Focus on finding authoritative sources, official documents, and current data.\n\n"
            f"Goal: {goal}\n\n"
            "Return the queries as a JSON array of strings."
        )

        response = await self.llm.extract_json(
            text=prompt,
            goal="Generate search queries",
            schema={"type": "array", "items": {"type": "string"}}
        )

        if isinstance(response, list):
            return response[:count]
        elif isinstance(response, dict) and "queries" in response:
            return response["queries"][:count]
        else:
            # Fallback to simple goal if LLM fails
            return [goal]

    async def discover_urls(self, queries: List[str], limit: int = 10) -> List[str]:
        """Execute search queries and return discovered URLs.

        Args:
            queries: Search queries to run.
            limit: Maximum URLs to return.

        Returns:
            List of unique URLs found.
        """
        if DDGS is None:
            logger.warning("duckduckgo_search is not installed. Discovery will be limited.")
            return []

        urls = []
        seen = set()

        try:
            with DDGS() as ddgs:
                for query in queries:
                    results = ddgs.text(query, max_results=limit // len(queries) + 2)
                    for r in results:
                        url = r.get("href")
                        if url and url not in seen:
                            urls.append(url)
                            seen.add(url)
                            if len(urls) >= limit:
                                break
                    if len(urls) >= limit:
                        break
        except Exception as e:
            logger.error(f"Discovery search failed: {e}")

        return urls

    async def rank_links(
        self,
        links: List[Dict[str, str]],
        goal: str,
        limit: int = 5
    ) -> List[str]:
        """Use LLM to rank links by relevance to the goal.

        Args:
            links: List of {'text': '...', 'url': '...'} dicts.
            goal: The scouting goal.
            limit: Maximum links to return.

        Returns:
            List of URLs sorted by relevance.
        """
        if not links:
            return []

        # Prepare link data for LLM
        link_previews = [
            {"index": i, "text": l.get("text", "")[:50], "url": l.get("url", "")}
            for i, l in enumerate(links[:50]) # Limit to 50 links for context
        ]

        prompt = (
            "You are an expert web navigator. Given a goal and a list of links "
            "from a webpage, identify the top most promising links to follow "
            "to achieve the goal.\n\n"
            f"Goal: {goal}\n\n"
            f"Links:\n{json.dumps(link_previews, indent=2)}\n\n"
            "Return a JSON array of indices for the top 5 links in order of relevance."
        )

        indices = await self.llm.extract_json(
            text=prompt,
            goal="Rank links",
            schema={"type": "array", "items": {"type": "integer"}}
        )

        if not isinstance(indices, list):
            return []

        ranked_urls = []
        for idx in indices:
            if 0 <= idx < len(links):
                ranked_urls.append(links[idx]["url"])

        return ranked_urls[:limit]
