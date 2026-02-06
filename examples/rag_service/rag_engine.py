"""RAG Engine using Gweta.

This module demonstrates the correct way to use Gweta's
ChromaStore for building RAG applications.

Usage:
    from rag_engine import RAGEngine

    engine = RAGEngine()
    await engine.ingest_json("data.json")
    results = await engine.query("your question")
"""
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

from gweta import ChromaStore, Chunk
from gweta.ingest.stores.base import StoreStats


@dataclass
class QueryRequest:
    """Query request parameters."""
    query: str
    top_k: int = 5
    category: Optional[str] = None
    pathway: Optional[str] = None


@dataclass
class QueryResult:
    """Single query result."""
    text: str
    source: str
    score: float
    metadata: dict = field(default_factory=dict)


class RAGEngine:
    """RAG engine powered by Gweta.

    Example:
        >>> engine = RAGEngine()
        >>> await engine.ingest_json("knowledge.json")
        >>> results = await engine.query(QueryRequest(query="How to...?"))
    """

    def __init__(
        self,
        collection_name: str = "knowledge-base",
        persist_dir: Optional[str] = "./data/chroma",
    ):
        """Initialize the RAG engine.

        Args:
            collection_name: Name for the ChromaDB collection
            persist_dir: Directory for persistence (None for in-memory)
        """
        self.store = ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )

        # Track metadata internally since ChromaStore doesn't have
        # get_all_metadata() method
        self._sources: set[str] = set()
        self._categories: set[str] = set()
        self._pathways: set[str] = set()
        self._total_chunks = 0
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize and load existing stats."""
        if self._initialized:
            return

        # Load existing chunks to rebuild internal tracking
        try:
            existing = await self.store.get_all()
            for chunk in existing:
                self._sources.add(chunk.source)
                if "category" in chunk.metadata:
                    self._categories.add(chunk.metadata["category"])
                if "pathway" in chunk.metadata:
                    self._pathways.add(chunk.metadata["pathway"])
            self._total_chunks = len(existing)
        except Exception:
            pass

        self._initialized = True

    async def ingest_json(self, json_path: str) -> int:
        """Ingest content from a JSON file.

        Expected JSON format:
        [
            {
                "text": "Content here...",
                "source": "source-name",
                "category": "category-name",    # optional
                "pathway": "pathway-name",       # optional
                "metadata": {...}                # optional
            }
        ]

        Args:
            json_path: Path to JSON file

        Returns:
            Number of chunks added
        """
        await self.initialize()

        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        for item in data:
            metadata = item.get("metadata", {})
            if "category" in item:
                metadata["category"] = item["category"]
            if "pathway" in item:
                metadata["pathway"] = item["pathway"]

            chunk = Chunk(
                text=item["text"],
                source=item.get("source", path.stem),
                metadata=metadata,
            )
            chunks.append(chunk)

            # Track internally
            self._sources.add(chunk.source)
            if metadata.get("category"):
                self._categories.add(metadata["category"])
            if metadata.get("pathway"):
                self._pathways.add(metadata["pathway"])

        if chunks:
            result = await self.store.add(chunks)
            self._total_chunks += result.added
            return result.added

        return 0

    async def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Ingest a list of Chunk objects.

        Args:
            chunks: List of Chunk objects to ingest

        Returns:
            Number of chunks added
        """
        await self.initialize()

        if not chunks:
            return 0

        # Track internally
        for chunk in chunks:
            self._sources.add(chunk.source)
            if chunk.metadata.get("category"):
                self._categories.add(chunk.metadata["category"])
            if chunk.metadata.get("pathway"):
                self._pathways.add(chunk.metadata["pathway"])

        result = await self.store.add(chunks)
        self._total_chunks += result.added
        return result.added

    async def query(self, request: QueryRequest) -> list[QueryResult]:
        """Query the knowledge base.

        Args:
            request: QueryRequest with query parameters

        Returns:
            List of QueryResult objects
        """
        await self.initialize()

        # Build filter dict
        # IMPORTANT: Use 'filter' not 'where'
        filter_dict: dict[str, Any] = {}
        if request.category:
            filter_dict["category"] = request.category
        if request.pathway:
            filter_dict["pathway"] = request.pathway

        # Query store
        chunks = await self.store.query(
            query=request.query,
            n_results=request.top_k,
            filter=filter_dict if filter_dict else None,
        )

        # Convert to results with approximate scores
        results = []
        for i, chunk in enumerate(chunks):
            # Approximate score from position (Chroma doesn't return scores)
            score = 1.0 - (i * 0.1)
            results.append(QueryResult(
                text=chunk.text,
                source=chunk.source,
                score=max(0.0, score),
                metadata=chunk.metadata,
            ))

        return results

    def get_health(self) -> dict:
        """Get health status.

        Returns:
            Dict with health information

        Note:
            get_stats() is SYNC, not async!
        """
        stats = self.store.get_stats()  # No await!
        return {
            "status": "healthy",
            "total_chunks": stats.chunk_count,
            "total_sources": len(self._sources),
            "categories": sorted(list(self._categories)),
            "pathways": sorted(list(self._pathways)),
        }

    async def get_categories(self) -> list[str]:
        """Get all categories in the knowledge base."""
        await self.initialize()
        return sorted(list(self._categories))

    async def get_pathways(self) -> list[str]:
        """Get all pathways in the knowledge base."""
        await self.initialize()
        return sorted(list(self._pathways))

    async def get_sources(self) -> list[str]:
        """Get all sources in the knowledge base."""
        await self.initialize()
        return sorted(list(self._sources))

    async def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a specific source.

        Note: ChromaStore.delete() only accepts IDs, not filters.
        This method gets all chunks and filters manually.

        Args:
            source: Source identifier to delete

        Returns:
            Number of chunks deleted
        """
        await self.initialize()

        # Get all chunks and find matching IDs
        all_chunks = await self.store.get_all()
        ids_to_delete = [
            chunk.id for chunk in all_chunks
            if chunk.source == source
        ]

        if ids_to_delete:
            deleted = await self.store.delete(ids_to_delete)
            self._total_chunks -= deleted
            self._sources.discard(source)
            return deleted

        return 0


# Example usage
if __name__ == "__main__":
    import asyncio

    async def main():
        engine = RAGEngine(persist_dir=None)  # In-memory for demo

        # Create sample data
        chunks = [
            Chunk(
                text="To register a business in Zimbabwe, submit Form CR6 to the Companies Registry.",
                source="zimra-guide",
                metadata={"category": "business", "pathway": "entrepreneurship"},
            ),
            Chunk(
                text="EcoCash merchant registration is free at any Econet shop.",
                source="ecocash-guide",
                metadata={"category": "payments", "pathway": "entrepreneurship"},
            ),
        ]

        # Ingest
        added = await engine.ingest_chunks(chunks)
        print(f"Added {added} chunks")

        # Query
        results = await engine.query(QueryRequest(query="How to register business?"))
        print(f"Found {len(results)} results:")
        for r in results:
            print(f"  - {r.text[:50]}... (score: {r.score:.2f})")

        # Health
        health = engine.get_health()
        print(f"Health: {health}")

    asyncio.run(main())
