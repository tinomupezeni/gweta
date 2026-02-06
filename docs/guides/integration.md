# Integration Guide

**Building RAG Services with Gweta**

This guide shows exactly how to integrate Gweta into your applications. It covers the exact API signatures, common patterns, and working code examples.

## Installation

```bash
# Core + ChromaDB + Intelligence Layer
pip install gweta[chroma,intelligence]

# Or everything
pip install gweta[all]
```

## Quick Reference

### ChromaStore API

```python
from gweta import ChromaStore, Chunk

# Initialize
store = ChromaStore(
    collection_name="my-kb",
    persist_directory="./data",  # Optional: for persistence
)

# Add chunks
result = await store.add(chunks)  # Returns AddResult
print(f"Added: {result.added}")

# Query (EXACT SIGNATURE)
results = await store.query(
    query="your search query",    # Required: search text
    n_results=10,                 # Optional: default 10
    filter={"category": "tech"},  # Optional: metadata filter (NOT 'where')
)
# Returns: list[Chunk]

# Get statistics
stats = store.get_stats()  # Returns StoreStats (sync, not async)
print(f"Total chunks: {stats.chunk_count}")

# Get all chunks
all_chunks = await store.get_all()  # Returns list[Chunk]

# Delete by IDs
deleted = await store.delete(["chunk-1", "chunk-2"])  # Returns int
```

### Chunk Data Type

```python
from gweta import Chunk

chunk = Chunk(
    id="unique-id",           # Optional: auto-generated if not provided
    text="Content here...",   # Required: the actual text
    source="document.pdf",    # Required: source identifier
    metadata={                # Optional: arbitrary key-value pairs
        "category": "business",
        "page": 1,
    },
    quality_score=0.85,       # Optional: 0.0 to 1.0
)

# Access fields
print(chunk.id)
print(chunk.text)
print(chunk.source)
print(chunk.metadata)
print(chunk.quality_score)
```

### AddResult and StoreStats

```python
from gweta.ingest.stores.base import AddResult, StoreStats

# AddResult (returned by store.add())
result = await store.add(chunks)
print(result.added)    # int: chunks added
print(result.skipped)  # int: chunks skipped (duplicates)
print(result.errors)   # list[str] | None: error messages

# StoreStats (returned by store.get_stats())
stats = store.get_stats()
print(stats.collection_name)  # str
print(stats.chunk_count)      # int
print(stats.dimension)        # int | None
print(stats.metadata)         # dict | None
```

## Building a RAG Service

Here's a complete example of building a FastAPI RAG service with Gweta.

### Project Structure

```
rag-service/
├── main.py           # FastAPI app
├── rag_engine.py     # Gweta integration
├── data/             # Persist directory
│   └── chroma/
└── knowledge/        # Source documents
    └── content.json
```

### rag_engine.py

```python
"""RAG Engine using Gweta."""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from gweta import ChromaStore, Chunk
from gweta.ingest.stores.base import StoreStats


@dataclass
class QueryRequest:
    """Query request."""
    query: str
    top_k: int = 5
    category: Optional[str] = None


@dataclass
class QueryResult:
    """Query result."""
    text: str
    source: str
    score: float
    metadata: dict


class RAGEngine:
    """RAG engine powered by Gweta."""

    def __init__(
        self,
        collection_name: str = "knowledge-base",
        persist_dir: str = "./data/chroma",
    ):
        self.store = ChromaStore(
            collection_name=collection_name,
            persist_directory=persist_dir,
        )
        # Track data internally (ChromaStore doesn't have get_all_metadata)
        self._sources: set[str] = set()
        self._categories: set[str] = set()
        self._total_ingested = 0

    async def ingest_json(self, json_path: str) -> int:
        """Ingest content from JSON file.

        Expected format:
        [
            {
                "text": "Content here...",
                "source": "source-name",
                "category": "category-name",
                "metadata": {...}
            }
        ]
        """
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {json_path}")

        with open(path) as f:
            data = json.load(f)

        chunks = []
        for item in data:
            chunk = Chunk(
                text=item["text"],
                source=item.get("source", path.stem),
                metadata={
                    "category": item.get("category", "general"),
                    **item.get("metadata", {}),
                },
            )
            chunks.append(chunk)

            # Track internally
            self._sources.add(chunk.source)
            if "category" in chunk.metadata:
                self._categories.add(chunk.metadata["category"])

        if chunks:
            result = await self.store.add(chunks)
            self._total_ingested += result.added
            return result.added

        return 0

    async def query(self, request: QueryRequest) -> list[QueryResult]:
        """Query the knowledge base."""
        # Build filter if category specified
        filter_dict = None
        if request.category:
            filter_dict = {"category": request.category}

        # Query store (note: 'filter' not 'where')
        chunks = await self.store.query(
            query=request.query,
            n_results=request.top_k,
            filter=filter_dict,
        )

        # Convert to results
        results = []
        for i, chunk in enumerate(chunks):
            results.append(QueryResult(
                text=chunk.text,
                source=chunk.source,
                score=1.0 - (i * 0.1),  # Approximate score from position
                metadata=chunk.metadata,
            ))

        return results

    def get_health(self) -> dict:
        """Get health status."""
        stats = self.store.get_stats()  # Note: sync method, not async
        return {
            "status": "healthy",
            "total_chunks": stats.chunk_count,
            "sources": sorted(list(self._sources)),
            "categories": sorted(list(self._categories)),
        }

    async def get_categories(self) -> list[str]:
        """Get all categories."""
        return sorted(list(self._categories))

    async def get_sources(self) -> list[str]:
        """Get all sources."""
        return sorted(list(self._sources))
```

### main.py

```python
"""FastAPI RAG Service."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from rag_engine import RAGEngine, QueryRequest

app = FastAPI(title="RAG Service", version="1.0.0")
engine = RAGEngine()


class QueryBody(BaseModel):
    query: str
    top_k: int = 5
    category: Optional[str] = None


class QueryResponse(BaseModel):
    results: list[dict]
    total: int


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    sources: list[str]
    categories: list[str]


@app.on_event("startup")
async def startup():
    """Load initial data on startup."""
    try:
        # Load knowledge from JSON files
        added = await engine.ingest_json("knowledge/content.json")
        print(f"Loaded {added} chunks")
    except FileNotFoundError:
        print("No initial data found")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return engine.get_health()


@app.post("/query", response_model=QueryResponse)
async def query(body: QueryBody):
    """Query the knowledge base."""
    try:
        request = QueryRequest(
            query=body.query,
            top_k=body.top_k,
            category=body.category,
        )
        results = await engine.query(request)
        return {
            "results": [
                {
                    "text": r.text,
                    "source": r.source,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in results
            ],
            "total": len(results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/categories")
async def get_categories():
    """Get all categories."""
    return await engine.get_categories()


@app.get("/sources")
async def get_sources():
    """Get all sources."""
    return await engine.get_sources()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### knowledge/content.json

```json
[
    {
        "text": "To register a Private Business Corporation (PBC) in Zimbabwe, submit Form CR6 to the Companies Registry with a $50-80 fee.",
        "source": "zimra-guide",
        "category": "business-registration",
        "metadata": {"country": "Zimbabwe"}
    },
    {
        "text": "EcoCash merchant registration is free at any Econet shop. Transaction fees are 1-2%.",
        "source": "ecocash-guide",
        "category": "payments",
        "metadata": {"country": "Zimbabwe"}
    }
]
```

## Common Mistakes

### 1. Wrong query parameter name

```python
# WRONG - 'where' doesn't exist
results = await store.query(
    query="search text",
    where={"category": "tech"},  # ❌ Wrong!
)

# CORRECT - use 'filter'
results = await store.query(
    query="search text",
    filter={"category": "tech"},  # ✅ Correct!
)
```

### 2. Calling get_stats() as async

```python
# WRONG - get_stats is sync
stats = await store.get_stats()  # ❌ Wrong!

# CORRECT - no await needed
stats = store.get_stats()  # ✅ Correct!
```

### 3. Expecting non-existent methods

```python
# These methods DON'T EXIST:
store.get_collection_info()  # ❌ Doesn't exist
store.get_all_metadata()     # ❌ Doesn't exist
store.delete(where={...})    # ❌ Wrong signature

# Use these instead:
stats = store.get_stats()           # ✅ For collection info
chunks = await store.get_all()      # ✅ For all data
deleted = await store.delete(ids)   # ✅ Delete by ID list
```

### 4. Not awaiting async methods

```python
# WRONG - missing await
result = store.add(chunks)    # ❌ Returns coroutine, not result
results = store.query("q")    # ❌ Returns coroutine, not results

# CORRECT - use await
result = await store.add(chunks)    # ✅ Returns AddResult
results = await store.query("q")    # ✅ Returns list[Chunk]
```

## With Intelligence Layer

Add intent-aware filtering to your RAG service:

```python
from gweta import ChromaStore, Chunk
from gweta.intelligence import Pipeline, SystemIntent

# Define your system's intent
intent = SystemIntent(
    name="Zimbabwe Career Platform",
    description="Career guidance for Zimbabwean graduates",
    core_questions=[
        "How do I register a business in Zimbabwe?",
        "What are ZIMRA requirements?",
    ],
    relevant_topics=["Zimbabwe business", "ZIMRA", "EcoCash"],
    irrelevant_topics=["US regulations", "cryptocurrency"],
)

# Create pipeline
store = ChromaStore("my-kb", persist_directory="./data")
pipeline = Pipeline(intent=intent, store=store)

# Ingest with filtering
async def ingest_with_filtering(chunks: list[Chunk]):
    result = await pipeline.ingest(chunks)
    print(f"Ingested: {result.ingested}")
    print(f"Rejected: {result.rejected_count}")
    return result

# Or preview before ingesting
def preview_filter(chunks: list[Chunk]):
    report = pipeline.filter_only(chunks)
    print(f"Would accept: {report.accepted_count}")
    print(f"Would reject: {report.rejected_count}")

    for r in report.results:
        if r.rejected:
            print(f"  Reject: {r.chunk.source} - {r.rejection_reason}")

    return report
```

## Full API Reference

### ChromaStore

| Method | Signature | Returns | Notes |
|--------|-----------|---------|-------|
| `__init__` | `(collection_name, client=None, embedding_function=None, persist_directory=None, use_default_embeddings=True)` | - | Sync |
| `add` | `async (chunks: list[Chunk])` | `AddResult` | Async |
| `query` | `async (query: str, n_results: int = 10, filter: dict = None)` | `list[Chunk]` | Async |
| `delete` | `async (chunk_ids: list[str])` | `int` | Async |
| `get_all` | `async ()` | `list[Chunk]` | Async |
| `get_stats` | `()` | `StoreStats` | **Sync** |
| `update` | `async (chunk: Chunk)` | `bool` | Async |
| `search_by_metadata` | `async (filter: dict, limit: int = 100)` | `list[Chunk]` | Async |
| `collection_name` | property | `str` | Property |

### ChunkValidator

```python
from gweta import ChunkValidator, Chunk

validator = ChunkValidator(
    min_length=50,              # Minimum text length
    required_metadata=None,     # Required metadata keys
)

# Validate single chunk
result = validator.validate(chunk)
print(result.passed)         # bool
print(result.score)          # float
print(result.issues)         # list[QualityIssue]

# Validate batch
report = validator.validate_batch(chunks)
print(report.total_chunks)   # int
print(report.passed)         # int
print(report.failed)         # int
print(report.avg_quality_score)  # float
```

### Pipeline (Intelligence Layer)

```python
from gweta.intelligence import Pipeline, SystemIntent

intent = SystemIntent(...)
pipeline = Pipeline(intent=intent, store=store)

# Ingest with filtering
result = await pipeline.ingest(chunks)
print(result.ingested)           # int
print(result.rejected_count)     # int
print(result.acceptance_rate)    # float

# Filter only (preview)
report = pipeline.filter_only(chunks)
print(report.accepted_count)     # int
print(report.rejected_count)     # int

# Score single chunk
scores = pipeline.score_chunk(chunk)
print(scores['quality_score'])    # int or None
print(scores['relevance_score'])  # float
print(scores['would_ingest'])     # bool
```

## Testing Your Integration

```python
import asyncio
from gweta import ChromaStore, Chunk

async def test_integration():
    # Create store
    store = ChromaStore("test-collection")

    # Add a chunk
    chunk = Chunk(
        text="Test content about Zimbabwe business registration.",
        source="test",
        metadata={"category": "test"},
    )
    result = await store.add([chunk])
    assert result.added == 1, f"Expected 1, got {result.added}"

    # Query
    results = await store.query(
        query="Zimbabwe business",
        n_results=5,
    )
    assert len(results) >= 1, "Expected at least 1 result"
    assert "Zimbabwe" in results[0].text

    # Get stats
    stats = store.get_stats()
    assert stats.chunk_count >= 1

    print("All tests passed!")

asyncio.run(test_integration())
```

## Troubleshooting

### "ChromaStore.query() got an unexpected keyword argument"

You're using wrong parameter names. Check:
- Use `query=` not `query_text=` or `text=`
- Use `n_results=` not `top_k=`
- Use `filter=` not `where=`

### "You must provide an embedding function"

Update to gweta >= 0.1.1:
```bash
pip install --upgrade gweta[chroma]
```

### "sentence-transformers not installed"

Install the dependency:
```bash
pip install sentence-transformers
```

### Async/await issues

Remember:
- `add()`, `query()`, `delete()`, `get_all()` are **async** - use `await`
- `get_stats()` is **sync** - don't use `await`

### Metadata not persisting

ChromaDB metadata must be `str`, `int`, `float`, or `bool`. Complex types are converted to strings automatically.
