# RAG Service Example

A complete example of building a RAG service with Gweta.

## Quick Start

```bash
# Install dependencies
pip install gweta[chroma] fastapi uvicorn

# Run the service
cd examples/rag_service
uvicorn main:app --reload
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How to register a business?", "top_k": 5}'
```

### Ingest Content
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "text": "To register a business in Zimbabwe...",
    "source": "guide",
    "category": "business"
  }'
```

### List Categories/Sources/Pathways
```bash
curl http://localhost:8000/categories
curl http://localhost:8000/sources
curl http://localhost:8000/pathways
```

## Files

- `rag_engine.py` - Core RAG engine using Gweta's ChromaStore
- `main.py` - FastAPI application
- `data/sample.json` - Sample data (optional)

## Key Implementation Details

### Correct ChromaStore.query() Usage

```python
# CORRECT
results = await store.query(
    query="search text",     # Required
    n_results=10,            # Optional
    filter={"key": "value"}, # Optional - NOT 'where'!
)

# WRONG - 'where' doesn't exist
results = await store.query(query="text", where={"key": "value"})
```

### get_stats() is Sync

```python
# CORRECT - no await
stats = store.get_stats()

# WRONG
stats = await store.get_stats()
```

### Tracking Metadata

ChromaStore doesn't have `get_all_metadata()`. Track metadata internally:

```python
class RAGEngine:
    def __init__(self):
        self._categories: set[str] = set()
        self._sources: set[str] = set()

    async def ingest(self, chunk):
        self._sources.add(chunk.source)
        if chunk.metadata.get("category"):
            self._categories.add(chunk.metadata["category"])
        await self.store.add([chunk])
```

## Sample Data Format

Create `data/sample.json`:

```json
[
    {
        "text": "Content here...",
        "source": "source-name",
        "category": "category-name",
        "pathway": "pathway-name",
        "metadata": {
            "extra": "data"
        }
    }
]
```
