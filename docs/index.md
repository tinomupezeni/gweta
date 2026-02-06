# Gweta

**The Missing Middleware for RAG Pipelines**

Gweta is a quality-aware framework that handles data acquisition, validation, and ingestion as a single pipeline, exposed over MCP for AI agent integration.

## The Problem

```
BEFORE: Document → Parser → Chunks → Vector DB → Bad answers → Debug → Repeat
```

RAG pipelines often fail silently. You parse documents, chunk them, load them into a vector store — and only discover quality issues when your AI starts hallucinating.

Even worse: **storing everything** pollutes your vector store with irrelevant content, leading to noisy retrieval and degraded answers.

## The Solution

```
AFTER:  Document → Gweta (acquire + validate + filter + ingest) → Quality chunks → Good answers
```

Gweta validates and filters data at every stage:

1. **Extraction Quality** - Is the text properly extracted?
2. **Chunk Quality** - Are chunks coherent and information-dense?
3. **Intent Relevance** - Does the content match your system's purpose?
4. **Domain Rules** - Does the content match known facts?
5. **KB Health** - Is the knowledge base fresh and complete?

## Quick Start

```bash
pip install gweta[intelligence]
```

```python
from gweta.intelligence import Pipeline, SystemIntent
from gweta import ChromaStore

# Define what your RAG system is meant to do
intent = SystemIntent(
    name="My Knowledge Base",
    description="Answers questions about Zimbabwe business registration",
    core_questions=["How do I register a business in Zimbabwe?"],
    relevant_topics=["Zimbabwe business", "ZIMRA", "EcoCash"],
    irrelevant_topics=["US regulations", "cryptocurrency"],
)

# Create intent-aware pipeline
store = ChromaStore(collection_name="my-kb")
pipeline = Pipeline(intent=intent, store=store)

# Ingest with automatic relevance filtering
result = await pipeline.ingest(chunks)
print(f"Ingested: {result.ingested} relevant chunks")
print(f"Rejected: {result.rejected_count} irrelevant chunks")
```

## Key Features

### Intelligence Layer (NEW in v0.2.0)

Gweta understands your system's purpose and filters content for relevance:

- **SystemIntent** - Define what your RAG system is meant to do (YAML-based)
- **RelevanceFilter** - Score chunks by semantic similarity to your intent
- **Pipeline** - Unified API for intent-aware ingestion

```python
# Load intent from YAML
intent = SystemIntent.from_yaml("intents/my_system.yaml")

# Filter chunks by relevance
filter = RelevanceFilter(intent)
report = filter.filter_batch(chunks)

# Only relevant chunks get stored
accepted = report.accepted()  # Chunks with score >= 0.6
rejected = report.rejected_count  # Irrelevant content filtered out
```

### Multi-Source Acquisition

- **Web Crawling** - JavaScript-rendered pages with Crawl4AI
- **PDF Extraction** - Tables and text with quality scoring
- **Database Connector** - SQL extraction with safety guards
- **API Client** - REST endpoint fetching

### 4-Layer Validation

| Layer | What it Checks |
|-------|----------------|
| **Extraction** | OCR quality, encoding, gibberish detection |
| **Chunks** | Coherence, density, boundary quality |
| **Domain Rules** | YAML-based rules, known fact verification |
| **KB Health** | Staleness, duplicates, coverage gaps |

### Vector Store Integration

- ChromaDB
- Qdrant
- Pinecone
- Weaviate

### MCP Server

Expose Gweta to AI agents like Claude Desktop:

```json
{
  "mcpServers": {
    "gweta": {
      "command": "gweta",
      "args": ["serve"]
    }
  }
}
```

## Documentation

- [Getting Started](getting-started.md) - 5-minute quickstart
- [Intelligence Layer](guides/intelligence.md) - Intent-aware filtering guide
- [Architecture](concepts/architecture.md) - How Gweta works
- [API Reference](api/reference.md) - Full API documentation
- [Examples](examples/full-pipeline.md) - Complete pipeline example

## Design Principles

| Principle | Implementation |
|-----------|----------------|
| Parser-agnostic | Works with any document parser |
| Chunker-agnostic | Works with any chunking strategy |
| Store-agnostic | Loads to any vector database |
| Lightweight core | Heuristics by default, optional LLM validation |
| Declarative rules | YAML-based domain rules |

## License

MIT License - see [LICENSE](https://github.com/tinomupezeni/gweta/blob/main/LICENSE)
