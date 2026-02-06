# Gweta

**The Missing Middleware for RAG Pipelines**

Gweta is a quality-aware framework that handles data acquisition, validation, and ingestion as a single pipeline, exposed over MCP for AI agent integration.

## The Problem

```
BEFORE: Document → Parser → Chunks → Vector DB → Bad answers → Debug → Repeat
```

RAG pipelines often fail silently. You parse documents, chunk them, load them into a vector store — and only discover quality issues when your AI starts hallucinating.

## The Solution

```
AFTER:  Document → Gweta (acquire + validate + ingest) → Quality chunks → Good answers
```

Gweta validates data at every stage:

1. **Extraction Quality** - Is the text properly extracted?
2. **Chunk Quality** - Are chunks coherent and information-dense?
3. **Domain Rules** - Does the content match known facts?
4. **KB Health** - Is the knowledge base fresh and complete?

## Quick Start

```bash
pip install gweta
```

```python
from gweta import ChunkValidator, Chunk

# Validate chunks before loading
validator = ChunkValidator()
chunks = [
    Chunk(text="Your content here...", source="document.pdf", metadata={})
]

report = validator.validate_batch(chunks)
print(f"Quality Score: {report.avg_quality_score}")
print(f"Passed: {report.passed}/{report.total_chunks}")
```

## Key Features

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
