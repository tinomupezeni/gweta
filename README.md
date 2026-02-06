# Gweta

[![PyPI version](https://badge.fury.io/py/gweta.svg)](https://badge.fury.io/py/gweta)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/yourusername/gweta/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/gweta/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/yourusername/gweta/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/gweta)

**Acquire. Validate. Ingest.**

Gweta is a RAG data quality and ingestion framework that handles the full data lifecycle for knowledge bases. It provides quality-aware acquisition from various sources, multi-stage validation, and ingestion into vector stores—all exposed via MCP (Model Context Protocol) for AI agent integration.

## Features

- **Multi-Source Acquisition**: Web crawling (Crawl4AI), PDF extraction, REST/GraphQL APIs, databases
- **4-Stage Validation Pipeline**:
  - Extraction quality (encoding, completeness, structure)
  - Chunk quality (length, density, coherence, gibberish detection)
  - Domain rules (custom YAML-based validation, known facts)
  - Knowledge base health (coverage, staleness, retrieval quality)
- **Vector Store Integration**: Chroma, Qdrant, Pinecone, Weaviate
- **MCP Server**: Expose all capabilities to AI agents via Model Context Protocol
- **Framework Adapters**: LangChain, LlamaIndex, Chonkie integration

## Installation

```bash
# Basic installation
pip install gweta

# With specific vector store
pip install gweta[chroma]
pip install gweta[qdrant]
pip install gweta[pinecone]

# With framework adapters
pip install gweta[langchain]
pip install gweta[llamaindex]
pip install gweta[chonkie]

# With database support
pip install gweta[db]

# Full installation
pip install gweta[all]
```

## Quick Start

### Validate Chunks

```python
from gweta import Chunk, ChunkValidator

# Create chunks
chunks = [
    Chunk(text="Your content here...", source="https://example.com"),
    Chunk(text="More content...", source="https://example.com/page2"),
]

# Validate
validator = ChunkValidator()
report = validator.validate_batch(chunks)

print(f"Passed: {report.passed}/{report.total_chunks}")
print(f"Average quality: {report.avg_quality_score:.2f}")

# Get only validated chunks
good_chunks = report.accepted()
```

### Crawl and Ingest

```python
import asyncio
from gweta import GwetaCrawler, ChromaStore

async def main():
    # Crawl with validation
    crawler = GwetaCrawler()
    result = await crawler.crawl("https://docs.example.com", depth=2)

    print(f"Extracted {len(result.chunks)} validated chunks")

    # Ingest to vector store
    store = ChromaStore(collection_name="my-knowledge-base")
    await store.add(result.chunks)

asyncio.run(main())
```

### Use with LangChain

```python
from langchain.document_loaders import WebBaseLoader
from gweta import ChunkValidator
from gweta.adapters import LangChainAdapter

# Load documents with LangChain
loader = WebBaseLoader("https://example.com")
documents = loader.load()

# Convert and validate with Gweta
adapter = LangChainAdapter()
chunks = adapter.from_documents(documents)

validator = ChunkValidator()
report = validator.validate_batch(chunks)

# Convert back to LangChain format
validated_docs = adapter.to_documents(report.accepted())
```

### Use with LlamaIndex

```python
from llama_index.core import SimpleDirectoryReader
from gweta import ChunkValidator
from gweta.adapters import LlamaIndexAdapter

# Load with LlamaIndex
reader = SimpleDirectoryReader("./docs")
nodes = reader.load_data()

# Validate with Gweta
adapter = LlamaIndexAdapter()
chunks = adapter.from_nodes(nodes)

validator = ChunkValidator()
report = validator.validate_batch(chunks)

# Convert back to LlamaIndex nodes
validated_nodes = adapter.to_nodes(report.accepted())
```

## CLI Usage

```bash
# Validate chunks from a file
gweta validate chunks.json --threshold 0.7

# Crawl a website
gweta crawl https://docs.example.com --depth 2 --output chunks.json

# Ingest to vector store
gweta ingest https://docs.example.com my-collection

# Check knowledge base health
gweta health my-collection --golden golden_dataset.json

# Start MCP server
gweta serve --transport stdio
gweta serve --transport http --port 8080
```

## MCP Integration

Gweta exposes its capabilities via MCP for AI agent integration:

```json
{
  "mcpServers": {
    "gweta": {
      "command": "gweta",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

### Available MCP Tools

- `crawl_and_ingest`: Crawl URL and ingest to vector store
- `validate_chunks`: Validate chunk quality
- `check_health`: Check knowledge base health
- `query_knowledge`: Query the knowledge base
- `list_sources`: List configured sources

### Available MCP Resources

- `gweta://sources`: Configured data sources
- `gweta://quality/{collection}`: Quality metrics for a collection
- `gweta://health/{collection}`: Health status of a collection

## Configuration

### Environment Variables

```bash
GWETA_LOG_LEVEL=INFO
GWETA_LOG_FORMAT=plain  # or "json"
GWETA_DEFAULT_MIN_QUALITY=0.6
GWETA_DEFAULT_CHUNK_SIZE=512
GWETA_CHROMA_PERSIST_DIR=./chroma_data
```

### Source Authority Registry

Create a `sources.yaml` to define trusted sources:

```yaml
sources:
  - url: https://docs.python.org
    name: Python Documentation
    authority_score: 1.0
    refresh_hours: 168
    tags: [python, official]

  - url: https://developer.mozilla.org
    name: MDN Web Docs
    authority_score: 0.95
    refresh_hours: 24
    tags: [web, official]
```

### Domain Rules

Create domain-specific validation rules:

```yaml
name: my-domain
version: "1.0"

rules:
  - id: no-pii
    name: No PII
    pattern: '\b\d{3}-\d{2}-\d{4}\b'
    action: reject
    severity: error

  - id: require-attribution
    name: Require Attribution
    condition: "'source' in chunk.metadata"
    action: warn
    severity: warning

known_facts:
  - claim: "Python 3.12 was released in 2023"
    confidence: 1.0
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      MCP Interface                          │
│              (Tools, Resources, Prompts)                    │
├─────────────────────────────────────────────────────────────┤
│                    Ingestion Layer                          │
│         (Chunkers, Pipeline, Vector Stores)                 │
├─────────────────────────────────────────────────────────────┤
│                   Validation Layer                          │
│    (Extraction → Chunk → Domain Rules → KB Health)          │
├─────────────────────────────────────────────────────────────┤
│                   Acquisition Layer                         │
│         (Crawler, PDF, API, Database)                       │
├─────────────────────────────────────────────────────────────┤
│                      Core Layer                             │
│        (Types, Config, Registry, Logging)                   │
└─────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/gweta.git
cd gweta

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=gweta

# Type checking
mypy src/gweta

# Linting
ruff check src/gweta
ruff format src/gweta
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
