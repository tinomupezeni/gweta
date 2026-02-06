# API Reference

Complete API documentation for Gweta.

## Core Types

### Chunk

The universal chunk representation.

```python
from gweta import Chunk

chunk = Chunk(
    id="chunk-001",              # Optional unique ID
    text="Content here...",      # Required text content
    source="document.pdf",       # Required source identifier
    metadata={"page": 1},        # Optional metadata dict
    quality_score=0.85,          # Optional quality score (0-1)
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str \| None` | Unique identifier |
| `text` | `str` | Chunk content |
| `source` | `str` | Source identifier |
| `metadata` | `dict[str, Any]` | Arbitrary metadata |
| `quality_score` | `float \| None` | Quality score (0.0 - 1.0) |

### QualityIssue

Represents a single quality problem.

```python
from gweta import QualityIssue

issue = QualityIssue(
    code="LOW_DENSITY",
    severity="warning",
    message="Information density below threshold",
    location="paragraph 2",
)
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `code` | `str` | Issue code (e.g., "LOW_DENSITY") |
| `severity` | `Literal["error", "warning", "info"]` | Severity level |
| `message` | `str` | Human-readable message |
| `location` | `str \| None` | Location in chunk |

---

## Validation

### ChunkValidator

Validates chunks for quality.

```python
from gweta import ChunkValidator

validator = ChunkValidator(
    min_length=50,
    required_metadata=["source", "date"],
)

# Single chunk
result = validator.validate(chunk)

# Batch validation
report = validator.validate_batch(chunks)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `validate(chunk)` | `ChunkResult` | Validate single chunk |
| `validate_batch(chunks)` | `QualityReport` | Validate multiple chunks |

### DomainRuleEngine

YAML-based domain validation rules.

```python
from gweta.validate.rules import DomainRuleEngine

# Load from YAML
engine = DomainRuleEngine.from_yaml("rules/domain.yaml")

# Or create programmatically
engine = DomainRuleEngine(rules=[...], known_facts=[...])

# Validate chunk
result = engine.validate_chunk(chunk)

# Validate AI response against known facts
result = engine.validate_response(response_text)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_yaml(path)` | `DomainRuleEngine` | Load from YAML file |
| `from_dict(data)` | `DomainRuleEngine` | Load from dictionary |
| `validate_chunk(chunk)` | `RuleValidationResult` | Validate chunk |
| `validate_response(text)` | `RuleValidationResult` | Validate AI response |
| `add_rule(rule)` | `None` | Add rule dynamically |
| `add_fact(fact)` | `None` | Add known fact |
| `to_yaml()` | `str` | Export as YAML |

### GoldenDatasetRunner

Test retrieval quality with golden Q&A pairs.

```python
from gweta.validate.golden import GoldenDatasetRunner

runner = GoldenDatasetRunner(
    store=my_store,
    dataset_path="golden/test.json",
)

# Run tests
report = await runner.run(k=5)

# Export results
junit_xml = runner.to_junit_xml(report)
json_output = runner.to_json(report)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `load_dataset(path)` | `list[GoldenPair]` | Load from JSON |
| `run(k, threshold)` | `GoldenTestReport` | Run all tests |
| `to_junit_xml(report)` | `str` | Export as JUnit XML |
| `to_json(report)` | `str` | Export as JSON |

---

## Acquisition

### GwetaCrawler

Web crawling with quality validation.

```python
from gweta.acquire import GwetaCrawler

crawler = GwetaCrawler()

result = await crawler.crawl(
    url="https://example.com",
    depth=2,
    follow_pdfs=True,
    allowed_domains=["example.com"],
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `crawl(url, **kwargs)` | `CrawlResult` | Async crawl |
| `crawl_sync(url, **kwargs)` | `CrawlResult` | Sync wrapper |

### PDFExtractor

PDF text and table extraction.

```python
from gweta.acquire import PDFExtractor

extractor = PDFExtractor()

result = await extractor.extract(
    source="document.pdf",
    extract_tables=True,
)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `extract(source, **kwargs)` | `PDFExtractionResult` | Async extract |
| `extract_sync(source, **kwargs)` | `PDFExtractionResult` | Sync wrapper |

### DatabaseSource

SQL database connector.

```python
from gweta.acquire import DatabaseSource

async with DatabaseSource(dsn="postgresql://...") as db:
    result = await db.query("SELECT * FROM docs")
    chunks = await db.extract_and_validate(
        query="SELECT content FROM articles",
        text_column="content",
    )
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `connect()` | `None` | Open connection |
| `disconnect()` | `None` | Close connection |
| `query(sql, params)` | `QueryResult` | Execute query |
| `extract_and_validate(...)` | `list[Chunk]` | Extract chunks |

---

## Vector Stores

All stores implement `BaseStore`:

```python
class BaseStore(ABC):
    @property
    def collection_name(self) -> str: ...
    async def add(self, chunks: list[Chunk]) -> AddResult: ...
    async def query(self, query: str, n_results: int) -> list[Chunk]: ...
    async def delete(self, chunk_ids: list[str]) -> int: ...
    async def get_all(self) -> list[Chunk]: ...
    def get_stats(self) -> StoreStats: ...
```

### ChromaStore

```python
from gweta import ChromaStore

# Default: Uses SentenceTransformer "all-MiniLM-L6-v2" embeddings
store = ChromaStore(collection_name="my_docs")

# With persistence
store = ChromaStore(
    collection_name="my_docs",
    persist_directory="./chroma_data",
)

# With custom embedding function
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
store = ChromaStore(
    collection_name="my_docs",
    embedding_function=embed_fn,
)

# With OpenAI embeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
embed_fn = OpenAIEmbeddingFunction(api_key="sk-...")
store = ChromaStore(collection_name="my_docs", embedding_function=embed_fn)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection_name` | `str` | required | Name of the collection |
| `client` | `chromadb.Client` | `None` | Existing ChromaDB client |
| `embedding_function` | `EmbeddingFunction` | `SentenceTransformer` | Custom embedding function |
| `persist_directory` | `str` | `None` | Directory for persistence |
| `use_default_embeddings` | `bool` | `True` | Use default embeddings if none provided |

### QdrantStore

```python
from gweta.ingest.stores import QdrantStore

store = QdrantStore(
    collection_name="my_docs",
    url="http://localhost:6333",
    api_key="optional-key",
)
```

### PineconeStore

```python
from gweta.ingest.stores import PineconeStore

store = PineconeStore(
    index_name="my-index",
    api_key="your-api-key",
    namespace="optional-namespace",
)
```

### WeaviateStore

```python
from gweta.ingest.stores import WeaviateStore

store = WeaviateStore(
    class_name="Document",
    url="http://localhost:8080",
)
```

---

## MCP Server

### Starting the Server

```python
from gweta.mcp import create_server, run_stdio, run_http

# stdio transport (for Claude Desktop)
run_stdio()

# HTTP transport
run_http(port=8080)
```

### Available Tools

| Tool | Description |
|------|-------------|
| `crawl_and_ingest` | Crawl website and load to vector store |
| `validate_chunks` | Validate chunks without loading |
| `check_health` | Get KB health report |
| `crawl_site` | Crawl without loading |
| `ingest_from_database` | DB to vector store |
| `query_database` | Execute read-only query |
| `extract_pdf` | Extract PDF content |
| `fetch_api` | Fetch from REST API |
| `fetch_sitemap` | Parse sitemap |
| `fetch_rss` | Parse RSS/Atom feed |

### Available Resources

| URI | Description |
|-----|-------------|
| `gweta://sources` | List registered sources |
| `gweta://quality/{collection}` | Quality report |
| `gweta://rules/{domain}` | Domain rules |
| `gweta://config` | Current config |

---

## CLI Commands

```bash
# Validate chunks
gweta validate <path> [--threshold 0.6] [--output report.json]

# Crawl website
gweta crawl <url> [--depth 2] [--output chunks.json]

# Check KB health
gweta health <collection> [--store chroma] [--golden golden.json]

# Ingest data
gweta ingest <source> <target> [--collection default]

# Start MCP server
gweta serve [--transport stdio|http] [--port 8080]
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GWETA_MIN_QUALITY_SCORE` | `0.6` | Minimum quality threshold |
| `GWETA_MIN_DENSITY_SCORE` | `0.3` | Minimum density threshold |
| `GWETA_LOG_LEVEL` | `INFO` | Logging level |

### GwetaSettings

```python
from gweta import GwetaSettings

settings = GwetaSettings(
    min_quality_score=0.7,
    min_density_score=0.4,
    log_level="DEBUG",
)
```
