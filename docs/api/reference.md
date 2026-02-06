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

## Intelligence Layer

### SystemIntent

Defines what your RAG system is meant to do.

```python
from gweta.intelligence import SystemIntent

# Create programmatically
intent = SystemIntent(
    name="My Knowledge Base",
    description="Answers questions about Zimbabwe business",
    target_users=["graduates", "entrepreneurs"],
    core_questions=[
        "How do I register a business in Zimbabwe?",
        "What are ZIMRA tax requirements?",
    ],
    relevant_topics=["Zimbabwe business", "ZIMRA", "EcoCash"],
    irrelevant_topics=["US regulations", "cryptocurrency"],
)

# Load from YAML
intent = SystemIntent.from_yaml("intents/my_system.yaml")

# Check if topic is irrelevant
intent.is_irrelevant_topic("Invest in Bitcoin now!")  # True
```

**Attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | System name |
| `description` | `str` | What the system does |
| `target_users` | `list[str]` | Who uses the system |
| `core_questions` | `list[str]` | Questions it should answer well |
| `relevant_topics` | `list[str]` | Topics to include |
| `irrelevant_topics` | `list[str]` | Topics to reject |
| `min_relevance_score` | `float` | Accept threshold (default: 0.6) |
| `review_threshold` | `float` | Review threshold (default: 0.4) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `from_yaml(path)` | `SystemIntent` | Load from YAML file |
| `from_dict(data)` | `SystemIntent` | Load from dictionary |
| `to_yaml()` | `str` | Export as YAML |
| `to_dict()` | `dict` | Export as dictionary |
| `is_irrelevant_topic(text)` | `bool` | Check if text contains irrelevant topics |

### RelevanceFilter

Scores and filters chunks by semantic similarity to intent.

```python
from gweta.intelligence import RelevanceFilter

filter = RelevanceFilter(intent=intent)

# Filter single chunk
result = filter.filter(chunk)
print(f"Score: {result.relevance_score:.2f}")
print(f"Decision: {result.decision}")  # ACCEPT, REVIEW, or REJECT

# Filter batch
report = filter.filter_batch(chunks)
print(f"Accepted: {report.accepted_count}")
print(f"Rejected: {report.rejected_count}")

# Get accepted chunks with metadata
accepted = report.accepted()
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `filter(chunk)` | `RelevanceResult` | Filter single chunk |
| `filter_batch(chunks)` | `RelevanceReport` | Filter multiple chunks |
| `score(chunk)` | `float` | Get relevance score only |

### RelevanceResult

Result of filtering a single chunk.

| Attribute | Type | Description |
|-----------|------|-------------|
| `chunk` | `Chunk` | The chunk that was filtered |
| `relevance_score` | `float` | Relevance score (0.0 - 1.0) |
| `decision` | `RelevanceDecision` | ACCEPT, REVIEW, or REJECT |
| `matched_topics` | `list[str]` | Topics found in chunk |
| `rejection_reason` | `str \| None` | Why it was rejected |
| `accepted` | `bool` | True if decision is ACCEPT |
| `needs_review` | `bool` | True if decision is REVIEW |
| `rejected` | `bool` | True if decision is REJECT |

### RelevanceReport

Report from filtering multiple chunks.

| Attribute | Type | Description |
|-----------|------|-------------|
| `results` | `list[RelevanceResult]` | All results |
| `total_chunks` | `int` | Total processed |
| `accepted_count` | `int` | Number accepted |
| `review_count` | `int` | Number needing review |
| `rejected_count` | `int` | Number rejected |
| `acceptance_rate` | `float` | Accepted / total |
| `rejection_rate` | `float` | Rejected / total |
| `avg_relevance_score` | `float` | Average score |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `accepted()` | `list[Chunk]` | Get accepted chunks with metadata |
| `for_review()` | `list[Chunk]` | Get chunks needing review |

### EmbeddingEngine

Wrapper for sentence-transformers.

```python
from gweta.intelligence import EmbeddingEngine

# Default model: all-MiniLM-L6-v2
engine = EmbeddingEngine()

# Custom model
engine = EmbeddingEngine(model_name="all-mpnet-base-v2")

# Embed text
vector = engine.embed("Zimbabwe business registration")

# Batch embed
vectors = engine.embed_batch(["text1", "text2"])

# Compute similarity
similarity = engine.similarity(vector1, vector2)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `embed(text)` | `np.ndarray` | Embed single text |
| `embed_batch(texts)` | `np.ndarray` | Embed multiple texts |
| `similarity(v1, v2)` | `float` | Cosine similarity |
| `similarity_to_reference(vectors, ref)` | `np.ndarray` | Batch similarity |

### Pipeline

Unified API for intent-aware ingestion.

```python
from gweta.intelligence import Pipeline

# With store - full pipeline
pipeline = Pipeline(intent=intent, store=store)
result = await pipeline.ingest(chunks)

# Without store - filter only
pipeline = Pipeline(intent=intent, store=None)
report = pipeline.filter_only(chunks)

# Score single chunk
scores = pipeline.score_chunk(chunk)
```

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| `ingest(chunks)` | `PipelineResult` | Filter and ingest to store |
| `filter_only(chunks)` | `RelevanceReport` | Filter without storing |
| `score_chunk(chunk)` | `dict` | Get quality and relevance scores |

### PipelineResult

Result from pipeline ingestion.

| Attribute | Type | Description |
|-----------|------|-------------|
| `ingested` | `int` | Chunks successfully stored |
| `rejected_count` | `int` | Chunks rejected |
| `review_count` | `int` | Chunks needing review |
| `acceptance_rate` | `float` | Ingested / total |
| `relevance_report` | `RelevanceReport` | Full relevance report |

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
