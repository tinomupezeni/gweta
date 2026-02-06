# Architecture Overview

Gweta is designed as a modular pipeline with four validation layers.

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                             │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────┤
│    Web      │    PDF      │   Database  │    API      │  Custom │
│  Crawler    │  Extractor  │  Connector  │   Client    │ Source  │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴────┬────┘
       │             │             │             │           │
       └─────────────┴─────────────┴─────────────┴───────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │     VALIDATION PIPELINE      │
                    ├──────────────────────────────┤
                    │  Layer 1: Extraction Quality │
                    │  Layer 2: Chunk Quality      │
                    │  Layer 3: Domain Rules       │
                    │  Layer 4: KB Health          │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │       VECTOR STORES          │
                    ├──────────────────────────────┤
                    │  ChromaDB │ Qdrant │ Pinecone │
                    └──────────────────────────────┘
```

## Module Structure

```
gweta/
├── core/           # Types, config, logging
├── acquire/        # Data acquisition
├── validate/       # Validation layers
├── ingest/         # Chunking and stores
├── mcp/            # MCP server
└── cli/            # Command-line interface
```

## Data Flow

### 1. Acquisition

Raw data enters through acquisition modules:

```python
# Web crawling
crawler = GwetaCrawler()
result = await crawler.crawl(url, depth=2)

# PDF extraction
extractor = PDFExtractor()
result = await extractor.extract("doc.pdf")

# Database query
async with DatabaseSource(dsn) as db:
    chunks = await db.extract_and_validate(query, text_column="content")
```

### 2. Validation

Data passes through validation layers:

```python
# Layer 1: Extraction quality
extraction_validator = ExtractionValidator()
extraction_result = extraction_validator.validate(raw_text)

# Layer 2: Chunk quality
chunk_validator = ChunkValidator()
report = chunk_validator.validate_batch(chunks)

# Layer 3: Domain rules
rule_engine = DomainRuleEngine.from_yaml("rules.yaml")
rule_result = rule_engine.validate_chunk(chunk)

# Layer 4: KB health (post-ingestion)
health_checker = HealthChecker(store)
health_report = await health_checker.full_health_check()
```

### 3. Ingestion

Validated chunks go to vector stores:

```python
store = ChromaStore("my_collection")
result = await store.add(validated_chunks)
```

## The Chunk Object

The `Chunk` is the universal data unit:

```python
@dataclass
class Chunk:
    id: str | None              # Unique identifier
    text: str                   # Content
    source: str                 # Source identifier
    metadata: dict[str, Any]    # Arbitrary metadata
    quality_score: float | None # 0.0 - 1.0
    quality_details: QualityDetails | None
```

## Validation Layers

### Layer 1: Extraction Quality

Validates raw extracted text:

- Text length (not empty, not truncated)
- Encoding (valid UTF-8)
- Gibberish detection (OCR failures)
- Language consistency
- Boilerplate ratio

### Layer 2: Chunk Quality

Validates individual chunks:

- Coherence (standalone comprehension)
- Information density
- Metadata completeness
- Boundary quality
- Duplicate detection

### Layer 3: Domain Rules

YAML-based validation rules:

- Regex patterns
- Numerical ranges
- Required fields
- Date freshness
- Known fact cross-referencing

### Layer 4: KB Health

Monitors knowledge base over time:

- Staleness detection
- Duplicate identification
- Coverage gap analysis
- Quality drift tracking
- Golden dataset testing

## Configuration

Gweta uses environment variables and YAML config:

```yaml
# gweta.yaml
quality:
  min_score: 0.6
  min_density: 0.3
  max_duplicate_similarity: 0.92

chunking:
  strategy: recursive
  size: 500
  overlap: 50

logging:
  level: INFO
```

## Async-First Design

All I/O operations are async:

```python
# Async by default
result = await crawler.crawl(url)
chunks = await store.query(query)

# Sync wrappers available
result = crawler.crawl_sync(url)
```

## Extensibility

### Custom Validators

```python
class MyValidator:
    def validate(self, chunk: Chunk) -> ChunkResult:
        # Custom validation logic
        pass
```

### Custom Stores

```python
class MyStore(BaseStore):
    async def add(self, chunks: list[Chunk]) -> AddResult:
        # Custom store logic
        pass
```

### Custom Acquisition

```python
class MySource:
    async def fetch(self) -> list[Chunk]:
        # Custom acquisition logic
        pass
```
