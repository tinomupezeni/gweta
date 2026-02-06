# Gweta Implementation Plan

## Complete Build Roadmap for the RAG Data Pipeline Framework

**Version:** 1.0
**Status:** Ready for Implementation
**Target:** `pip install gweta`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Technology Stack](#2-technology-stack)
3. [Repository Structure](#3-repository-structure)
4. [Phase 1: Foundation](#4-phase-1-foundation-weeks-1-3)
5. [Phase 2: Acquisition Layer](#5-phase-2-acquisition-layer-weeks-4-6)
6. [Phase 3: MCP Server](#6-phase-3-mcp-server-weeks-7-8)
7. [Phase 4: Intelligence Layer](#7-phase-4-intelligence-layer-weeks-9-12)
8. [Phase 5: Polish & Launch](#8-phase-5-polish--launch-week-13)
9. [Testing Strategy](#9-testing-strategy)
10. [Dependencies](#10-dependencies)
11. [Configuration Schema](#11-configuration-schema)
12. [API Reference](#12-api-reference)

---

## 1. Project Overview

### 1.1 Mission Statement

Gweta is the **missing middleware** for RAG pipelines — a quality-aware framework that handles data acquisition, validation, and ingestion as a single pipeline, exposed over MCP for AI agent integration.

### 1.2 Core Value Proposition

```
BEFORE: Document → Parser → Chunks → Vector DB → Bad answers → Debug → Repeat
AFTER:  Document → Gweta (acquire + validate + ingest) → Quality chunks → Good answers
```

### 1.3 Design Principles

| Principle | Implementation |
|-----------|----------------|
| Parser-agnostic | Works with Unstructured, pdfplumber, PyPDF, custom parsers |
| Chunker-agnostic | Works with Chonkie, LangChain, LlamaIndex, raw text |
| Store-agnostic | Loads to Chroma, Qdrant, Pinecone, Weaviate |
| Framework-agnostic | Plugs into any RAG pipeline |
| Lightweight core | Heuristics by default, optional LLM validation |
| Declarative rules | YAML-based domain rules inspired by Great Expectations |

---

## 2. Technology Stack

### 2.1 Core Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Runtime | Python 3.10+ | Minimum supported version |
| Type hints | Pydantic v2 | Data validation and settings |
| Async | asyncio, anyio | Async-first internals |
| HTTP | httpx | Async HTTP client |
| CLI | Typer | Command-line interface |
| Config | PyYAML, tomli | Configuration parsing |

### 2.2 Acquisition Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Web crawling | Crawl4AI | JavaScript-rendered crawling |
| PDF extraction | pdfplumber | PDF text and table extraction |
| Database | SQLAlchemy 2.0 | Database connectivity |
| HTML parsing | BeautifulSoup4 | HTML cleanup |

### 2.3 Validation Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Duplicate detection | datasketch | MinHash LSH |
| Language detection | langdetect | Language identification |
| Text stats | textstat | Readability metrics |
| Encoding | chardet | Encoding detection |

### 2.4 Ingestion Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Chunking (optional) | Chonkie | Advanced chunking strategies |
| Chroma | chromadb | Vector store |
| Qdrant | qdrant-client | Vector store |
| Pinecone | pinecone-client | Vector store |

### 2.5 MCP Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| MCP Server | mcp[server] | FastMCP server SDK |

---

## 3. Repository Structure

```
gweta/
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                    # Test on PR
│   │   ├── release.yml               # PyPI publish
│   │   └── docs.yml                  # Docs deployment
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.md
│       └── feature_request.md
│
├── docs/
│   ├── index.md                      # Landing page
│   ├── getting-started.md            # Quick start guide
│   ├── concepts/
│   │   ├── architecture.md
│   │   ├── quality-scoring.md
│   │   └── domain-rules.md
│   ├── guides/
│   │   ├── crawling.md
│   │   ├── validation.md
│   │   ├── ingestion.md
│   │   └── mcp-integration.md
│   └── api/
│       └── reference.md
│
├── examples/
│   ├── basic_crawl.py
│   ├── validate_chunks.py
│   ├── full_pipeline.py
│   ├── mcp_client.py
│   └── langchain_integration.py
│
├── src/
│   └── gweta/
│       ├── __init__.py               # Public API exports
│       ├── _version.py               # Version info
│       ├── py.typed                  # PEP 561 marker
│       │
│       ├── core/
│       │   ├── __init__.py
│       │   ├── types.py              # Chunk, Source, QualityReport
│       │   ├── config.py             # Settings management
│       │   ├── registry.py           # Source authority registry
│       │   ├── exceptions.py         # Custom exceptions
│       │   └── logging.py            # Structured logging
│       │
│       ├── acquire/
│       │   ├── __init__.py
│       │   ├── crawler.py            # Crawl4AI wrapper
│       │   ├── pdf.py                # PDF extraction
│       │   ├── api.py                # REST/GraphQL client
│       │   ├── database.py           # SQLAlchemy connector
│       │   └── fetchers/
│       │       ├── __init__.py
│       │       ├── url.py            # Single URL fetch
│       │       ├── sitemap.py        # Sitemap crawler
│       │       └── rss.py            # RSS/Atom feeds
│       │
│       ├── validate/
│       │   ├── __init__.py
│       │   ├── extraction.py         # Layer 1: Extraction quality
│       │   ├── chunks.py             # Layer 2: Chunk validation
│       │   ├── rules.py              # Layer 3: Domain rules
│       │   ├── health.py             # Layer 4: KB health
│       │   ├── golden.py             # Golden dataset testing
│       │   └── detectors/
│       │       ├── __init__.py
│       │       ├── gibberish.py      # Garbled text detection
│       │       ├── duplicates.py     # Near-duplicate detection
│       │       ├── density.py        # Information density
│       │       ├── coherence.py      # Chunk coherence
│       │       └── staleness.py      # Freshness tracking
│       │
│       ├── ingest/
│       │   ├── __init__.py
│       │   ├── chunkers.py           # Chunking strategies
│       │   ├── pipeline.py           # Full ingestion pipeline
│       │   └── stores/
│       │       ├── __init__.py
│       │       ├── base.py           # Abstract store interface
│       │       ├── chroma.py
│       │       ├── qdrant.py
│       │       ├── pinecone.py
│       │       └── weaviate.py
│       │
│       ├── mcp/
│       │   ├── __init__.py
│       │   ├── server.py             # FastMCP server
│       │   ├── tools.py              # MCP tools
│       │   ├── resources.py          # MCP resources
│       │   └── prompts.py            # MCP prompts
│       │
│       ├── adapters/
│       │   ├── __init__.py
│       │   ├── langchain.py
│       │   ├── llamaindex.py
│       │   ├── chonkie.py
│       │   └── unstructured.py
│       │
│       └── cli/
│           ├── __init__.py
│           └── main.py               # Typer CLI app
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                   # Pytest fixtures
│   ├── unit/
│   │   ├── test_types.py
│   │   ├── test_extraction.py
│   │   ├── test_chunks.py
│   │   ├── test_duplicates.py
│   │   └── test_rules.py
│   ├── integration/
│   │   ├── test_crawler.py
│   │   ├── test_database.py
│   │   ├── test_chroma.py
│   │   └── test_mcp.py
│   └── fixtures/
│       ├── sample_pdfs/
│       ├── sample_html/
│       └── golden_datasets/
│
├── pyproject.toml                    # Project config
├── README.md                         # Main documentation
├── LICENSE                           # MIT License
├── CHANGELOG.md                      # Version history
└── CONTRIBUTING.md                   # Contribution guide
```

---

## 4. Phase 1: Foundation (Weeks 1-3)

> **Goal:** Core types, extraction validation, chunk validation, Chroma adapter, CLI

### 4.1 Week 1: Core Types & Project Setup

#### 4.1.1 Repository Initialization

```bash
# Tasks
- [ ] Initialize git repository
- [ ] Create pyproject.toml with all extras defined
- [ ] Set up src/ layout
- [ ] Configure pytest, ruff, mypy
- [ ] Set up GitHub Actions CI
- [ ] Create README.md with project description
```

#### 4.1.2 Core Types (`src/gweta/core/types.py`)

```python
# Data models to implement

@dataclass
class Chunk:
    """Universal chunk representation."""
    id: str                           # Unique identifier
    text: str                         # Chunk content
    metadata: dict[str, Any]          # Arbitrary metadata
    source: str                       # Source identifier
    quality_score: float | None       # 0.0 - 1.0
    quality_details: QualityDetails | None
    created_at: datetime

@dataclass
class QualityDetails:
    """Multi-dimensional quality breakdown."""
    extraction_score: float           # Layer 1 score
    coherence_score: float            # Layer 2 score
    density_score: float              # Information density
    duplicate_score: float            # 1.0 = unique, 0.0 = exact duplicate
    issues: list[QualityIssue]

@dataclass
class QualityIssue:
    """Single quality problem."""
    code: str                         # e.g., "LOW_DENSITY"
    severity: Literal["error", "warning", "info"]
    message: str
    location: str | None              # Where in the chunk

@dataclass
class Source:
    """Data source with authority tracking."""
    id: str
    name: str
    url: str | None
    authority_tier: int               # 1-5 (5 = primary legislation)
    freshness_days: int               # Max age before stale
    last_crawled: datetime | None

@dataclass
class QualityReport:
    """Validation results for a batch."""
    total_chunks: int
    passed: int
    failed: int
    warnings: int
    avg_quality_score: float
    issues_by_type: dict[str, int]
    chunks: list[ChunkResult]

@dataclass
class ChunkResult:
    """Single chunk validation result."""
    chunk: Chunk
    passed: bool
    quality_score: float
    issues: list[QualityIssue]
```

#### 4.1.3 Configuration (`src/gweta/core/config.py`)

```python
# Settings model to implement

class GwetaSettings(BaseSettings):
    """Global configuration."""

    # Quality thresholds
    min_quality_score: float = 0.6
    min_density_score: float = 0.3
    max_duplicate_similarity: float = 0.92

    # Extraction settings
    min_text_length: int = 50
    max_gibberish_ratio: float = 0.3

    # Chunking defaults
    default_chunk_size: int = 500
    default_chunk_overlap: int = 50

    # Authority registry path
    authority_registry: Path | None = None

    # Logging
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_prefix="GWETA_",
        env_file=".env",
    )
```

### 4.2 Week 2: Validation Layer (Layers 1 & 2)

#### 4.2.1 Extraction Validator (`src/gweta/validate/extraction.py`)

```python
# Functions to implement

class ExtractionValidator:
    """Layer 1: Validates raw extracted text before chunking."""

    def __init__(self, config: GwetaSettings | None = None): ...

    def validate(
        self,
        text: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult: ...

    async def async_validate(
        self,
        text: str,
        source_metadata: dict[str, Any] | None = None,
    ) -> ExtractionResult: ...

# Checks to implement:
# 1. Text length validation (not empty, not truncated)
# 2. Encoding detection (is it valid UTF-8?)
# 3. Gibberish detection (OCR failures, garbled text)
# 4. Language detection and consistency
# 5. Boilerplate ratio (headers, footers, nav)
# 6. Table extraction fidelity (if tables present)
```

#### 4.2.2 Gibberish Detector (`src/gweta/validate/detectors/gibberish.py`)

```python
# Heuristics to implement

def detect_gibberish(text: str) -> GibberishResult:
    """
    Detect OCR failures and encoding corruption.

    Checks:
    1. Character entropy (random chars = high entropy)
    2. Consecutive consonant ratio
    3. Dictionary word ratio
    4. Special character density
    5. Repeated character sequences
    """
    ...

def estimate_ocr_confidence(text: str) -> float:
    """
    Estimate OCR quality from text characteristics.

    Signals:
    - Clean word boundaries
    - Proper punctuation
    - Recognizable sentence structure
    - Low special character ratio
    """
    ...
```

#### 4.2.3 Chunk Validator (`src/gweta/validate/chunks.py`)

```python
# Functions to implement

class ChunkValidator:
    """Layer 2: Validates chunks before vector store loading."""

    def __init__(
        self,
        config: GwetaSettings | None = None,
        required_metadata: list[str] | None = None,
    ): ...

    def validate(self, chunk: Chunk) -> ChunkResult: ...

    def validate_batch(
        self,
        chunks: list[Chunk],
        parallel: bool = True,
    ) -> QualityReport: ...

    async def async_validate_batch(
        self,
        chunks: list[Chunk],
    ) -> QualityReport: ...

# Checks to implement:
# 1. Coherence scoring (does chunk make sense standalone?)
# 2. Information density (signal vs noise)
# 3. Metadata completeness (required fields present?)
# 4. Boundary quality (proper sentence boundaries?)
# 5. Minimum/maximum length
```

#### 4.2.4 Duplicate Detector (`src/gweta/validate/detectors/duplicates.py`)

```python
# Functions to implement

class DuplicateDetector:
    """Near-duplicate detection using MinHash LSH."""

    def __init__(
        self,
        threshold: float = 0.92,
        num_perm: int = 128,
    ): ...

    def add(self, chunk_id: str, text: str) -> None:
        """Add chunk to the index."""
        ...

    def find_duplicates(self, text: str) -> list[DuplicateMatch]:
        """Find near-duplicates of given text."""
        ...

    def get_duplicate_groups(self) -> list[list[str]]:
        """Get all duplicate groups in the index."""
        ...

# Use datasketch library for MinHash implementation
```

#### 4.2.5 Information Density (`src/gweta/validate/detectors/density.py`)

```python
# Functions to implement

def calculate_density(text: str) -> DensityResult:
    """
    Calculate information density score.

    Metrics:
    1. Unique word ratio (vocabulary richness)
    2. Stop word ratio (low = more content words)
    3. Average word length (longer = more specific)
    4. Sentence complexity (clauses, conjunctions)
    5. Named entity density (if spaCy available)
    """
    ...

def calculate_coherence(text: str) -> float:
    """
    Score how well the chunk stands alone.

    Checks:
    - Complete sentences (starts capital, ends punctuation)
    - No dangling references ("it", "this" without antecedent)
    - Topic consistency (embedding similarity of sentences)
    """
    ...
```

### 4.3 Week 3: Ingestion Layer & CLI

#### 4.3.1 Built-in Chunker (`src/gweta/ingest/chunkers.py`)

```python
# Functions to implement

class RecursiveChunker:
    """Default chunker: recursive text splitting."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] | None = None,
    ): ...

    def chunk(self, text: str, metadata: dict | None = None) -> list[Chunk]:
        """Split text into chunks."""
        ...

# Optional Chonkie integration
def get_chonkie_chunker(strategy: str = "semantic") -> "ChonkieChunker":
    """Get Chonkie chunker if installed."""
    try:
        from chonkie import ...
    except ImportError:
        raise ImportError("Install gweta[chonkie] for advanced chunking")
```

#### 4.3.2 Chroma Store (`src/gweta/ingest/stores/chroma.py`)

```python
# Functions to implement

class ChromaStore(BaseStore):
    """Chroma vector store adapter."""

    def __init__(
        self,
        collection_name: str,
        client: chromadb.Client | None = None,
        embedding_function: Any | None = None,
    ): ...

    async def add(self, chunks: list[Chunk]) -> AddResult: ...

    async def query(
        self,
        query: str,
        n_results: int = 10,
    ) -> list[Chunk]: ...

    async def delete(self, chunk_ids: list[str]) -> None: ...

    async def get_all(self) -> list[Chunk]: ...

    def get_stats(self) -> StoreStats: ...
```

#### 4.3.3 CLI (`src/gweta/cli/main.py`)

```python
# Commands to implement

app = typer.Typer(name="gweta", help="RAG data quality framework")

@app.command()
def validate(
    path: Path,
    format: str = "json",
    threshold: float = 0.6,
    output: Path | None = None,
):
    """Validate chunks from a file or directory."""
    ...

@app.command()
def crawl(
    url: str,
    depth: int = 2,
    output: Path | None = None,
    validate: bool = True,
):
    """Crawl a URL and extract validated content."""
    ...

@app.command()
def ingest(
    source: str,
    target: str,
    collection: str = "default",
):
    """Ingest validated data into a vector store."""
    ...

@app.command()
def health(
    collection: str,
    store: str = "chroma",
    golden: Path | None = None,
):
    """Check health of a knowledge base."""
    ...

@app.command()
def serve(
    transport: str = "stdio",
    port: int = 8080,
):
    """Start the MCP server."""
    ...
```

#### 4.3.4 Public API (`src/gweta/__init__.py`)

```python
# Public exports

from gweta.core.types import (
    Chunk,
    Source,
    QualityReport,
    QualityDetails,
    QualityIssue,
)
from gweta.core.config import GwetaSettings
from gweta.validate.extraction import ExtractionValidator
from gweta.validate.chunks import ChunkValidator
from gweta.validate.rules import DomainRuleEngine
from gweta.validate.health import HealthChecker
from gweta.ingest.chunkers import RecursiveChunker
from gweta.ingest.stores.chroma import ChromaStore

__version__ = "0.1.0"

__all__ = [
    "Chunk",
    "Source",
    "QualityReport",
    "GwetaSettings",
    "ExtractionValidator",
    "ChunkValidator",
    "DomainRuleEngine",
    "HealthChecker",
    "RecursiveChunker",
    "ChromaStore",
]
```

---

## 5. Phase 2: Acquisition Layer (Weeks 4-6)

> **Goal:** Web crawler, PDF extractor, database connector, source registry

### 5.1 Week 4: Web Crawler

#### 5.1.1 Crawl4AI Wrapper (`src/gweta/acquire/crawler.py`)

```python
# Functions to implement

class GwetaCrawler:
    """Quality-aware web crawler wrapping Crawl4AI."""

    def __init__(
        self,
        config: GwetaSettings | None = None,
        authority_registry: Path | str | None = None,
    ): ...

    async def crawl(
        self,
        url: str,
        depth: int = 2,
        follow_pdfs: bool = True,
        allowed_domains: list[str] | None = None,
        rules: str | None = None,
    ) -> CrawlResult: ...

    def crawl_sync(self, url: str, **kwargs) -> CrawlResult:
        """Sync wrapper for crawl()."""
        return asyncio.run(self.crawl(url, **kwargs))

@dataclass
class CrawlResult:
    """Results from a crawl operation."""
    url: str
    pages_crawled: int
    pages_passed: int
    pages_failed: int
    quality_score: float
    chunks: list[Chunk]
    rejected_chunks: list[Chunk]
    errors: list[CrawlError]
    duration_seconds: float

    def load_to(self, store: BaseStore) -> None:
        """Load validated chunks to a store."""
        ...
```

#### 5.1.2 Pre-crawl Validation

```python
# Functions to implement

class SourceAuthorityRegistry:
    """Registry of trusted sources with authority tiers."""

    def __init__(self, config_path: Path | None = None): ...

    @classmethod
    def from_yaml(cls, path: Path) -> "SourceAuthorityRegistry": ...

    def is_allowed(self, url: str) -> bool:
        """Check if URL is from allowed domain."""
        ...

    def get_authority(self, url: str) -> int:
        """Get authority tier for URL (1-5)."""
        ...

    def get_freshness_window(self, url: str) -> timedelta:
        """Get freshness window for source."""
        ...

# Example YAML format:
# sources:
#   - domain: "zimra.co.zw"
#     authority: 5
#     freshness_days: 30
#     name: "ZIMRA Official"
#   - domain: "*.gov.zw"
#     authority: 4
#     freshness_days: 90
```

#### 5.1.3 Post-crawl Quality Scoring

```python
# Functions to implement

class CrawlQualityScorer:
    """Score quality of crawled pages."""

    def score_page(
        self,
        url: str,
        content: str,
        metadata: dict,
    ) -> PageQualityScore: ...

@dataclass
class PageQualityScore:
    url: str
    extraction_score: float
    content_completeness: float
    javascript_rendered: bool
    has_main_content: bool
    boilerplate_ratio: float
    issues: list[QualityIssue]
    passed: bool
```

### 5.2 Week 5: PDF & API Extraction

#### 5.2.1 PDF Extractor (`src/gweta/acquire/pdf.py`)

```python
# Functions to implement

class PDFExtractor:
    """Extract and validate PDF content."""

    def __init__(self, config: GwetaSettings | None = None): ...

    async def extract(
        self,
        source: Path | str | bytes,
        extract_tables: bool = True,
        extract_images: bool = False,
    ) -> PDFExtractionResult: ...

    def extract_sync(self, source: Path | str | bytes, **kwargs) -> PDFExtractionResult:
        return asyncio.run(self.extract(source, **kwargs))

@dataclass
class PDFExtractionResult:
    pages: list[PDFPage]
    tables: list[PDFTable]
    metadata: dict
    quality_score: float
    issues: list[QualityIssue]

@dataclass
class PDFPage:
    number: int
    text: str
    quality_score: float
    is_scanned: bool
    ocr_confidence: float | None

@dataclass
class PDFTable:
    page: int
    data: list[list[str]]
    quality_score: float
    headers: list[str] | None
```

#### 5.2.2 API Client (`src/gweta/acquire/api.py`)

```python
# Functions to implement

class APIClient:
    """Fetch and validate data from REST/GraphQL APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        headers: dict | None = None,
        timeout: float = 30.0,
    ): ...

    async def fetch(
        self,
        url: str,
        method: str = "GET",
        params: dict | None = None,
        json: dict | None = None,
    ) -> APIResponse: ...

    async def fetch_paginated(
        self,
        url: str,
        page_param: str = "page",
        max_pages: int = 100,
    ) -> list[APIResponse]: ...

@dataclass
class APIResponse:
    url: str
    status_code: int
    data: Any
    headers: dict
    quality_score: float
    issues: list[QualityIssue]
```

### 5.3 Week 6: Database Connector

#### 5.3.1 Database Source (`src/gweta/acquire/database.py`)

```python
# Functions to implement

class DatabaseSource:
    """Extract data from SQL databases."""

    def __init__(
        self,
        dsn: str,
        read_only: bool = True,
        timeout: float = 30.0,
        max_rows: int = 10000,
    ): ...

    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    async def query(
        self,
        sql: str,
        params: dict | None = None,
    ) -> QueryResult: ...

    async def extract_and_validate(
        self,
        query: str,
        text_column: str,
        metadata_columns: list[str] | None = None,
    ) -> list[Chunk]: ...

    async def ingest(
        self,
        query: str,
        text_column: str,
        target: BaseStore,
        chunk_strategy: str = "recursive",
    ) -> IngestResult: ...

    # Context manager support
    async def __aenter__(self) -> "DatabaseSource": ...
    async def __aexit__(self, *args) -> None: ...

@dataclass
class QueryResult:
    rows: list[dict]
    columns: list[str]
    row_count: int
    execution_time: float
```

#### 5.3.2 Safety Measures

```python
# Security implementations

class QuerySanitizer:
    """Ensure queries are read-only and safe."""

    FORBIDDEN_KEYWORDS = [
        "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
        "CREATE", "TRUNCATE", "EXEC", "EXECUTE",
    ]

    def validate(self, query: str) -> None:
        """Raise if query contains forbidden operations."""
        ...

    def is_read_only(self, query: str) -> bool:
        """Check if query is read-only."""
        ...
```

---

## 6. Phase 3: MCP Server (Weeks 7-8)

> **Goal:** FastMCP server with tools, resources, prompts; stdio + HTTP transports

### 6.1 Week 7: MCP Tools

#### 6.1.1 Server Setup (`src/gweta/mcp/server.py`)

```python
# Server implementation

from mcp.server.fastmcp import FastMCP

gweta_mcp = FastMCP(
    name="gweta",
    version="0.1.0",
    description="RAG data quality and ingestion framework",
)

def create_server() -> FastMCP:
    """Create and configure the MCP server."""
    from gweta.mcp.tools import register_tools
    from gweta.mcp.resources import register_resources
    from gweta.mcp.prompts import register_prompts

    register_tools(gweta_mcp)
    register_resources(gweta_mcp)
    register_prompts(gweta_mcp)

    return gweta_mcp

def run_stdio():
    """Run server with stdio transport."""
    server = create_server()
    server.run()

def run_http(port: int = 8080):
    """Run server with HTTP transport."""
    server = create_server()
    server.run(transport="http", port=port)
```

#### 6.1.2 MCP Tools (`src/gweta/mcp/tools.py`)

```python
# Tool implementations

def register_tools(mcp: FastMCP):

    @mcp.tool()
    async def crawl_and_ingest(
        url: str,
        depth: int = 2,
        target_collection: str = "default",
        authority_tier: int = 3,
        rules: str | None = None,
    ) -> dict:
        """
        Crawl a website, validate extracted content, and load
        quality chunks into the target vector database collection.

        Args:
            url: Starting URL to crawl
            depth: How many links deep to follow (1-5)
            target_collection: Name of the vector DB collection
            authority_tier: Source authority level (1=blog, 5=legislation)
            rules: Optional domain rule set name

        Returns:
            Quality report with crawl statistics
        """
        crawler = GwetaCrawler()
        result = await crawler.crawl(url, depth=depth)

        store = ChromaStore(collection_name=target_collection)
        await store.add(result.chunks)

        return {
            "pages_crawled": result.pages_crawled,
            "chunks_loaded": len(result.chunks),
            "chunks_rejected": len(result.rejected_chunks),
            "quality_score": result.quality_score,
        }

    @mcp.tool()
    async def ingest_from_database(
        dsn: str,
        query: str,
        target_collection: str = "default",
        text_column: str = "content",
        metadata_columns: list[str] | None = None,
    ) -> dict:
        """
        Read data from a database, validate it, and load into
        the knowledge base.

        Args:
            dsn: Database connection string
            query: SQL query to extract data
            target_collection: Target vector DB collection
            text_column: Column containing text to embed
            metadata_columns: Columns to preserve as metadata
        """
        ...

    @mcp.tool()
    async def validate_chunks(
        chunks: list[dict],
        rules: str | None = None,
    ) -> dict:
        """
        Validate a list of chunks without loading them.

        Args:
            chunks: List of chunk dicts with 'text' and 'metadata'
            rules: Optional domain rule set name

        Returns:
            Validation report with pass/fail per chunk
        """
        ...

    @mcp.tool()
    async def check_health(
        collection: str,
        golden_dataset: str | None = None,
    ) -> dict:
        """
        Check health of a knowledge base collection.

        Returns quality scores, stale sources, coverage gaps,
        and specific chunks that fail validation.
        """
        ...

    @mcp.tool()
    async def crawl_site(
        url: str,
        depth: int = 2,
        output_format: str = "markdown",
    ) -> dict:
        """
        Crawl a website and return validated content without
        loading it. Useful for preview/review.
        """
        ...

    @mcp.tool()
    async def query_database(
        dsn: str,
        query: str,
    ) -> dict:
        """
        Execute a read-only query against a database.
        Returns results as structured data.
        """
        ...
```

### 6.2 Week 8: Resources, Prompts & Transports

#### 6.2.1 MCP Resources (`src/gweta/mcp/resources.py`)

```python
# Resource implementations

def register_resources(mcp: FastMCP):

    @mcp.resource("gweta://sources")
    async def list_sources() -> str:
        """
        List all registered data sources with their authority tiers,
        last crawl dates, and freshness status.
        """
        registry = SourceAuthorityRegistry.get_default()
        sources = registry.get_all()

        return yaml.dump([s.to_dict() for s in sources])

    @mcp.resource("gweta://quality/{collection}")
    async def quality_report(collection: str) -> str:
        """
        Get the latest quality report for a collection.
        """
        store = ChromaStore(collection_name=collection)
        report = await HealthChecker(store).generate_report()

        return report.to_json()

    @mcp.resource("gweta://rules/{domain}")
    async def domain_rules(domain: str) -> str:
        """
        Get the validation rules for a specific domain.
        Returns YAML-formatted rule definitions.
        """
        engine = DomainRuleEngine.load(domain)
        return engine.to_yaml()

    @mcp.resource("gweta://config")
    async def current_config() -> str:
        """
        Get current Gweta configuration.
        """
        settings = GwetaSettings()
        return settings.model_dump_json(indent=2)
```

#### 6.2.2 MCP Prompts (`src/gweta/mcp/prompts.py`)

```python
# Prompt implementations

def register_prompts(mcp: FastMCP):

    @mcp.prompt()
    async def plan_ingestion(
        sources: str,
        target: str,
    ) -> str:
        """
        Generate an ingestion plan for the given sources and target.

        Analyzes source types, recommends crawl strategies, and
        estimates quality thresholds.
        """
        return f"""
You are planning a data ingestion job for Gweta.

**Sources to ingest:**
{sources}

**Target collection:** {target}

Please analyze the sources and create a plan covering:
1. Source type identification (web, PDF, database, API)
2. Recommended crawl depth for web sources
3. Authority tier assignment for each source
4. Suggested quality threshold
5. Estimated chunk count
6. Potential quality issues to watch for

Use the gweta tools to execute the plan once approved.
"""

    @mcp.prompt()
    async def quality_review(
        collection: str,
    ) -> str:
        """
        Review the quality of a knowledge base collection.
        """
        return f"""
You are reviewing the quality of the "{collection}" knowledge base.

Use the check_health tool to get the current quality report, then:
1. Identify sources with low quality scores
2. Find duplicate or near-duplicate chunks
3. Detect stale content that needs refreshing
4. Recommend specific actions to improve quality

Provide a summary with actionable recommendations.
"""
```

#### 6.2.3 CLI Integration

```python
# Update CLI to support MCP

@app.command()
def serve(
    transport: Annotated[str, typer.Option(help="Transport: stdio or http")] = "stdio",
    port: Annotated[int, typer.Option(help="HTTP port (if transport=http)")] = 8080,
):
    """Start the Gweta MCP server."""
    from gweta.mcp.server import run_stdio, run_http

    if transport == "stdio":
        run_stdio()
    elif transport == "http":
        run_http(port=port)
    else:
        raise typer.BadParameter(f"Unknown transport: {transport}")
```

---

## 7. Phase 4: Intelligence Layer (Weeks 9-12)

> **Goal:** Domain rules, golden dataset testing, KB health monitoring, additional stores

### 7.1 Week 9: Domain Rule Engine

#### 7.1.1 Rule Engine (`src/gweta/validate/rules.py`)

```python
# Rule engine implementation

class DomainRuleEngine:
    """Layer 3: Domain-specific validation rules."""

    def __init__(self, rules: list[Rule] | None = None): ...

    @classmethod
    def from_yaml(cls, path: Path) -> "DomainRuleEngine": ...

    @classmethod
    def from_dict(cls, config: dict) -> "DomainRuleEngine": ...

    def validate_chunk(self, chunk: Chunk) -> RuleValidationResult: ...

    def validate_response(
        self,
        response: str,
        source_chunks: list[Chunk],
    ) -> RuleValidationResult: ...

    def add_rule(self, rule: Rule) -> None: ...

    def to_yaml(self) -> str: ...

@dataclass
class Rule:
    name: str
    description: str
    condition: str  # Python expression or rule type
    severity: Literal["error", "warning", "info"]
    message: str

@dataclass
class KnownFact:
    """Verified fact for cross-referencing."""
    key: str
    value: Any
    source: str
    verified_date: date
    tolerance: float = 0.0  # For numerical values
```

#### 7.1.2 Rule YAML Schema

```yaml
# Example: zimbabwe_business.yaml

domain: zimbabwe_business
version: "1.0"
description: "Validation rules for Zimbabwe business information"

known_facts:
  - key: pbc_registration_fee
    value: 100
    unit: USD
    source: "ZIMRA Official Tariff 2024"
    verified_date: "2024-01-15"
    tolerance: 0

  - key: minimum_wage_domestic
    value: 100
    unit: USD
    source: "Labour Act SI 2024"
    verified_date: "2024-06-01"
    tolerance: 5

rules:
  - name: startup_cost_ceiling
    description: "Startup costs should be realistic for Zimbabwe informal sector"
    type: numerical_range
    field: startup_cost
    min: 0
    max: 5000
    severity: warning
    message: "Startup cost ${value} seems unrealistic for informal sector"

  - name: currency_format
    description: "Currency mentions should use standard format"
    type: regex
    pattern: '\$[\d,]+(\.\d{2})?|USD\s*[\d,]+|ZWL\s*[\d,]+'
    severity: info
    message: "Currency format may be ambiguous"

  - name: statutory_reference
    description: "Legal references should match Zimbabwe statute format"
    type: regex
    pattern: '(SI|Act)\s+\d+/\d{4}'
    required: false
    severity: info
    message: "Statutory reference format check"

  - name: date_freshness
    description: "Regulatory information should be recent"
    type: date_check
    max_age_days: 365
    fields: ["effective_date", "published_date"]
    severity: warning
    message: "Information may be outdated"

cross_references:
  - name: pbc_fee_consistency
    description: "PBC fees mentioned should match known rates"
    pattern: 'PBC.*?\$(\d+)'
    known_fact: pbc_registration_fee
    tolerance: 0
    severity: error
    message: "PBC fee ${extracted} doesn't match official rate ${known}"
```

### 7.2 Week 10: Golden Dataset Testing

#### 7.2.1 Golden Dataset Runner (`src/gweta/validate/golden.py`)

```python
# Golden dataset implementation

class GoldenDatasetRunner:
    """Test knowledge base against golden Q&A pairs."""

    def __init__(
        self,
        store: BaseStore,
        dataset_path: Path | None = None,
    ): ...

    def load_dataset(self, path: Path) -> list[GoldenPair]: ...

    async def run(
        self,
        k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> GoldenTestReport: ...

    def to_junit_xml(self, report: GoldenTestReport) -> str:
        """Export as JUnit XML for CI/CD."""
        ...

    def to_json(self, report: GoldenTestReport) -> str:
        """Export as JSON report."""
        ...

@dataclass
class GoldenPair:
    """Single Q&A test case."""
    id: str
    question: str
    expected_answer: str
    expected_sources: list[str]  # Source IDs that should be retrieved
    tags: list[str]

@dataclass
class GoldenTestReport:
    total_tests: int
    passed: int
    failed: int
    retrieval_accuracy: float  # % with correct sources in top-k
    mrr: float  # Mean Reciprocal Rank
    precision_at_k: dict[int, float]
    failed_tests: list[FailedTest]
    coverage_gaps: list[str]  # Topics with no matching chunks
```

#### 7.2.2 Golden Dataset JSON Schema

```json
{
  "$schema": "gweta-golden-v1",
  "name": "Zimbabwe Business Pathway",
  "version": "1.0",
  "created": "2024-02-01",
  "pairs": [
    {
      "id": "pbc-001",
      "question": "How much does it cost to register a Private Business Corporation in Zimbabwe?",
      "expected_answer": "PBC registration costs $100 USD at ZIMRA",
      "expected_sources": ["zimra-pbc-guide-2024"],
      "tags": ["registration", "pbc", "costs"]
    },
    {
      "id": "tax-001",
      "question": "What is the VAT rate in Zimbabwe?",
      "expected_answer": "The standard VAT rate in Zimbabwe is 15%",
      "expected_sources": ["zimra-vat-guide-2024"],
      "tags": ["tax", "vat"]
    }
  ]
}
```

### 7.3 Week 11: KB Health Monitoring

#### 7.3.1 Health Checker (`src/gweta/validate/health.py`)

```python
# Health monitoring implementation

class HealthChecker:
    """Layer 4: Ongoing knowledge base health monitoring."""

    def __init__(
        self,
        store: BaseStore,
        authority_registry: SourceAuthorityRegistry | None = None,
    ): ...

    async def check_staleness(self) -> StalenessReport:
        """Find sources past their freshness window."""
        ...

    async def check_duplicates(self) -> DuplicateReport:
        """Find duplicate/near-duplicate chunks."""
        ...

    async def check_coverage(
        self,
        expected_topics: list[str],
    ) -> CoverageReport:
        """Check if expected topics are covered."""
        ...

    async def check_quality_drift(
        self,
        baseline_date: datetime,
    ) -> DriftReport:
        """Compare current quality to historical baseline."""
        ...

    async def full_health_check(
        self,
        golden_dataset: Path | None = None,
    ) -> HealthReport: ...

@dataclass
class HealthReport:
    timestamp: datetime
    collection: str
    total_chunks: int
    avg_quality_score: float
    staleness: StalenessReport
    duplicates: DuplicateReport
    coverage: CoverageReport | None
    golden_results: GoldenTestReport | None
    recommendations: list[str]
```

### 7.4 Week 12: Additional Vector Stores

#### 7.4.1 Qdrant Store (`src/gweta/ingest/stores/qdrant.py`)

```python
class QdrantStore(BaseStore):
    """Qdrant vector store adapter."""

    def __init__(
        self,
        collection_name: str,
        url: str = "http://localhost:6333",
        api_key: str | None = None,
    ): ...

    # Implement BaseStore interface
```

#### 7.4.2 Pinecone Store (`src/gweta/ingest/stores/pinecone.py`)

```python
class PineconeStore(BaseStore):
    """Pinecone vector store adapter."""

    def __init__(
        self,
        index_name: str,
        api_key: str,
        environment: str,
    ): ...

    # Implement BaseStore interface
```

#### 7.4.3 Weaviate Store (`src/gweta/ingest/stores/weaviate.py`)

```python
class WeaviateStore(BaseStore):
    """Weaviate vector store adapter."""

    def __init__(
        self,
        class_name: str,
        url: str = "http://localhost:8080",
    ): ...

    # Implement BaseStore interface
```

---

## 8. Phase 5: Polish & Launch (Week 13+)

> **Goal:** Documentation, examples, testing, PyPI release

### 8.1 Documentation

```markdown
# Documentation Structure

docs/
├── index.md                    # Landing page with value prop
├── getting-started.md          # 5-minute quickstart
├── installation.md             # All install options
│
├── concepts/
│   ├── architecture.md         # System design explanation
│   ├── quality-scoring.md      # How scoring works
│   ├── authority-tiers.md      # Source authority system
│   └── domain-rules.md         # Rule engine concepts
│
├── guides/
│   ├── crawling.md             # Web crawling guide
│   ├── pdf-extraction.md       # PDF handling
│   ├── database-ingestion.md   # DB connector guide
│   ├── validation.md           # Validation deep-dive
│   ├── mcp-integration.md      # MCP setup guide
│   ├── langchain.md            # LangChain integration
│   └── golden-datasets.md      # Testing with golden data
│
├── api/
│   └── reference.md            # Full API documentation
│
└── examples/
    ├── basic-validation.md
    ├── full-pipeline.md
    └── custom-rules.md
```

### 8.2 Examples

```python
# examples/basic_crawl.py
"""Basic web crawling example."""

from gweta import GwetaCrawler, ChromaStore

async def main():
    # Crawl a website
    crawler = GwetaCrawler(quality_threshold=0.6)
    result = await crawler.crawl(
        url="https://example.com/docs",
        depth=2,
    )

    print(f"Crawled {result.pages_crawled} pages")
    print(f"Quality score: {result.quality_score}")
    print(f"Chunks: {len(result.chunks)} passed, {len(result.rejected_chunks)} rejected")

    # Load to Chroma
    store = ChromaStore("my_knowledge_base")
    result.load_to(store)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

```python
# examples/full_pipeline.py
"""Complete ingestion pipeline example."""

from gweta import (
    GwetaCrawler,
    PDFExtractor,
    DatabaseSource,
    ChunkValidator,
    DomainRuleEngine,
    ChromaStore,
)

async def ingest_all():
    store = ChromaStore("comprehensive_kb")
    validator = ChunkValidator()
    rules = DomainRuleEngine.from_yaml("rules/my_domain.yaml")

    # 1. Crawl websites
    crawler = GwetaCrawler()
    web_result = await crawler.crawl("https://docs.example.com", depth=3)

    # 2. Extract PDFs
    pdf = PDFExtractor()
    pdf_result = await pdf.extract("documents/guide.pdf")

    # 3. Query database
    async with DatabaseSource("postgresql://...") as db:
        db_chunks = await db.extract_and_validate(
            query="SELECT content FROM articles WHERE published = true",
            text_column="content",
        )

    # 4. Validate all with domain rules
    all_chunks = web_result.chunks + pdf_result.chunks + db_chunks
    validated = validator.validate_batch(all_chunks)

    # 5. Apply domain rules
    for chunk in validated.accepted():
        rule_result = rules.validate_chunk(chunk)
        if rule_result.passed:
            await store.add([chunk])

if __name__ == "__main__":
    import asyncio
    asyncio.run(ingest_all())
```

### 8.3 Release Checklist

```markdown
# Pre-Release Checklist

## Code Quality
- [ ] All tests passing (>90% coverage)
- [ ] Type hints complete (mypy strict)
- [ ] Linting clean (ruff)
- [ ] No security vulnerabilities (bandit)

## Documentation
- [ ] README.md complete with badges
- [ ] All docstrings written
- [ ] API reference generated
- [ ] Examples tested and working
- [ ] CHANGELOG.md updated

## Package
- [ ] pyproject.toml metadata complete
- [ ] LICENSE file present (MIT)
- [ ] py.typed marker present
- [ ] Version bumped in _version.py

## Testing
- [ ] Unit tests for all modules
- [ ] Integration tests for stores
- [ ] MCP server tests
- [ ] CLI tests

## Release
- [ ] Tag version in git
- [ ] Build wheel and sdist
- [ ] Test install from TestPyPI
- [ ] Publish to PyPI
- [ ] Create GitHub release
- [ ] Update documentation site
```

### 8.4 Launch Plan

```markdown
# Launch Strategy

## Week 13: Soft Launch
- [ ] Publish v0.1.0 to PyPI
- [ ] Create GitHub repository (public)
- [ ] Deploy documentation site
- [ ] Test in Simuka production

## Week 14: Community Launch
- [ ] Write launch blog post
- [ ] Post to Hacker News
- [ ] Post to r/MachineLearning
- [ ] Post to r/LangChain
- [ ] Share on Twitter/X
- [ ] Share on LinkedIn

## Ongoing
- [ ] Monitor GitHub issues
- [ ] Respond to community feedback
- [ ] Plan v0.2.0 features
- [ ] Build gweta-zimbabwe package
```

---

## 9. Testing Strategy

### 9.1 Test Structure

```
tests/
├── conftest.py                 # Shared fixtures
├── unit/
│   ├── test_types.py           # Core type tests
│   ├── test_config.py          # Configuration tests
│   ├── test_extraction.py      # Extraction validator
│   ├── test_chunks.py          # Chunk validator
│   ├── test_duplicates.py      # Duplicate detection
│   ├── test_density.py         # Density scoring
│   ├── test_rules.py           # Rule engine
│   └── test_golden.py          # Golden dataset runner
├── integration/
│   ├── test_crawler.py         # Crawl4AI integration
│   ├── test_pdf.py             # PDF extraction
│   ├── test_database.py        # Database connector
│   ├── test_chroma.py          # Chroma store
│   ├── test_qdrant.py          # Qdrant store
│   └── test_mcp.py             # MCP server
└── fixtures/
    ├── sample_pdfs/
    │   ├── clean.pdf
    │   ├── scanned.pdf
    │   └── tables.pdf
    ├── sample_html/
    │   ├── clean.html
    │   └── javascript_heavy.html
    └── golden_datasets/
        └── test_golden.json
```

### 9.2 Key Test Cases

```python
# tests/unit/test_chunks.py

class TestChunkValidator:
    def test_valid_chunk_passes(self):
        """Well-formed chunk should pass validation."""
        chunk = Chunk(
            id="test-1",
            text="This is a well-formed chunk with sufficient content.",
            metadata={"source": "test"},
            source="test",
        )
        validator = ChunkValidator()
        result = validator.validate(chunk)
        assert result.passed
        assert result.quality_score > 0.6

    def test_empty_chunk_fails(self):
        """Empty chunk should fail validation."""
        chunk = Chunk(id="test-2", text="", metadata={}, source="test")
        validator = ChunkValidator()
        result = validator.validate(chunk)
        assert not result.passed
        assert "EMPTY_CONTENT" in [i.code for i in result.issues]

    def test_low_density_chunk_warned(self):
        """Low information density should trigger warning."""
        chunk = Chunk(
            id="test-3",
            text="the the the the the the the",
            metadata={},
            source="test",
        )
        validator = ChunkValidator()
        result = validator.validate(chunk)
        assert "LOW_DENSITY" in [i.code for i in result.issues]

    def test_missing_required_metadata(self):
        """Missing required metadata should fail."""
        chunk = Chunk(id="test-4", text="Valid text", metadata={}, source="test")
        validator = ChunkValidator(required_metadata=["source", "date"])
        result = validator.validate(chunk)
        assert "MISSING_METADATA" in [i.code for i in result.issues]
```

### 9.3 CI Configuration

```yaml
# .github/workflows/ci.yml

name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev,all]"

      - name: Run linting
        run: |
          ruff check src tests
          ruff format --check src tests

      - name: Run type checking
        run: mypy src

      - name: Run tests
        run: pytest --cov=gweta --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run security scan
        run: |
          pip install bandit
          bandit -r src
```

---

## 10. Dependencies

### 10.1 pyproject.toml

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "gweta"
dynamic = ["version"]
description = "RAG data quality and ingestion framework"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = [
    "rag",
    "llm",
    "data-quality",
    "vector-database",
    "knowledge-base",
    "mcp",
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    # Core
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "httpx>=0.25",
    "typer>=0.9",
    "rich>=13.0",

    # Crawling (core, not optional)
    "crawl4ai>=0.3",
    "beautifulsoup4>=4.12",

    # PDF
    "pdfplumber>=0.10",

    # Validation
    "datasketch>=1.6",
    "langdetect>=1.0",
    "chardet>=5.0",
]

[project.optional-dependencies]
# Database connectivity
db = [
    "sqlalchemy>=2.0",
    "asyncpg>=0.29",
    "aiomysql>=0.2",
    "aiosqlite>=0.19",
]

# MCP server
mcp = [
    "mcp>=1.0",
]

# Advanced chunking
chonkie = [
    "chonkie>=0.2",
]

# Vector stores
chroma = ["chromadb>=0.4"]
qdrant = ["qdrant-client>=1.7"]
pinecone = ["pinecone-client>=3.0"]
weaviate = ["weaviate-client>=4.0"]

# All vector stores
stores = [
    "gweta[chroma]",
    "gweta[qdrant]",
    "gweta[pinecone]",
    "gweta[weaviate]",
]

# Everything
all = [
    "gweta[db]",
    "gweta[mcp]",
    "gweta[chonkie]",
    "gweta[stores]",
]

# Development
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "pytest-cov>=4.0",
    "mypy>=1.0",
    "ruff>=0.1",
    "bandit>=1.7",
    "pre-commit>=3.0",
]

[project.scripts]
gweta = "gweta.cli.main:app"

[project.urls]
Homepage = "https://github.com/yourusername/gweta"
Documentation = "https://gweta.readthedocs.io"
Repository = "https://github.com/yourusername/gweta"

[tool.hatch.version]
path = "src/gweta/_version.py"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true

[tool.ruff]
target-version = "py310"
line-length = 88
select = ["E", "F", "I", "N", "W", "UP", "B", "C4", "SIM"]

[tool.ruff.isort]
known-first-party = ["gweta"]
```

---

## 11. Configuration Schema

### 11.1 Global Config (`gweta.yaml`)

```yaml
# gweta.yaml - Global configuration

version: "1"

# Quality thresholds
quality:
  min_score: 0.6
  min_density: 0.3
  max_duplicate_similarity: 0.92
  min_text_length: 50
  max_gibberish_ratio: 0.3

# Chunking defaults
chunking:
  strategy: recursive  # or semantic, sentence
  size: 500
  overlap: 50

# Authority registry
authority:
  path: sources.yaml
  default_tier: 3
  default_freshness_days: 90

# Logging
logging:
  level: INFO
  format: structured  # or plain

# MCP server
mcp:
  transport: stdio
  http_port: 8080
```

### 11.2 Source Registry (`sources.yaml`)

```yaml
# sources.yaml - Source authority registry

sources:
  # Government sources (high authority)
  - domain: "zimra.co.zw"
    name: "ZIMRA Official"
    authority: 5
    freshness_days: 30

  - domain: "*.gov.zw"
    name: "Zimbabwe Government"
    authority: 4
    freshness_days: 90

  # Professional bodies
  - domain: "icaz.org.zw"
    name: "ICAZ"
    authority: 4
    freshness_days: 180

  # News and media (lower authority)
  - domain: "herald.co.zw"
    name: "The Herald"
    authority: 2
    freshness_days: 7

  # Blogs (lowest authority)
  - pattern: "*.blogspot.com"
    authority: 1
    freshness_days: 365

# Blocked domains
blocked:
  - "spam-site.com"
  - "unreliable-source.net"
```

---

## 12. API Reference

### 12.1 Core Types

| Type | Description |
|------|-------------|
| `Chunk` | Universal chunk representation with quality metadata |
| `Source` | Data source with authority tracking |
| `QualityReport` | Batch validation results |
| `QualityDetails` | Multi-dimensional quality breakdown |
| `QualityIssue` | Single quality problem |

### 12.2 Validators

| Class | Purpose |
|-------|---------|
| `ExtractionValidator` | Layer 1: Raw text quality |
| `ChunkValidator` | Layer 2: Chunk quality |
| `DomainRuleEngine` | Layer 3: Domain rules |
| `HealthChecker` | Layer 4: KB health |
| `GoldenDatasetRunner` | Golden dataset testing |

### 12.3 Acquisition

| Class | Purpose |
|-------|---------|
| `GwetaCrawler` | Web crawling with quality |
| `PDFExtractor` | PDF extraction |
| `APIClient` | REST/GraphQL fetching |
| `DatabaseSource` | SQL database connector |

### 12.4 Ingestion

| Class | Purpose |
|-------|---------|
| `RecursiveChunker` | Default text chunking |
| `ChromaStore` | Chroma adapter |
| `QdrantStore` | Qdrant adapter |
| `PineconeStore` | Pinecone adapter |
| `WeaviateStore` | Weaviate adapter |

### 12.5 CLI Commands

| Command | Description |
|---------|-------------|
| `gweta validate` | Validate chunks from file |
| `gweta crawl` | Crawl URL and extract |
| `gweta ingest` | Full ingestion pipeline |
| `gweta health` | KB health check |
| `gweta serve` | Start MCP server |

### 12.6 MCP Tools

| Tool | Description |
|------|-------------|
| `crawl_and_ingest` | Crawl + validate + load |
| `ingest_from_database` | DB → vector store |
| `validate_chunks` | Validate without loading |
| `check_health` | KB health report |
| `crawl_site` | Crawl without loading |
| `query_database` | Read-only DB query |

---

## Summary

This implementation plan covers the complete build of Gweta over 13+ weeks:

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1** | Weeks 1-3 | Core types, validation, Chroma, CLI |
| **Phase 2** | Weeks 4-6 | Crawler, PDF, database connector |
| **Phase 3** | Weeks 7-8 | MCP server (stdio + HTTP) |
| **Phase 4** | Weeks 9-12 | Rules, golden tests, health, stores |
| **Phase 5** | Week 13+ | Docs, polish, PyPI launch |

**Key Success Metrics:**
- All 4 validation layers functional
- MCP server working with Claude Desktop
- >90% test coverage
- Full documentation
- Published to PyPI as `pip install gweta`

---

*Document: Gweta Implementation Plan v1.0*
*Status: Ready for implementation*
*Next: Scaffold repository structure*
