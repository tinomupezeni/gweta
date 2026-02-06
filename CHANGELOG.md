# Changelog

All notable changes to Gweta will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-02-06

### Added

#### Intelligence Layer (Phase 1)
Gweta now understands your system's purpose and filters content for relevance.

- **SystemIntent**: Define what your RAG system is meant to do
  - YAML-based configuration
  - Core questions, relevant/irrelevant topics
  - Geographic focus and quality thresholds

- **EmbeddingEngine**: Wrapper for sentence-transformers
  - Lazy loading for efficiency
  - Multiple model options (fast, balanced, accurate)
  - Batch processing support

- **RelevanceFilter**: Score and filter chunks by intent
  - Embedding-based semantic similarity
  - Automatic rejection of irrelevant topics
  - Accept/Review/Reject decisions with thresholds

- **Pipeline**: Unified API for intent-aware ingestion
  - Combines quality validation + relevance filtering
  - Single entry point for full pipeline
  - Detailed result reporting

- New optional dependency: `gweta[intelligence]`

### Example Usage

```python
from gweta.intelligence import Pipeline, SystemIntent
from gweta import ChromaStore

# Define your system's intent
intent = SystemIntent(
    name="Simuka Career Platform",
    description="Career guidance for Zimbabwean graduates",
    core_questions=[
        "How do I register a business in Zimbabwe?",
        "What freelance services can I offer?",
    ],
    relevant_topics=["Zimbabwe business", "ZIMRA", "entrepreneurship"],
    irrelevant_topics=["US regulations", "cryptocurrency"],
)

# Create intent-aware pipeline
pipeline = Pipeline(intent=intent, store=ChromaStore("my-kb"))

# Ingest with automatic relevance filtering
result = await pipeline.ingest(chunks)
print(f"Ingested {result.ingested} relevant chunks")
print(f"Rejected {result.rejected_count} irrelevant chunks")
```

---

## [0.1.1] - 2025-02-06

### Fixed

- **ChromaStore default embeddings**: ChromaStore now automatically uses
  SentenceTransformer `all-MiniLM-L6-v2` embeddings by default. Previously,
  users had to manually provide an embedding function.
- Added `sentence-transformers` to `gweta[chroma]` dependencies
- Added `use_default_embeddings` parameter to ChromaStore
- Improved ChromaStore documentation with embedding configuration examples

### Changed

- Updated project URLs to point to correct GitHub repository

---

## [0.1.0] - 2025-02-06

### Added

#### Core
- `Chunk` dataclass for universal chunk representation
- `QualityReport` and `QualityIssue` types
- `GwetaSettings` configuration management
- Structured logging with `get_logger()`

#### Validation (4 Layers)
- **Layer 1: Extraction Quality**
  - `ExtractionValidator` for raw text validation
  - Gibberish detection
  - Encoding validation
  - Language detection

- **Layer 2: Chunk Quality**
  - `ChunkValidator` for chunk validation
  - Information density scoring
  - Coherence checking
  - `DuplicateDetector` using MinHash LSH

- **Layer 3: Domain Rules**
  - `DomainRuleEngine` with YAML configuration
  - Regex pattern matching
  - Numerical range validation
  - Required field validation
  - Known fact cross-referencing

- **Layer 4: KB Health**
  - `HealthChecker` for ongoing monitoring
  - Staleness detection
  - Duplicate identification
  - `GoldenDatasetRunner` for retrieval testing
  - JUnit XML and JSON export

#### Acquisition
- `GwetaCrawler` wrapping Crawl4AI
- `PDFExtractor` with table extraction
- `DatabaseSource` with SQL safety
- `APIClient` for REST endpoints
- Sitemap and RSS fetchers

#### Ingestion
- `RecursiveChunker` for text splitting
- `ChromaStore` adapter
- `QdrantStore` adapter
- `PineconeStore` adapter
- `WeaviateStore` adapter
- `BaseStore` abstract interface

#### MCP Server
- FastMCP-based server
- 10 MCP tools for AI agent integration
- 4 MCP resources
- 3 MCP prompts
- stdio and HTTP transports

#### CLI
- `gweta validate` command
- `gweta crawl` command
- `gweta ingest` command
- `gweta health` command
- `gweta serve` command

### Documentation
- Getting started guide
- Architecture overview
- API reference
- Example scripts

---

## Version History

| Version | Date | Description |
|---------|------|-------------|
| 0.1.0 | 2025-02-06 | Initial release |

[Unreleased]: https://github.com/tinomupezeni/gweta/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/tinomupezeni/gweta/releases/tag/v0.1.0
