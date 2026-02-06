# Changelog

All notable changes to Gweta will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release preparation

## [0.1.0] - 2024-XX-XX

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
| 0.1.0 | TBD | Initial release |

[Unreleased]: https://github.com/yourusername/gweta/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/gweta/releases/tag/v0.1.0
