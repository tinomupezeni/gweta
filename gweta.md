# Gweta — Architecture Blueprint v2
## The Data Acquisition, Validation & Ingestion Framework for RAG Knowledge Bases

**Name:** Gweta (Shona: "to filter/strain") — `pip install gweta`
**Tagline:** Acquire. Validate. Ingest. The complete pipeline for RAG knowledge bases.

---

## What Changed from v1

The original concept was a **validation-only** library — "Great Expectations for RAG." Your additions fundamentally expand the scope:

| v1 (Validation Library) | v2 (Full Framework with MCP) |
|---|---|
| Validates data someone else extracted | **Acquires** data from URLs, APIs, databases |
| Works after parsing | Has its own **web crawler** |
| No connectivity | **MCP server** for AI agent integration |
| Library you call | Framework that **agents call** |

This makes Gweta the **only tool in the ecosystem that handles acquisition → validation → ingestion as a single, quality-aware pipeline** — and exposes the entire thing over MCP so any AI agent (Claude, GPT, open-source) can use it as a tool.

---

## 1. Why MCP Changes Everything

### What MCP Is
MCP (Model Context Protocol) is an open protocol by Anthropic that standardizes how LLM applications connect to external data sources and tools. Think of it as USB-C for AI — any MCP client (Claude Desktop, Cursor, VS Code Copilot, LangChain agents) can connect to any MCP server.

The Python SDK (v1.26 stable, v2 in development for Q1 2026) uses FastMCP for server creation. MCP servers expose three primitives:
- **Tools** — functions the LLM can call (like POST endpoints)
- **Resources** — data the LLM can read (like GET endpoints)
- **Prompts** — reusable interaction templates

### What This Means for Gweta
Instead of just being a library developers import into their code, Gweta **also** becomes an MCP server that any AI agent can connect to. A user in Claude Desktop or Cursor could say:

> "Crawl the ZIMRA website, extract all tax registration guides, validate the quality, and load them into my Chroma database"

And Gweta's MCP tools handle every step — the agent doesn't need to know about pdfplumber, chunking strategies, or quality thresholds. It just calls Gweta's tools.

### The MCP Ecosystem Context
Database MCP servers already exist — Anthropic's official postgres and sqlite servers, DBHub (universal gateway for Postgres/MySQL/SQLite/DuckDB), and dozens of community servers. But **no MCP server exists for quality-validated RAG ingestion.** The existing servers let you query databases; Gweta lets you build and maintain the knowledge bases those databases serve.

---

## 2. Why a Built-in Web Crawler

### The Landscape
Two tools dominate web crawling for RAG:

**Crawl4AI** — Open-source Python crawler, 58K+ GitHub stars. Local-first, outputs clean LLM-ready markdown. Supports BM25 content filtering, adaptive crawling, LLM-based extraction. Free, no API keys required. The "Scrapy for LLMs."

**Firecrawl** — API-first crawler that converts any URL to clean markdown/JSON. Handles JavaScript rendering, anti-bot measures automatically. Managed infrastructure. $29-499/month.

### Why Gweta Needs Its Own Crawler (or a Deep Integration)
The problem: both Crawl4AI and Firecrawl output clean markdown — but neither validates what they've crawled. They don't know if:
- The page they crawled was actually the current version (Zimbabwean government sites often serve cached/outdated pages)
- The extracted content is complete (JavaScript-rendered content sometimes partially loads)
- The data contradicts what's already in your knowledge base
- The source meets your authority requirements

**Gweta's approach: Don't rebuild the crawler. Wrap Crawl4AI and add the quality layer.**

Crawl4AI is open-source, Python-native, and free — perfect as Gweta's crawling backend. Gweta adds:
1. **Pre-crawl validation** — Is this URL from an authorized source? Is it on the allowed domain list?
2. **Post-crawl quality scoring** — Is the extracted content complete? Is it garbled? Does it meet minimum information density?
3. **Differential crawling** — Compare newly crawled content against what's already in the knowledge base. Flag changes, contradictions, or stale data.
4. **Crawl scheduling** — Different sources need different freshness windows (ZIMRA tax rates: monthly, ZNQF framework: yearly)

---

## 3. Revised Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MCP INTERFACE                                     │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Gweta MCP Server (FastMCP)                                      │    │
│  │                                                                   │    │
│  │  TOOLS:                                                           │    │
│  │  • gweta_crawl(url, depth, rules)     — Crawl + validate          │    │
│  │  • gweta_ingest(source, target_db)    — Full pipeline             │    │
│  │  • gweta_validate(chunks)             — Validate existing data    │    │
│  │  • gweta_health(knowledge_base)       — KB health check           │    │
│  │  • gweta_query_db(dsn, query)         — Read from user's DB       │    │
│  │  • gweta_status()                     — Pipeline status           │    │
│  │                                                                   │    │
│  │  RESOURCES:                                                       │    │
│  │  • gweta://sources                    — Registered data sources   │    │
│  │  • gweta://quality-report             — Latest quality scores     │    │
│  │  • gweta://rules/{domain}             — Domain validation rules   │    │
│  │                                                                   │    │
│  │  PROMPTS:                                                         │    │
│  │  • ingest_planning                    — Plan an ingestion job     │    │
│  │  • quality_review                     — Review KB quality          │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
┌────────────────────────────▼─────────────────────────────────────────────┐
│                      GWETA CORE ENGINE                                    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │  ACQUISITION LAYER (New in v2)                                      │ │
│  │                                                                      │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │ │
│  │  │  Web Crawler  │  │  API Client  │  │  DB Connector │              │ │
│  │  │  (Crawl4AI)   │  │  (httpx)     │  │  (SQLAlchemy) │              │ │
│  │  │              │  │              │  │               │              │ │
│  │  │  • Site crawl │  │  • REST APIs │  │  • Postgres   │              │ │
│  │  │  • PDF fetch  │  │  • GraphQL   │  │  • MySQL      │              │ │
│  │  │  • Sitemap    │  │  • Webhooks  │  │  • SQLite     │              │ │
│  │  │  • RSS/Atom   │  │  • CSV/JSON  │  │  • User DBs   │              │ │
│  │  └──────┬───────┘  └──────┬───────┘  └───────┬───────┘              │ │
│  │         └─────────────────┼──────────────────┘                       │ │
│  │                           ▼                                          │ │
│  │              ┌────────────────────────┐                              │ │
│  │              │  Source Authority       │                              │ │
│  │              │  Registry               │                              │ │
│  │              │  (Is this source        │                              │ │
│  │              │   trusted? What tier?)  │                              │ │
│  │              └────────────┬───────────┘                              │ │
│  └───────────────────────────┼──────────────────────────────────────────┘ │
│                              ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  VALIDATION LAYER (from v1)                                        │   │
│  │                                                                     │   │
│  │  Stage 1: Extraction Quality         Stage 2: Chunk Quality         │   │
│  │  ┌─────────────────────────┐        ┌─────────────────────────┐    │   │
│  │  │ • OCR confidence         │        │ • Coherence scoring      │    │   │
│  │  │ • Encoding detection     │        │ • Information density    │    │   │
│  │  │ • Completeness check     │        │ • Duplicate detection    │    │   │
│  │  │ • Table fidelity         │        │ • Metadata completeness  │    │   │
│  │  │ • Language consistency   │        │ • Boundary quality       │    │   │
│  │  │ • Boilerplate filtering  │        │ • Relevance scoring      │    │   │
│  │  └─────────────────────────┘        └─────────────────────────┘    │   │
│  │                                                                     │   │
│  │  Stage 3: Domain Rules              Stage 4: KB Health              │   │
│  │  ┌─────────────────────────┐        ┌─────────────────────────┐    │   │
│  │  │ • Known facts checking   │        │ • Golden dataset tests   │    │   │
│  │  │ • Numerical validation   │        │ • Staleness detection    │    │   │
│  │  │ • Constraint enforcement │        │ • Coverage analysis      │    │   │
│  │  │ • Cross-reference check  │        │ • Drift monitoring       │    │   │
│  │  └─────────────────────────┘        └─────────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────────┘   │
│                              ▼                                            │
│  ┌───────────────────────────────────────────────────────────────────┐   │
│  │  INGESTION LAYER                                                    │   │
│  │                                                                     │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │   │
│  │  │  Chroma   │  │  Qdrant  │  │ Pinecone │  │ Weaviate │  + more  │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │   │
│  │                                                                     │   │
│  │  Only validated, quality-scored chunks get loaded                    │   │
│  │  Each chunk carries: quality_score, authority_level, freshness      │   │
│  └───────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. MCP Server Design

### Tools (What Agents Can Call)

```python
from mcp.server.fastmcp import FastMCP

gweta = FastMCP("gweta")

@gweta.tool()
async def crawl_and_ingest(
    url: str,
    depth: int = 2,
    target_collection: str = "default",
    authority_tier: int = 3,
    rules: str | None = None,
) -> dict:
    """Crawl a website, validate extracted content, and load quality chunks
    into the target vector database collection.

    Args:
        url: Starting URL to crawl
        depth: How many links deep to follow (1-5)
        target_collection: Name of the vector DB collection to load into
        authority_tier: Source authority level (1=blog, 5=primary legislation)
        rules: Optional domain rule set name (e.g., 'zimbabwe_business')
    """
    # 1. Crawl with Crawl4AI
    # 2. Extract + score extraction quality
    # 3. Chunk with configurable strategy
    # 4. Validate chunks against rules
    # 5. Load passing chunks into vector store
    # 6. Return quality report
    ...

@gweta.tool()
async def ingest_from_database(
    dsn: str,
    query: str,
    target_collection: str = "default",
    text_column: str = "content",
    metadata_columns: list[str] | None = None,
) -> dict:
    """Read data from a user's database, validate it, and load into
    the knowledge base.

    Args:
        dsn: Database connection string (postgres://, mysql://, sqlite://)
        query: SQL query to extract data
        target_collection: Target vector DB collection
        text_column: Column containing the text to embed
        metadata_columns: Columns to preserve as chunk metadata
    """
    ...

@gweta.tool()
async def validate_existing(
    collection: str,
    rules: str | None = None,
    golden_dataset: str | None = None,
) -> dict:
    """Run quality checks on an existing knowledge base collection.

    Returns quality scores, stale sources, coverage gaps, and
    specific chunks that fail validation.
    """
    ...

@gweta.tool()
async def crawl_site(
    url: str,
    depth: int = 2,
    output_format: str = "markdown",
) -> dict:
    """Crawl a website and return clean, validated content without
    loading it into any database. Useful for preview/review.

    Returns extracted content with quality scores per page.
    """
    ...

@gweta.tool()
async def query_source_db(
    dsn: str,
    query: str,
) -> dict:
    """Execute a read-only query against a connected database.
    Returns results as structured data with schema info.
    """
    ...

@gweta.tool()
async def check_health() -> dict:
    """Get the health status of all connected knowledge bases,
    including staleness warnings and quality score trends."""
    ...
```

### Resources (What Agents Can Read)

```python
@gweta.resource("gweta://sources")
async def list_sources() -> str:
    """List all registered data sources with their authority tiers,
    last crawl dates, and freshness status."""
    ...

@gweta.resource("gweta://quality/{collection}")
async def quality_report(collection: str) -> str:
    """Get the latest quality report for a collection, including
    per-source scores, validation failures, and recommendations."""
    ...

@gweta.resource("gweta://rules/{domain}")
async def domain_rules(domain: str) -> str:
    """Get the validation rules for a specific domain.
    Returns YAML-formatted rule definitions."""
    ...
```

### Prompts (Interaction Templates)

```python
@gweta.prompt()
async def plan_ingestion(
    sources: str,
    target: str,
) -> str:
    """Generate an ingestion plan for the given sources and target.
    Analyzes source types, recommends crawl strategies, and
    estimates quality thresholds."""
    ...
```

---

## 5. Web Crawler Integration

### Architecture Decision: Wrap Crawl4AI, Don't Rebuild

Crawl4AI has 58K+ stars, handles JavaScript rendering, outputs LLM-ready markdown, and is free/open-source. Rebuilding this would be months of work with no differentiation. Instead, Gweta wraps Crawl4AI and adds what it lacks:

```python
# What users write with Gweta
from gweta import GwetaCrawler

crawler = GwetaCrawler(
    authority_registry="sources.yaml",
    quality_threshold=0.6,
    chunk_strategy="semantic",
    chunk_size=300,
    chunk_overlap=0.17,
)

# Crawl, validate, and get quality-scored results
results = await crawler.crawl(
    url="https://www.zimra.co.zw/tax-registration",
    depth=2,
    follow_pdfs=True,  # Also download and extract linked PDFs
)

print(results.pages_crawled)       # 12
print(results.pages_passed)        # 9
print(results.pages_failed)        # 3 (reasons logged)
print(results.quality_score)       # 0.78
print(results.chunks)              # List of validated, quality-scored chunks
print(results.rejected_chunks)     # Chunks that failed validation (for review)

# Load only the good stuff
results.load_to(chroma_collection)
```

### What Gweta Adds on Top of Crawl4AI

| Capability | Crawl4AI Alone | Gweta + Crawl4AI |
|---|---|---|
| Crawl websites | ✅ | ✅ |
| Output clean markdown | ✅ | ✅ |
| JavaScript rendering | ✅ | ✅ |
| Quality score per page | ❌ | ✅ |
| Authority verification | ❌ | ✅ (is this source trusted?) |
| Chunk validation | ❌ | ✅ (coherence, density, duplication) |
| Diff against existing KB | ❌ | ✅ (what changed since last crawl?) |
| PDF extraction + validation | ❌ (markdown only) | ✅ (pdfplumber integration) |
| Domain rule enforcement | ❌ | ✅ (custom rules per domain) |
| Load to vector DB | ❌ | ✅ (with quality metadata) |
| MCP server exposure | ❌ | ✅ |
| Scheduled re-crawling | ❌ | ✅ (per-source freshness windows) |

---

## 6. Database Connectivity

### Why It Matters
Many RAG knowledge bases need to incorporate structured data — user records, product catalogs, transaction histories. Currently, developers write ad-hoc scripts to extract from databases, manually chunk, and load. Gweta standardizes this.

### Design: SQLAlchemy + Read-Only Safety

```python
from gweta import DatabaseSource

# Connect to user's database
source = DatabaseSource(
    dsn="postgresql://user:pass@localhost/simuka_db",
    read_only=True,  # Safety: only SELECT queries allowed
)

# Extract and ingest
results = await source.extract_and_ingest(
    query="SELECT content, category, created_at FROM knowledge_articles WHERE active = true",
    text_column="content",
    metadata_columns=["category", "created_at"],
    target=chroma_collection,
    chunk_strategy="semantic",
    quality_threshold=0.5,
)
```

### Supported Databases (via SQLAlchemy)
- PostgreSQL
- MySQL / MariaDB
- SQLite
- DuckDB (for analytics/OLAP data)

### Safety Measures
- **Read-only by default** — only SELECT queries permitted unless explicitly overridden
- **Query timeout** — configurable max execution time (default 30s)
- **Row limit** — configurable max rows returned (default 10,000)
- **No credential storage** — DSNs passed at runtime, never persisted

---

## 7. Package Structure

```
gweta/
├── __init__.py                    # Public API
├── py.typed                       # PEP 561 type stubs
├── core/
│   ├── __init__.py
│   ├── config.py                  # Configuration management (YAML + Python)
│   ├── types.py                   # Core types: Chunk, Source, QualityReport
│   └── registry.py                # Source authority registry
│
├── acquire/                       # ACQUISITION LAYER
│   ├── __init__.py
│   ├── crawler.py                 # Crawl4AI wrapper with quality layer
│   ├── pdf_extractor.py           # PDF extraction (pdfplumber)
│   ├── api_client.py              # REST/GraphQL API data fetching
│   ├── database.py                # SQLAlchemy database connector
│   └── fetchers/                  # Pluggable fetchers
│       ├── __init__.py
│       ├── url.py                 # Single URL fetch
│       ├── sitemap.py             # Sitemap-based crawling
│       └── rss.py                 # RSS/Atom feed ingestion
│
├── validate/                      # VALIDATION LAYER
│   ├── __init__.py
│   ├── extraction.py              # Layer 1: Extraction quality scoring
│   ├── chunks.py                  # Layer 2: Chunk quality validation
│   ├── rules.py                   # Layer 3: Domain rule engine
│   ├── health.py                  # Layer 4: KB health monitoring
│   ├── golden.py                  # Golden dataset test runner
│   └── detectors/                 # Heuristic detectors
│       ├── __init__.py
│       ├── gibberish.py           # OCR/encoding garbage detection
│       ├── duplicates.py          # Near-duplicate detection (MinHash)
│       ├── density.py             # Information density scoring
│       └── staleness.py           # Freshness tracking
│
├── ingest/                        # INGESTION LAYER
│   ├── __init__.py
│   ├── chunkers.py                # Chunking strategies (wraps Chonkie)
│   └── stores/                    # Vector store adapters
│       ├── __init__.py
│       ├── chroma.py
│       ├── qdrant.py
│       ├── pinecone.py
│       └── weaviate.py
│
├── mcp/                           # MCP SERVER
│   ├── __init__.py
│   ├── server.py                  # FastMCP server definition
│   ├── tools.py                   # MCP tool implementations
│   ├── resources.py               # MCP resource definitions
│   └── prompts.py                 # MCP prompt templates
│
├── adapters/                      # FRAMEWORK ADAPTERS
│   ├── __init__.py
│   ├── langchain.py               # LangChain Document ↔ Gweta Chunk
│   ├── llamaindex.py              # LlamaIndex Node ↔ Gweta Chunk
│   ├── chonkie.py                 # Chonkie Chunk ↔ Gweta Chunk
│   └── unstructured.py            # Unstructured Element ↔ Gweta Chunk
│
├── cli.py                         # CLI: gweta crawl, gweta validate, etc.
└── _version.py                    # Version info
```

---

## 8. Installation Options

```bash
# Core (validation only, minimal dependencies)
pip install gweta

# With web crawling
pip install gweta[crawl]           # adds crawl4ai

# With database connectivity
pip install gweta[db]              # adds sqlalchemy + drivers

# With MCP server
pip install gweta[mcp]             # adds mcp sdk

# With specific vector stores
pip install gweta[chroma]
pip install gweta[qdrant]

# Everything
pip install gweta[all]

# Start the MCP server
gweta serve                        # stdio transport (for Claude Desktop)
gweta serve --http --port 8080     # HTTP transport (for remote agents)
```

---

## 9. Competitive Positioning (Updated)

```
BEFORE GWETA:
Developer wants to build a RAG knowledge base from various sources.

Step 1: Write crawler script (Crawl4AI / Scrapy / custom)
Step 2: Write PDF extractor (pdfplumber / PyPDF)
Step 3: Write database ETL (custom SQL scripts)
Step 4: Write chunking logic (or use Chonkie / LangChain)
Step 5: Hope data quality is okay (no validation)
Step 6: Load into vector store (custom per store)
Step 7: Discover bad data when users complain
Step 8: Write evaluation scripts (RAGAS)
Step 9: Go back to Step 1 with patches

AFTER GWETA:
gweta crawl https://example.com --depth 2 --target my_collection
gweta ingest --db postgres://... --query "SELECT..." --target my_collection
gweta validate my_collection --rules my_domain.yaml
gweta health my_collection --golden tests/golden.json

Or, from any MCP client:
"Crawl the ZIMRA website and add tax guides to my knowledge base"
→ Gweta handles everything.
```

### Where Gweta Sits in the Ecosystem

| Tool | What It Does | Gweta's Relationship |
|---|---|---|
| Crawl4AI | Crawls websites → markdown | Gweta wraps it (crawling backend) |
| Unstructured | Parses documents → elements | Gweta validates its output |
| Chonkie | Chunks text for RAG | Gweta wraps it (chunking backend) |
| Chroma/Qdrant | Stores vectors | Gweta loads validated data into them |
| RAGAS | Evaluates RAG output quality | Gweta prevents bad input (complementary) |
| Great Expectations | Validates structured data | Gweta does the same for unstructured data |
| LangChain/LlamaIndex | Orchestrates RAG pipelines | Gweta plugs into their ingestion step |

**Gweta is the missing middleware.** It doesn't replace any of these tools — it connects them with a quality-aware pipeline and exposes everything over MCP.

---

## 10. Revised Build Plan

### Phase 1: Foundation (Weeks 1-3) — "It Works for Simuka"
- Core types (Chunk, Source, QualityReport)
- Extraction quality scoring (heuristic)
- Chunk validation (density, duplicates, metadata)
- Chroma adapter
- CLI: `gweta validate`
- **Simuka integration:** Replace ad-hoc pipeline validation

### Phase 2: Acquisition (Weeks 4-6) — "It Crawls"
- Crawl4AI wrapper with quality layer
- PDF extraction with pdfplumber
- SQLAlchemy database connector (read-only)
- Source authority registry (YAML)
- CLI: `gweta crawl`, `gweta ingest`
- **Simuka integration:** Replace `zw_knowledge_pipeline.py` with Gweta

### Phase 3: MCP Server (Weeks 7-8) — "Agents Can Use It"
- FastMCP server with core tools
- stdio + HTTP transports
- Claude Desktop / Cursor integration tested
- CLI: `gweta serve`
- **Simuka integration:** LangGraph agent uses Gweta tools via MCP

### Phase 4: Intelligence (Weeks 9-12) — "It Gets Smarter"
- Domain rule engine (YAML definitions)
- Golden dataset test runner
- KB health monitoring
- Staleness detection + scheduled re-crawling
- Additional vector store adapters (Qdrant, Pinecone)
- **Simuka integration:** Golden dataset for Business pathway, automated quality monitoring

### Phase 5: Community Launch (Week 13+)
- PyPI publish
- Documentation site
- GitHub README with demos
- Launch posts (HN, r/MachineLearning, r/LangChain)
- Chonkie pipeline integration PR

---
Crawling is core, not optional — Crawl4AI goes in the main install, not behind gweta[crawl]. This means Playwright is a dependency for everyone, which adds weight but matches your vision of Gweta as a full pipeline, not just a validator. We'll also include a lightweight httpx fetcher for simple static sites where a headless browser is overkill.

Both MCP transports from day one — gweta serve for stdio (Claude Desktop, Cursor) and gweta serve --http for remote agents. This means any LangGraph agent, any IDE, or any custom client can connect. Doubles the addressable market from launch.

Chonkie as optional backend — pip install gweta[chonkie] adds it. Core package ships with a minimal built-in chunker (recursive text splitting — good enough for 80% of cases). Users who want semantic/neural chunking install the Chonkie extra. Keeps the base package lighter.

gweta-zimbabwe as separate package — Smart move. The core gweta stays universal, gweta-zimbabwe contains ZIMRA rules, PBC fee facts, ZNQF mappings, ZIMSTAT thresholds. Other communities could build gweta-kenya, gweta-nigeria, etc. This is also the pattern that makes Gweta feel like a real ecosystem, not a one-country tool.

Both sync and async APIs — Like httpx's dual interface. gweta.crawl() for scripts and notebooks, await gweta.async_crawl() for agents and servers. The internals are async-first (Crawl4AI and MCP both require it), with sync wrappers using asyncio.run().

---

*Document: Gweta Architecture Blueprint v2*
*Status: Research complete → Ready for scaffolding*
*Next: Pick answers to open questions → Scaffold repository → Build Phase 1*