# Getting Started

This guide will get you up and running with Gweta in 5 minutes.

## Installation

### Basic Installation

```bash
pip install gweta
```

### With Intelligence Layer (Recommended)

```bash
pip install gweta[intelligence]
```

This enables intent-aware filtering with ML-powered relevance scoring.

### With Vector Store Support

```bash
# ChromaDB (default)
pip install gweta[chroma]

# Qdrant
pip install gweta[qdrant]

# Pinecone
pip install gweta[pinecone]

# All stores
pip install gweta[stores]
```

### Full Installation

```bash
pip install gweta[all]
```

## Basic Usage

### 1. Intent-Aware Ingestion (Recommended)

The most powerful way to use Gweta is with intent-aware filtering:

```python
from gweta.intelligence import Pipeline, SystemIntent
from gweta import ChromaStore

# Define your system's intent
intent = SystemIntent(
    name="My Knowledge Base",
    description="Answers questions about Zimbabwe business",
    core_questions=[
        "How do I register a business in Zimbabwe?",
        "What are ZIMRA tax requirements?",
    ],
    relevant_topics=["Zimbabwe business", "ZIMRA", "EcoCash"],
    irrelevant_topics=["US regulations", "cryptocurrency", "forex"],
)

# Or load from YAML file
# intent = SystemIntent.from_yaml("intents/my_system.yaml")

# Create pipeline with store
store = ChromaStore(collection_name="my-kb")
pipeline = Pipeline(intent=intent, store=store)

# Ingest chunks - irrelevant content is automatically filtered
import asyncio

async def ingest_data(chunks):
    result = await pipeline.ingest(chunks)
    print(f"Ingested: {result.ingested} relevant chunks")
    print(f"Rejected: {result.rejected_count} irrelevant chunks")
    print(f"Acceptance rate: {result.acceptance_rate:.0%}")
    return result

asyncio.run(ingest_data(my_chunks))
```

### 2. Validate Chunks (Basic)

For simple validation without intent filtering:

```python
from gweta import ChunkValidator, Chunk

# Create some chunks
chunks = [
    Chunk(
        text="Zimbabwe's VAT rate is 15% as of 2024.",
        source="zimra-guide.pdf",
        metadata={"page": 1, "section": "Tax Rates"}
    ),
    Chunk(
        text="Short",  # Too short - will fail validation
        source="notes.txt",
        metadata={}
    ),
]

# Validate
validator = ChunkValidator()
report = validator.validate_batch(chunks)

# Check results
print(f"Total: {report.total_chunks}")
print(f"Passed: {report.passed}")
print(f"Failed: {report.failed}")
print(f"Average Quality: {report.avg_quality_score:.2f}")

# Get details on failures
for result in report.chunks:
    if not result.passed:
        print(f"  Failed: {result.chunk.source}")
        for issue in result.issues:
            print(f"    [{issue.severity}] {issue.message}")
```

### 2. Crawl a Website

```python
import asyncio
from gweta.acquire import GwetaCrawler

async def main():
    crawler = GwetaCrawler()

    result = await crawler.crawl(
        url="https://docs.example.com",
        depth=2,
        allowed_domains=["docs.example.com"],
    )

    print(f"Crawled {result.pages_crawled} pages")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Chunks: {len(result.chunks)} passed, {len(result.rejected_chunks)} rejected")

    return result

result = asyncio.run(main())
```

### 3. Load to Vector Store

```python
from gweta import ChromaStore

# Initialize store
store = ChromaStore(collection_name="my_docs")

# Add validated chunks
import asyncio

async def load_chunks(chunks):
    result = await store.add(chunks)
    print(f"Added: {result.added}")
    return result

asyncio.run(load_chunks(result.chunks))
```

### 4. Extract from PDF

```python
import asyncio
from gweta.acquire import PDFExtractor

async def extract_pdf():
    extractor = PDFExtractor()

    result = await extractor.extract(
        "document.pdf",
        extract_tables=True,
    )

    print(f"Pages: {len(result.pages)}")
    print(f"Tables: {len(result.tables)}")
    print(f"Quality Score: {result.quality_score:.2f}")

    return result

result = asyncio.run(extract_pdf())
```

### 5. Domain-Specific Validation

Create a rules file `rules/my_domain.yaml`:

```yaml
rules:
  - name: price_range
    description: Prices should be reasonable
    type: numerical_range
    condition:
      min: 0
      max: 100000
    severity: warning

  - name: required_date
    description: Documents must have dates
    type: required_field
    field: date
    severity: error

known_facts:
  - key: vat_rate
    value: 15
    source: ZIMRA 2024
    verified_date: "2024-01-01"
    tolerance: 0
```

Use in Python:

```python
from gweta.validate.rules import DomainRuleEngine

# Load rules
engine = DomainRuleEngine.from_yaml("rules/my_domain.yaml")

# Validate a chunk
result = engine.validate_chunk(chunk)
print(f"Passed: {result.passed}")
print(f"Score: {result.score:.2f}")

# Validate an AI response against known facts
response = "The VAT rate in Zimbabwe is 15%."
fact_result = engine.validate_response(response)
print(f"Facts checked: {fact_result.facts_checked}")
```

## CLI Usage

Gweta includes a command-line interface:

```bash
# Validate chunks from a JSON file
gweta validate chunks.json --threshold 0.7

# Crawl a website
gweta crawl https://example.com --depth 2

# Check KB health
gweta health my_collection

# Start MCP server
gweta serve
```

## MCP Integration

To use Gweta with Claude Desktop, add to your config:

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

Then Claude can use tools like:

- `crawl_and_ingest` - Crawl and load to vector store
- `validate_chunks` - Validate without loading
- `check_health` - Get KB health report

## Next Steps

- [Intelligence Layer Guide](guides/intelligence.md) - Deep dive into intent-aware filtering
- [Architecture Overview](concepts/architecture.md) - Understand how Gweta works
- [API Reference](api/reference.md) - Full API documentation
- [Full Pipeline Example](examples/full-pipeline.md) - Complete end-to-end example
