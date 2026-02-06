# Full Pipeline Example

This example demonstrates a complete ingestion pipeline combining multiple data sources with validation and loading.

## What This Example Does

1. Crawls a website for documentation
2. Extracts content from PDF files
3. Queries a database for articles
4. Validates all content with domain rules
5. Loads quality chunks to ChromaDB
6. Runs health checks on the knowledge base

## Prerequisites

```bash
pip install gweta[all]
```

## Complete Code

```python
import asyncio
from pathlib import Path

from gweta import ChunkValidator, Chunk, ChromaStore
from gweta.acquire import GwetaCrawler, PDFExtractor, DatabaseSource
from gweta.validate.rules import DomainRuleEngine, Rule, KnownFact
from gweta.validate.health import HealthChecker
from datetime import date


async def crawl_documentation(url: str) -> list[Chunk]:
    """Crawl a documentation website."""
    print(f"\n[1/5] Crawling {url}...")

    crawler = GwetaCrawler()

    try:
        result = await crawler.crawl(
            url=url,
            depth=2,
            follow_pdfs=True,
        )

        print(f"  Pages crawled: {result.pages_crawled}")
        print(f"  Quality score: {result.quality_score:.2f}")
        print(f"  Chunks: {len(result.chunks)} passed")

        return result.chunks

    except Exception as e:
        print(f"  Crawl failed: {e}")
        return []


async def extract_pdfs(pdf_paths: list[Path]) -> list[Chunk]:
    """Extract content from PDF files."""
    print(f"\n[2/5] Extracting {len(pdf_paths)} PDFs...")

    extractor = PDFExtractor()
    all_chunks = []

    for pdf_path in pdf_paths:
        try:
            result = await extractor.extract(
                source=pdf_path,
                extract_tables=True,
            )

            print(f"  {pdf_path.name}: {len(result.pages)} pages")
            all_chunks.extend(result.chunks)

        except Exception as e:
            print(f"  {pdf_path.name}: Failed - {e}")

    return all_chunks


async def query_database(dsn: str, query: str) -> list[Chunk]:
    """Extract content from database."""
    print(f"\n[3/5] Querying database...")

    try:
        async with DatabaseSource(dsn, read_only=True) as db:
            chunks = await db.extract_and_validate(
                query=query,
                text_column="content",
                metadata_columns=["title", "author", "published_date"],
            )

            print(f"  Retrieved {len(chunks)} chunks")
            return chunks

    except Exception as e:
        print(f"  Database query failed: {e}")
        return []


def setup_domain_rules() -> DomainRuleEngine:
    """Create domain-specific validation rules."""
    rules = [
        Rule(
            name="content_length",
            description="Content should be substantial",
            rule_type="numerical_range",
            field="text_length",
            condition={"min": 100, "max": 10000},
            severity="warning",
            message="Content length outside expected range",
        ),
        Rule(
            name="has_source",
            description="All chunks must have a source",
            rule_type="required_field",
            field="source",
            severity="error",
            message="Missing source information",
        ),
    ]

    known_facts = [
        KnownFact(
            key="framework_version",
            value="1.0",
            source="Official Documentation",
            verified_date=date.today(),
        ),
    ]

    return DomainRuleEngine(rules=rules, known_facts=known_facts)


async def validate_and_filter(
    chunks: list[Chunk],
    rule_engine: DomainRuleEngine,
) -> list[Chunk]:
    """Validate chunks and filter out failures."""
    print(f"\n[4/5] Validating {len(chunks)} chunks...")

    # Basic validation
    validator = ChunkValidator()
    report = validator.validate_batch(chunks)

    print(f"  Basic validation: {report.passed}/{report.total_chunks} passed")
    print(f"  Average quality: {report.avg_quality_score:.2f}")

    # Filter to passed chunks
    passed_chunks = [
        r.chunk for r in report.chunks
        if r.passed and r.quality_score >= 0.6
    ]

    # Apply domain rules
    final_chunks = []
    domain_failures = 0

    for chunk in passed_chunks:
        result = rule_engine.validate_chunk(chunk)
        if result.passed:
            final_chunks.append(chunk)
        else:
            domain_failures += 1

    print(f"  Domain rules: {len(final_chunks)} passed, {domain_failures} failed")
    return final_chunks


async def load_to_store(chunks: list[Chunk], collection_name: str) -> ChromaStore:
    """Load validated chunks to ChromaDB."""
    print(f"\n[5/5] Loading {len(chunks)} chunks to '{collection_name}'...")

    store = ChromaStore(collection_name=collection_name)
    result = await store.add(chunks)

    print(f"  Added: {result.added}")
    print(f"  Skipped: {result.skipped}")

    return store


async def run_health_check(store: ChromaStore):
    """Run health check on the knowledge base."""
    print("\n[Bonus] Running health check...")

    checker = HealthChecker(store)
    report = await checker.full_health_check()

    print(f"  Total chunks: {report.total_chunks}")
    print(f"  Average quality: {report.avg_quality_score:.2f}")
    print(f"  Duplicates found: {report.duplicates.duplicate_groups}")


async def main():
    """Run the complete pipeline."""
    print("=" * 60)
    print("Gweta Full Pipeline Example")
    print("=" * 60)

    # Configuration
    collection_name = "example_kb"
    docs_url = "https://docs.python.org/3/tutorial/"

    # Collect chunks from all sources
    all_chunks = []

    # Web crawling
    try:
        web_chunks = await crawl_documentation(docs_url)
        all_chunks.extend(web_chunks)
    except Exception:
        print("  Skipping web crawl")

    # If no chunks, create demo chunks
    if not all_chunks:
        print("\n[Demo] Creating sample chunks...")
        all_chunks = [
            Chunk(
                text="Python is a high-level programming language known for its clear syntax.",
                source="python-docs",
                metadata={"topic": "introduction"},
            ),
            Chunk(
                text="Lists in Python are ordered, mutable sequences.",
                source="python-docs",
                metadata={"topic": "data-structures"},
            ),
        ]

    # Set up domain rules
    rule_engine = setup_domain_rules()

    # Validate and filter
    valid_chunks = await validate_and_filter(all_chunks, rule_engine)

    if not valid_chunks:
        print("\nNo valid chunks to load.")
        return

    # Load to vector store
    store = await load_to_store(valid_chunks, collection_name)

    # Run health check
    await run_health_check(store)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
```

## Running the Example

```bash
python examples/full_pipeline.py
```

## Expected Output

```
============================================================
Gweta Full Pipeline Example
============================================================

[1/5] Crawling https://docs.python.org/3/tutorial/...
  Pages crawled: 15
  Quality score: 0.87
  Chunks: 142 passed

[4/5] Validating 142 chunks...
  Basic validation: 138/142 passed
  Average quality: 0.82
  Domain rules: 135 passed, 3 failed

[5/5] Loading 135 chunks to 'example_kb'...
  Added: 135
  Skipped: 0

[Bonus] Running health check...
  Total chunks: 135
  Average quality: 0.82
  Duplicates found: 0

============================================================
Pipeline Complete!
============================================================
```

## Next Steps

- Add custom domain rules for your use case
- Configure golden dataset testing for retrieval quality
- Set up scheduled health checks
- Integrate with your AI agent via MCP
